#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "attention/attention_ops.h"

#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

#include "sgl_kernel_cuda_stream.h"
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "pytorch_extension_utils.h"
#endif

namespace {

inline cudaStream_t GetCurrentCUDAStream(const SglTensor& tensor) {
#ifdef TORCH_TARGET_VERSION
  return sgl_kernel::stable::get_current_cuda_stream(tensor.get_device_index());
#else
  return at::cuda::getCurrentCUDAStream();
#endif
}

}  // namespace

// Helper functions to convert between different data types
// (float, half, bfloat16) for the merge attention states kernel.
inline __device__ float to_float(float u) {
  return u;
}
inline __device__ float to_float(half u) {
  return __half2float(u);
}
inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}
inline __device__ void from_float(float& d, float s) {
  d = s;
}
inline __device__ void from_float(half& d, float s) {
  d = __float2half(s);
}
inline __device__ void from_float(__nv_bfloat16& d, float s) {
  d = __float2bfloat16(s);
}

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
template <typename scalar_t, const uint NUM_THREADS>
__global__ void merge_attn_states_kernel(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    const uint num_tokens,
    const uint num_heads,
    const uint head_size) {
  using pack_128b_t = uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint head_offset = token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;

  // float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  // float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  float p_lse = prefix_lse[token_idx * num_heads + head_idx];
  float s_lse = suffix_lse[token_idx * num_heads + head_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      // float -> half(uint16_t), bfloat16, float.
      from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[token_idx * num_heads + head_idx] = out_lse;
  }
}

// The following macro is used to dispatch the conversion function based on
// the output data type. The FN is a macro that calls a function with
// template<typename scalar_t>.
#define DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                          \
  {                                                                         \
    if (scalar_dtype == SglScalarType::Float) {                             \
      fn(float);                                                            \
    } else if (scalar_dtype == SglScalarType::Half) {                       \
      fn(half);                                                             \
    } else if (scalar_dtype == SglScalarType::BFloat16) {                   \
      fn(__nv_bfloat16);                                                    \
    } else {                                                                \
      SGL_TORCH_CHECK(false, "Unsupported data type of O: ", scalar_dtype); \
    }                                                                       \
  }

#define LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS)                          \
  {                                                                              \
    merge_attn_states_kernel<scalar_t, NUM_THREADS><<<grid, block, 0, stream>>>( \
        reinterpret_cast<scalar_t*>(SGL_MUTABLE_DATA_PTR(output)),               \
        reinterpret_cast<float*>(SGL_MUTABLE_DATA_PTR(output_lse)),              \
        reinterpret_cast<const scalar_t*>(SGL_CONST_DATA_PTR(prefix_output)),    \
        reinterpret_cast<const float*>(SGL_CONST_DATA_PTR(prefix_lse)),          \
        reinterpret_cast<const scalar_t*>(SGL_CONST_DATA_PTR(suffix_output)),    \
        reinterpret_cast<const float*>(SGL_CONST_DATA_PTR(suffix_lse)),          \
        num_tokens,                                                              \
        num_heads,                                                               \
        head_size);                                                              \
  }

/*@brief Merges the attention states from prefix and suffix
 * into the output tensor. NUM_TOKENS: n, NUM_HEADS: h, HEAD_SIZE: d
 *
 * @param output [n,h,d] The output tensor to store the merged attention states.
 * @param output_lse [h,d] Optional tensor to store the log-sum-exp values.
 * @param prefix_output [n,h,d] The prefix attention states.
 * @param prefix_lse [n,h] The log-sum-exp values for the prefix attention
 * states.
 * @param suffix_output [n,h,d] The suffix attention states.
 * @param suffix_lse [n,h] The log-sum-exp values for the suffix attention
 * states.
 */
template <typename scalar_t>
void merge_attn_states_launcher(
    const SglTensor& prefix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const SglTensor& prefix_lse,     // [NUM_TOKENS, NUM_HEADS]
    const SglTensor& suffix_output,  // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    const SglTensor& suffix_lse,     // [NUM_TOKENS, NUM_HEADS]
    SglTensor& output,               // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    SglTensor& output_lse            // [NUM_TOKENS, NUM_HEADS]
) {
  constexpr uint NUM_THREADS = 128;
  const uint num_tokens = output.size(0);
  const uint num_heads = output.size(1);
  const uint head_size = output.size(2);
  const uint pack_size = 16 / sizeof(scalar_t);
  SGL_TORCH_CHECK(head_size % pack_size == 0, "headsize must be multiple of pack_size:", pack_size);
  // Process one pack elements per thread. for float, the
  // pack_size is 4 for half/bf16, the pack_size is 8.
  const uint threads_per_head = head_size / pack_size;
  const uint total_threads = num_tokens * num_heads * threads_per_head;

  dim3 block(NUM_THREADS);
  dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);

#ifdef TORCH_TARGET_VERSION
  const torch::stable::accelerator::DeviceGuard device_guard(prefix_output.get_device_index());
#else
  const c10::cuda::OptionalCUDAGuard device_guard(prefix_output.device());
#endif
  auto stream = GetCurrentCUDAStream(prefix_output);

  LAUNCH_MERGE_ATTN_STATES(scalar_t, NUM_THREADS);
}

#define CALL_MERGE_ATTN_STATES_LAUNCHER(scalar_t)                                 \
  {                                                                               \
    merge_attn_states_launcher<scalar_t>(v_a, s_a, v_b, s_b, v_merged, s_merged); \
  }

void merge_state_v2(
    SglTensor v_a, SglTensor s_a, SglTensor v_b, SglTensor s_b, SglTensor v_merged, SglTensor s_merged) {
  // Input tensors must be contiguous
#ifdef TORCH_TARGET_VERSION
  STD_TORCH_CHECK(v_a.is_cuda(), "v_a must be a CUDA tensor");
  STD_TORCH_CHECK(v_a.is_contiguous(), "v_a must be contiguous");
  STD_TORCH_CHECK(s_a.is_cuda(), "s_a must be a CUDA tensor");
  STD_TORCH_CHECK(s_a.is_contiguous(), "s_a must be contiguous");
  STD_TORCH_CHECK(v_b.is_cuda(), "v_b must be a CUDA tensor");
  STD_TORCH_CHECK(v_b.is_contiguous(), "v_b must be contiguous");
  STD_TORCH_CHECK(s_b.is_cuda(), "s_b must be a CUDA tensor");
  STD_TORCH_CHECK(s_b.is_contiguous(), "s_b must be contiguous");
  const auto device_index = v_a.get_device_index();
  const auto s_a_device_index = s_a.get_device_index();
  const auto v_b_device_index = v_b.get_device_index();
  const auto s_b_device_index = s_b.get_device_index();
  STD_TORCH_CHECK(
      s_a_device_index == device_index,
      "CHECK_EQ(s_a.device(), device) failed. cuda:",
      s_a_device_index,
      " vs cuda:",
      device_index);
  STD_TORCH_CHECK(
      v_b_device_index == device_index,
      "CHECK_EQ(v_b.device(), device) failed. cuda:",
      v_b_device_index,
      " vs cuda:",
      device_index);
  STD_TORCH_CHECK(
      s_b_device_index == device_index,
      "CHECK_EQ(s_b.device(), device) failed. cuda:",
      s_b_device_index,
      " vs cuda:",
      device_index);
  const auto v_a_dim = v_a.dim();
  const auto s_a_dim = s_a.dim();
  const auto v_b_dim = v_b.dim();
  const auto s_b_dim = s_b.dim();
  STD_TORCH_CHECK(v_a_dim == 3, "v_a must be a 3D tensor");
  STD_TORCH_CHECK(s_a_dim == 2, "s_a must be a 2D tensor");
  STD_TORCH_CHECK(v_b_dim == 3, "v_b must be a 3D tensor");
  STD_TORCH_CHECK(s_b_dim == 2, "s_b must be a 2D tensor");
  STD_TORCH_CHECK(v_a_dim == v_b_dim, "v_a.dim() != v_b.dim(). ", v_a_dim, " vs ", v_b_dim);
  const auto v_a_sizes = v_a.sizes();
  const auto v_b_sizes = v_b.sizes();
  for (int i = 0; i < v_a_dim; ++i) {
    STD_TORCH_CHECK(v_a_sizes[i] == v_b_sizes[i], "v_a.size(", i, ") != v_b.size(", i, ")");
  }
  STD_TORCH_CHECK(s_a_dim == s_b_dim, "s_a.dim() != s_b.dim(). ", s_a_dim, " vs ", s_b_dim);
  const auto s_a_sizes = s_a.sizes();
  const auto s_b_sizes = s_b.sizes();
  for (int i = 0; i < s_a_dim; ++i) {
    STD_TORCH_CHECK(s_a_sizes[i] == s_b_sizes[i], "s_a.size(", i, ") != s_b.size(", i, ")");
  }
  const auto v_a_tokens = v_a_sizes[0];
  const auto s_a_tokens = s_a_sizes[0];
  STD_TORCH_CHECK(
      v_a_tokens == s_a_tokens, "CHECK_EQ(v_a.size(0), s_a.size(0)) failed. ", v_a_tokens, " vs ", s_a_tokens);
  const auto v_a_heads = v_a_sizes[1];
  const auto s_b_heads = s_b_sizes[1];
  STD_TORCH_CHECK(v_a_heads == s_b_heads, "CHECK_EQ(v_a.size(1), s_b.size(1)) failed. ", v_a_heads, " vs ", s_b_heads);
#else
  CHECK_INPUT(v_a);  // v_a prefix_output (seq_len, num_heads, head_dim)
  CHECK_INPUT(s_a);  // s_a prefix_lse (seq_len, num_heads)
  CHECK_INPUT(v_b);  // v_b suffix_output (seq_len, num_heads, head_dim)
  CHECK_INPUT(s_b);  // s_b suffix_lse (seq_len, num_heads)
  // v_merged output (seq_len, num_heads, head_dim)
  // s_merged output_lse (seq_len, num_heads)
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
#endif
#ifdef TORCH_TARGET_VERSION
  const auto output_scalar_type = v_merged.scalar_type();
  if (output_scalar_type == SglScalarType::Float) {
    CALL_MERGE_ATTN_STATES_LAUNCHER(float);
  } else if (output_scalar_type == SglScalarType::Half) {
    CALL_MERGE_ATTN_STATES_LAUNCHER(half);
  } else if (output_scalar_type == SglScalarType::BFloat16) {
    CALL_MERGE_ATTN_STATES_LAUNCHER(__nv_bfloat16);
  } else {
    const char* output_type_name = torch::headeronly::toString(output_scalar_type);
    switch (output_scalar_type) {
      case SglScalarType::Byte:
        output_type_name = "unsigned char";
        break;
      case SglScalarType::Char:
        output_type_name = "signed char";
        break;
      case SglScalarType::Short:
        output_type_name = "short int";
        break;
      case SglScalarType::Int:
        output_type_name = "int";
        break;
      case SglScalarType::Long:
        output_type_name = "long int";
        break;
      case SglScalarType::Double:
        output_type_name = "double";
        break;
      case SglScalarType::Bool:
        output_type_name = "bool";
        break;
      case SglScalarType::ComplexFloat:
        output_type_name = "c10::complex<float>";
        break;
      case SglScalarType::ComplexDouble:
        output_type_name = "c10::complex<double>";
        break;
      default:
        break;
    }
    STD_TORCH_CHECK(false, "Unsupported data type of O: ", output_type_name);
  }
#else
  DISPATCH_BY_SCALAR_DTYPE(v_merged.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
#endif
}
