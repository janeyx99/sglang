#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Half.h>

#include "gemm/gemm_ops.h"
#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

#define CHECK_CUDA(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_EQ(a, b) STD_TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)
#define SGL_CURRENT_CUDA_STREAM() sgl_kernel::stable::get_current_cuda_stream()
#define SGL_INPUT_PTR(tensor, type) stable_input_ptr<type>(tensor)
#define SGL_OUTPUT_Q_PTR(tensor, type) stable_output_q_ptr<type>(tensor)
#define SGL_OUTPUT_S_PTR(tensor, type) stable_output_s_ptr<type>(tensor)
#ifdef FLASHINFER_ENABLE_BF16
#define SGL_DISPATCH_CASE_BF16(c_type, ...) \
  case ScalarType::BFloat16: {              \
    using c_type = nv_bfloat16;             \
    return __VA_ARGS__();                   \
  }
#else
#define SGL_DISPATCH_CASE_BF16(c_type, ...)
#endif
#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...)                                                       \
  [&]() -> bool {                                                                                      \
    const auto scalar_type = TYPE;                                                                     \
    switch (scalar_type) {                                                                             \
      case ScalarType::Float: {                                                                        \
        using scalar_t = float;                                                                        \
        return __VA_ARGS__();                                                                          \
      }                                                                                                \
      case ScalarType::Half: {                                                                         \
        using scalar_t = nv_half;                                                                      \
        return __VA_ARGS__();                                                                          \
      }                                                                                                \
        SGL_DISPATCH_CASE_BF16(scalar_t, __VA_ARGS__)                                                  \
      default:                                                                                         \
        STD_TORCH_CHECK(                                                                               \
            false,                                                                                     \
            NAME                                                                                       \
            "(at::Tensor, at::Tensor, at::Tensor, int64_t, double, double, double, bool)::<lambda()> " \
            "failed to "                                                                               \
            "dispatch data type ",                                                                     \
            torch::headeronly::toString(scalar_type));                                                 \
        return false;                                                                                  \
    }                                                                                                  \
  }()

template <typename T>
const T* stable_input_ptr(const Tensor& tensor) {
  if constexpr (std::is_same_v<T, nv_half>) {
    return reinterpret_cast<const T*>(tensor.const_data_ptr<torch::headeronly::Half>());
  } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
    return reinterpret_cast<const T*>(tensor.const_data_ptr<torch::headeronly::BFloat16>());
  } else {
    return tensor.const_data_ptr<T>();
  }
}

template <typename T>
void* stable_output_q_ptr(Tensor& tensor) {
  if constexpr (std::is_same_v<T, int8_t>) {
    return tensor.mutable_data_ptr<int8_t>();
  } else {
    static_assert(std::is_same_v<T, __nv_fp8_e4m3>);
    return reinterpret_cast<T*>(tensor.mutable_data_ptr<torch::headeronly::Float8_e4m3fn>());
  }
}

template <typename T>
T* stable_output_s_ptr(Tensor& tensor) {
  if constexpr (std::is_same_v<T, uint32_t>) {
    return reinterpret_cast<T*>(tensor.mutable_data_ptr<int32_t>());
  } else {
    static_assert(std::is_same_v<T, float>);
    return tensor.mutable_data_ptr<float>();
  }
}
#else
#include <ATen/cuda/CUDAContext.h>

#include "utils.h"

using Tensor = torch::Tensor;
using ScalarType = at::ScalarType;

#define SGL_CURRENT_CUDA_STREAM() at::cuda::getCurrentCUDAStream()
#define SGL_INPUT_PTR(tensor, type) static_cast<type*>(tensor.data_ptr())
#define SGL_OUTPUT_Q_PTR(tensor, type) tensor.data_ptr()
#define SGL_OUTPUT_S_PTR(tensor, type) static_cast<type*>(tensor.data_ptr())
#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...) DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(TYPE, scalar_t, __VA_ARGS__)
#endif

#include <cuda_fp8.h>

#include <cassert>
#include <cmath>
#include <flashinfer/vec_dtypes.cuh>
#include <type_traits>

__device__ __forceinline__ float GroupReduceMax(float val, const int tid) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

template <
    typename T,
    typename DST_DTYPE,
    bool IS_COLUMN_MAJOR = false,
    bool SCALE_UE8M0 = false,
    typename scale_packed_t = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int num_groups_per_row = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;
  const int64_t block_group_offset = global_group_id * group_size;

  float local_absmax = eps;

  using scale_element_t = std::conditional_t<SCALE_UE8M0, uint8_t, float>;
  static_assert(sizeof(scale_packed_t) % sizeof(scale_element_t) == 0);

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output = static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack = static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int row_idx = global_group_id / num_groups_per_row;
    const int col_idx_unpacked = global_group_id % num_groups_per_row;
    const int col_idx = col_idx_unpacked / num_elems_per_pack;
    const int pack_idx = col_idx_unpacked % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
  } else {
    static_assert(!SCALE_UE8M0);
    scale_output = output_s + global_group_id;
  }

  constexpr uint32_t vec_size = 16 / sizeof(T);
  using vec_t = flashinfer::vec_t<T, vec_size>;

  const int32_t num_vec_elems = group_size / vec_size;

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float abs_val = fabsf(val);
      local_absmax = fmaxf(local_absmax, abs_val);
    }
  }

  local_absmax = GroupReduceMax(local_absmax, lane_id);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(y_s, 1e-10f))));
  }

  // TODO can optimize
  scale_element_t y_s_quant;
  if constexpr (SCALE_UE8M0) {
    y_s_quant = (uint8_t)(((int)log2f(y_s)) + 127);
  } else {
    y_s_quant = y_s;
  }

  if (lane_id == 0) {
    *scale_output = y_s_quant;
  }

  for (int32_t i = lane_id; i < num_vec_elems; i += 16) {
    vec_t input_vec;
    input_vec.cast_load(group_input + i * vec_size);

#pragma unroll
    for (uint32_t j = 0; j < vec_size; ++j) {
      float val = static_cast<float>(input_vec[j]);
      float q_val = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
      group_output[i * vec_size + j] = DST_DTYPE(q_val);
    }
  }
}

void sgl_per_token_group_quant_8bit(
    Tensor input,
    Tensor output_q,
    Tensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);

  const int num_groups = input.numel() / group_size;

  CHECK_EQ(input.numel() % group_size, 0);
  CHECK_EQ(output_s.dim(), 2);

  cudaStream_t stream = SGL_CURRENT_CUDA_STREAM();

  constexpr int THREADS_PER_GROUP = 16;

  int groups_per_block = 1;

  if (num_groups % 16 == 0) {
    groups_per_block = 16;
  } else if (num_groups % 8 == 0) {
    groups_per_block = 8;
  } else if (num_groups % 4 == 0) {
    groups_per_block = 4;
  } else if (num_groups % 2 == 0) {
    groups_per_block = 2;
  }

  auto dst_type = output_q.scalar_type();
  const int num_blocks = num_groups / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int hidden_dim = input.size(input.dim() - 1);
  const int num_groups_per_row = hidden_dim / group_size;
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                                               \
  do {                                                                                            \
    dim3 grid(num_blocks);                                                                        \
    dim3 block(num_threads);                                                                      \
    if (is_column_major) {                                                                        \
      if (scale_ue8m0) {                                                                          \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true><<<grid, block, 0, stream>>>(  \
            SGL_INPUT_PTR(input, T),                                                              \
            SGL_OUTPUT_Q_PTR(output_q, DST_DTYPE),                                                \
            SGL_OUTPUT_S_PTR(output_s, uint32_t),                                                 \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            num_groups_per_row,                                                                   \
            scale_stride);                                                                        \
      } else {                                                                                    \
        per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false><<<grid, block, 0, stream>>>( \
            SGL_INPUT_PTR(input, T),                                                              \
            SGL_OUTPUT_Q_PTR(output_q, DST_DTYPE),                                                \
            SGL_OUTPUT_S_PTR(output_s, float),                                                    \
            group_size,                                                                           \
            num_groups,                                                                           \
            groups_per_block,                                                                     \
            (float)eps,                                                                           \
            (float)min_8bit,                                                                      \
            (float)max_8bit,                                                                      \
            num_groups_per_row,                                                                   \
            scale_stride);                                                                        \
      }                                                                                           \
    } else {                                                                                      \
      assert(!scale_ue8m0);                                                                       \
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, false><<<grid, block, 0, stream>>>(         \
          SGL_INPUT_PTR(input, T),                                                                \
          SGL_OUTPUT_Q_PTR(output_q, DST_DTYPE),                                                  \
          SGL_OUTPUT_S_PTR(output_s, float),                                                      \
          group_size,                                                                             \
          num_groups,                                                                             \
          groups_per_block,                                                                       \
          (float)eps,                                                                             \
          (float)min_8bit,                                                                        \
          (float)max_8bit);                                                                       \
    }                                                                                             \
  } while (0)

  SGL_DISPATCH_FLOAT_FP16(input.scalar_type(), "sgl_per_token_group_quant_8bit", [&] {
    if (dst_type == ScalarType::Char) {
      LAUNCH_KERNEL(scalar_t, int8_t);
      return true;
    } else if (dst_type == ScalarType::Float8_e4m3fn) {
      LAUNCH_KERNEL(scalar_t, __nv_fp8_e4m3);
      return true;
    }
    return false;
  });

#undef LAUNCH_KERNEL
}

#ifdef TORCH_TARGET_VERSION
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef CHECK_EQ
#undef SGL_DISPATCH_CASE_BF16
#endif
#undef SGL_CURRENT_CUDA_STREAM
#undef SGL_INPUT_PTR
#undef SGL_OUTPUT_Q_PTR
#undef SGL_OUTPUT_S_PTR
#undef SGL_DISPATCH_FLOAT_FP16
