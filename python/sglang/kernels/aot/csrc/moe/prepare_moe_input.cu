#include <cudaTypedefs.h>

#include <algorithm>
#include <flashinfer/vec_dtypes.cuh>
#include <iostream>
#include <optional>
#include <type_traits>

#include "cutlass/array.h"

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include "moe/moe_ops.h"
#undef SGL_CONST_DATA_PTR
#undef SGL_MUTABLE_DATA_PTR
#include "sgl_kernel_cuda_stream.h"

using TorchTensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Half = torch::headeronly::Half;
using BFloat16 = torch::headeronly::BFloat16;
using OptionalTensor = std::optional<TorchTensor>;

#define SGL_CHECK(...) STD_TORCH_CHECK(__VA_ARGS__)
#define SGL_CURRENT_CUDA_STREAM() sgl_kernel::stable::get_current_cuda_stream()
#define SGL_TENSOR_CUDA_STREAM(tensor_) sgl_kernel::stable::get_current_cuda_stream((tensor_).get_device_index())
#define SGL_NEW_ZEROS_INT32(self_, size_) \
  torch::stable::new_zeros((self_), {(size_)}, ScalarType::Int, torch::headeronly::Layout::Strided)
#define SGL_CONST_DATA_PTR(tensor_, type_) (tensor_).const_data_ptr<type_>()
#define SGL_MUTABLE_DATA_PTR(tensor_, type_) (tensor_).mutable_data_ptr<type_>()
#define SGL_CONST_UNTYPED_DATA_PTR(tensor_) (tensor_).const_data_ptr()
#define SGL_MUTABLE_UNTYPED_DATA_PTR(tensor_) (tensor_).mutable_data_ptr()

template <typename T>
struct stable_cuda_type {
  using type = T;
};

template <>
struct stable_cuda_type<Half> {
  using type = nv_half;
};

template <>
struct stable_cuda_type<BFloat16> {
  using type = nv_bfloat16;
};

#define SGL_CUDA_TYPE(type_) typename stable_cuda_type<type_>::type

#ifdef FLASHINFER_ENABLE_F16
#define SGL_DISPATCH_CASE_F16(...) THO_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__)
#else
#define SGL_DISPATCH_CASE_F16(...)
#endif

#ifdef FLASHINFER_ENABLE_BF16
#define SGL_DISPATCH_CASE_BF16(...) THO_DISPATCH_CASE(ScalarType::BFloat16, __VA_ARGS__)
#else
#define SGL_DISPATCH_CASE_BF16(...)
#endif

#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...)                                           \
  THO_DISPATCH_SWITCH(                                                                     \
      TYPE,                                                                                \
      NAME,                                                                                \
      THO_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__) SGL_DISPATCH_CASE_F16(__VA_ARGS__) \
          SGL_DISPATCH_CASE_BF16(__VA_ARGS__))
#else
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>

#include "utils.h"

using TorchTensor = torch::Tensor;
using ScalarType = at::ScalarType;
using OptionalTensor = std::optional<TorchTensor>;

#define SGL_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#define SGL_CURRENT_CUDA_STREAM() at::cuda::getCurrentCUDAStream().stream()
#define SGL_TENSOR_CUDA_STREAM(tensor_) at::cuda::getCurrentCUDAStream((tensor_).device().index())
#define SGL_NEW_ZEROS_INT32(self_, size_) \
  torch::zeros((size_), torch::TensorOptions().dtype(torch::kInt32).device((self_).device()))
#define SGL_CONST_DATA_PTR(tensor_, type_) (tensor_).data_ptr<type_>()
#define SGL_MUTABLE_DATA_PTR(tensor_, type_) (tensor_).data_ptr<type_>()
#define SGL_CONST_UNTYPED_DATA_PTR(tensor_) (tensor_).data_ptr()
#define SGL_MUTABLE_UNTYPED_DATA_PTR(tensor_) (tensor_).data_ptr()
#define SGL_CUDA_TYPE(type_) type_
#define SGL_DISPATCH_FLOAT_FP16(TYPE, NAME, ...) DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(TYPE, scalar_t, __VA_ARGS__)
#endif

constexpr uint64_t THREADS_PER_EXPERT = 512;

__global__ void compute_problem_sizes(
    const int* __restrict__ topk_ids,
    int32_t* problem_sizes1,
    int32_t* problem_sizes2,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t n,
    const int64_t k) {
  int expert_id = blockIdx.x;

  int occurrences = 0;
  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    occurrences += (topk_ids[i] == expert_id);
  }
  atomicAdd(&atomic_buffer[expert_id], occurrences);
  __syncthreads();

  if (threadIdx.x == 0) {
    int final_occurrences = atomic_buffer[expert_id];
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = static_cast<int32_t>(2 * n);
    problem_sizes1[expert_id * 3 + 2] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3] = final_occurrences;
    problem_sizes2[expert_id * 3 + 1] = static_cast<int32_t>(k);
    problem_sizes2[expert_id * 3 + 2] = static_cast<int32_t>(n);
  }
}

__global__ void compute_expert_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  expert_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    tot_offset += problem_sizes1[i * 3];
    expert_offsets[i + 1] = tot_offset;
  }
}

__global__ void compute_expert_blockscale_offsets(
    const int32_t* __restrict__ problem_sizes1,
    int32_t* expert_offsets,
    int32_t* blockscale_offsets,
    int32_t* atomic_buffer,
    const int64_t num_experts) {
  int32_t tot_offset = 0;
  int32_t tot_rounded_offset = 0;
  expert_offsets[0] = 0;
  blockscale_offsets[0] = 0;
  for (int i = 0; i < num_experts; ++i) {
    atomic_buffer[i] = tot_offset;
    int num_tokens = problem_sizes1[i * 3];
    int rounded_num_tokens = (num_tokens + (128 - 1)) / 128 * 128;
    tot_offset += num_tokens;
    tot_rounded_offset += rounded_num_tokens;
    expert_offsets[i + 1] = tot_offset;
    blockscale_offsets[i + 1] = tot_rounded_offset;
  }
}

__global__ void compute_arg_sorts(
    const int32_t* __restrict__ topk_ids,
    int32_t* input_permutation,
    int32_t* output_permutation,
    int32_t* atomic_buffer,
    const int64_t topk_length,
    const int64_t topk) {
  int expert_id = blockIdx.x;

  for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
    if (topk_ids[i] == expert_id) {
      int start = atomicAdd(&atomic_buffer[expert_id], 1);
      input_permutation[start] = i / topk;
      output_permutation[i] = start;
    }
  }
}

void get_moe_prepare_input_caller(
    const TorchTensor& topk_ids,
    TorchTensor& expert_offsets,
    const OptionalTensor& blockscale_offsets,
    TorchTensor& problem_sizes1,
    TorchTensor& problem_sizes2,
    TorchTensor& input_permutation,
    TorchTensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  auto stream = SGL_TENSOR_CUDA_STREAM(topk_ids);
  TorchTensor atomic_buffer = SGL_NEW_ZEROS_INT32(topk_ids, num_experts);

  uint32_t num_threads = static_cast<uint32_t>(min(THREADS_PER_EXPERT, topk_ids.numel()));
  uint32_t num_blocks = static_cast<uint32_t>(num_experts);

  compute_problem_sizes<<<num_blocks, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(SGL_CONST_UNTYPED_DATA_PTR(topk_ids)),
      static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(problem_sizes1)),
      static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(problem_sizes2)),
      SGL_MUTABLE_DATA_PTR(atomic_buffer, int32_t),
      topk_ids.numel(),
      n,
      k);
  if (blockscale_offsets.has_value()) {
    compute_expert_blockscale_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(SGL_CONST_UNTYPED_DATA_PTR(problem_sizes1)),
        static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(expert_offsets)),
        static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(blockscale_offsets.value())),
        SGL_MUTABLE_DATA_PTR(atomic_buffer, int32_t),
        num_experts);
  } else {
    compute_expert_offsets<<<1, 1, 0, stream>>>(
        static_cast<const int32_t*>(SGL_CONST_UNTYPED_DATA_PTR(problem_sizes1)),
        static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(expert_offsets)),
        SGL_MUTABLE_DATA_PTR(atomic_buffer, int32_t),
        num_experts);
  }
  compute_arg_sorts<<<num_blocks, num_threads, 0, stream>>>(
      static_cast<const int32_t*>(SGL_CONST_UNTYPED_DATA_PTR(topk_ids)),
      static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(input_permutation)),
      static_cast<int32_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(output_permutation)),
      SGL_MUTABLE_DATA_PTR(atomic_buffer, int32_t),
      topk_ids.numel(),
      topk_ids.size(1));
}

void prepare_moe_input(
    const TorchTensor& topk_ids,
    TorchTensor& expert_offsets,
    const OptionalTensor& blockscale_offsets,
    TorchTensor& problem_sizes1,
    TorchTensor& problem_sizes2,
    TorchTensor& input_permutation,
    TorchTensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
#ifdef TORCH_TARGET_VERSION
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == ScalarType::Int,
      "Expected topk_ids.dtype() == torch::kInt32 to be true, but got false.  (Could this error message be "
      "improved?  If so, please report an enhancement request to PyTorch.)");
#else
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32);
#endif
  get_moe_prepare_input_caller(
      topk_ids,
      expert_offsets,
      blockscale_offsets,
      problem_sizes1,
      problem_sizes2,
      input_permutation,
      output_permutation,
      num_experts,
      n,
      k);
  return;
}

template <typename T>
__global__ void shuffleRowsKernel(
    const T* input,
    const int32_t* dst2src_map,
    T* output,
    int64_t num_src_rows,
    int64_t num_dst_rows,
    int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  int64_t const source_row_idx = dst2src_map[dest_row_idx];

  if (blockIdx.x < num_dst_rows) {
    // Load 128-bits per thread
    constexpr uint64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    // Duplicate and permute rows
    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    auto* dest_row_ptr = reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);

    auto const start_offset = threadIdx.x;
    auto const stride = blockDim.x;
    auto const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    for (auto elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

#define DECLARE_SHUFFLE_ROWS(T)      \
  __global__ void shuffleRowsKernel( \
      const T* input,                \
      const int32_t* dst2src_map,    \
      T* output,                     \
      int64_t num_src_rows,          \
      int64_t num_dest_rows,         \
      int64_t num_cols);

DECLARE_SHUFFLE_ROWS(float);
DECLARE_SHUFFLE_ROWS(half);
DECLARE_SHUFFLE_ROWS(__nv_bfloat16);
DECLARE_SHUFFLE_ROWS(__nv_fp8_e4m3);
DECLARE_SHUFFLE_ROWS(uint8_t);

#define SHUFFLE_ROWS(T)                                                     \
  shuffleRowsKernel<T><<<blocks, threads, 0, stream>>>(                     \
      reinterpret_cast<const T*>(input),                                    \
      static_cast<const int32_t*>(SGL_CONST_UNTYPED_DATA_PTR(dst2src_map)), \
      reinterpret_cast<T*>(output),                                         \
      num_src_rows,                                                         \
      num_dst_rows,                                                         \
      num_cols)

#define DTYPE_DISPATCH_CASE(T, CUDA_T) \
  case T:                              \
    SHUFFLE_ROWS(CUDA_T);              \
    break;

void shuffle_rows_caller(const TorchTensor& input_tensor, const TorchTensor& dst2src_map, TorchTensor& output_tensor) {
  SGL_CHECK(
      input_tensor.scalar_type() == output_tensor.scalar_type(),
      "Input and output tensors must have the same data type");
  auto stream = SGL_CURRENT_CUDA_STREAM();
  uint32_t blocks = static_cast<uint32_t>(output_tensor.size(0));
  uint32_t threads = 256;
  int64_t num_dst_rows = output_tensor.size(0);
  int64_t num_src_rows = input_tensor.size(0);
  int64_t num_cols = input_tensor.size(1);
  const void* input = SGL_CONST_UNTYPED_DATA_PTR(input_tensor);
  void* output = SGL_MUTABLE_UNTYPED_DATA_PTR(output_tensor);
  switch (input_tensor.scalar_type()) {
    DTYPE_DISPATCH_CASE(ScalarType::Half, half);
    DTYPE_DISPATCH_CASE(ScalarType::BFloat16, __nv_bfloat16);
    DTYPE_DISPATCH_CASE(ScalarType::Float, float);
    DTYPE_DISPATCH_CASE(ScalarType::Float8_e4m3fn, __nv_fp8_e4m3);
    DTYPE_DISPATCH_CASE(ScalarType::Byte, uint8_t);
    default:
      SGL_CHECK(false, "[moe replicate input] data type dispatch fail!");
  }
  return;
}

void shuffle_rows(const TorchTensor& input_tensor, const TorchTensor& dst2src_map, TorchTensor& output_tensor) {
  shuffle_rows_caller(input_tensor, dst2src_map, output_tensor);
  return;
}

template <typename scalar_t>
__global__ void apply_shuffle_mul_sum_kernel(
    const scalar_t* __restrict__ input_tensor,  // [m * topk, k] (expert-major layout)
    scalar_t* __restrict__ output_tensor,       // [m, k] (token-major layout)
    const int32_t* __restrict__ permutation,    // [m * topk] (c_map: token-major-idx -> expert-major-idx)
    int m,
    int topk,
    int row_stride,
    const scalar_t* __restrict__ factors)  // [m * topk] (topk_weights, token-major layout)
{
  int i = blockIdx.x;
  if (i >= m) {
    return;
  }

  constexpr uint32_t vec_size = 16 / sizeof(scalar_t);
  using t = float;
  using vec_t = flashinfer::vec_t<t, vec_size>;
  int thread_idx = threadIdx.x;
  int stride = blockDim.x;

  for (int d_vec_idx = thread_idx; d_vec_idx < row_stride / vec_size; d_vec_idx += stride) {
    int d = d_vec_idx * vec_size;
    vec_t sum_vec;
    sum_vec.fill(0.0f);

    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];

      vec_t val_vec;
      val_vec.cast_load(input_tensor + src_row * row_stride + d);

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }

#pragma unroll
      for (int k = 0; k < vec_size; ++k) {
        sum_vec[k] += factor * val_vec[k];
      }
    }
    sum_vec.cast_store(output_tensor + i * row_stride + d);
  }

  //  remainder part
  int remainder_start = (row_stride / vec_size) * vec_size;
  for (int d = remainder_start + thread_idx; d < row_stride; d += stride) {
    t sum_val = 0.0;
    for (int j = 0; j < topk; ++j) {
      int token_major_idx = i * topk + j;
      int src_row = permutation[token_major_idx];
      t val = input_tensor[src_row * row_stride + d];

      t factor = 1.0;
      if (factors != nullptr) {
        factor = factors[token_major_idx];
      }
      sum_val += factor * val;
    }
    output_tensor[i * row_stride + d] = sum_val;
  }
}

void get_apply_shuffle_mul_sum_caller(
    const TorchTensor& input_tensor,    // [m * topk, row_stride], bf16/f16
    TorchTensor& output_tensor,         // [m, row_stride], bf16/f16
    const TorchTensor& permutation,     // [m * topk], int32
    const OptionalTensor& factors_opt)  // optional [m * topk], bf16/f16
{
  SGL_CHECK(input_tensor.dim() == 2, "input_tensor must be 2D [m * topk, row_stride]");
  SGL_CHECK(output_tensor.dim() == 2, "output_tensor must be 2D [m, row_stride]");
  SGL_CHECK(permutation.dim() == 1, "permutation must be 1D [m * topk]");

  int m = output_tensor.size(0);
  int topk = int(permutation.size(0) / m);
  int row_stride = output_tensor.size(1);

  SGL_CHECK(permutation.size(0) == m * topk, "permutation size must match m * topk");

  auto scalar_type = output_tensor.scalar_type();
  uint32_t vec_size = 16 / sizeof(scalar_type);
  auto blockDim = std::min(row_stride / vec_size, 1024U);
  dim3 block(blockDim);

  dim3 grid(m);  // blockIdx.x = j, blockIdx.y = i
  auto stream = SGL_TENSOR_CUDA_STREAM(input_tensor);

  const int32_t* perm_ptr = SGL_CONST_DATA_PTR(permutation, int32_t);

  const void* factors_ptr = nullptr;
  if (factors_opt.has_value()) {
    SGL_CHECK(factors_opt->scalar_type() == output_tensor.scalar_type(), "Factors must match output dtype");
    SGL_CHECK(factors_opt->numel() == m * topk, "Factors must have shape [m * topk]");
    factors_ptr = SGL_CONST_UNTYPED_DATA_PTR(*factors_opt);
  }

  SGL_DISPATCH_FLOAT_FP16(output_tensor.scalar_type(), "apply_shuffle_mul_sum", [&] {
    using cuda_scalar_t = SGL_CUDA_TYPE(scalar_t);
    apply_shuffle_mul_sum_kernel<cuda_scalar_t><<<grid, block, 0, stream>>>(
        static_cast<const cuda_scalar_t*>(SGL_CONST_UNTYPED_DATA_PTR(input_tensor)),
        static_cast<cuda_scalar_t*>(SGL_MUTABLE_UNTYPED_DATA_PTR(output_tensor)),
        perm_ptr,
        m,
        topk,
        row_stride,
        static_cast<const cuda_scalar_t*>(factors_ptr));
    return true;
  });
}

/**
 * @brief Applies a permutation-based shuffle, element-wise multiplication, and reduction over the second dimension.
 *
 * This function performs the equivalent of the following PyTorch expression:
 *
 *     (c2[c_map].view(m, topk, k) * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1)
 *
 * Specifically:
 * - `input` is shuffled using the `permutation` tensor.
 * - The shuffled tensor is reshaped and multiplied element-wise with `factors` (e.g., top-k weights).
 * - The result is summed along dimension 1 (the top-k dimension), and stored in `output`.
 *
 * @param input        Input tensor of shape (m * topk, k), representing c2.
 * @param output       Output tensor of shape (m, k), where the final reduced results are stored.
 * @param permutation  Index tensor (e.g., c_map) that maps positions in `input` to shuffled layout.
 * @param factors      Optional scaling factors (e.g., top-k weights), shape (m * topk) or (m, topk).
 */
void apply_shuffle_mul_sum(
    const TorchTensor& input, TorchTensor& output, const TorchTensor& permutation, const OptionalTensor& factors) {
  get_apply_shuffle_mul_sum_caller(input, output, permutation, factors);
}

#undef SGL_CHECK
#undef SGL_CURRENT_CUDA_STREAM
#undef SGL_TENSOR_CUDA_STREAM
#undef SGL_NEW_ZEROS_INT32
#undef SGL_CONST_DATA_PTR
#undef SGL_MUTABLE_DATA_PTR
#undef SGL_CONST_UNTYPED_DATA_PTR
#undef SGL_MUTABLE_UNTYPED_DATA_PTR
#undef SGL_CUDA_TYPE
#undef SGL_DISPATCH_FLOAT_FP16
#ifdef TORCH_TARGET_VERSION
#undef SGL_DISPATCH_CASE_F16
#undef SGL_DISPATCH_CASE_BF16
#endif
