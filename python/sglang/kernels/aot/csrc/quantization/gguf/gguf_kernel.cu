// Adatped from
// https://github.com/vllm-project/vllm/blob/755ed7b05be4743237d3339c4ff8c22bcaae04f4/csrc/quantization/gguf/gguf_kernel.cu
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>

#include "quantization/gguf/gguf_ops.h"

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Device = torch::stable::Device;
using SglBFloat16 = torch::headeronly::BFloat16;

#define WARP_SIZE 32
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) __shfl_xor_sync((mask), (var), (lane_mask), (width))
#define GGUF_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                                       \
  THO_DISPATCH_SWITCH(                                                                                   \
      TYPE,                                                                                              \
      NAME,                                                                                              \
      THO_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__) THO_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__) \
          THO_DISPATCH_CASE(ScalarType::BFloat16, __VA_ARGS__))
#else
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// dont use clang-format here, it breaks the include order
// clang-format off
#include "utils.h"

using Tensor = torch::Tensor;
using ScalarType = at::ScalarType;
using Device = c10::Device;
using SglBFloat16 = c10::BFloat16;

#define GGUF_DISPATCH_FLOAT_TYPES DISPATCH_FLOAT_TYPES
#endif

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"
// clang-format off

namespace {

template <typename T>
const T* ReadPtr(const Tensor& tensor) {
  return static_cast<const T*>(SGL_CONST_DATA_PTR(tensor));
}

template <typename T>
T* MutablePtr(const Tensor& tensor) {
  return static_cast<T*>(SGL_MUTABLE_DATA_PTR(tensor));
}

#ifdef TORCH_TARGET_VERSION
torch::stable::accelerator::DeviceIndex CheckedCudaDeviceIndex(const Tensor& tensor) {
  STD_TORCH_CHECK(tensor.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  return tensor.get_device_index();
}

class TensorDeviceGuard {
 public:
  explicit TensorDeviceGuard(const Tensor& tensor)
      : device_index_(CheckedCudaDeviceIndex(tensor)), guard_(device_index_) {}

  cudaStream_t current_stream() const {
    return sgl_kernel::stable::get_current_cuda_stream(device_index_);
  }

 private:
  torch::stable::accelerator::DeviceIndex device_index_;
  torch::stable::accelerator::DeviceGuard guard_;
};

Tensor Empty(const Device& device, std::initializer_list<int64_t> size, ScalarType dtype) {
  return torch::stable::empty(
      torch::headeronly::IntHeaderOnlyArrayRef(size.begin(), size.size()),
      dtype,
      torch::headeronly::Layout::Strided,
      device);
}

Tensor Zeros(const Device& device, std::initializer_list<int64_t> size, ScalarType dtype) {
  const auto sizes = torch::headeronly::IntHeaderOnlyArrayRef(size.begin(), size.size());
  std::array<StableIValue, 5> stack{
      torch::stable::detail::from(sizes),
      torch::stable::detail::from(SglOptional<ScalarType>(dtype)),
      torch::stable::detail::from(SglOptional<torch::headeronly::Layout>(torch::headeronly::Layout::Strided)),
      torch::stable::detail::from(SglOptional<Device>(device)),
      torch::stable::detail::from(std::optional<bool>())};
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher("aten::zeros", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}
#else
class TensorDeviceGuard {
 public:
  explicit TensorDeviceGuard(const Tensor& tensor) : guard_(device_of(tensor)) {}

  cudaStream_t current_stream() const {
    return at::cuda::getCurrentCUDAStream().stream();
  }

 private:
  at::cuda::OptionalCUDAGuard guard_;
};

Tensor Empty(const Device& device, std::initializer_list<int64_t> size, ScalarType dtype) {
  return torch::empty(size, torch::TensorOptions().dtype(dtype).device(device));
}

Tensor Zeros(const Device& device, std::initializer_list<int64_t> size, ScalarType dtype) {
  return torch::zeros(size, torch::TensorOptions().dtype(dtype).device(device));
}
#endif

}  // namespace

// Q8 gemv
template <typename scalar_t>
static __global__ void
quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), amax, mask, 32));
    sum += SGLANG_SHFL_XOR_SYNC_WIDTH(uint32_t(-1), sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

Tensor ggml_dequantize(
    Tensor W,  // quant weight
    int64_t type,
    int64_t m,
    int64_t n,
    SglOptional<SglScalarType> const& dtype) {
  const TensorDeviceGuard device_guard(W);
  auto dtype_ = dtype.value_or(ScalarType::Half);
  const auto output_device = W.device();
  Tensor DW = Empty(output_device, {m, n}, dtype_);
  cudaStream_t stream = device_guard.current_stream();

  GGUF_DISPATCH_FLOAT_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda(ReadPtr<void>(W), MutablePtr<scalar_t>(DW), m * n, stream);
  });

  return DW;
}

Tensor ggml_mul_mat_vec_a8(
    Tensor W,  // quant weight
    Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int vecs = X.sizes()[0];
  const int padded = (col + 512 - 1) / 512 * 512;
  const TensorDeviceGuard device_guard(X);
  const auto input_dtype = X.scalar_type();
  const auto output_device = W.device();
  Tensor Y = Empty(output_device, {vecs, row}, input_dtype);
  cudaStream_t stream = device_guard.current_stream();
  Tensor quant_X = Empty(output_device, {vecs, padded / 32 * 9}, ScalarType::Int);
  GGUF_DISPATCH_FLOAT_TYPES(input_dtype, "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(ReadPtr<scalar_t>(X), MutablePtr<void>(quant_X), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W), ReadPtr<void>(quant_X), MutablePtr<scalar_t>(Y), col, row, vecs, stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_mul_mat_a8(
    Tensor W,  // quant weight
    Tensor X,  // input
    int64_t type,
    int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const TensorDeviceGuard device_guard(X);
  const auto input_dtype = X.scalar_type();
  const auto output_device = W.device();
  Tensor Y = Empty(output_device, {batch, row}, input_dtype);
  cudaStream_t stream = device_guard.current_stream();
  Tensor quant_X = Empty(output_device, {batch, padded / 32 * 9}, ScalarType::Int);
  GGUF_DISPATCH_FLOAT_TYPES(input_dtype, "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda(ReadPtr<scalar_t>(X), MutablePtr<void>(quant_X), col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            col,
            row,
            batch,
            padded,
            row,
            stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_moe_a8(
    Tensor X,  // input
    Tensor W,  // expert weights
    Tensor sorted_token_ids,
    Tensor expert_ids,
    Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  const TensorDeviceGuard device_guard(X);
  const auto input_dtype = X.scalar_type();
  const auto output_device = W.device();
  Tensor Y = Empty(output_device, {tokens * top_k, row}, input_dtype);
  cudaStream_t stream = device_guard.current_stream();
  Tensor quant_X = Empty(output_device, {tokens, padded / 32 * 9}, ScalarType::Int);
  GGUF_DISPATCH_FLOAT_TYPES(input_dtype, "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda(ReadPtr<scalar_t>(X), MutablePtr<void>(quant_X), col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            ReadPtr<void>(quant_X),
            ReadPtr<void>(W),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(sorted_token_ids),
            ReadPtr<int>(expert_ids),
            ReadPtr<int>(num_tokens_post_padded),
            W.stride(0),
            col,
            row,
            tokens,
            padded,
            row,
            top_k,
            sorted_token_ids.sizes()[0],
            stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_moe_a8_vec(
    Tensor X,  // input
    Tensor W,  // expert weights
    Tensor topk_ids,
    int64_t top_k,
    int64_t type,
    int64_t row,
    int64_t tokens) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const TensorDeviceGuard device_guard(X);
  const auto input_dtype = X.scalar_type();
  const auto output_device = W.device();
  Tensor Y = Zeros(output_device, {tokens * top_k, row}, input_dtype);
  cudaStream_t stream = device_guard.current_stream();
  Tensor quant_X = Empty(output_device, {tokens, padded / 32 * 9}, ScalarType::Int);
  GGUF_DISPATCH_FLOAT_TYPES(input_dtype, "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(ReadPtr<scalar_t>(X), MutablePtr<void>(quant_X), col, tokens, stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            ReadPtr<void>(W),
            ReadPtr<void>(quant_X),
            MutablePtr<scalar_t>(Y),
            ReadPtr<int>(topk_ids),
            top_k,
            tokens,
            col,
            row,
            quant_X.stride(0),
            stream);
        break;
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}

#undef GGUF_DISPATCH_FLOAT_TYPES
#ifdef TORCH_TARGET_VERSION
#undef SGLANG_SHFL_XOR_SYNC_WIDTH
#undef SGLANG_SHFL_XOR_SYNC
#undef WARP_SIZE
#endif
