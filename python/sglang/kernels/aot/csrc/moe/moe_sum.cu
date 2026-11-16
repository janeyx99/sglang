#ifdef TORCH_TARGET_VERSION
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <algorithm>

#include "moe/moe_ops.h"
#include "sgl_kernel_cuda_stream.h"

#define SGLANG_LDG(arg) __ldg(arg)

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
#else
#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <ATen/cuda/Atomic.cuh>
#include <cub/cub.cuh>

#ifdef USE_MUSA
#include <musa_runtime.h>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#endif

#include "gguf_moe_weights.cuh"
#include "utils.h"
#endif

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., topk, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += SGLANG_LDG(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

#ifdef TORCH_TARGET_VERSION
#define STABLE_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                                     \
  THO_DISPATCH_SWITCH(                                                                                   \
      TYPE,                                                                                              \
      NAME,                                                                                              \
      THO_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__) THO_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__) \
          THO_DISPATCH_CASE(ScalarType::BFloat16, __VA_ARGS__))

void moe_sum(
    Tensor& input,   // [num_tokens, topk, hidden_size]
    Tensor& output)  // [num_tokens, hidden_size]
{
  const int64_t input_ndim = input.dim();
  STD_TORCH_CHECK(input_ndim > 0, "Dimension specified as -1 but tensor has no dimensions");
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  STD_TORCH_CHECK(input_ndim > 1, "Dimension out of range (expected to be in range of [-1, 0], but got 1)");
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  STD_TORCH_CHECK(output.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const torch::stable::accelerator::DeviceGuard device_guard(output.get_device_index());
  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();

  switch (topk) {
    case 2:
      STABLE_DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 2><<<grid, block, 0, stream>>>(
            output.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 3:
      STABLE_DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
            output.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 4:
      STABLE_DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 4><<<grid, block, 0, stream>>>(
            output.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(), hidden_size);
      });
      break;

    default: {
      STD_TORCH_CHECK(input.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
      const int64_t dim = 1;
      const torch::headeronly::IntHeaderOnlyArrayRef dims(&dim, 1);
      torch::stable::sum_out(output, input, dims);
      break;
    }
  }
}

#undef STABLE_DISPATCH_FLOAT_TYPES
#undef SGLANG_LDG
#else
void moe_sum(
    torch::Tensor& input,   // [num_tokens, topk, hidden_size]
    torch::Tensor& output)  // [num_tokens, hidden_size]
{
  const int hidden_size = input.size(-1);
  const auto num_tokens = output.numel() / hidden_size;
  const int topk = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (topk) {
    case 2:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 2>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 3:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 3>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    case 4:
      DISPATCH_FLOAT_TYPES(input.scalar_type(), "moe_sum_kernel", [&] {
        moe_sum_kernel<scalar_t, 4>
            <<<grid, block, 0, stream>>>(output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), hidden_size);
      });
      break;

    default:
      at::sum_out(output, input, 1);
      break;
  }
}

void apply_gguf_moe_weights(torch::Tensor& outputs, const torch::Tensor& weights, const torch::Tensor& ids) {
#ifdef USE_MUSA
  TORCH_CHECK(outputs.device().type() == torch::kMUSA, "outputs must be a MUSA tensor");
#else
  TORCH_CHECK(outputs.is_cuda(), "outputs must be a CUDA tensor");
#endif
  TORCH_CHECK(weights.device() == outputs.device() && ids.device() == outputs.device(), "tensors must share a device");
  TORCH_CHECK(
      outputs.layout() == at::kStrided && weights.layout() == at::kStrided && ids.layout() == at::kStrided,
      "tensors must have strided layout");
  TORCH_CHECK(outputs.dim() == 2 && outputs.is_contiguous(), "outputs must be contiguous [tokens * topk, hidden_size]");
  TORCH_CHECK(weights.dim() == 2 && ids.dim() == 2, "weights and ids must have shape [tokens, topk]");
  TORCH_CHECK(weights.sizes() == ids.sizes(), "weights and ids shapes must match");
  TORCH_CHECK(outputs.size(0) == weights.numel(), "outputs rows must equal tokens * topk");
  const auto supported_float = [](at::ScalarType dtype) {
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kFloat;
  };
  TORCH_CHECK(supported_float(outputs.scalar_type()), "outputs must be float16, bfloat16, or float32");
  TORCH_CHECK(supported_float(weights.scalar_type()), "weights must be float16, bfloat16, or float32");
  TORCH_CHECK(ids.scalar_type() == at::kInt || ids.scalar_type() == at::kLong, "ids must be int32 or int64");
  TORCH_CHECK(
      weights.stride(0) >= 0 && weights.stride(1) >= 0 && ids.stride(0) >= 0 && ids.stride(1) >= 0,
      "weights and ids strides must be nonnegative");
  at::assert_no_overlap(outputs, weights);
  at::assert_no_overlap(outputs, ids);
  const int64_t numel = outputs.numel();
  if (numel == 0) {
    return;
  }
#ifdef USE_MUSA
  const c10::musa::OptionalMUSAGuard device_guard(device_of(outputs));
  const musaStream_t stream = c10::musa::getCurrentMUSAStream();
#else
  const at::cuda::OptionalCUDAGuard device_guard(device_of(outputs));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif
  constexpr int threads = 256;
  // Reuse each route's ID and score across four coalesced column groups.
  constexpr int columns_per_block = threads * 4;
  const dim3 blocks(
      static_cast<unsigned int>(std::min<int64_t>(outputs.size(0), 65535)),
      static_cast<unsigned int>(std::min<int64_t>((outputs.size(1) - 1) / columns_per_block + 1, 65535)));
  DISPATCH_FLOAT_TYPES(outputs.scalar_type(), "apply_gguf_moe_weights_output", [&] {
    using output_t = scalar_t;
    DISPATCH_FLOAT_TYPES(weights.scalar_type(), "apply_gguf_moe_weights_weight", [&] {
      using weight_t = scalar_t;
      const auto launch = [&](auto* id_data) {
        apply_gguf_moe_weights_kernel<<<blocks, threads, 0, stream>>>(
            outputs.data_ptr<output_t>(),
            weights.data_ptr<weight_t>(),
            id_data,
            outputs.size(0),
            outputs.size(1),
            weights.size(1),
            weights.stride(0),
            weights.stride(1),
            ids.stride(0),
            ids.stride(1));
      };
      if (ids.scalar_type() == at::kInt) {
        launch(ids.data_ptr<int32_t>());
      } else {
        launch(ids.data_ptr<int64_t>());
      }
    });
  });
#ifdef USE_MUSA
  const auto error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "apply_gguf_moe_weights launch failed: ", musaGetErrorString(error));
#else
  const auto error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "apply_gguf_moe_weights launch failed: ", cudaGetErrorString(error));
#endif
}
#endif
