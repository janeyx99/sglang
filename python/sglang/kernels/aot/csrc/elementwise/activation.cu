/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <algorithm>
#include <flashinfer/activation.cuh>

#include "elementwise/elementwise_ops.h"
#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#ifndef USE_ROCM

#include <flashinfer/activation.cuh>

#include "utils.h"

#else
#include "hip/hip_act_and_mul.cuh"
#endif
#endif

// Adapted from flashinfer activation
// https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/csrc/activation.cu#L44

namespace detail {

template <typename T>
__device__ __forceinline__ float to_f32(const T& x) {
#if USE_ROCM
  return castToFloat(x);
#else
  return static_cast<float>(x);
#endif
}

template <typename T>
__device__ __forceinline__ T from_f32(float f32) {
#if USE_ROCM
  return castFromFloat<T>(f32);
#else
  return static_cast<T>(f32);
#endif
}

}  // namespace detail

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val)));
}

template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}

// gelu_quick(x) = x * torch.sigmoid(1.702 * x)
template <typename T>
__device__ __forceinline__ T gelu_quick_act(const T& x) {
  float f32_val = detail::to_f32(x);
  return detail::from_f32<T>(f32_val / (1.0f + expf(-f32_val * 1.702f)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = detail::to_f32(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return detail::from_f32<T>(f32_val * cdf);
}

#ifdef TORCH_TARGET_VERSION
#ifdef FLASHINFER_ENABLE_BF16
#define STABLE_DISPATCH_BF16_CASE(c_type, ...) \
  case ScalarType::BFloat16: {                 \
    using c_type = nv_bfloat16;                \
    return __VA_ARGS__();                      \
  }
#else
#define STABLE_DISPATCH_BF16_CASE(c_type, ...)
#endif

#define STABLE_DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(pytorch_dtype, c_type, legacy_function, ...) \
  [&]() -> bool {                                                                                      \
    switch (pytorch_dtype) {                                                                           \
      case ScalarType::Float: {                                                                        \
        using c_type = float;                                                                          \
        return __VA_ARGS__();                                                                          \
      }                                                                                                \
      case ScalarType::Half: {                                                                         \
        using c_type = nv_half;                                                                        \
        return __VA_ARGS__();                                                                          \
      }                                                                                                \
        STABLE_DISPATCH_BF16_CASE(c_type, __VA_ARGS__)                                                 \
      default:                                                                                         \
        STD_TORCH_CHECK(                                                                               \
            false,                                                                                     \
            legacy_function,                                                                           \
            "(at::Tensor&, at::Tensor&)::<lambda()> failed to dispatch data type ",                    \
            torch::headeronly::toString(pytorch_dtype));                                               \
        return false;                                                                                  \
    }                                                                                                  \
  }()

void silu_and_mul(Tensor& out, Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();
  STD_TORCH_CHECK(input.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());

  STABLE_DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, "silu_and_mul", [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, silu><<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.mutable_data_ptr()), static_cast<const c_type*>(input.const_data_ptr()), d);
    return true;
  });
}

void gelu_tanh_and_mul(Tensor& out, Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();
  STD_TORCH_CHECK(input.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());

  STABLE_DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, "gelu_tanh_and_mul", [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh><<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.mutable_data_ptr()), static_cast<const c_type*>(input.const_data_ptr()), d);
    return true;
  });
}

void gelu_and_mul(Tensor& out, Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();
  STD_TORCH_CHECK(input.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());

  STABLE_DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, "gelu_and_mul", [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    flashinfer::activation::act_and_mul_kernel<c_type, gelu><<<grid, block, 0, stream>>>(
        static_cast<c_type*>(out.mutable_data_ptr()), static_cast<const c_type*>(input.const_data_ptr()), d);
    return true;
  });
}

#undef STABLE_DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16
#undef STABLE_DISPATCH_BF16_CASE
#else
void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif

    return true;
  });
}
#endif

#if USE_ROCM
void gelu_quick(at::Tensor& out, const at::Tensor& input) {
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    sgl_hip::activation::act_only_kernel<c_type, gelu_quick_act>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}
#endif
