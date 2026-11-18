// Adapted from
// https://github.com/mlc-ai/xgrammar/blob/v0.1.18/python/xgrammar/kernels/apply_token_bitmask_inplace_cuda.cu

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "grammar/apply_token_bitmask_inplace_cuda.h"

#ifdef TORCH_TARGET_VERSION
#include <cuda.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include "sgl_kernel_cuda_stream.h"

using SglBFloat16 = torch::headeronly::BFloat16;
using SglHalf = torch::headeronly::Half;
using SglScalarType = torch::headeronly::ScalarType;
#else
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>

using SglBFloat16 = at::BFloat16;
using SglHalf = at::Half;
using SglScalarType = at::ScalarType;
#endif

template <typename T>
T* MutableDataPtr(const SglTensor& tensor) {
#ifdef TORCH_TARGET_VERSION
  return tensor.mutable_data_ptr<T>();
#else
  return tensor.data_ptr<T>();
#endif
}

template <typename T>
const T* ConstDataPtr(const SglTensor& tensor) {
#ifdef TORCH_TARGET_VERSION
  return tensor.const_data_ptr<T>();
#else
  return tensor.data_ptr<T>();
#endif
}

inline cudaStream_t GetCurrentCUDAStream() {
#ifdef TORCH_TARGET_VERSION
  return sgl_kernel::stable::get_current_cuda_stream();
#else
  return at::cuda::getCurrentCUDAStream().stream();
#endif
}


#if !defined(USE_ROCM) && (!defined(CUDA_VERSION) || CUDA_VERSION < 12040)
void ApplyTokenBitmaskInplace(SglTensor logits, SglTensor bitmask, SglOptional<SglTensor> indices = {}) {
  SGL_TORCH_CHECK(false, "CUDA version must be >= 12.4 for ApplyTokenBitmaskInplace");
}
#else

#ifndef CUDART_INF_FP16
#ifndef USE_ROCM
#define CUDART_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#endif
#endif

#ifndef CUDART_INF_BF16
#ifndef USE_ROCM
#define CUDART_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#endif
#endif

constexpr int32_t BITS_PER_BLOCK = 32;
constexpr int32_t THREADS_PER_THREAD_BLOCK = 256;

template <typename T>
__device__ T NegativeInfinity() {
  return -INFINITY;
}

template <>
__device__ __half NegativeInfinity<__half>() {
#ifdef USE_ROCM
  return __float2half(-INFINITY);
#else
  return -CUDART_INF_FP16;
#endif
}

template <>
__device__ __nv_bfloat16 NegativeInfinity<__nv_bfloat16>() {
#ifdef USE_ROCM
  return __nv_bfloat16(-INFINITY);
#else
  return -CUDART_INF_BF16;
#endif
}

template <typename T, typename PackedT>
__device__ PackedT PackedNegativeInfinity() {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  T packed[kAlignment];
#pragma unroll
  for (int i = 0; i < kAlignment; i++) {
    packed[i] = NegativeInfinity<T>();
  }
  return *reinterpret_cast<PackedT*>(packed);
}

template <typename T, typename PackedT, int32_t kBitsPerThread>
__global__ void __launch_bounds__(THREADS_PER_THREAD_BLOCK) LogitsBitmaskKernel(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  constexpr uint32_t kPackedMask = (1 << kAlignment) - 1;

  const int batch_idx = (indices == nullptr) ? blockIdx.y : indices[blockIdx.y];

  const int block_offset = blockIdx.x * THREADS_PER_THREAD_BLOCK * kBitsPerThread;
  T* logits_gmem_ptr = logits + batch_idx * logits_stride + block_offset;
  const int32_t* bitmask_gmem_ptr = bitmask + batch_idx * bitmask_stride + block_offset / BITS_PER_BLOCK;
  const int bitmask_inner_idx = threadIdx.x % (BITS_PER_BLOCK / kAlignment);
  T logits_reg[kAlignment];

#pragma unroll
  for (int offset = threadIdx.x * kAlignment; offset < THREADS_PER_THREAD_BLOCK * kBitsPerThread;
       offset += THREADS_PER_THREAD_BLOCK * kAlignment) {
    if (block_offset + offset >= vocab_size) {
      break;
    }

    const uint32_t bitmask_val =
        (~bitmask_gmem_ptr[offset / BITS_PER_BLOCK] >> (bitmask_inner_idx * kAlignment)) & kPackedMask;

    if (bitmask_val == 0) {
      continue;
    }

    if (bitmask_val == kPackedMask) {
      *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = PackedNegativeInfinity<T, PackedT>();
      continue;
    }

    *reinterpret_cast<PackedT*>(logits_reg) = *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset);
#pragma unroll
    for (int i = 0; i < kAlignment; i++) {
      if (((bitmask_val >> i) & 1)) {
        logits_reg[i] = NegativeInfinity<T>();
      }
    }
    *reinterpret_cast<PackedT*>(logits_gmem_ptr + offset) = *reinterpret_cast<PackedT*>(logits_reg);
  }
}

template <typename T, typename = std::enable_if_t<std::is_integral<T>::value>>
constexpr auto CeilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename T, typename PackedT>
void ApplyTokenBitmaskInplaceDispatchToBitsPerThread(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride,
    int32_t num_rows) {
  constexpr int kAlignment = sizeof(PackedT) / sizeof(T);
  const int32_t num_blocks_per_row = CeilDiv(2048 / THREADS_PER_THREAD_BLOCK * 128, num_rows);
  const int32_t num_bits_per_thread = CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * num_blocks_per_row);

  const dim3 block(THREADS_PER_THREAD_BLOCK);
  cudaStream_t stream = GetCurrentCUDAStream();

  if (num_bits_per_thread <= 4 && kAlignment <= 4) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 4), num_rows);
    LogitsBitmaskKernel<T, PackedT, 4>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else if (num_bits_per_thread <= 8 && kAlignment <= 8) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 8), num_rows);
    LogitsBitmaskKernel<T, PackedT, 8>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else if (num_bits_per_thread <= 16 && kAlignment <= 16) {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 16), num_rows);
    LogitsBitmaskKernel<T, PackedT, 16>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  } else {
    const dim3 grid(CeilDiv(vocab_size, THREADS_PER_THREAD_BLOCK * 32), num_rows);
    LogitsBitmaskKernel<T, PackedT, 32>
        <<<grid, block, 0, stream>>>(logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride);
  }
}

template <typename T>
void ApplyTokenBitmaskInplaceDispatchToPackedT(
    T* __restrict__ logits,
    const int32_t* __restrict__ bitmask,
    const int32_t* __restrict__ indices,
    int32_t vocab_size,
    int32_t logits_stride,
    int32_t bitmask_stride,
    int32_t num_rows) {
  if (logits_stride % (sizeof(float4) / sizeof(T)) == 0) {
    ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, float4>(
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride, num_rows);
  } else {
    ApplyTokenBitmaskInplaceDispatchToBitsPerThread<T, T>(
        logits, bitmask, indices, vocab_size, logits_stride, bitmask_stride, num_rows);
  }
}

void ApplyTokenBitmaskInplace(SglTensor logits, SglTensor bitmask, SglOptional<SglTensor> indices = {}) {
  SGL_TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor.");
  SGL_TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous.");
  SGL_TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be a 1D or 2D tensor.");
  std::pair<int32_t, int32_t> logits_shape =
      logits.dim() == 2 ? std::make_pair(static_cast<int32_t>(logits.size(0)), static_cast<int32_t>(logits.size(1)))
                        : std::make_pair(1, static_cast<int32_t>(logits.size(0)));

  SGL_TORCH_CHECK(bitmask.is_cuda(), "bitmask must be a CUDA tensor.");
  SGL_TORCH_CHECK(bitmask.is_contiguous(), "bitmask must be contiguous.");
  SGL_TORCH_CHECK(bitmask.dim() == 1 || bitmask.dim() == 2, "bitmask must be a 1D or 2D tensor.");
  std::pair<int32_t, int32_t> bitmask_shape =
      bitmask.dim() == 2 ? std::make_pair(static_cast<int32_t>(bitmask.size(0)), static_cast<int32_t>(bitmask.size(1)))
                         : std::make_pair(1, static_cast<int32_t>(bitmask.size(0)));

  SGL_TORCH_CHECK(bitmask.scalar_type() == SglScalarType::Int, "bitmask must be of type int32.");

  SGL_TORCH_CHECK(
      (logits_shape.second + BITS_PER_BLOCK - 1) / BITS_PER_BLOCK >= bitmask_shape.second,
      "The provided logits's vocab size should be no less than the bitmask's vocab size "
      "(converted from bitmask size). But got vocab size ",
      logits_shape.second,
      " vs bitmask size ",
      bitmask_shape.second);

  int vocab_size = std::min(logits_shape.second, bitmask_shape.second * BITS_PER_BLOCK);

  int32_t num_rows = logits_shape.first;
  const int32_t* indices_ptr = nullptr;
  if (indices) {
    SGL_TORCH_CHECK(indices->is_cuda(), "indices must be a CUDA tensor.");
    SGL_TORCH_CHECK(indices->is_contiguous(), "indices must be contiguous.");
    SGL_TORCH_CHECK(indices->dim() == 1, "indices must be a 1D tensor.");
    SGL_TORCH_CHECK(indices->scalar_type() == SglScalarType::Int, "indices must be of type int32.");
    num_rows = indices->size(0);
    indices_ptr = ConstDataPtr<int32_t>(*indices);
  } else {
    SGL_TORCH_CHECK(logits_shape.first == bitmask_shape.first, "logits and bitmask must have the same batch size.");
  }

  switch (logits.scalar_type()) {
    case SglScalarType::Float: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          MutableDataPtr<float>(logits),
          ConstDataPtr<int32_t>(bitmask),
          indices_ptr,
          vocab_size,
          logits_shape.second,
          bitmask_shape.second,
          num_rows);
      break;
    }
    case SglScalarType::Half: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__half*>(MutableDataPtr<SglHalf>(logits)),
          ConstDataPtr<int32_t>(bitmask),
          indices_ptr,
          vocab_size,
          logits_shape.second,
          bitmask_shape.second,
          num_rows);
      break;
    }
    case SglScalarType::BFloat16: {
      ApplyTokenBitmaskInplaceDispatchToPackedT(
          reinterpret_cast<__nv_bfloat16*>(MutableDataPtr<SglBFloat16>(logits)),
          ConstDataPtr<int32_t>(bitmask),
          indices_ptr,
          vocab_size,
          logits_shape.second,
          bitmask_shape.second,
          num_rows);
      break;
    }
    default:
      SGL_TORCH_CHECK(false, "logits dtype must be float, half or bfloat16.");
      break;
  }
}
#endif
// clang-format on
