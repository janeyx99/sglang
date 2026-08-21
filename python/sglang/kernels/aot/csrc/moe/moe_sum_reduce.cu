#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include "moe/moe_ops.h"
#undef SGL_CONST_DATA_PTR
#undef SGL_MUTABLE_DATA_PTR
#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using HalfType = torch::headeronly::Half;
using BFloat16Type = torch::headeronly::BFloat16;

#define SGL_CHECK(...) STD_TORCH_CHECK(__VA_ARGS__)
#define SGL_CURRENT_CUDA_STREAM() sgl_kernel::stable::get_current_cuda_stream()
#define SGL_CONST_DATA_PTR(tensor, type) tensor.const_data_ptr<type>()
#define SGL_MUTABLE_DATA_PTR(tensor, type) tensor.mutable_data_ptr<type>()
#define SGL_DISPATCH_FLOATING_TYPES_AND2(TYPE, NAME, ...)                                                  \
  THO_DISPATCH_SWITCH(                                                                                     \
      TYPE,                                                                                                \
      NAME,                                                                                                \
      THO_DISPATCH_CASE(ScalarType::Double, __VA_ARGS__) THO_DISPATCH_CASE(ScalarType::Float, __VA_ARGS__) \
          THO_DISPATCH_CASE(ScalarType::Half, __VA_ARGS__) THO_DISPATCH_CASE(ScalarType::BFloat16, __VA_ARGS__))
#else
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "utils.h"

using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
using HalfType = at::Half;
using BFloat16Type = at::BFloat16;

#define SGL_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#define SGL_CURRENT_CUDA_STREAM() at::cuda::getCurrentCUDAStream()
#define SGL_CONST_DATA_PTR(tensor, type) tensor.data_ptr<type>()
#define SGL_MUTABLE_DATA_PTR(tensor, type) tensor.data_ptr<type>()
#define SGL_DISPATCH_FLOATING_TYPES_AND2(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, TYPE, NAME, __VA_ARGS__)
#endif

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>

#include "cutlass/array.h"

#ifdef TORCH_TARGET_VERSION
template <typename T>
struct stable_opmath_type {
  using type = T;
};

template <>
struct stable_opmath_type<HalfType> {
  using type = float;
};

template <>
struct stable_opmath_type<BFloat16Type> {
  using type = float;
};
#endif

#ifdef TORCH_TARGET_VERSION
template <typename T>
using opmath_t = typename stable_opmath_type<T>::type;
#else
template <typename T>
using opmath_t = at::opmath_type<T>;
#endif

template <typename T>
__device__ __forceinline__ opmath_t<T> to_acc(T x) {
  return static_cast<opmath_t<T>>(x);
}

template <typename T>
__device__ __forceinline__ T from_acc(opmath_t<T> x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ opmath_t<HalfType> to_acc<HalfType>(HalfType x) {
  return __half2float(__nv_half(x));
}
template <>
__device__ __forceinline__ HalfType from_acc<HalfType>(opmath_t<HalfType> x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ opmath_t<BFloat16Type> to_acc<BFloat16Type>(BFloat16Type x) {
  return __bfloat162float(__nv_bfloat16(x));
}
template <>
__device__ __forceinline__ BFloat16Type from_acc<BFloat16Type>(opmath_t<BFloat16Type> x) {
  return __float2bfloat16_rn(x);
}

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

union Pack16B {
  uint4 v;
  __nv_bfloat16 u16[8];
};

template <int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_per_token_vec_kernel(
    const BFloat16Type* __restrict__ x,
    BFloat16Type* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t topk_num,
    const int64_t stride_token,      // in elements
    const int64_t stride_topk,       // in elements
    const int64_t out_stride_token,  // in elements
    const float scale) {
  constexpr int VEC = 16;
  constexpr int PACKS = VEC / 8;

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  const int64_t n_chunks = hidden_dim / VEC;

  for (int64_t chunk = (int64_t)blockIdx.x * 32 + lane; chunk < n_chunks; chunk += (int64_t)gridDim.x * 32) {
    const int64_t d = chunk * VEC;
    const int64_t base = t * stride_token + d;

    float acc[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] = 0.f;

#pragma unroll
    for (int k = 0; k < topk_num; ++k) {
#pragma unroll
      for (int p = 0; p < PACKS; ++p) {
        const int64_t offset = base + (int64_t)k * stride_topk + p * 8;
        Pack16B pack = {ldg_cg(reinterpret_cast<const uint4*>(x + offset))};

#pragma unroll
        for (int i = 0; i < 8; ++i) {
          acc[p * 8 + i] += __bfloat162float(pack.u16[i]);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < VEC; ++i)
      acc[i] *= scale;

#pragma unroll
    for (int p = 0; p < PACKS; ++p) {
      Pack16B outp;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        outp.u16[i] = __float2bfloat16_rn(acc[p * 8 + i]);
      }
      const int64_t dst = t * out_stride_token + d + p * 8;
      *reinterpret_cast<uint4*>(y + dst) = outp.v;
    }
  }
}

template <typename scalar_t, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_topk(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;

#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_reduce_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);

#pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }

      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

// -------------------- general-topk fallback kernels --------------------
// small-token
template <typename scalar_t>
__global__ void moe_sum_reduce_kernel_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {
  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
#pragma unroll 1
      for (int k = 0; k < topk_num; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }
      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

// warp-per-token
template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_kernel_warp_token_general(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {
  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;
#pragma unroll 1
    for (int k = 0; k < topk_num; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

void moe_sum_reduce(Tensor& input, Tensor& output, double routed_scaling_factor) {
  SGL_CHECK(input.is_cuda(), "input must be CUDA tensor");
  SGL_CHECK(output.is_cuda(), "output must be CUDA tensor");
  SGL_CHECK(input.dim() == 3, "input must be a 3D tensor like [token_num, topk_num, hidden_dim]");
  SGL_CHECK(output.dim() == 2, "output must be [token_num, hidden_dim]");
  SGL_CHECK(input.size(0) == output.size(0), "token dim mismatch");
  SGL_CHECK(input.size(2) == output.size(1), "hidden_dim mismatch");

  SGL_CHECK(input.is_contiguous(), "expect input to be contiguous");
  SGL_CHECK(output.is_contiguous(), "expect output to be contiguous");

  const int64_t token_num = input.size(0);
  const int64_t topk_num = input.size(1);
  const int64_t hidden_dim = input.size(2);

  const int64_t in_stride_token = input.stride(0);
  const int64_t in_stride_topk = input.stride(1);
  const int64_t out_stride_token = output.stride(0);

  auto stream = SGL_CURRENT_CUDA_STREAM();

  const bool fast_bf16_vec_ok =
      (input.scalar_type() == ScalarType::BFloat16) && (token_num > 256) && (hidden_dim % 8 == 0);

  // Fast path for bf16 vectorize
  if (fast_bf16_vec_ok) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    const int64_t n_chunks = hidden_dim / 8;
    int64_t grid_x = (n_chunks + 32 - 1) / 32;
    if (grid_x > 65535) grid_x = 65535;

    int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (grid_y > 65535) grid_y = 65535;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    auto stream = SGL_CURRENT_CUDA_STREAM();

    const float scale = static_cast<float>(routed_scaling_factor);
    moe_sum_reduce_warp_per_token_vec_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
        reinterpret_cast<const BFloat16Type*>(SGL_CONST_DATA_PTR(input, BFloat16Type)),
        reinterpret_cast<BFloat16Type*>(SGL_MUTABLE_DATA_PTR(output, BFloat16Type)),
        token_num,
        hidden_dim,
        topk_num,
        in_stride_token,
        in_stride_topk,
        out_stride_token,
        scale);

    SGL_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (bf16 vec) launch failed");
    return;
  }

  const bool per_token_use_one_warp = (token_num > 128);

  if (!per_token_use_one_warp) {
    // ---------- small-token ----------
    const int block_size = 256;
    int64_t grid_x = (hidden_dim + block_size - 1) / block_size;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num < 65535 ? token_num : 65535;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

#define LAUNCH_SMALL_TOKEN_KERNEL(TOPK)                               \
  moe_sum_reduce_kernel<scalar_t_, TOPK><<<grid, block, 0, stream>>>( \
      SGL_CONST_DATA_PTR(input, scalar_t_),                           \
      SGL_MUTABLE_DATA_PTR(output, scalar_t_),                        \
      token_num,                                                      \
      hidden_dim,                                                     \
      in_stride_token,                                                \
      in_stride_topk,                                                 \
      out_stride_token,                                               \
      scale);

    SGL_DISPATCH_FLOATING_TYPES_AND2(input.scalar_type(), "moe_sum_reduce_cuda_small_token", [&] {
      using scalar_t_ = scalar_t;
      using acc_t_ = opmath_t<scalar_t_>;
      const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

      switch (topk_num) {
        case 2:
          LAUNCH_SMALL_TOKEN_KERNEL(2);
          break;
        case 4:
          LAUNCH_SMALL_TOKEN_KERNEL(4);
          break;
        case 8:
          LAUNCH_SMALL_TOKEN_KERNEL(8);
          break;
        case 9:
          LAUNCH_SMALL_TOKEN_KERNEL(9);
          break;
        default:  // launch general kernel
          moe_sum_reduce_kernel_general<scalar_t_><<<grid, block, 0, stream>>>(
              SGL_CONST_DATA_PTR(input, scalar_t_),
              SGL_MUTABLE_DATA_PTR(output, scalar_t_),
              token_num,
              hidden_dim,
              in_stride_token,
              in_stride_topk,
              out_stride_token,
              static_cast<int>(topk_num),
              scale);
      }
    });
#undef LAUNCH_SMALL_TOKEN_KERNEL

    SGL_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (small-token) launch failed");

  } else {
    // ---------- warp-per-token ----------
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int64_t gx = (hidden_dim + 32 - 1) / 32;
    gx = gx > 65535 ? 65535 : gx;

    int64_t gy = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gy = gy > 65535 ? 65535 : gy;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));

#define LAUNCH_WARP_PER_TOKEN_KERNEL(TOPK)                                                             \
  moe_sum_reduce_kernel_warp_token_topk<scalar_t_, TOPK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>( \
      SGL_CONST_DATA_PTR(input, scalar_t_),                                                            \
      SGL_MUTABLE_DATA_PTR(output, scalar_t_),                                                         \
      token_num,                                                                                       \
      hidden_dim,                                                                                      \
      in_stride_token,                                                                                 \
      in_stride_topk,                                                                                  \
      out_stride_token,                                                                                \
      scale);

    SGL_DISPATCH_FLOATING_TYPES_AND2(input.scalar_type(), "moe_sum_reduce_cuda_large_token", [&] {
      using scalar_t_ = scalar_t;
      using acc_t_ = opmath_t<scalar_t_>;
      const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

      switch (topk_num) {
        case 2:
          LAUNCH_WARP_PER_TOKEN_KERNEL(2);
          break;
        case 4:
          LAUNCH_WARP_PER_TOKEN_KERNEL(4);
          break;
        case 8:
          LAUNCH_WARP_PER_TOKEN_KERNEL(8);
          break;
        case 9:
          LAUNCH_WARP_PER_TOKEN_KERNEL(9);
          break;
        default:  // launch general kernel
          moe_sum_reduce_kernel_warp_token_general<scalar_t_, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
              SGL_CONST_DATA_PTR(input, scalar_t_),
              SGL_MUTABLE_DATA_PTR(output, scalar_t_),
              token_num,
              hidden_dim,
              in_stride_token,
              in_stride_topk,
              out_stride_token,
              static_cast<int>(topk_num),
              scale);
      }
    });
#undef LAUNCH_WARP_PER_TOKEN_KERNEL

    SGL_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (warp-token) launch failed");
  }
}

#undef SGL_CHECK
#undef SGL_CURRENT_CUDA_STREAM
#undef SGL_CONST_DATA_PTR
#undef SGL_MUTABLE_DATA_PTR
#undef SGL_DISPATCH_FLOATING_TYPES_AND2
