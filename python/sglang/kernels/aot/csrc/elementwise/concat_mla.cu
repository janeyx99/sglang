#ifdef TORCH_TARGET_VERSION
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "elementwise/elementwise_ops.h"
#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
#else
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>

#include "pytorch_extension_utils.h"
#endif
#include "utils.cuh"

constexpr int NUM_LOCAL_HEADS = 128;
constexpr int QK_NOPE_HEAD_DIM = 128;
constexpr int QK_ROPE_HEAD_DIM = 64;
constexpr int K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM;

constexpr int HEAD_CHUNK_SIZE = 16;
constexpr int NUM_HEAD_CHUNKS = NUM_LOCAL_HEADS / HEAD_CHUNK_SIZE;

__global__ void concat_mla_k_kernel(
    nv_bfloat16* __restrict__ k,
    const nv_bfloat16* __restrict__ k_nope,
    const nv_bfloat16* __restrict__ k_rope,
    const int num_tokens,
    const int64_t k_stride_0,
    const int k_stride_1,
    const int64_t k_nope_stride_0,
    const int k_nope_stride_1,
    const int64_t k_rope_stride_0) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int token_id = flat_warp_id / NUM_HEAD_CHUNKS;
  const int head_chunk_id = flat_warp_id % NUM_HEAD_CHUNKS;
  const int lane_id = get_lane_id();
  if (token_id >= num_tokens) return;

  using NopeVec = int2;  // 8B/thread，32 thread = 256B/row
  using RopeVec = int;   // 4B/thread，32 thread = 128B/row
  static_assert(sizeof(NopeVec) * 32 == QK_NOPE_HEAD_DIM * sizeof(nv_bfloat16), "nope vec mismatch");
  static_assert(sizeof(RopeVec) * 32 == QK_ROPE_HEAD_DIM * sizeof(nv_bfloat16), "rope vec mismatch");

  const int head_row0 = head_chunk_id * HEAD_CHUNK_SIZE;

  const int2* __restrict__ nope_src =
      reinterpret_cast<const int2*>(k_nope + token_id * k_nope_stride_0 + head_row0 * k_nope_stride_1) + lane_id;

  int2* __restrict__ nope_dst = reinterpret_cast<int2*>(k + token_id * k_stride_0 + head_row0 * k_stride_1) + lane_id;

  int* __restrict__ rope_dst =
      reinterpret_cast<int*>(k + token_id * k_stride_0 + head_row0 * k_stride_1 + QK_NOPE_HEAD_DIM) + lane_id;

  const int nope_src_stride_v = (k_nope_stride_1 >> 2);  // int2 covers 4 bf16
  const int nope_dst_stride_v = (k_stride_1 >> 2);
  const int rope_dst_stride_v = (k_stride_1 >> 1);  // int covers 2 bf16

  const int* rope_base = reinterpret_cast<const int*>(k_rope + token_id * k_rope_stride_0);
  const RopeVec rope_val = ld_na_global_v1(rope_base + lane_id);

  prefetch_L2(nope_src);
  NopeVec cur = ld_na_global_v2(nope_src);

#pragma unroll
  for (int i = 0; i < HEAD_CHUNK_SIZE; ++i) {
    NopeVec next;
    if (i + 1 < HEAD_CHUNK_SIZE) {
      const int2* next_src = nope_src + nope_src_stride_v;
      prefetch_L2(next_src);
      next = ld_na_global_v2(next_src);
    }

    st_na_global_v2(nope_dst, cur);
    st_na_global_v1(rope_dst, rope_val);

    nope_src += nope_src_stride_v;
    nope_dst += nope_dst_stride_v;
    rope_dst += rope_dst_stride_v;

    cur = next;
  }
}

#ifdef TORCH_TARGET_VERSION
inline const char* legacy_type_meta_name(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::Byte:
      return "unsigned char";
    case ScalarType::Char:
      return "signed char";
    case ScalarType::Short:
      return "short int";
    case ScalarType::Int:
      return "int";
    case ScalarType::Long:
      return "long int";
    case ScalarType::Half:
      return "c10::Half";
    case ScalarType::Float:
      return "float";
    case ScalarType::Double:
      return "double";
    case ScalarType::ComplexHalf:
      return "c10::complex<c10::Half>";
    case ScalarType::ComplexFloat:
      return "c10::complex<float>";
    case ScalarType::ComplexDouble:
      return "c10::complex<double>";
    case ScalarType::Bool:
      return "bool";
    case ScalarType::QInt8:
      return "c10::qint8";
    case ScalarType::QUInt8:
      return "c10::quint8";
    case ScalarType::QInt32:
      return "c10::qint32";
    case ScalarType::BFloat16:
      return "c10::BFloat16";
    case ScalarType::QUInt4x2:
      return "c10::quint4x2";
    case ScalarType::QUInt2x4:
      return "c10::quint2x4";
    case ScalarType::Bits1x8:
      return "c10::bits1x8";
    case ScalarType::Bits2x4:
      return "c10::bits2x4";
    case ScalarType::Bits4x2:
      return "c10::bits4x2";
    case ScalarType::Bits8:
      return "c10::bits8";
    case ScalarType::Bits16:
      return "c10::bits16";
    case ScalarType::Float8_e5m2:
      return "c10::Float8_e5m2";
    case ScalarType::Float8_e4m3fn:
      return "c10::Float8_e4m3fn";
    case ScalarType::Float8_e5m2fnuz:
      return "c10::Float8_e5m2fnuz";
    case ScalarType::Float8_e4m3fnuz:
      return "c10::Float8_e4m3fnuz";
    case ScalarType::UInt16:
      return "short unsigned int";
    case ScalarType::UInt32:
      return "unsigned int";
    case ScalarType::UInt64:
      return "long unsigned int";
    case ScalarType::Float8_e8m0fnu:
      return "c10::Float8_e8m0fnu";
    case ScalarType::Float4_e2m1fn_x2:
      return "c10::Float4_e2m1fn_x2";
    default:
      return torch::headeronly::toString(dtype);
  }
}

template <typename Lhs, typename Rhs>
inline void check_eq(bool condition, const char* expression, const Lhs& lhs, const Rhs& rhs) {
  STD_TORCH_CHECK(condition, "Check failed: ", expression, " (", lhs, " vs. ", rhs, "). ");
}

inline void check_tensor(const Tensor& t, int64_t shape0, int64_t shape1, int64_t shape2, ScalarType dtype) {
  const auto dim = t.dim();
  check_eq(dim == 3, "t.dim() == 3", dim, 3);
  const auto size0 = t.size(0);
  check_eq(size0 == shape0, "t.size(0) == shape0", size0, shape0);
  const auto size1 = t.size(1);
  check_eq(size1 == shape1, "t.size(1) == shape1", size1, shape1);
  const auto size2 = t.size(2);
  check_eq(size2 == shape2, "t.size(2) == shape2", size2, shape2);
  const auto actual_dtype = t.scalar_type();
  check_eq(
      actual_dtype == dtype,
      "t.dtype() == dtype",
      legacy_type_meta_name(actual_dtype),
      torch::headeronly::toString(dtype));
  STD_TORCH_CHECK(
      t.is_cuda(),
      "Expected t.device().is_cuda() to be true, but got false.  "
      "(Could this error message be improved?  If so, please report an enhancement request to PyTorch.)");
  const auto alignment = reinterpret_cast<int64_t>(t.const_data_ptr()) % 16;
  check_eq(alignment == 0, "((int64_t)t.data_ptr()) % 16 == 0", alignment, 0);
}

void concat_mla_k(Tensor k, Tensor k_nope, Tensor k_rope) {
  const int num_tokens = k.size(0);

  check_tensor(k, num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, ScalarType::BFloat16);
  check_tensor(k_nope, num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, ScalarType::BFloat16);
  check_tensor(k_rope, num_tokens, 1, QK_ROPE_HEAD_DIM, ScalarType::BFloat16);
  const auto k_stride2 = k.stride(2);
  check_eq(k_stride2 == 1, "k.stride(2) == 1", k_stride2, 1);
  const auto k_nope_stride2 = k_nope.stride(2);
  check_eq(k_nope_stride2 == 1, "k_nope.stride(2) == 1", k_nope_stride2, 1);
  const auto k_rope_stride2 = k_rope.stride(2);
  check_eq(k_rope_stride2 == 1, "k_rope.stride(2) == 1", k_rope_stride2, 1);

  const auto stream = sgl_kernel::stable::get_current_cuda_stream();

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  concat_mla_k_kernel<<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<nv_bfloat16*>(k.mutable_data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(k_nope.const_data_ptr()),
      reinterpret_cast<const nv_bfloat16*>(k_rope.const_data_ptr()),
      num_tokens,
      k.stride(0),
      k.stride(1),
      k_nope.stride(0),
      k_nope.stride(1),
      k_rope.stride(0));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
#else
inline void check_tensor(const at::Tensor& t, int64_t shape0, int64_t shape1, int64_t shape2, c10::ScalarType dtype) {
  TORCH_CHECK_EQ(t.dim(), 3);
  TORCH_CHECK_EQ(t.size(0), shape0);
  TORCH_CHECK_EQ(t.size(1), shape1);
  TORCH_CHECK_EQ(t.size(2), shape2);
  TORCH_CHECK_EQ(t.dtype(), dtype);
  TORCH_CHECK(t.device().is_cuda());
  TORCH_CHECK_EQ(((int64_t)t.data_ptr()) % 16, 0);  // alignment
}

void concat_mla_k(at::Tensor k, at::Tensor k_nope, at::Tensor k_rope) {
  const int num_tokens = k.size(0);

  check_tensor(k, num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM, at::kBFloat16);
  check_tensor(k_nope, num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM, at::kBFloat16);
  check_tensor(k_rope, num_tokens, 1, QK_ROPE_HEAD_DIM, at::kBFloat16);
  TORCH_CHECK_EQ(k.stride(2), 1);
  TORCH_CHECK_EQ(k_nope.stride(2), 1);
  TORCH_CHECK_EQ(k_rope.stride(2), 1);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_tokens * NUM_HEAD_CHUNKS, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  concat_mla_k_kernel<<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<nv_bfloat16*>(k.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(k_nope.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(k_rope.data_ptr()),
      num_tokens,
      k.stride(0),
      k.stride(1),
      k_nope.stride(0),
      k_nope.stride(1),
      k_rope.stride(0));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
#endif

// ============================== concat_mla_absorb_q ==============================

// TODO give a name prefix, also maybe refactor code above
constexpr int A_LAST_DIM = 512;
constexpr int B_LAST_DIM = 64;

__global__ void concat_mla_absorb_q_kernel(
    nv_bfloat16* a,
    nv_bfloat16* b,
    nv_bfloat16* out,
    const int num_items,
    const int dim_1,
    const int64_t a_stride_0,
    const int a_stride_1,
    const int64_t b_stride_0,
    const int b_stride_1,
    const int64_t out_stride_0,
    const int out_stride_1) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane_id = get_lane_id();

  const int idx_0 = flat_warp_id / dim_1;
  const int idx_1 = flat_warp_id % dim_1;

  if (flat_warp_id >= num_items) {
    return;
  }

  using ABufType = int4;
  constexpr int A_NUM_UNROLL = 2;
  static_assert(sizeof(ABufType) * A_NUM_UNROLL == A_LAST_DIM * sizeof(a[0]) / 32);
  ABufType a_buf[A_NUM_UNROLL];

  using BBufType = int;
  constexpr int B_NUM_UNROLL = 1;
  static_assert(sizeof(BBufType) * B_NUM_UNROLL == B_LAST_DIM * sizeof(b[0]) / 32);
  BBufType b_buf;

  {
    const BBufType* base_addr = reinterpret_cast<BBufType*>(b + idx_0 * b_stride_0 + idx_1 * b_stride_1);
    b_buf = *(base_addr + lane_id);
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    const ABufType* base_addr = reinterpret_cast<ABufType*>(a + idx_0 * a_stride_0 + idx_1 * a_stride_1);
    a_buf[i] = *(base_addr + i * 32 + lane_id);
  }

  {
    BBufType* base_addr = reinterpret_cast<BBufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1 + A_LAST_DIM);
    *(base_addr + lane_id) = b_buf;
  }

#pragma unroll
  for (int i = 0; i < A_NUM_UNROLL; ++i) {
    ABufType* base_addr = reinterpret_cast<ABufType*>(out + idx_0 * out_stride_0 + idx_1 * out_stride_1);
    *(base_addr + i * 32 + lane_id) = a_buf[i];
  }
}

#ifdef TORCH_TARGET_VERSION
inline void check_tensor_concat_mla_absorb_q(const Tensor& t, int64_t shape2) {
  const auto dim = t.dim();
  check_eq(dim == 3, "t.dim() == 3", dim, 3);
  const auto size2 = t.size(2);
  check_eq(size2 == shape2, "t.size(2) == shape2", size2, shape2);
  const auto stride2 = t.stride(2);
  check_eq(stride2 == 1, "t.stride(2) == 1", stride2, 1);
  const auto actual_dtype = t.scalar_type();
  check_eq(
      actual_dtype == ScalarType::BFloat16,
      "t.dtype() == at::kBFloat16",
      legacy_type_meta_name(actual_dtype),
      "BFloat16");
  STD_TORCH_CHECK(
      t.is_cuda(),
      "Expected t.device().is_cuda() to be true, but got false.  "
      "(Could this error message be improved?  If so, please report an enhancement request to PyTorch.)");
  const auto alignment = reinterpret_cast<int64_t>(t.const_data_ptr()) % 16;
  check_eq(alignment == 0, "((int64_t)t.data_ptr()) % 16 == 0", alignment, 0);
}

// TODO further optimize it later
void concat_mla_absorb_q(Tensor a, Tensor b, Tensor out) {
  check_tensor_concat_mla_absorb_q(a, A_LAST_DIM);
  check_tensor_concat_mla_absorb_q(b, B_LAST_DIM);
  check_tensor_concat_mla_absorb_q(out, A_LAST_DIM + B_LAST_DIM);

  const auto stream = sgl_kernel::stable::get_current_cuda_stream();

  const auto a_num_items = a.size(0) * a.size(1);
  const auto b_num_items = b.size(0) * b.size(1);
  check_eq(a_num_items == b_num_items, "a.size(0) * a.size(1) == b.size(0) * b.size(1)", a_num_items, b_num_items);
  const auto a_size1 = a.size(1);
  const auto b_size1 = b.size(1);
  check_eq(a_size1 == b_size1, "a.size(1) == b.size(1)", a_size1, b_size1);
  const int num_items = a_num_items;

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_items, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  concat_mla_absorb_q_kernel<<<grid_size, block_size, 0, stream>>>(
      const_cast<nv_bfloat16*>(reinterpret_cast<const nv_bfloat16*>(a.const_data_ptr())),
      const_cast<nv_bfloat16*>(reinterpret_cast<const nv_bfloat16*>(b.const_data_ptr())),
      reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()),
      num_items,
      a.size(1),
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      out.stride(0),
      out.stride(1));
  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
#else
inline void check_tensor_concat_mla_absorb_q(const at::Tensor& t, int64_t shape2) {
  TORCH_CHECK_EQ(t.dim(), 3);
  TORCH_CHECK_EQ(t.size(2), shape2);
  TORCH_CHECK_EQ(t.stride(2), 1);
  TORCH_CHECK_EQ(t.dtype(), at::kBFloat16);
  TORCH_CHECK(t.device().is_cuda());
  TORCH_CHECK_EQ(((int64_t)t.data_ptr()) % 16, 0);  // alignment
}

// TODO further optimize it later
void concat_mla_absorb_q(at::Tensor a, at::Tensor b, at::Tensor out) {
  check_tensor_concat_mla_absorb_q(a, A_LAST_DIM);
  check_tensor_concat_mla_absorb_q(b, B_LAST_DIM);
  check_tensor_concat_mla_absorb_q(out, A_LAST_DIM + B_LAST_DIM);

  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(a.size(0) * a.size(1), b.size(0) * b.size(1));
  TORCH_CHECK_EQ(a.size(1), b.size(1));
  const int num_items = a.size(0) * a.size(1);

  constexpr int num_warps_per_block = 32;
  const int grid_size = ceil_div(num_items, num_warps_per_block);
  const int block_size = num_warps_per_block * 32;

  concat_mla_absorb_q_kernel<<<grid_size, block_size, 0, stream>>>(
      reinterpret_cast<nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(b.data_ptr()),
      reinterpret_cast<nv_bfloat16*>(out.data_ptr()),
      num_items,
      a.size(1),
      a.stride(0),
      a.stride(1),
      b.stride(0),
      b.stride(1),
      out.stride(0),
      out.stride(1));
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}
#endif
