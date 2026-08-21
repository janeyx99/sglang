#pragma once

#include <cuda.h>
#include <torch/headeronly/util/Exception.h>

#include "cutlass/bfloat16.h"
#include "cutlass/float8.h"
#include "moe/moe_ops.h"
#include "sgl_kernel_cuda_stream.h"

#define W4A8_CHECK_NO_MSG(condition_, condition_text_)                                                \
  STD_TORCH_CHECK(                                                                                    \
      (condition_),                                                                                   \
      "Expected " condition_text_                                                                     \
      " to be true, but got false.  (Could this error message be improved?  If so, please report an " \
      "enhancement request to PyTorch.)")
#define W4A8_MUTABLE_RAW_PTR(tensor_) (tensor_).mutable_data_ptr()

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
__global__ void int4_fp8_get_group_gemm_starts(
    const int32_t* expert_offsets,
    ElementA** a_offsets,
    ElementB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    ElementA* a_base_as_int,
    ElementB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int,
    int64_t n,
    int64_t k,
    bool per_act_token,
    bool per_out_ch) {
  int expert_id = threadIdx.x;
  int32_t expert_offset = expert_offsets[expert_id];

  a_offsets[expert_id] = a_base_as_int + expert_offset * k;
  b_offsets[expert_id] = b_base_as_int + expert_id * k * n / 2;
  out_offsets[expert_id] = out_base_as_int + expert_offset * n;
  a_scales_offsets[expert_id] = a_scales_base_as_int + (per_act_token ? expert_offset : 0);
  b_scales_offsets[expert_id] = b_scales_base_as_int + (per_out_ch ? expert_id * n * k / 128 : expert_id);
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
__global__ void int4_fp8_get_group_gemm_starts_3d(
    ElementA** a_offsets,
    ElementB** b_offsets,
    ElementC** out_offsets,
    ElementAccumulator** a_scales_offsets,
    cutlass::bfloat16_t** b_scales_offsets,
    ElementA* a_base_as_int,
    ElementB* b_base_as_int,
    ElementC* out_base_as_int,
    ElementAccumulator* a_scales_base_as_int,
    cutlass::bfloat16_t* b_scales_base_as_int,
    int64_t n,
    int64_t m,
    int64_t k,
    bool per_act_token,
    bool per_out_ch,
    int num_experts) {
  int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert_id >= num_experts) return;

  int64_t a_offset = expert_id * m * k;
  int64_t b_offset = expert_id * k * n / 2;
  int64_t out_offset = expert_id * m * n;
  int64_t a_scales_offset = 0;
  int64_t b_scales_offset = per_out_ch ? expert_id * n * 4 * k / 512 : expert_id;

  a_offsets[expert_id] = a_base_as_int + a_offset;
  b_offsets[expert_id] = b_base_as_int + b_offset;
  out_offsets[expert_id] = out_base_as_int + out_offset;
  a_scales_offsets[expert_id] = a_scales_base_as_int + a_scales_offset;
  b_scales_offsets[expert_id] = b_scales_base_as_int + b_scales_offset;
}

#define __CALL_W4A8_GET_STARTS_KERNEL(TENSOR_C_TYPE, C_TYPE)                                    \
  else if (out_tensors.scalar_type() == TENSOR_C_TYPE) {                                        \
    int4_fp8_get_group_gemm_starts<cutlass::float_e4m3_t, cutlass::int8_t, C_TYPE, float>       \
        <<<1, num_experts, 0, stream>>>(                                                        \
            static_cast<const int32_t*>(expert_offsets.const_data_ptr()),                       \
            static_cast<cutlass::float_e4m3_t**>(W4A8_MUTABLE_RAW_PTR(a_ptrs)),                 \
            static_cast<cutlass::int8_t**>(W4A8_MUTABLE_RAW_PTR(b_ptrs)),                       \
            static_cast<C_TYPE**>(W4A8_MUTABLE_RAW_PTR(out_ptrs)),                              \
            static_cast<float**>(W4A8_MUTABLE_RAW_PTR(a_scales_ptrs)),                          \
            static_cast<cutlass::bfloat16_t**>(W4A8_MUTABLE_RAW_PTR(b_scales_ptrs)),            \
            static_cast<cutlass::float_e4m3_t*>(const_cast<void*>(a_tensors.const_data_ptr())), \
            static_cast<cutlass::int8_t*>(const_cast<void*>(b_tensors.const_data_ptr())),       \
            static_cast<C_TYPE*>(W4A8_MUTABLE_RAW_PTR(out_tensors)),                            \
            static_cast<float*>(const_cast<void*>(a_scales.const_data_ptr())),                  \
            static_cast<cutlass::bfloat16_t*>(const_cast<void*>(b_scales.const_data_ptr())),    \
            out_tensors.size(1),                                                                \
            a_tensors.size(1),                                                                  \
            per_act_token,                                                                      \
            per_out_ch);                                                                        \
  }

#define __CALL_W4A8_GET_STARTS_KERNEL_3D(TENSOR_C_TYPE, C_TYPE)                                 \
  else if (out_tensors.scalar_type() == TENSOR_C_TYPE) {                                        \
    int4_fp8_get_group_gemm_starts_3d<cutlass::float_e4m3_t, cutlass::int8_t, C_TYPE, float>    \
        <<<1, num_experts, 0, stream>>>(                                                        \
            static_cast<cutlass::float_e4m3_t**>(W4A8_MUTABLE_RAW_PTR(a_ptrs)),                 \
            static_cast<cutlass::int8_t**>(W4A8_MUTABLE_RAW_PTR(b_ptrs)),                       \
            static_cast<C_TYPE**>(W4A8_MUTABLE_RAW_PTR(out_ptrs)),                              \
            static_cast<float**>(W4A8_MUTABLE_RAW_PTR(a_scales_ptrs)),                          \
            static_cast<cutlass::bfloat16_t**>(W4A8_MUTABLE_RAW_PTR(b_scales_ptrs)),            \
            static_cast<cutlass::float_e4m3_t*>(const_cast<void*>(a_tensors.const_data_ptr())), \
            static_cast<cutlass::int8_t*>(const_cast<void*>(b_tensors.const_data_ptr())),       \
            static_cast<C_TYPE*>(W4A8_MUTABLE_RAW_PTR(out_tensors)),                            \
            static_cast<float*>(const_cast<void*>(a_scales.const_data_ptr())),                  \
            static_cast<cutlass::bfloat16_t*>(const_cast<void*>(b_scales.const_data_ptr())),    \
            out_tensors.size(2),                                                                \
            a_tensors.size(1),                                                                  \
            a_tensors.size(2),                                                                  \
            per_act_token,                                                                      \
            per_out_ch,                                                                         \
            num_experts);                                                                       \
  }

namespace {

void run_int4_fp8_get_group_gemm_starts(
    const SglTensor& expert_offsets,
    SglTensor& a_ptrs,
    SglTensor& b_ptrs,
    SglTensor& out_ptrs,
    SglTensor& a_scales_ptrs,
    SglTensor& b_scales_ptrs,
    const SglTensor& a_tensors,
    const SglTensor& b_tensors,
    SglTensor& out_tensors,
    const SglTensor& a_scales,
    const SglTensor& b_scales) {
  W4A8_CHECK_NO_MSG(
      a_tensors.scalar_type() == SglScalarType::Float8_e4m3fn, "a_tensors.dtype() == torch::kFloat8_e4m3fn");
  W4A8_CHECK_NO_MSG(b_tensors.scalar_type() == SglScalarType::Char, "b_tensors.dtype() == torch::kInt8");
  W4A8_CHECK_NO_MSG(a_scales.scalar_type() == SglScalarType::Float, "a_scales.dtype() == torch::kFloat32");
  W4A8_CHECK_NO_MSG(b_scales.scalar_type() == SglScalarType::BFloat16, "b_scales.dtype() == torch::kBFloat16");

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream(expert_offsets.get_device_index());

  if (a_tensors.dim() == 3) {
    if (false) {
    }
    __CALL_W4A8_GET_STARTS_KERNEL_3D(SglScalarType::BFloat16, cutlass::bfloat16_t)
    __CALL_W4A8_GET_STARTS_KERNEL_3D(SglScalarType::Half, half)
    else {
      STD_TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
    }
  } else {
    if (false) {
    }
    __CALL_W4A8_GET_STARTS_KERNEL(SglScalarType::BFloat16, cutlass::bfloat16_t)
    __CALL_W4A8_GET_STARTS_KERNEL(SglScalarType::Half, half)
    else {
      STD_TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
    }
  }
}

}  // namespace

#undef W4A8_CHECK_NO_MSG
#undef W4A8_MUTABLE_RAW_PTR
