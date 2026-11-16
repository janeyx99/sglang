/* Copyright (c) 2024 by FlashInfer team.
 * Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef FLASHINFER_ENABLE_F16
#include <cuda_fp16.h>
#endif
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <cstdint>
#include <flashinfer/norm.cuh>
#include <string>

#include "flashinfer/flashinfer_ops.h"
#include "sgl_kernel_cuda_stream.h"

using torch::headeronly::ScalarType;

namespace {

#define CHECK_CUDA(x) STD_TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_LAST_DIM_CONTIGUOUS(x) check_last_dim_contiguous((x), #x)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) STD_TORCH_CHECK((x).dim() == (d), #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) STD_TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", (a), " vs ", (b))

void check_last_dim_contiguous(const SglTensor& tensor, const char* name) {
  const auto strides = tensor.strides();
  STD_TORCH_CHECK(strides[strides.size() - 1] == 1, name, "must be contiguous at last dimension");
}

struct NormMetadata {
  torch::stable::accelerator::DeviceIndex device_index;
  unsigned int batch_size;
  unsigned int hidden_size;
};

template <typename storage_t, typename c_type>
void rmsnorm_impl(
    SglTensor& output,
    SglTensor& input,
    SglTensor& weight,
    unsigned int batch_size,
    unsigned int hidden_size,
    double eps,
    bool enable_pdl,
    cudaStream_t stream) {
  auto* input_ptr = reinterpret_cast<c_type*>(const_cast<storage_t*>(input.const_data_ptr<storage_t>()));
  auto* weight_ptr = reinterpret_cast<c_type*>(const_cast<storage_t*>(weight.const_data_ptr<storage_t>()));
  auto* output_ptr = reinterpret_cast<c_type*>(output.mutable_data_ptr<storage_t>());
  const cudaError_t status = flashinfer::norm::RMSNorm(
      input_ptr,
      weight_ptr,
      output_ptr,
      batch_size,
      hidden_size,
      input.stride(0),
      output.stride(0),
      eps,
      enable_pdl,
      stream);
  STD_TORCH_CHECK(status == cudaSuccess, "RMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
}

template <typename storage_t, typename c_type>
void gemma_rmsnorm_impl(
    SglTensor& output,
    SglTensor& input,
    SglTensor& weight,
    unsigned int batch_size,
    unsigned int hidden_size,
    double eps,
    bool enable_pdl,
    cudaStream_t stream) {
  auto* input_ptr = reinterpret_cast<c_type*>(const_cast<storage_t*>(input.const_data_ptr<storage_t>()));
  auto* weight_ptr = reinterpret_cast<c_type*>(const_cast<storage_t*>(weight.const_data_ptr<storage_t>()));
  auto* output_ptr = reinterpret_cast<c_type*>(output.mutable_data_ptr<storage_t>());
  const cudaError_t status = flashinfer::norm::GemmaRMSNorm(
      input_ptr,
      weight_ptr,
      output_ptr,
      batch_size,
      hidden_size,
      input.stride(0),
      output.stride(0),
      eps,
      enable_pdl,
      stream);
  STD_TORCH_CHECK(
      status == cudaSuccess, "GemmaRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
}

template <typename storage_t, typename c_type>
void gemma_fused_add_rmsnorm_impl(
    SglTensor& input,
    SglTensor& residual,
    SglTensor& weight,
    unsigned int batch_size,
    unsigned int hidden_size,
    double eps,
    bool enable_pdl,
    cudaStream_t stream) {
  auto* input_ptr = reinterpret_cast<c_type*>(input.mutable_data_ptr<storage_t>());
  auto* residual_ptr = reinterpret_cast<c_type*>(residual.mutable_data_ptr<storage_t>());
  auto* weight_ptr = reinterpret_cast<c_type*>(const_cast<storage_t*>(weight.const_data_ptr<storage_t>()));
  const cudaError_t status = flashinfer::norm::GemmaFusedAddRMSNorm(
      input_ptr,
      residual_ptr,
      weight_ptr,
      batch_size,
      hidden_size,
      input.stride(0),
      residual.stride(0),
      eps,
      enable_pdl,
      stream);
  STD_TORCH_CHECK(
      status == cudaSuccess, "GemmaFusedAddRMSNorm failed with error code " + std::string(cudaGetErrorString(status)));
}

template <typename storage_t_, typename c_type_>
struct DTypePair {
  using storage_t = storage_t_;
  using c_type = c_type_;
};

template <typename F>
void dispatch_fp16_bf16(ScalarType scalar_type, const char* name, F&& f) {
  switch (scalar_type) {
#ifdef FLASHINFER_ENABLE_F16
    case ScalarType::Half:
      f(DTypePair<torch::headeronly::Half, nv_half>{});
      return;
#endif
#ifdef FLASHINFER_ENABLE_BF16
    case ScalarType::BFloat16:
      f(DTypePair<torch::headeronly::BFloat16, nv_bfloat16>{});
      return;
#endif
    default:
      STD_TORCH_CHECK(false, name, " failed to dispatch data type ", torch::headeronly::toString(scalar_type));
  }
}

NormMetadata check_norm_inputs(SglTensor& output, SglTensor& input, SglTensor& weight) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  const auto device_index = input.get_device_index();
  CHECK_EQ(weight.get_device_index(), device_index);
  CHECK_DIM(2, input);
  CHECK_DIM(1, weight);
  const auto input_hidden_size = input.size(1);
  CHECK_EQ(input_hidden_size, weight.size(0));
  const auto batch_size = static_cast<unsigned int>(input.size(0));
  const auto hidden_size = static_cast<unsigned int>(input_hidden_size);
  CHECK_EQ(output.size(0), batch_size);
  CHECK_EQ(output.size(1), hidden_size);
  return {device_index, batch_size, hidden_size};
}

}  // namespace

void rmsnorm(SglTensor& output, SglTensor& input, SglTensor& weight, double eps, bool enable_pdl) {
  const auto metadata = check_norm_inputs(output, input, weight);
  const torch::stable::accelerator::DeviceGuard device_guard(metadata.device_index);
  const auto stream = sgl_kernel::stable::get_current_cuda_stream(metadata.device_index);
  dispatch_fp16_bf16(input.scalar_type(), "rmsnorm", [&](auto dtype) {
    using storage_t = typename decltype(dtype)::storage_t;
    using c_type = typename decltype(dtype)::c_type;
    rmsnorm_impl<storage_t, c_type>(
        output, input, weight, metadata.batch_size, metadata.hidden_size, eps, enable_pdl, stream);
  });
}

void gemma_rmsnorm(SglTensor& output, SglTensor& input, SglTensor& weight, double eps, bool enable_pdl) {
  const auto metadata = check_norm_inputs(output, input, weight);
  const torch::stable::accelerator::DeviceGuard device_guard(metadata.device_index);
  const auto stream = sgl_kernel::stable::get_current_cuda_stream(metadata.device_index);
  dispatch_fp16_bf16(input.scalar_type(), "gemma_rmsnorm", [&](auto dtype) {
    using storage_t = typename decltype(dtype)::storage_t;
    using c_type = typename decltype(dtype)::c_type;
    gemma_rmsnorm_impl<storage_t, c_type>(
        output, input, weight, metadata.batch_size, metadata.hidden_size, eps, enable_pdl, stream);
  });
}

void gemma_fused_add_rmsnorm(SglTensor& input, SglTensor& residual, SglTensor& weight, double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(residual);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  const auto device_index = input.get_device_index();
  CHECK_EQ(residual.get_device_index(), device_index);
  CHECK_EQ(weight.get_device_index(), device_index);
  CHECK_DIM(2, input);
  CHECK_DIM(2, residual);
  CHECK_DIM(1, weight);
  const auto input_batch_size = input.size(0);
  CHECK_EQ(input_batch_size, residual.size(0));
  const auto input_hidden_size = input.size(1);
  CHECK_EQ(input_hidden_size, residual.size(1));
  CHECK_EQ(input_hidden_size, weight.size(0));
  const auto batch_size = static_cast<unsigned int>(input_batch_size);
  const auto hidden_size = static_cast<unsigned int>(input_hidden_size);

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = sgl_kernel::stable::get_current_cuda_stream(device_index);
  dispatch_fp16_bf16(input.scalar_type(), "gemma_fused_add_rmsnorm", [&](auto dtype) {
    using storage_t = typename decltype(dtype)::storage_t;
    using c_type = typename decltype(dtype)::c_type;
    gemma_fused_add_rmsnorm_impl<storage_t, c_type>(
        input, residual, weight, batch_size, hidden_size, eps, enable_pdl, stream);
  });
}

#undef CHECK_CUDA
#undef CHECK_LAST_DIM_CONTIGUOUS
#undef CHECK_LAST_DIM_CONTIGUOUS_INPUT
#undef CHECK_DIM
#undef CHECK_EQ
