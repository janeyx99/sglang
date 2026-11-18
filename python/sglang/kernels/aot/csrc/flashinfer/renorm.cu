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

#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <flashinfer/sampling.cuh>
#include <string>

#include "flashinfer/flashinfer_ops.h"
#include "sgl_kernel_cuda_stream.h"

namespace {

#define CHECK_CUDA(x) STD_TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) STD_TORCH_CHECK((x).dim() == (d), #x " must be a " #d "D tensor")

}  // namespace

void top_p_renorm_probs(
    SglTensor probs, SglTensor renorm_probs, SglOptional<SglTensor> maybe_top_p_arr, double top_p_val) {
  CHECK_INPUT(probs);
  const auto device_index = probs.get_device_index();
  CHECK_DIM(2, probs);
  const auto batch_size = static_cast<unsigned int>(probs.size(0));
  const auto vocab_size = static_cast<unsigned int>(probs.size(1));
  const bool has_top_p_arr = maybe_top_p_arr.has_value();

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = sgl_kernel::stable::get_current_cuda_stream(device_index);
  const cudaError_t status = flashinfer::sampling::TopPRenormProb<float>(
      const_cast<float*>(probs.const_data_ptr<float>()),
      renorm_probs.mutable_data_ptr<float>(),
      has_top_p_arr ? const_cast<float*>(maybe_top_p_arr->const_data_ptr<float>()) : nullptr,
      batch_size,
      top_p_val,
      vocab_size,
      stream);
  STD_TORCH_CHECK(
      status == cudaSuccess, "TopPRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
}

void top_k_renorm_probs(
    SglTensor probs, SglTensor renorm_probs, SglOptional<SglTensor> maybe_top_k_arr, int64_t top_k_val) {
  CHECK_INPUT(probs);
  const auto device_index = probs.get_device_index();
  CHECK_DIM(2, probs);
  const auto batch_size = static_cast<unsigned int>(probs.size(0));
  const auto vocab_size = static_cast<unsigned int>(probs.size(1));
  const bool has_top_k_arr = maybe_top_k_arr.has_value();

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = sgl_kernel::stable::get_current_cuda_stream(device_index);
  const cudaError_t status = flashinfer::sampling::TopKRenormProb<float>(
      const_cast<float*>(probs.const_data_ptr<float>()),
      renorm_probs.mutable_data_ptr<float>(),
      has_top_k_arr ? const_cast<int*>(maybe_top_k_arr->const_data_ptr<int>()) : nullptr,
      batch_size,
      top_k_val,
      vocab_size,
      stream);
  STD_TORCH_CHECK(
      status == cudaSuccess, "TopKRenormProb failed with error code " + std::string(cudaGetErrorString(status)));
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef CHECK_DIM
