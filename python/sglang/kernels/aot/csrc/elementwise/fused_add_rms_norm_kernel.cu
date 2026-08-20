/* Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <flashinfer/norm.cuh>

#include "elementwise/elementwise_ops.h"
#include "sgl_kernel_cuda_stream.h"

using namespace flashinfer;

void sgl_fused_add_rmsnorm(
    torch::stable::Tensor input,
    torch::stable::Tensor residual,
    torch::stable::Tensor weight,
    double eps,
    bool enable_pdl) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  STD_TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  STD_TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  STD_TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  const auto device_index = input.get_device_index();
  const auto residual_device_index = residual.get_device_index();
  STD_TORCH_CHECK(
      residual_device_index == device_index,
      "CHECK_EQ(residual.device(), device) failed. cuda:",
      residual_device_index,
      " vs cuda:",
      device_index);
  const auto weight_device_index = weight.get_device_index();
  STD_TORCH_CHECK(
      weight_device_index == device_index,
      "CHECK_EQ(weight.device(), device) failed. cuda:",
      weight_device_index,
      " vs cuda:",
      device_index);
  STD_TORCH_CHECK(input.dim() == 2, "input must be a 2D tensor");
  STD_TORCH_CHECK(residual.dim() == 2, "residual must be a 2D tensor");
  STD_TORCH_CHECK(weight.dim() == 1, "weight must be a 1D tensor");
  const auto batch_size = input.size(0);
  const auto residual_batch_size = residual.size(0);
  STD_TORCH_CHECK(
      batch_size == residual_batch_size,
      "CHECK_EQ(input.size(0), residual.size(0)) failed. ",
      batch_size,
      " vs ",
      residual_batch_size);
  const auto hidden_size = input.size(1);
  const auto residual_hidden_size = residual.size(1);
  STD_TORCH_CHECK(
      hidden_size == residual_hidden_size,
      "CHECK_EQ(input.size(1), residual.size(1)) failed. ",
      hidden_size,
      " vs ",
      residual_hidden_size);
  const auto weight_size = weight.size(0);
  STD_TORCH_CHECK(
      hidden_size == weight_size, "CHECK_EQ(input.size(1), weight.size(0)) failed. ", hidden_size, " vs ", weight_size);

  cudaStream_t torch_current_stream = sgl_kernel::stable::get_current_cuda_stream();
  // support float16, bfloat16 and float32
#define LAUNCH_FUSED_ADD_RMSNORM(c_type)                                                                           \
  do {                                                                                                             \
    cudaError_t status = norm::FusedAddRMSNorm(                                                                    \
        reinterpret_cast<c_type*>(input.mutable_data_ptr()),                                                       \
        reinterpret_cast<c_type*>(residual.mutable_data_ptr()),                                                    \
        const_cast<c_type*>(reinterpret_cast<const c_type*>(weight.const_data_ptr())),                             \
        batch_size,                                                                                                \
        hidden_size,                                                                                               \
        input.stride(0),                                                                                           \
        residual.stride(0),                                                                                        \
        eps,                                                                                                       \
        enable_pdl,                                                                                                \
        torch_current_stream);                                                                                     \
    STD_TORCH_CHECK(status == cudaSuccess, "FusedAddRMSNorm failed with error code ", cudaGetErrorString(status)); \
    return;                                                                                                        \
  } while (false)

  const auto scalar_type = input.scalar_type();
  switch (scalar_type) {
    case torch::headeronly::ScalarType::Float:
      LAUNCH_FUSED_ADD_RMSNORM(float);
    case torch::headeronly::ScalarType::Half:
      LAUNCH_FUSED_ADD_RMSNORM(__half);
#ifdef FLASHINFER_ENABLE_BF16
    case torch::headeronly::ScalarType::BFloat16:
      LAUNCH_FUSED_ADD_RMSNORM(nv_bfloat16);
#endif
    default:
      STD_TORCH_CHECK(
          false,
          "sgl_fused_add_rmsnorm(at::Tensor, at::Tensor, at::Tensor, double, bool)::<lambda()> failed to dispatch "
          "data type ",
          torch::headeronly::toString(scalar_type));
  }

#undef LAUNCH_FUSED_ADD_RMSNORM
}
