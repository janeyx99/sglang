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

#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace sgl_kernel::moe::stable {

template <std::size_t Rank>
inline torch::stable::Tensor empty_contiguous(
    const torch::stable::Device& device, const std::array<int64_t, Rank>& sizes, torch::headeronly::ScalarType dtype) {
  std::array<int64_t, Rank> strides{};
  int64_t stride = 1;
  for (std::size_t i = Rank; i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }

  const auto shim_dtype = torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
  const auto shim_device_type = torch::stable::detail::to<int32_t>(torch::stable::detail::from(device.type()));
  AtenTensorHandle tensor_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      Rank, sizes.data(), strides.data(), shim_dtype, shim_device_type, device.index(), &tensor_handle));
  return torch::stable::Tensor(tensor_handle);
}

inline torch::stable::Tensor
empty_contiguous_like(const torch::stable::Tensor& reference, int64_t size, torch::headeronly::ScalarType dtype) {
  return empty_contiguous(reference.device(), std::array<int64_t, 1>{size}, dtype);
}

}  // namespace sgl_kernel::moe::stable
