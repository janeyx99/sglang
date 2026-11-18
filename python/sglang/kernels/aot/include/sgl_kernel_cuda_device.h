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

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/Exception.h>

#include <deque>
#include <mutex>
#include <vector>

namespace sgl_kernel::stable {
namespace detail {

inline std::deque<std::once_flag> device_property_flags;
inline std::vector<cudaDeviceProp> device_properties;
inline std::once_flag device_property_vectors_init_flag;

inline void init_device_property_vectors() {
  int device_count = 0;
  const cudaError_t error = cudaGetDeviceCount(&device_count);
  STD_TORCH_CHECK(error == cudaSuccess, "cudaGetDeviceCount failed: ", cudaGetErrorString(error));
  device_property_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void init_device_property(int device_index) {
  cudaDeviceProp device_property{};
  const cudaError_t error = cudaGetDeviceProperties(&device_property, device_index);
  STD_TORCH_CHECK(error == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(error));
  device_properties[device_index] = device_property;
}

}  // namespace detail

inline const cudaDeviceProp& get_cached_device_properties(torch::stable::accelerator::DeviceIndex device_index) {
  std::call_once(detail::device_property_vectors_init_flag, detail::init_device_property_vectors);
  STD_TORCH_CHECK(
      device_index >= 0 && static_cast<size_t>(device_index) < detail::device_properties.size(),
      "CUDA device index ",
      device_index,
      " out of range [0, ",
      detail::device_properties.size(),
      ")");
  std::call_once(detail::device_property_flags[device_index], detail::init_device_property, device_index);
  return detail::device_properties[device_index];
}

inline const cudaDeviceProp& get_cached_device_properties() {
  return get_cached_device_properties(torch::stable::accelerator::getCurrentDeviceIndex());
}

}  // namespace sgl_kernel::stable
