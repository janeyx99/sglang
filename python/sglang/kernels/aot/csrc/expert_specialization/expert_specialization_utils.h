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
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>

#include <array>
#include <initializer_list>

namespace expert_specialization {

inline torch::stable::Tensor
empty(const torch::stable::Device& device, std::initializer_list<int64_t> sizes, torch::headeronly::ScalarType dtype) {
  std::array<int64_t, 2> strides{};
  STD_TORCH_CHECK(sizes.size() <= strides.size(), "expert-specialization allocations support rank <= 2");
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= *(sizes.begin() + i - 1);
  }

  const auto shim_dtype = torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
  const auto shim_device_type = torch::stable::detail::to<int32_t>(torch::stable::detail::from(device.type()));
  AtenTensorHandle tensor_handle = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      sizes.size(), sizes.begin(), strides.data(), shim_dtype, shim_device_type, device.index(), &tensor_handle));
  return torch::stable::Tensor(tensor_handle);
}

class CUDAEvent {
 public:
  CUDAEvent() = default;
  CUDAEvent(const CUDAEvent&) = delete;
  CUDAEvent& operator=(const CUDAEvent&) = delete;

  ~CUDAEvent() {
    if (event_ != nullptr) {
      int original_device = -1;
      if (cudaGetDevice(&original_device) == cudaSuccess && original_device != device_index_) {
        cudaSetDevice(device_index_);
        cudaEventDestroy(event_);
        cudaSetDevice(original_device);
      } else {
        cudaEventDestroy(event_);
      }
    }
  }

  void record_once(cudaStream_t stream) {
    if (recorded_) {
      return;
    }
    STD_CUDA_CHECK(cudaGetDevice(&device_index_));
    STD_CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    STD_CUDA_CHECK(cudaEventRecord(event_, stream));
    recorded_ = true;
  }

  void block(cudaStream_t stream) const {
    if (event_ != nullptr) {
      STD_CUDA_CHECK(cudaStreamWaitEvent(stream, event_, cudaEventWaitDefault));
    }
  }

 private:
  int device_index_ = -1;
  cudaEvent_t event_ = nullptr;
  bool recorded_ = false;
};

}  // namespace expert_specialization
