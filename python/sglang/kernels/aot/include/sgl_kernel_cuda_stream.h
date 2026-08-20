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
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/util/shim_utils.h>

namespace sgl_kernel::stable {

inline cudaStream_t get_current_cuda_stream(torch::stable::accelerator::DeviceIndex device_index) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
  return static_cast<cudaStream_t>(stream_ptr);
}

// Preserve legacy no-argument stream selection for kernels whose existing
// contract is tied to the process's current CUDA device.
inline cudaStream_t get_current_cuda_stream() {
  return get_current_cuda_stream(torch::stable::accelerator::getCurrentDeviceIndex());
}

}  // namespace sgl_kernel::stable
