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

// Adapted from: https://github.com/vllm-project/vllm/blob/main/csrc/ops.h

#include "memory/weak_ref_tensor.h"

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/ops.h>
#else
#include <vector>
#endif

SglTensor weak_ref_tensor(const SglTensor& tensor) {
  SGL_TORCH_CHECK(tensor.is_cuda(), "weak_ref_tensor expects a CUDA tensor");

  void* data_ptr = SGL_MUTABLE_DATA_PTR(tensor);
#ifdef TORCH_TARGET_VERSION
  SGL_TORCH_CHECK(
      data_ptr != nullptr, "The specified pointer resides on host memory and is not registered with any CUDA device.");

  return torch::stable::from_blob(
      data_ptr, tensor.sizes(), tensor.strides(), tensor.device(), tensor.scalar_type(), 0, tensor.layout());
#else
  std::vector<int64_t> sizes = tensor.sizes().vec();
  std::vector<int64_t> strides = tensor.strides().vec();

  auto options = tensor.options();

  auto new_tensor = at::from_blob(data_ptr, sizes, strides, options);

  return new_tensor;
#endif
}
