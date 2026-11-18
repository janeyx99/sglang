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

#include <optional>

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

using SglTensor = torch::stable::Tensor;
using SglScalarType = torch::headeronly::ScalarType;
template <typename T>
using SglOptional = std::optional<T>;
#define SGL_TORCH_CHECK STD_TORCH_CHECK
#define SGL_CONST_DATA_PTR(tensor) (tensor).const_data_ptr()
#define SGL_MUTABLE_DATA_PTR(tensor) (tensor).mutable_data_ptr()
#else
#include <ATen/ATen.h>

using SglTensor = at::Tensor;
using SglScalarType = at::ScalarType;
template <typename T>
using SglOptional = at::optional<T>;
#define SGL_TORCH_CHECK TORCH_CHECK
#define SGL_CONST_DATA_PTR(tensor) (tensor).data_ptr()
#define SGL_MUTABLE_DATA_PTR(tensor) (tensor).data_ptr()
#endif
