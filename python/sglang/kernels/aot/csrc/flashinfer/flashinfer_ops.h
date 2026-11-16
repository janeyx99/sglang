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

#include "sgl_kernel_torch_compat.h"

void rmsnorm(SglTensor& output, SglTensor& input, SglTensor& weight, double eps, bool enable_pdl);
void gemma_rmsnorm(SglTensor& output, SglTensor& input, SglTensor& weight, double eps, bool enable_pdl);
void gemma_fused_add_rmsnorm(SglTensor& input, SglTensor& residual, SglTensor& weight, double eps, bool enable_pdl);

void top_k_renorm_probs(
    SglTensor probs, SglTensor renorm_probs, SglOptional<SglTensor> maybe_top_k_arr, int64_t top_k_val);
void top_p_renorm_probs(
    SglTensor probs, SglTensor renorm_probs, SglOptional<SglTensor> maybe_top_p_arr, double top_p_val);
