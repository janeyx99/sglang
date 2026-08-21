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

#include <torch/csrc/stable/tensor.h>

void es_fp8_blockwise_scaled_grouped_mm(
    torch::stable::Tensor& output,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& scales_a,
    const torch::stable::Tensor& scales_b,
    const torch::stable::Tensor& stride_a,
    const torch::stable::Tensor& stride_b,
    const torch::stable::Tensor& stride_d,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& workspace);

void es_sm100_mxfp8_blockscaled_grouped_mm(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& sfa,
    const torch::stable::Tensor& sfb,
    torch::stable::Tensor& d,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets);

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets,
    torch::stable::Tensor& quant_output,
    torch::stable::Tensor& scale_factor);
