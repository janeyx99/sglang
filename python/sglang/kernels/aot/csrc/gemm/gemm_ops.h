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

#include <cstdint>

#include "sgl_kernel_torch_compat.h"

SglTensor awq_dequantize(SglTensor qweight, SglTensor scales, SglTensor qzeros);

SglTensor int8_scaled_mm(
    const SglTensor& mat_a,
    const SglTensor& mat_b,
    const SglTensor& scales_a,
    const SglTensor& scales_b,
    const SglScalarType& out_dtype,
    const SglOptional<SglTensor>& bias);

SglTensor fp8_scaled_mm(
    const SglTensor& mat_a,
    const SglTensor& mat_b,
    const SglTensor& scales_a,
    const SglTensor& scales_b,
    const SglScalarType& out_dtype,
    const SglOptional<SglTensor>& bias);

void sgl_per_token_group_quant_8bit(
    SglTensor input,
    SglTensor output_q,
    SglTensor output_s,
    int64_t group_size,
    double eps,
    double fp8_min,
    double fp8_max,
    bool scale_ue8m0);

void sgl_per_token_group_quant_8bit_v2(
    SglTensor input,
    SglTensor output_q,
    SglTensor output_s,
    int64_t group_size,
    double eps,
    double min_8bit,
    double max_8bit,
    bool scale_ue8m0,
    bool fuse_silu_and_mul,
    const SglOptional<SglTensor>& masked_m);

void sgl_per_token_quant_fp8(SglTensor input, SglTensor output_q, SglTensor output_s);

SglTensor gptq_gemm(
    SglTensor a,
    SglTensor b_q_weight,
    SglTensor b_gptq_qzeros,
    SglTensor b_gptq_scales,
    SglTensor b_g_idx,
    bool use_shuffle,
    int64_t bit);

void gptq_shuffle(SglTensor q_weight, SglTensor q_perm, int64_t bit);
