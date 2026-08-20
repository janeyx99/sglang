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

void silu_and_mul(SglTensor& out, SglTensor& input);
void gelu_tanh_and_mul(SglTensor& out, SglTensor& input);
void gelu_and_mul(SglTensor& out, SglTensor& input);

void concat_mla_k(SglTensor k, SglTensor k_nope, SglTensor k_rope);
void concat_mla_absorb_q(SglTensor a, SglTensor b, SglTensor out);

void copy_to_gpu_no_ce(const SglTensor& input, SglTensor& output);

void dsv4_fused_q_norm_rope(
    const SglTensor& q_input, SglTensor& q_output, const SglTensor& freqs_cis, const SglTensor& positions, double eps);

void dsv4_fused_k_norm_rope_flashmla(
    const SglTensor& kv,
    const SglTensor& kv_weight,
    const SglTensor& freqs_cis,
    const SglTensor& positions,
    const SglTensor& out_loc,
    SglTensor& kvcache,
    double eps,
    int64_t page_size);

void dsv4_fused_q_indexer_rope_hadamard_quant(
    const SglTensor& q_input,
    SglTensor& q_fp8,
    const SglTensor& weight,
    SglTensor& weights_out,
    double weight_scale,
    const SglTensor& freqs_cis,
    const SglTensor& positions);

void sgl_fused_add_rmsnorm(SglTensor input, SglTensor residual, SglTensor weight, double eps, bool enable_pdl);

void rotary_embedding(
    SglTensor& positions,
    SglTensor& query,
    SglOptional<SglTensor> key,
    int64_t head_size,
    SglTensor& cos_sin_cache,
    bool is_neox);

void fast_topk_interface(
    const SglTensor& score, SglTensor& indices, const SglTensor& lengths, SglOptional<SglTensor> row_starts_opt);

void fast_topk_transform_interface(
    const SglTensor& score,
    const SglTensor& lengths,
    SglTensor& dst_page_table,
    const SglTensor& src_page_table,
    const SglTensor& cu_seqlens_q,
    SglOptional<SglTensor> row_starts_opt);

void fast_topk_transform_ragged_interface(
    const SglTensor& score,
    const SglTensor& lengths,
    SglTensor& topk_indices_ragged,
    const SglTensor& topk_indices_offset,
    SglOptional<SglTensor> row_starts_opt);
