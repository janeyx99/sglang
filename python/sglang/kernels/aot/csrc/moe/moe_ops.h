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

void moe_align_block_size(
    SglTensor topk_ids,
    int64_t num_experts,
    int64_t block_size,
    SglTensor sorted_token_ids,
    SglTensor experts_ids,
    SglTensor num_tokens_post_pad,
    SglTensor cumsum_buffer,
    bool pad_sorted_token_ids,
    bool ignore_invalid_expert);

void topk_softmax(
    SglTensor& topk_weights,
    SglTensor& topk_indices,
    SglTensor& gating_output,
    bool renormalize,
    double moe_softcapping,
    const SglOptional<SglTensor>& correction_bias);

void topk_sigmoid(
    SglTensor& topk_weights,
    SglTensor& topk_indices,
    SglTensor& gating_output,
    bool renormalize,
    const SglOptional<SglTensor>& correction_bias);

void moe_sum_reduce(SglTensor& input, SglTensor& output, double routed_scaling_factor);

void moe_sum(SglTensor& input, SglTensor& output);

void fp8_blockwise_scaled_grouped_mm(
    SglTensor& output,
    SglTensor& a_ptrs,
    SglTensor& b_ptrs,
    SglTensor& out_ptrs,
    SglTensor& a_scales_ptrs,
    SglTensor& b_scales_ptrs,
    const SglTensor& a,
    const SglTensor& b,
    const SglTensor& scales_a,
    const SglTensor& scales_b,
    const SglTensor& stride_a,
    const SglTensor& stride_b,
    const SglTensor& stride_c,
    const SglTensor& layout_sfa,
    const SglTensor& layout_sfb,
    const SglTensor& problem_sizes,
    const SglTensor& expert_offsets,
    const SglTensor& workspace);

void prepare_moe_input(
    const SglTensor& topk_ids,
    SglTensor& expert_offsets,
    const SglOptional<SglTensor>& blockscale_offsets,
    SglTensor& problem_sizes1,
    SglTensor& problem_sizes2,
    SglTensor& input_permutation,
    SglTensor& output_permutation,
    int64_t num_experts,
    int64_t n,
    int64_t k);

void shuffle_rows(const SglTensor& input_tensor, const SglTensor& dst2src_map, SglTensor& output_tensor);

void apply_shuffle_mul_sum(
    const SglTensor& input, SglTensor& output, const SglTensor& permutation, const SglOptional<SglTensor>& factors);

void fused_qk_norm_rope(
    SglTensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    SglTensor& q_weight,
    SglTensor& k_weight,
    double base,
    bool is_neox,
    SglTensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim);

void get_cutlass_w4a8_moe_mm_data(
    const SglTensor& topk_ids,
    SglTensor& expert_offsets,
    SglTensor& problem_sizes1,
    SglTensor& problem_sizes2,
    SglTensor& input_permutation,
    SglTensor& output_permutation,
    int64_t num_experts,
    int64_t n,
    int64_t k);

void cutlass_w4a8_moe_mm(
    SglTensor& d_tensors,
    const SglTensor& a_tensors,
    const SglTensor& b_tensors,
    const SglTensor& a_scales,
    const SglTensor& b_scales,
    const SglTensor& expert_offsets,
    const SglTensor& problem_sizes,
    const SglTensor& a_strides,
    const SglTensor& b_strides,
    const SglTensor& d_strides,
    const SglTensor& s_strides,
    int64_t chunk_size,
    int64_t topk);
