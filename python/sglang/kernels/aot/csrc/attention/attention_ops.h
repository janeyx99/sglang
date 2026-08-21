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

void merge_state_v2(SglTensor v_a, SglTensor s_a, SglTensor v_b, SglTensor s_b, SglTensor v_merged, SglTensor s_merged);

void cutlass_mla_decode(
    const SglTensor& out,
    const SglTensor& q_nope,
    const SglTensor& q_pe,
    const SglTensor& kv_c_and_k_pe_cache,
    const SglTensor& seq_lens,
    const SglTensor& page_table,
    const SglTensor& workspace,
    double sm_scale,
    int64_t num_kv_splits = 1);

int64_t cutlass_mla_get_workspace_size(
    int64_t max_seq_len, int64_t num_batches, int64_t sm_count = 0, int64_t num_kv_splits = 1);

void convert_vertical_slash_indexes(
    SglTensor& block_count,
    SglTensor& block_offset,
    SglTensor& column_count,
    SglTensor& column_index,
    SglTensor q_seqlens,
    SglTensor kv_seqlens,
    SglTensor vertical_indexes,
    SglTensor slash_indexes,
    int64_t context_size,
    int64_t block_size_M,
    int64_t block_size_N,
    bool causal);

void convert_vertical_slash_indexes_mergehead(
    SglTensor& block_count,
    SglTensor& block_offset,
    SglTensor& column_count,
    SglTensor& column_index,
    SglTensor q_seqlens,
    SglTensor kv_seqlens,
    SglTensor vertical_indexes,
    SglTensor slash_indexes,
    SglTensor vertical_indices_count,
    SglTensor slash_indices_count,
    int64_t context_size,
    int64_t block_size_M,
    int64_t block_size_N,
    bool causal);
