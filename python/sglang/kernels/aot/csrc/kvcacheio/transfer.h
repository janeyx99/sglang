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
#include <vector>

#include "sgl_kernel_torch_compat.h"

void transfer_kv_per_layer(
    const SglTensor src_k,
    SglTensor dst_k,
    const SglTensor src_v,
    SglTensor dst_v,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_pf_lf(
    const SglTensor src_k,
    SglTensor dst_k,
    const SglTensor src_v,
    SglTensor dst_v,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_ph_lf(
    const SglTensor src_k,
    SglTensor dst_k,
    const SglTensor src_v,
    SglTensor dst_v,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer(
    const SglTensor src_k_layers,
    const SglTensor dst_k_layers,
    const SglTensor src_v_layers,
    const SglTensor dst_v_layers,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_lf_pf(
    const SglTensor src_k_layers,
    SglTensor dst_k,
    const SglTensor src_v_layers,
    SglTensor dst_v,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_lf_ph(
    const SglTensor src_k_layers,
    SglTensor dst_k,
    const SglTensor src_v_layers,
    SglTensor dst_v,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t page_size,
    int64_t head_num,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla(
    const SglTensor src,
    SglTensor dst,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_per_layer_mla_pf_lf(
    const SglTensor src,
    SglTensor dst,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t layer_id,
    int64_t item_size,
    int64_t src_layout_dim,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla(
    const SglTensor src_layers,
    const SglTensor dst_layers,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_all_layer_mla_lf_pf(
    const SglTensor src_layers,
    SglTensor dst,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t item_size,
    int64_t dst_layout_dim,
    int64_t num_layers,
    int64_t block_quota,
    int64_t num_warps_per_block);

void transfer_kv_direct(
    const std::vector<SglTensor>& src_layers,
    std::vector<SglTensor> dst_layers,
    const SglTensor src_indices,
    const SglTensor dst_indices,
    int64_t page_size);

void transfer_embedding_ranges_direct(
    const SglTensor& src,
    SglTensor& dst,
    const std::vector<int64_t>& src_starts,
    const std::vector<int64_t>& dst_starts,
    const std::vector<int64_t>& lengths);

void transfer_kv_per_layer_direct_pf_lf(
    const std::vector<SglTensor>& src_ptrs,
    std::vector<SglTensor> dst_ptrs,
    const SglTensor& src_indices,
    const SglTensor& dst_indices,
    int64_t layer_id,
    int64_t page_size);

void transfer_kv_all_layer_direct_lf_pf(
    const std::vector<SglTensor>& src_ptrs,
    std::vector<SglTensor> dst_ptrs,
    const SglTensor& src_indices,
    const SglTensor& dst_indices,
    int64_t page_size);
