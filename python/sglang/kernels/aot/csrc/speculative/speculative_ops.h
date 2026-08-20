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

void tree_speculative_sampling_target_only(
    SglTensor predicts,
    SglTensor accept_index,
    SglTensor accept_token_num,
    SglTensor candidates,
    SglTensor retrive_index,
    SglTensor retrive_next_token,
    SglTensor retrive_next_sibling,
    SglTensor uniform_samples,
    SglTensor uniform_samples_for_final_sampling,
    SglTensor target_probs,
    SglTensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic);

void verify_tree_greedy(
    SglTensor predicts,
    SglTensor accept_index,
    SglTensor accept_token_num,
    SglTensor candidates,
    SglTensor retrive_index,
    SglTensor retrive_next_token,
    SglTensor retrive_next_sibling,
    SglTensor target_predict);

void reconstruct_indices_from_tree_mask(
    SglTensor tree_mask,
    SglTensor verified_seq_len,
    SglTensor positions,
    SglTensor retrive_index,
    SglTensor retrive_next_token,
    SglTensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num);

void build_tree_kernel_efficient(
    SglTensor parent_list,
    SglTensor selected_index,
    SglTensor verified_seq_len,
    SglTensor tree_mask,
    SglTensor positions,
    SglTensor retrive_index,
    SglTensor retrive_next_token,
    SglTensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode);

void segment_packbits(
    SglTensor x, SglTensor input_indptr, SglTensor output_indptr, SglTensor y, int64_t batch_size, int64_t cuda_stream);
