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

#include <torch/csrc/stable/tensor_struct.h>

#include <cstdint>

void infllm_v2_max_pooling_1d_varlen(
    torch::stable::Tensor input,
    torch::stable::Tensor output,
    torch::stable::Tensor cu_seqlens_q,
    torch::stable::Tensor cu_seqlens_k,
    torch::stable::Tensor cache_lens,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t block_size,
    int64_t local_blocks,
    int64_t init_blocks,
    int64_t total_q);
