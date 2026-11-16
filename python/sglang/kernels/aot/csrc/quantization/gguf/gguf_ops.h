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

SglTensor ggml_dequantize(SglTensor W, int64_t type, int64_t m, int64_t n, const SglOptional<SglScalarType>& dtype);

SglTensor ggml_mul_mat_vec_a8(SglTensor W, SglTensor X, int64_t type, int64_t row);

SglTensor ggml_mul_mat_a8(SglTensor W, SglTensor X, int64_t type, int64_t row);

SglTensor ggml_moe_a8(
    SglTensor X,
    SglTensor W,
    SglTensor sorted_token_ids,
    SglTensor expert_ids,
    SglTensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens);

SglTensor
ggml_moe_a8_vec(SglTensor X, SglTensor W, SglTensor topk_ids, int64_t top_k, int64_t type, int64_t row, int64_t tokens);

int64_t ggml_moe_get_block_size(int64_t type);
