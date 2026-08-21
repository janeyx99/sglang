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
#include <tuple>
#include <vector>

#include "sgl_kernel_torch_compat.h"

using fptr_t = int64_t;

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, SglTensor& rank_data, int64_t rank, bool full_nvlink);

void all_reduce(fptr_t fa, SglTensor& inp, SglTensor& out, fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);

void dispose(fptr_t fa);

int64_t meta_size();

void register_buffer(fptr_t fa, const std::vector<fptr_t>& fake_ipc_ptrs);

std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t fa);

void register_graph_buffers(
    fptr_t fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets);
