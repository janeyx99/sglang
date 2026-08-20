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
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "infllm_v2_max_pooling_1d_varlen(Tensor input, Tensor! output, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
      "Tensor cache_lens, int max_seqlen_q, int max_seqlen_k, int kernel_size, int stride, int padding, "
      "int block_size, int local_blocks, int init_blocks, int total_q) -> ()");
  m.impl("infllm_v2_max_pooling_1d_varlen", torch::kCUDA, &infllm_v2_max_pooling_1d_varlen);
}
