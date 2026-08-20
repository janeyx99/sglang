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
      "fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor? alibi_slopes, "
      "float p_dropout, float softmax_scale, bool is_causal, "
      "float softcap, bool return_softmax, Generator? gen)"
      "-> Tensor[]");
  m.impl("fwd_sparse", torch::kCUDA, &flash::mha_fwd_sparse);

  m.def(
      "varlen_fwd_sparse(Tensor! q, Tensor k, Tensor v, "
      "Tensor block_count, Tensor block_offset, Tensor column_count, Tensor column_index, "
      "Tensor!? out, Tensor cu_seqlens_q, "
      "Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, "
      "bool is_causal, float softcap, bool return_softmax, "
      "Generator? gen) -> Tensor[]");
  m.impl("varlen_fwd_sparse", torch::kCUDA, &flash::mha_varlen_fwd_sparse);
}
