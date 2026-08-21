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
#include "attention/attention_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.def(
      "cutlass_mla_decode(Tensor! out, Tensor q_nope, Tensor q_pe, Tensor kv_c_and_k_pe_cache, Tensor seq_lens, Tensor "
      "page_table, Tensor! workspace, float sm_scale, int num_kv_splits) -> ()");
  m.def("cutlass_mla_get_workspace_size(int _0, int _1, int _2, int _3) -> int _0");

  m.def(
      "convert_vertical_slash_indexes("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");

  m.def(
      "convert_vertical_slash_indexes_mergehead("
      "   Tensor! block_count, Tensor! block_offset, "
      "   Tensor! column_count, Tensor! column_index, "
      "   Tensor q_seqlens, Tensor q_seqlens, "
      "   Tensor vertical_indexes, Tensor slash_indexes, "
      "   Tensor vertical_indices_count, Tensor slash_indices_count, "
      "   int context_size, int block_size_M, int block_size_N, "
      "   bool causal) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CompositeImplicitAutograd, m) {
  m.impl("cutlass_mla_get_workspace_size", TORCH_BOX(&cutlass_mla_get_workspace_size));
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("merge_state_v2", TORCH_BOX(&merge_state_v2));
  m.impl("cutlass_mla_decode", TORCH_BOX(&cutlass_mla_decode));
  m.impl("convert_vertical_slash_indexes", TORCH_BOX(&convert_vertical_slash_indexes));
  m.impl("convert_vertical_slash_indexes_mergehead", TORCH_BOX(&convert_vertical_slash_indexes_mergehead));
}
