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
#include "elementwise/elementwise_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  m.def("copy_to_gpu_no_ce(Tensor input, Tensor! output) -> ()");
  m.def("concat_mla_k(Tensor! k, Tensor k_nope, Tensor k_rope) -> ()");
  m.def("concat_mla_absorb_q(Tensor a, Tensor b, Tensor! out) -> ()");
  m.def("fast_topk(Tensor score, Tensor indices, Tensor lengths, Tensor? row_starts) -> ()");
  m.def(
      "fast_topk_transform_fused(Tensor score, Tensor lengths, Tensor dst_page_table, Tensor src_page_table, Tensor "
      "cu_seqlens_q, Tensor? row_starts) -> ()");
  m.def(
      "fast_topk_transform_ragged_fused(Tensor score, Tensor lengths, Tensor topk_indices_ragged, Tensor "
      "topk_indices_offset, Tensor ? row_starts) -> ()");
  m.def(
      "dsv4_fused_q_norm_rope(Tensor q_input, Tensor! q_output, Tensor freqs_cis, Tensor positions, float eps) -> ()");
  m.def(
      "dsv4_fused_k_norm_rope_flashmla(Tensor kv, Tensor kv_weight, Tensor freqs_cis, Tensor positions, "
      "Tensor out_loc, Tensor! kvcache, float eps, int page_size) -> ()");
  m.def(
      "dsv4_fused_q_indexer_rope_hadamard_quant(Tensor q_input, Tensor! q_fp8, Tensor weight, "
      "Tensor! weights_out, float weight_scale, Tensor freqs_cis, Tensor positions) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("fused_add_rmsnorm", TORCH_BOX(&sgl_fused_add_rmsnorm));
  m.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
  m.impl("gelu_tanh_and_mul", TORCH_BOX(&gelu_tanh_and_mul));
  m.impl("gelu_and_mul", TORCH_BOX(&gelu_and_mul));
  m.impl("rotary_embedding", TORCH_BOX(&rotary_embedding));
  m.impl("copy_to_gpu_no_ce", TORCH_BOX(&copy_to_gpu_no_ce));
  m.impl("concat_mla_k", TORCH_BOX(&concat_mla_k));
  m.impl("concat_mla_absorb_q", TORCH_BOX(&concat_mla_absorb_q));
  m.impl("fast_topk", TORCH_BOX(&fast_topk_interface));
  m.impl("fast_topk_transform_fused", TORCH_BOX(&fast_topk_transform_interface));
  m.impl("fast_topk_transform_ragged_fused", TORCH_BOX(&fast_topk_transform_ragged_interface));
  m.impl("dsv4_fused_q_norm_rope", TORCH_BOX(&dsv4_fused_q_norm_rope));
  m.impl("dsv4_fused_k_norm_rope_flashmla", TORCH_BOX(&dsv4_fused_k_norm_rope_flashmla));
  m.impl("dsv4_fused_q_indexer_rope_hadamard_quant", TORCH_BOX(&dsv4_fused_q_indexer_rope_hadamard_quant));
}
