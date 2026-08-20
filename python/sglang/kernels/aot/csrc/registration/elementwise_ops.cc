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
  m.def("rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("rmsnorm", torch::kCUDA, &rmsnorm);

  m.def("fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("fused_add_rmsnorm", torch::kCUDA, &sgl_fused_add_rmsnorm);

  m.def("gemma_rmsnorm(Tensor! output, Tensor input, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_rmsnorm", torch::kCUDA, &gemma_rmsnorm);

  m.def("gemma_fused_add_rmsnorm(Tensor! input, Tensor! residual, Tensor weight, float eps, bool enable_pdl) -> ()");
  m.impl("gemma_fused_add_rmsnorm", torch::kCUDA, &gemma_fused_add_rmsnorm);

  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_tanh_and_mul", torch::kCUDA, &gelu_tanh_and_mul);

  m.def("gelu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("gelu_and_mul", torch::kCUDA, &gelu_and_mul);

  m.def(
      "rotary_embedding(Tensor positions, Tensor! query,"
      "                 Tensor!? key, int head_size,"
      "                 Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);

  m.def("copy_to_gpu_no_ce(Tensor input, Tensor! output) -> ()");
  m.impl("copy_to_gpu_no_ce", torch::kCUDA, &copy_to_gpu_no_ce);
  m.def("concat_mla_k(Tensor! k, Tensor k_nope, Tensor k_rope) -> ()");
  m.impl("concat_mla_k", torch::kCUDA, &concat_mla_k);

  m.def("concat_mla_absorb_q(Tensor a, Tensor b, Tensor! out) -> ()");
  m.impl("concat_mla_absorb_q", torch::kCUDA, &concat_mla_absorb_q);

  m.def("fast_topk(Tensor score, Tensor indices, Tensor lengths, Tensor? row_starts) -> ()");
  m.impl("fast_topk", torch::kCUDA, &fast_topk_interface);
  m.def(
      "fast_topk_transform_fused(Tensor score, Tensor lengths, Tensor dst_page_table, Tensor src_page_table, Tensor "
      "cu_seqlens_q, Tensor? row_starts) -> ()");
  m.impl("fast_topk_transform_fused", torch::kCUDA, &fast_topk_transform_interface);
  m.def(
      "fast_topk_transform_ragged_fused(Tensor score, Tensor lengths, Tensor topk_indices_ragged, Tensor "
      "topk_indices_offset, Tensor ? row_starts) -> ()");
  m.impl("fast_topk_transform_ragged_fused", torch::kCUDA, &fast_topk_transform_ragged_interface);

  // DeepSeek-V4 fused norm + rope
  m.def(
      "dsv4_fused_q_norm_rope(Tensor q_input, Tensor! q_output, Tensor freqs_cis, Tensor positions, float eps) -> ()");
  m.impl("dsv4_fused_q_norm_rope", torch::kCUDA, &dsv4_fused_q_norm_rope);

  m.def(
      "dsv4_fused_k_norm_rope_flashmla(Tensor kv, Tensor kv_weight, Tensor freqs_cis, Tensor positions, "
      "Tensor out_loc, Tensor! kvcache, float eps, int page_size) -> ()");
  m.impl("dsv4_fused_k_norm_rope_flashmla", torch::kCUDA, &dsv4_fused_k_norm_rope_flashmla);

  m.def(
      "dsv4_fused_q_indexer_rope_hadamard_quant(Tensor q_input, Tensor! q_fp8, Tensor weight, "
      "Tensor! weights_out, float weight_scale, Tensor freqs_cis, Tensor positions) -> ()");
  m.impl("dsv4_fused_q_indexer_rope_hadamard_quant", torch::kCUDA, &dsv4_fused_q_indexer_rope_hadamard_quant);
}
