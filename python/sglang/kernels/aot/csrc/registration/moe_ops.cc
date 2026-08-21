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
#include "moe/moe_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts, int block_size, Tensor! sorted_token_ids, Tensor! "
      "experts_ids, Tensor! num_tokens_post_pad, Tensor! cumsum_buffer, bool "
      "pad_sorted_token_ids, bool ignore_invalid_expert) -> ()");

  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, float "
      "moe_softcapping, Tensor? correction_bias) -> ()");

  m.def(
      "topk_sigmoid(Tensor! topk_weights, Tensor! topk_indices, Tensor gating_output, bool renormalize, Tensor? "
      "correction_bias) -> ()");

  m.def("moe_sum_reduce(Tensor input, Tensor output, float routed_scaling_factor) -> ()");

  m.def("moe_sum(Tensor input, Tensor! output) -> ()");

  // moe_fused_gate / kimi_k2_moe_fused_gate (AOT) retired: the CUDA gate/topk path
  // now routes through the unified Triton router
  // (python/sglang/kernels/ops/moe/moe_fused_gate.py).

  m.def(
      "fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a_ptrs, Tensor b_ptrs, Tensor out_ptrs, Tensor "
      "a_scales_ptrs, Tensor b_scales_ptrs, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_c, Tensor layout_sfa, Tensor layout_sfb, Tensor problem_sizes, Tensor "
      "expert_offsets, Tensor workspace) -> ()");

  m.def(
      "prepare_moe_input(Tensor topk_ids, Tensor expert_offsets, Tensor? blockscale_offsets, Tensor problem_sizes1,"
      " Tensor problem_sizes2, Tensor input_permutation, Tensor output_permutation, int num_experts, int n, int k) -> "
      "()");

  m.def("shuffle_rows(Tensor input, Tensor dst2src_map, Tensor output) -> ()");

  m.def("apply_shuffle_mul_sum(Tensor input, Tensor output, Tensor permutation, Tensor? factors) -> ()");

  m.def(
      "fused_qk_norm_rope(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, float base, "
      "bool is_neox, Tensor position_ids, float factor, float low, float high, float attention_factor, int rotary_dim) "
      "-> ()");

  m.def(
      "get_cutlass_w4a8_moe_mm_data(Tensor topk_ids, Tensor! expert_offsets, "
      "                        Tensor! problem_sizes1, Tensor! problem_sizes2, "
      "                        Tensor! input_permutation, "
      "                        Tensor! output_permutation, int num_experts, "
      "                        int n, int k) -> ()");

  m.def(
      "cutlass_w4a8_moe_mm(Tensor! d, Tensor a, Tensor b, "
      "               Tensor a_scales, Tensor b_scales, Tensor expert_offsets, "
      "               Tensor problem_sizes, Tensor a_strides, "
      "               Tensor b_strides, Tensor d_strides, Tensor s_strides,"
      "               int chunk_size, int topk) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
  m.impl("topk_softmax", TORCH_BOX(&topk_softmax));
  m.impl("topk_sigmoid", TORCH_BOX(&topk_sigmoid));
  m.impl("moe_sum_reduce", TORCH_BOX(&moe_sum_reduce));
  m.impl("moe_sum", TORCH_BOX(&moe_sum));
  m.impl("fp8_blockwise_scaled_grouped_mm", TORCH_BOX(&fp8_blockwise_scaled_grouped_mm));
  m.impl("prepare_moe_input", TORCH_BOX(&prepare_moe_input));
  m.impl("shuffle_rows", TORCH_BOX(&shuffle_rows));
  m.impl("apply_shuffle_mul_sum", TORCH_BOX(&apply_shuffle_mul_sum));
  m.impl("fused_qk_norm_rope", TORCH_BOX(&fused_qk_norm_rope));
  m.impl("get_cutlass_w4a8_moe_mm_data", TORCH_BOX(&get_cutlass_w4a8_moe_mm_data));
  m.impl("cutlass_w4a8_moe_mm", TORCH_BOX(&cutlass_w4a8_moe_mm));
}
