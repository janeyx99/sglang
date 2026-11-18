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

// Stable dispatcher entry for the vendored InfLLM-V2 FlashAttention backend.
// Keep its `flash::` symbols in a separate library.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <vector>

using torch::stable::Tensor;

// Forward declarations of the FlashAttention entry points implemented in
// flash_attn/flash_api.cpp. Signatures must match exactly.
std::vector<Tensor> mha_varlen_fwd_stage1(
    Tensor& q,
    const Tensor& k,
    const Tensor& v,
    std::optional<Tensor>& out_,
    const Tensor& cu_seqlens_q,
    const Tensor& cu_seqlens_k,
    const Tensor& cu_seqlens_v,
    std::optional<Tensor>& seqused_k,
    std::optional<Tensor>& leftpad_k_,
    std::optional<Tensor>& block_table_,
    std::optional<Tensor>& alibi_slopes_,
    int max_seqlen_q,
    const int max_seqlen_k,
    const float p_dropout,
    const float softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    const float softcap,
    const bool return_softmax);

// Dispatcher scalars are int64/double; preserve the host wrapper's original
// int/float conversions. The public Python wrapper never supplies a generator:
// dropout is disabled in this inference-only build.
static std::vector<Tensor> varlen_fwd_stage1(
    Tensor q,
    const Tensor& k,
    const Tensor& v,
    std::optional<Tensor> out,
    const Tensor& cu_seqlens_q,
    const Tensor& cu_seqlens_k,
    const Tensor& cu_seqlens_v,
    std::optional<Tensor> seqused_k,
    std::optional<Tensor> leftpad_k,
    std::optional<Tensor> block_table,
    std::optional<Tensor> alibi_slopes,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    double p_dropout,
    double softmax_scale,
    bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    double softcap,
    bool return_softmax) {
  return mha_varlen_fwd_stage1(
      q,
      k,
      v,
      out,
      cu_seqlens_q,
      cu_seqlens_k,
      cu_seqlens_v,
      seqused_k,
      leftpad_k,
      block_table,
      alibi_slopes,
      max_seqlen_q,
      max_seqlen_k,
      p_dropout,
      softmax_scale,
      zero_tensors,
      is_causal,
      window_size_left,
      window_size_right,
      softcap,
      return_softmax);
}

STABLE_TORCH_LIBRARY(infllm_ops, m) {
  m.def(
      "varlen_fwd_stage1(Tensor q, Tensor k, Tensor v, Tensor? out, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
      "Tensor cu_seqlens_v, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, "
      "int max_seqlen_q, int max_seqlen_k, float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal, "
      "int window_size_left, int window_size_right, float softcap, bool return_softmax) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(infllm_ops, CUDA, m) {
  m.impl("varlen_fwd_stage1", TORCH_BOX(&varlen_fwd_stage1));
}
