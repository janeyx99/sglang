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
#include <torch/csrc/stable/library.h>

#include "quantization/gguf/gguf_ops.h"

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");

  m.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");

  m.def("ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");

  m.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");

  m.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");

  m.def("ggml_moe_get_block_size(int type) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("ggml_dequantize", TORCH_BOX(&ggml_dequantize));
  m.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8));
  m.impl("ggml_mul_mat_a8", TORCH_BOX(&ggml_mul_mat_a8));
  m.impl("ggml_moe_a8", TORCH_BOX(&ggml_moe_a8));
  m.impl("ggml_moe_a8_vec", TORCH_BOX(&ggml_moe_a8_vec));
  m.impl("ggml_moe_get_block_size", TORCH_BOX(&ggml_moe_get_block_size));
}
