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
#include "gemm/gemm_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("awq_dequantize(Tensor qweight, Tensor scales, Tensor qzeros) -> Tensor");

  m.def(
      "int8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");

  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");

  m.def(
      "sgl_per_token_group_quant_8bit(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0) -> ()");

  m.def(
      "sgl_per_token_group_quant_8bit_v2(Tensor input, Tensor! output_q, Tensor! output_s, int group_size,"
      " float eps, float fp8_min, float fp8_max, bool scale_ue8m0, bool fuse_silu_and_mul, Tensor? masked_m) -> ()");

  // Compatibility API: SGLang runtime dispatches to the JIT implementation,
  // but external sgl_kernel consumers still rely on this exported CUDA op.
  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor! output_q, Tensor! output_s) -> ()");

  m.def(
      "gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, bool "
      "use_shuffle, int bit) -> Tensor");

  m.def("gptq_shuffle(Tensor! q_weight, Tensor q_perm, int bit) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("awq_dequantize", TORCH_BOX(&awq_dequantize));
  m.impl("int8_scaled_mm", TORCH_BOX(&int8_scaled_mm));
  m.impl("fp8_scaled_mm", TORCH_BOX(&fp8_scaled_mm));
  m.impl("sgl_per_token_group_quant_8bit", TORCH_BOX(&sgl_per_token_group_quant_8bit));
  m.impl("sgl_per_token_group_quant_8bit_v2", TORCH_BOX(&sgl_per_token_group_quant_8bit_v2));
  m.impl("sgl_per_token_quant_fp8", TORCH_BOX(&sgl_per_token_quant_fp8));
  m.impl("gptq_gemm", TORCH_BOX(&gptq_gemm));
  m.impl("gptq_shuffle", TORCH_BOX(&gptq_shuffle));
}
