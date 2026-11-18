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
#include "expert_specialization/expert_specialization_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "es_fp8_blockwise_scaled_grouped_mm(Tensor output, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, Tensor "
      "stride_a, Tensor stride_b, Tensor stride_d, Tensor problem_sizes, Tensor expert_offsets, Tensor workspace) -> "
      "()");
  m.def(
      "es_sm100_mxfp8_blockscaled_grouped_mm(Tensor a, Tensor b, Tensor sfa, Tensor sfb, Tensor d, Tensor "
      "problem_sizes, Tensor expert_offsets, Tensor blockscale_offsets) -> ()");
  m.def(
      "es_sm100_mxfp8_blockscaled_grouped_quant(Tensor input, Tensor problem_sizes, Tensor expert_offsets, Tensor "
      "blockscale_offsets, Tensor quant_output, Tensor scale_factor) -> () ");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CompositeImplicitAutograd, m) {
  m.impl("es_fp8_blockwise_scaled_grouped_mm", TORCH_BOX(&es_fp8_blockwise_scaled_grouped_mm));
  m.impl("es_sm100_mxfp8_blockscaled_grouped_mm", TORCH_BOX(&es_sm100_mxfp8_blockscaled_grouped_mm));
  m.impl("es_sm100_mxfp8_blockscaled_grouped_quant", TORCH_BOX(&es_sm100_mxfp8_blockscaled_grouped_quant));
}
