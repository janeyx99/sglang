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

#include "mamba/causal_conv1d_ops.h"

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  // Compatibility API: SGLang dispatches to the JIT implementation, but external
  // sgl_kernel consumers still rely on these exported CUDA ops.
  m.def(
      "causal_conv1d_update(Tensor! x,"
      "Tensor! conv_state,"
      "Tensor! weight,"
      "Tensor? bias_,"
      "bool silu_activation,"
      "Tensor? cache_seqlens_,"
      "Tensor? conv_state_indices,"
      "int pad_slot_id) -> ()");
  m.def(
      "causal_conv1d_fwd(Tensor! x, Tensor! weight,"
      "Tensor? bias_,"
      "Tensor!? conv_states,"
      "Tensor? query_start_loc,"
      "Tensor? cache_indices,"
      "Tensor? has_initial_state,"
      "bool silu_activation,"
      "int pad_slot_id) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("causal_conv1d_update", TORCH_BOX(&causal_conv1d_update));
  m.impl("causal_conv1d_fwd", TORCH_BOX(&causal_conv1d_fwd));
}
