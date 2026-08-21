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
#include "allreduce/allreduce_ops.h"

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("get_graph_buffer_ipc_meta(int _0) -> (int[] _0, int[] _1)");
  m.def("register_graph_buffers(int _0, int[][] _1, int[][] _2) -> ()");
  m.def("dispose(int _0) -> ()");
  m.def("meta_size() -> int _0");
  m.def("register_buffer(int _0, int[] _1) -> ()");

  m.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool full_nvlink) -> int");

  m.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CompositeImplicitAutograd, m) {
  m.impl("get_graph_buffer_ipc_meta", TORCH_BOX(&get_graph_buffer_ipc_meta));
  m.impl("register_graph_buffers", TORCH_BOX(&register_graph_buffers));
  m.impl("dispose", TORCH_BOX(&dispose));
  m.impl("meta_size", TORCH_BOX(&meta_size));
  m.impl("register_buffer", TORCH_BOX(&register_buffer));
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("init_custom_ar", TORCH_BOX(&init_custom_ar));
  m.impl("all_reduce", TORCH_BOX(&all_reduce));
}
