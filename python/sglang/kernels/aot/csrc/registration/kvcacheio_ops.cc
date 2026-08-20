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

#include "kvcacheio/transfer.h"

STABLE_TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def(
      "transfer_kv_per_layer(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int item_size, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_per_layer_pf_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_per_layer_ph_lf(Tensor src_k, Tensor dst_k, Tensor src_v, Tensor dst_v, Tensor src_indices, Tensor "
      "dst_indices, int layer_id, int item_size, int src_layout_dim, int page_size, int head_num, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_all_layer(Tensor src_k_layers, Tensor dst_k_layers, Tensor src_v_layers, Tensor dst_v_layers, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_all_layer_lf_pf(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int block_quota, int "
      "num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_all_layer_lf_ph(Tensor src_k_layers, Tensor dst_k, Tensor src_v_layers, Tensor dst_v, "
      "Tensor src_indices, Tensor dst_indices, int item_size, int dst_layout_dim, int num_layers, int page_size, int "
      "head_num, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_per_layer_mla(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int item_size, int "
      "block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_per_layer_mla_pf_lf(Tensor src, Tensor dst, Tensor src_indices, Tensor dst_indices, int layer_id, "
      "int item_size, int src_layout_dim, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_all_layer_mla(Tensor src_layers, Tensor dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "item_size, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_all_layer_mla_lf_pf(Tensor src_layers, Tensor dst, Tensor src_indices, Tensor dst_indices, "
      "int item_size, int dst_layout_dim, int num_layers, int block_quota, int num_warps_per_block) -> ()");
  m.def(
      "transfer_kv_direct(Tensor[] src_layers, Tensor[] dst_layers, Tensor src_indices, Tensor dst_indices, int "
      "page_size) -> ()");
  m.def(
      "transfer_embedding_ranges_direct(Tensor src, Tensor! dst, int[] src_starts, int[] dst_starts, int[] "
      "lengths) -> ()");
  m.def(
      "transfer_kv_per_layer_direct_pf_lf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int layer_id, int page_size)->() ");
  m.def(
      "transfer_kv_all_layer_direct_lf_pf(Tensor[] src_ptrs, Tensor[] dst_ptrs, Tensor src_indices, "
      "Tensor dst_indices, int page_size) ->() ");
}

STABLE_TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("transfer_kv_per_layer", TORCH_BOX(&transfer_kv_per_layer));
  m.impl("transfer_kv_per_layer_pf_lf", TORCH_BOX(&transfer_kv_per_layer_pf_lf));
  m.impl("transfer_kv_per_layer_ph_lf", TORCH_BOX(&transfer_kv_per_layer_ph_lf));
  m.impl("transfer_kv_all_layer", TORCH_BOX(&transfer_kv_all_layer));
  m.impl("transfer_kv_all_layer_lf_pf", TORCH_BOX(&transfer_kv_all_layer_lf_pf));
  m.impl("transfer_kv_all_layer_lf_ph", TORCH_BOX(&transfer_kv_all_layer_lf_ph));
  m.impl("transfer_kv_per_layer_mla", TORCH_BOX(&transfer_kv_per_layer_mla));
  m.impl("transfer_kv_per_layer_mla_pf_lf", TORCH_BOX(&transfer_kv_per_layer_mla_pf_lf));
  m.impl("transfer_kv_all_layer_mla", TORCH_BOX(&transfer_kv_all_layer_mla));
  m.impl("transfer_kv_all_layer_mla_lf_pf", TORCH_BOX(&transfer_kv_all_layer_mla_lf_pf));
  m.impl("transfer_kv_direct", TORCH_BOX(&transfer_kv_direct));
  m.impl("transfer_embedding_ranges_direct", TORCH_BOX(&transfer_embedding_ranges_direct));
  m.impl("transfer_kv_per_layer_direct_pf_lf", TORCH_BOX(&transfer_kv_per_layer_direct_pf_lf));
  m.impl("transfer_kv_all_layer_direct_lf_pf", TORCH_BOX(&transfer_kv_all_layer_direct_lf_pf));
}
