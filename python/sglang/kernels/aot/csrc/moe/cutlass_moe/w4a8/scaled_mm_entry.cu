#include <cudaTypedefs.h>

#include "moe/moe_ops.h"

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

void cutlass_w4a8_moe_mm_sm90(
    SglTensor& d_tensors,
    const SglTensor& a_tensors,
    const SglTensor& b_tensors,
    const SglTensor& a_scales,
    const SglTensor& b_scales,
    const SglTensor& expert_offsets,
    const SglTensor& problem_sizes,
    const SglTensor& a_strides,
    const SglTensor& b_strides,
    const SglTensor& d_strides,
    const SglTensor& s_strides,
    int64_t chunk_size,
    int64_t topk);

void get_cutlass_w4a8_moe_mm_data_caller(
    const SglTensor& topk_ids,
    SglTensor& expert_offsets,
    SglTensor& problem_sizes1,
    SglTensor& problem_sizes2,
    SglTensor& input_permutation,
    SglTensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k);

void cutlass_w4a8_moe_mm(
    SglTensor& d_tensors,
    const SglTensor& a_tensors,
    const SglTensor& b_tensors,
    const SglTensor& a_scales,
    const SglTensor& b_scales,
    const SglTensor& expert_offsets,
    const SglTensor& problem_sizes,
    const SglTensor& a_strides,
    const SglTensor& b_strides,
    const SglTensor& d_strides,
    const SglTensor& s_strides,
    int64_t chunk_size,
    int64_t topk) {
  cutlass_w4a8_moe_mm_sm90(
      d_tensors,
      a_tensors,
      b_tensors,
      a_scales,
      b_scales,
      expert_offsets,
      problem_sizes,
      a_strides,
      b_strides,
      d_strides,
      s_strides,
      chunk_size,
      topk);
  return;
}

void get_cutlass_w4a8_moe_mm_data(
    const SglTensor& topk_ids,
    SglTensor& expert_offsets,
    SglTensor& problem_sizes1,
    SglTensor& problem_sizes2,
    SglTensor& input_permutation,
    SglTensor& output_permutation,
    const int64_t num_experts,
    const int64_t n,
    const int64_t k) {
  get_cutlass_w4a8_moe_mm_data_caller(
      topk_ids,
      expert_offsets,
      problem_sizes1,
      problem_sizes2,
      input_permutation,
      output_permutation,
      num_experts,
      n,
      k);
  return;
}
