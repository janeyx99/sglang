#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <string>
#include <tuple>

#include "es_fp8_blockwise_launcher.cuh"
#include "expert_specialization_ops.h"
#include "expert_specialization_utils.h"
#include "sgl_kernel_cuda_device.h"
#include "sgl_kernel_cuda_stream.h"

/**
 * @brief Performs blockwise grouped matrix multiplication on FP8 quantized inputs,
 *        with per-block scaling.
 *
 * This function dispatches to hardware-specific implementations (e.g., SM100 FP8)
 * to compute:
 *     C_i = scale_a[i] * A_i * scale_b[i] * B_i
 * for each expert group `i`, using input `problem_sizes` and `expert_offsets`
 * to describe the individual matrix dimensions and their offsets.
 *
 * Input tensors A and B must be quantized to 8-bit formats and dequantized before multiplication.
 * The output tensor is written with bfloat16 or half precision.
 *
 * @param output         Output tensor (must be of type bfloat16 or half).
 * @param a              Input tensor A (must be kFloat8_e4m3fn).
 * @param b              Input tensor B (must be kFloat8_e4m3fn).
 * @param scales_a       Scaling factors for tensor A, float32 per expert group.
 * @param scales_b       Scaling factors for tensor B, float32 per expert group.
 * @param stride_a       Stride information for tensor A (int32).
 * @param stride_b       Stride information for tensor B (int32).
 * @param stride_c       Stride information for output tensor C (int32).
 * @param problem_sizes  2D int32 tensor of shape (num_experts, 3), specifying (M, N, K)
 *                       for each grouped matrix multiplication problem.
 * @param expert_offsets 1D int32 tensor of size (num_experts), used to index into
 *                       the grouped input tensors for dispatch.
 */
void es_fp8_blockwise_scaled_grouped_mm(
    torch::stable::Tensor& output,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& scales_a,
    const torch::stable::Tensor& scales_b,
    const torch::stable::Tensor& stride_a,
    const torch::stable::Tensor& stride_b,
    const torch::stable::Tensor& stride_d,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& workspace) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED) && defined(CUTLASS_ARCH_MMA_MODIFIABLE_TMA_SM90_SUPPORTED)
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");
  STD_TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have shape (num_experts, 3)");
  STD_TORCH_CHECK(
      problem_sizes.size(0) == expert_offsets.size(0), "Number of experts in problem_sizes must match expert_offsets");
  STD_TORCH_CHECK(problem_sizes.scalar_type() == torch::headeronly::ScalarType::Int, "problem_sizes must be int32");
  STD_TORCH_CHECK(a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "a must be kFloat8_e4m3fn");
  STD_TORCH_CHECK(b.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn, "b must be kFloat8_e4m3fn");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 ||
          output.scalar_type() == torch::headeronly::ScalarType::Half,
      "output must be bfloat16 or half");

  int num_experts = (int)problem_sizes.size(0);
  const auto device = a.device();
  auto out_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto a_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto b_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto a_scales_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto b_scales_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);

  auto layout_sfa = expert_specialization::empty(device, {num_experts, 5}, torch::headeronly::ScalarType::Int);
  auto layout_sfb = expert_specialization::empty(device, {num_experts, 5}, torch::headeronly::ScalarType::Int);

  auto lm_problem_sizes = expert_specialization::empty(device, {num_experts, 3}, torch::headeronly::ScalarType::Int);
  auto mm_problem_sizes = expert_specialization::empty(device, {num_experts, 3}, torch::headeronly::ScalarType::Int);
  auto hm_problem_sizes = expert_specialization::empty(device, {num_experts, 3}, torch::headeronly::ScalarType::Int);

  auto backup_workspace_0 = torch::stable::empty_like(workspace);
  auto backup_workspace_1 = torch::stable::empty_like(workspace);

  const std::string H20_device_type_str("NVIDIA H20");
  bool is_h20_device = std::string(sgl_kernel::stable::get_cached_device_properties().name) == H20_device_type_str;

  auto stream = sgl_kernel::stable::get_current_cuda_stream();
  static auto backup_stream_0 = sgl_kernel::stable::get_cuda_stream_from_pool(false, -1);
  static auto backup_stream_1 = sgl_kernel::stable::get_cuda_stream_from_pool(false, -1);
  expert_specialization::CUDAEvent start_event;
  expert_specialization::CUDAEvent end_event_0;
  expert_specialization::CUDAEvent end_event_1;

  if (output.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_pre_compute<cutlass::bfloat16_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        output,
        a,
        b,
        scales_a,
        scales_b,
        problem_sizes,
        expert_offsets,
        is_h20_device,
        stream);
  } else if (output.scalar_type() == torch::headeronly::ScalarType::Half) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_pre_compute<cutlass::half_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        output,
        a,
        b,
        scales_a,
        scales_b,
        problem_sizes,
        expert_offsets,
        is_h20_device,
        stream);
  } else {
    STD_TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }

  start_event.record_once(stream);
  start_event.block(backup_stream_0);
  start_event.block(backup_stream_1);

  if (output.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype<cutlass::bfloat16_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        workspace,
        backup_workspace_0,
        backup_workspace_1,
        is_h20_device,
        stream,
        backup_stream_0,
        backup_stream_1);
  } else if (output.scalar_type() == torch::headeronly::ScalarType::Half) {
    expert_specialization::es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype<cutlass::half_t>(
        out_ptrs,
        a_ptrs,
        b_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        stride_a,
        stride_b,
        stride_d,
        layout_sfa,
        layout_sfb,
        lm_problem_sizes,
        mm_problem_sizes,
        hm_problem_sizes,
        workspace,
        backup_workspace_0,
        backup_workspace_1,
        is_h20_device,
        stream,
        backup_stream_0,
        backup_stream_1);
  } else {
    STD_TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
  }

  end_event_0.record_once(backup_stream_0);
  end_event_1.record_once(backup_stream_1);
  end_event_0.block(stream);
  end_event_1.block(stream);
#else
  STD_TORCH_CHECK(
      false, "NotImplementedError: No implemented fp8_blockwise_scaled_grouped_mm for current compute capability");
#endif
}
