#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "es_sm100_mxfp8_blockscaled_group_quant.cuh"
#include "expert_specialization_ops.h"
#include "sgl_kernel_cuda_stream.h"

void es_sm100_mxfp8_blockscaled_grouped_quant(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets,
    torch::stable::Tensor& quant_output,
    torch::stable::Tensor& scale_factor) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  STD_TORCH_CHECK(input.dim() == 2, "input must be 2D tensor");
  STD_TORCH_CHECK(input.size(1) % 128 == 0, "k must align to 128");
  STD_TORCH_CHECK(input.strides()[1] == 1, "input must be row major");
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  auto groups = problem_sizes.size(0);
  STD_TORCH_CHECK(
      expert_offsets.dim() == 1 && expert_offsets.size(0) == groups,
      "expert_offsets must be 1D and have size equal to the number of groups");
  STD_TORCH_CHECK(
      blockscale_offsets.dim() == 1 && blockscale_offsets.size(0) == groups,
      "blockscale_offsets must be 1D and have size equal to the number of groups");

  auto stream = sgl_kernel::stable::get_current_cuda_stream();
  if (input.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__nv_bfloat16>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else if (input.scalar_type() == torch::headeronly::ScalarType::Half) {
    expert_specialization::launch_es_sm100_mxfp8_blockscaled_grouped_quant<__half>(
        input, problem_sizes, expert_offsets, blockscale_offsets, quant_output, scale_factor);
  } else {
    STD_TORCH_CHECK(false, "dtype must be kFloat16 or kBFloat16");
  }
#else
  STD_TORCH_CHECK(false, "No implemented es_sm100_mxfp8_blockscaled_grouped_mm for current device");
#endif
}
