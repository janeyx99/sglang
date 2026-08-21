#pragma once

/**
 * @file w4a8_grouped_mm_c3x.cuh
 * @brief Implementation of grouped GEMM operation with int4 and fp8 mixed
 * precision
 *
 * This file implements a grouped GEMM operation that multiplies FP8 matrices
 * (A) with quantized INT4 matrices (B), applying per-block scaling factors.
 * The implementation is optimized for NVIDIA Hopper GPUs, leveraging Tensor
 * Cores for mixed precision arithmetic.
 *
 * Key features:
 * - Supports grouped GEMM operations with multiple experts
 * - Uses FP8 (e4m3) for matrix A
 * - Uses INT4 quantization for matrix B with per-block scaling
 * - Implements preprocessing for INT4 encoding and scale packing
 * - Optimized for Hopper architecture with Tensor Core operations
 */

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/headeronly/util/Exception.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder_mixed_input.hpp"
#include "moe/moe_ops.h"
#include "moe/moe_stable_utils.h"
#include "sgl_kernel_cuda_stream.h"
#include "w4a8_get_group_starts.cuh"

#define W4A8_MUTABLE_RAW_PTR(tensor_) (tensor_).mutable_data_ptr()

using namespace cute;

namespace {

// Type definitions
using MmaType = cutlass::float_e4m3_t;     // FP8 e4m3 type
using QuantType = cutlass::int4b_t;        // 4-bit integer type
using ElementAccumulator = float;          // Accumulator type
using ElementScale = cutlass::bfloat16_t;  // Scale type
using ElementC = cutlass::bfloat16_t;      // Output type
using ElementD = ElementC;                 // Output type
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

// Architecture-specific configurations
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
// constexpr int TileShapeK = 512;
// using TileShape = Shape<_128, _32, cute::Int<TileShapeK>>;
// using ClusterShape = Shape<_1, _1, _1>;

// Layout configurations
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = LayoutC;

// Transposed layouts
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutC_Transpose = typename cutlass::layout::LayoutTranspose<LayoutC>::type;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;

// Alignments
static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<MmaType>::value;
static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<QuantType>::value;
static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
struct cutlass_3x_w4a8_group_gemm {
  static constexpr int GroupSize = 128;
  static constexpr int PackedScalesNum = get<2>(TileShape{}) / GroupSize;
  using ElementScalePacked = cutlass::Array<ElementScale, PackedScalesNum>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementAccumulator,
      ElementC,
      LayoutC_Transpose*,
      AlignmentC,
      ElementD,
      LayoutD_Transpose*,
      AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloopScaleOnly = typename cutlass::gemm::collective::CollectiveBuilderMixedInput<
      ArchTag,
      OperatorClass,
      cute::tuple<QuantType, ElementScalePacked>,
      LayoutB_Transpose*,
      AlignmentB,
      MmaType,
      LayoutA_Transpose*,
      AlignmentA,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  // Define the final kernel and GEMM operation types
  using GemmKernelScaleOnly =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloopScaleOnly, CollectiveEpilogue>;

  using GemmScaleOnly = cutlass::gemm::device::GemmUniversalAdapter<GemmKernelScaleOnly>;

  using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
  using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
  using StrideC = typename GemmKernelScaleOnly::InternalStrideC;
  using StrideD = typename GemmKernelScaleOnly::InternalStrideD;
  using StrideS = typename CollectiveMainloopScaleOnly::StrideScale;
};

/**
 * @brief Main function to run int4 * fp8 grouped GEMM from PyTorch
 *
 * This function performs multiple GEMM operations in parallel where each
 * operation multiplies an FP8 matrix (A) with a quantized INT4 matrix (B),
 * applying per-channel scaling factors. It's designed for efficient execution
 * on NVIDIA Hopper GPUs, leveraging Tensor Cores for optimal performance with
 * mixed precision arithmetic.
 *
 * The function includes preprocessing steps for both INT4 tensors and scale
 * factors to ensure optimal performance and correct operation.
 *
 * @param d_tensors Output tensor D with shape [total_m, total_n]
 * @param a_tensors Tensor containing all A matrices (fp8_e4m3) with shape
 * [total_m, K]
 * @param b_tensors Tensor containing all B matrices (int4 packed as int8) with
 * shape [E, N, K/2]
 * @param a_scales Tensor containing A matrix scale factors
 * @param b_scales Tensor containing B matrix scale factors with shape [E,
 * K//512, N*4]
 * @param expert_offsets Tensor containing expert offsets for determining group
 * boundaries (int32)
 * @param problem_sizes Tensor containing problem sizes with shape [num_experts,
 * 3] (M, N, K for each group) (int32)
 * @param a_strides Stride information for A tensors
 * @param b_strides Stride information for B tensors
 * @param d_strides Stride information for D tensors
 * @param s_strides Stride information for scale tensors
 * @param chunk_size Size of each chunk for scales (K / number of scale chunks)
 */
// template <typename TileShape, typename ClusterShape, typename KernelSchedule, typename EpilogueSchedule>
template <typename Gemm>
void cutlass_w4a8_group_gemm_caller(
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
    int64_t chunk_size) {
  //   using Gemm = cutlass_3x_w4a8_group_gemm<TileShape, ClusterShape, KernelSchedule, EpilogueSchedule>;
  using Args = typename Gemm::GemmScaleOnly::Arguments;

  int num_experts = static_cast<int>(expert_offsets.size(0));
  bool per_act_token = a_scales.numel() != 1;
  bool per_out_ch = b_scales.numel() != num_experts;

  // Check inputs
  STD_TORCH_CHECK(a_tensors.dim() == 2 or a_tensors.dim() == 3, "A tensor must be 2D/3D");
  STD_TORCH_CHECK(b_tensors.dim() == 3, "B tensor must be 3D [E, N, K/2]");
  STD_TORCH_CHECK(b_scales.dim() == 3, "Scale tensor must be 3D [E, K//512, N*4]");
  STD_TORCH_CHECK(a_scales.dim() == 1, "A Scale tensor must be 1D [1]");
  STD_TORCH_CHECK(expert_offsets.dim() == 1, "expert_offsets must be a 1D tensor");
  STD_TORCH_CHECK(problem_sizes.dim() == 2, "problem_sizes must be 2D tensor");

  // Check tensor shapes
  STD_TORCH_CHECK(problem_sizes.size(0) == num_experts, "problem_sizes must have num_experts rows");
  STD_TORCH_CHECK(problem_sizes.size(1) == 3, "problem_sizes must have 3 columns (N, M, K)");
  STD_TORCH_CHECK(b_tensors.size(0) == num_experts, "B tensor first dimension must match number of groups");
  STD_TORCH_CHECK(b_scales.size(0) == num_experts, "Scale tensor first dimension must match number of groups");
  STD_TORCH_CHECK(
      b_tensors.size(2) * 2 == a_tensors.size(1) or b_tensors.size(2) * 2 == a_tensors.size(2),
      "B tensor K/2 dimension must match A tensor K dimension");

  // Check tensor types
  STD_TORCH_CHECK(a_tensors.scalar_type() == SglScalarType::Float8_e4m3fn, "A tensor must be fp8 (float_e4m3_t) type");
  STD_TORCH_CHECK(
      b_tensors.scalar_type() == SglScalarType::Char, "B tensor must contain packed int4 values (stored as int8)");
  STD_TORCH_CHECK(expert_offsets.scalar_type() == SglScalarType::Int, "Expert offsets must be int32 type");
  STD_TORCH_CHECK(problem_sizes.scalar_type() == SglScalarType::Int, "Problem sizes must be int32 type");

  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream(a_tensors.get_device_index());

  const auto device = a_tensors.device();
  const std::array<int64_t, 1> pointer_array_size{static_cast<int64_t>(num_experts)};
  SglTensor a_ptrs = sgl_kernel::moe::stable::empty_contiguous(device, pointer_array_size, SglScalarType::Long);
  SglTensor b_ptrs = sgl_kernel::moe::stable::empty_contiguous(device, pointer_array_size, SglScalarType::Long);
  SglTensor out_ptrs = sgl_kernel::moe::stable::empty_contiguous(device, pointer_array_size, SglScalarType::Long);
  SglTensor a_scales_ptrs = sgl_kernel::moe::stable::empty_contiguous(device, pointer_array_size, SglScalarType::Long);
  SglTensor b_scales_ptrs = sgl_kernel::moe::stable::empty_contiguous(device, pointer_array_size, SglScalarType::Long);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a_tensors.get_device_index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  Args arguments;
  decltype(arguments.epilogue.thread) fusion_args;
  fusion_args.alpha = 0;
  fusion_args.beta = 0;
  fusion_args.alpha_ptr = a_scales.const_data_ptr<float>();
  ;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes = const_cast<ProblemShape::UnderlyingProblemShape*>(
      static_cast<const ProblemShape::UnderlyingProblemShape*>(problem_sizes.const_data_ptr()));

  run_int4_fp8_get_group_gemm_starts(
      expert_offsets,
      a_ptrs,
      b_ptrs,
      out_ptrs,
      a_scales_ptrs,
      b_scales_ptrs,
      a_tensors,
      b_tensors,
      d_tensors,
      a_scales,
      b_scales);

  arguments = Args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      {static_cast<const QuantType**>(W4A8_MUTABLE_RAW_PTR(b_ptrs)),
       static_cast<typename Gemm::StrideB*>(const_cast<void*>(b_strides.const_data_ptr())),
       static_cast<const MmaType**>(W4A8_MUTABLE_RAW_PTR(a_ptrs)),
       static_cast<typename Gemm::StrideA*>(const_cast<void*>(a_strides.const_data_ptr())),
       static_cast<const typename Gemm::ElementScalePacked**>(W4A8_MUTABLE_RAW_PTR(b_scales_ptrs)),
       static_cast<typename Gemm::StrideS*>(const_cast<void*>(s_strides.const_data_ptr())),
       static_cast<int>(chunk_size)},
      {fusion_args,
       nullptr,
       nullptr,
       static_cast<ElementD**>(W4A8_MUTABLE_RAW_PTR(out_ptrs)),
       static_cast<typename Gemm::StrideD*>(const_cast<void*>(d_strides.const_data_ptr()))},
      hw_info};

  // Instantiate and run GEMM
  typename Gemm::GemmScaleOnly gemm;
  size_t workspace_size = Gemm::GemmScaleOnly::get_workspace_size(arguments);
  SglTensor workspace = sgl_kernel::moe::stable::empty_contiguous(
      device, std::array<int64_t, 1>{static_cast<int64_t>(workspace_size)}, SglScalarType::Byte);

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM implementation not supported");
  }

  status = gemm.initialize(arguments, workspace.mutable_data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM initialization failed");
  }

  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) {
    STD_TORCH_CHECK(false, "GEMM execution failed");
  }
}

}  // namespace

#undef W4A8_MUTABLE_RAW_PTR
