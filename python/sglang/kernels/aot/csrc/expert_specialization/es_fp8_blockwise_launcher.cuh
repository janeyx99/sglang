#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "es_fp8_blockwise_functor.cuh"

namespace expert_specialization {

using namespace cute;

template <typename T>
void es_sm90_fp8_blockwise_scaled_group_mm_pre_compute(
    // Output
    torch::stable::Tensor& out_ptrs,
    torch::stable::Tensor& a_ptrs,
    torch::stable::Tensor& b_ptrs,
    torch::stable::Tensor& a_scales_ptrs,
    torch::stable::Tensor& b_scales_ptrs,
    torch::stable::Tensor& layout_sfa,
    torch::stable::Tensor& layout_sfb,
    torch::stable::Tensor& lm_problem_sizes,
    torch::stable::Tensor& mm_problem_sizes,
    torch::stable::Tensor& hm_problem_sizes,
    // Input
    torch::stable::Tensor& out_tensors,
    torch::stable::Tensor const& a_tensors,
    torch::stable::Tensor const& b_tensors,
    torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& problem_sizes,
    torch::stable::Tensor const& expert_offsets,
    bool is_h20_device,
    cudaStream_t stream) {
  STD_TORCH_CHECK(a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(a_scales.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() == torch::headeronly::ScalarType::Float);

  // Creat Scale Factor Layout Functor
  using LayoutSFA = typename PerfConfigMiddleMH20::LayoutSFA;
  using LayoutSFB = typename PerfConfigMiddleMH20::LayoutSFB;
  struct Fp8BlockwiseGroupedGemmSFLayoutFunctor<PerfConfigMiddleMH20> sf_layout(
      reinterpret_cast<LayoutSFA*>(layout_sfa.mutable_data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.mutable_data_ptr()));

  int num_experts = (int)expert_offsets.size(0);
  STD_TORCH_CHECK(num_experts <= 1024, "Expert more than 1024");  // Max threads per block is 1024

  struct Fp8BlockwiseGroupedGemmOffsetFunctor<cutlass::float_e4m3_t, float, T> of(
      const_cast<int*>(static_cast<const int*>(expert_offsets.const_data_ptr())),
      const_cast<cutlass::float_e4m3_t*>(static_cast<const cutlass::float_e4m3_t*>(a_tensors.const_data_ptr())),
      const_cast<cutlass::float_e4m3_t*>(static_cast<const cutlass::float_e4m3_t*>(b_tensors.const_data_ptr())),
      static_cast<T*>(out_tensors.mutable_data_ptr()),
      const_cast<float*>(static_cast<const float*>(a_scales.const_data_ptr())),
      const_cast<float*>(static_cast<const float*>(b_scales.const_data_ptr())),
      static_cast<cutlass::float_e4m3_t**>(a_ptrs.mutable_data_ptr()),
      static_cast<cutlass::float_e4m3_t**>(b_ptrs.mutable_data_ptr()),
      static_cast<float**>(a_scales_ptrs.mutable_data_ptr()),
      static_cast<float**>(b_scales_ptrs.mutable_data_ptr()),
      static_cast<T**>(out_ptrs.mutable_data_ptr()));
  if (!is_h20_device) {
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMHx00> lm_psf(
        static_cast<int*>(lm_problem_sizes.mutable_data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMHx00> mm_psf(
        static_cast<int*>(mm_problem_sizes.mutable_data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMHx00> hm_psf(
        static_cast<int*>(hm_problem_sizes.mutable_data_ptr()));
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
        const_cast<int*>(static_cast<const int*>(problem_sizes.const_data_ptr())),
        of,
        sf_layout,
        lm_psf,
        mm_psf,
        hm_psf);
  } else {
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigLowMH20> lm_psf(
        static_cast<int*>(lm_problem_sizes.mutable_data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigMiddleMH20> mm_psf(
        static_cast<int*>(mm_problem_sizes.mutable_data_ptr()));
    struct Fp8BlockwiseGroupedGemmProblemSizeFilterFunctor<PerfConfigHighMH20> hm_psf(
        static_cast<int*>(hm_problem_sizes.mutable_data_ptr()));
    groupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
        const_cast<int*>(static_cast<const int*>(problem_sizes.const_data_ptr())),
        of,
        sf_layout,
        lm_psf,
        mm_psf,
        hm_psf);
  }
}

template <typename GemmTraits>
void launch_sm90_fp8_blockwise_scaled_group_mm(
    torch::stable::Tensor& out_ptrs,
    const torch::stable::Tensor& a_ptrs,
    const torch::stable::Tensor& b_ptrs,
    const torch::stable::Tensor& a_scales_ptrs,
    const torch::stable::Tensor& b_scales_ptrs,
    const torch::stable::Tensor& stride_a,
    const torch::stable::Tensor& stride_b,
    const torch::stable::Tensor& stride_d,
    const torch::stable::Tensor& layout_sfa,
    const torch::stable::Tensor& layout_sfb,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& workspace,
    cudaStream_t stream,
    int sm_count) {
  using ElementA = typename GemmTraits::ElementA;
  using StrideA = typename GemmTraits::StrideA;
  using ElementB = typename GemmTraits::ElementB;
  using StrideB = typename GemmTraits::StrideB;
  using ElementAccumulator = typename GemmTraits::ElementAccumulator;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using ElementD = typename GemmTraits::ElementD;
  using StrideD = typename GemmTraits::StrideD;
  using UnderlyingProblemShape = typename GemmTraits::ProblemShape::UnderlyingProblemShape;
  using Gemm = typename GemmTraits::Gemm;
  using GemmKernel = typename GemmTraits::GemmKernel;

  int num_experts = (int)problem_sizes.size(0);
  Gemm gemm_op;

  typename GemmKernel::MainloopArguments mainloop_args{
      static_cast<const ElementA**>(const_cast<void*>(a_ptrs.const_data_ptr())),
      static_cast<StrideA*>(const_cast<void*>(stride_a.const_data_ptr())),
      static_cast<const ElementB**>(const_cast<void*>(b_ptrs.const_data_ptr())),
      static_cast<StrideB*>(const_cast<void*>(stride_b.const_data_ptr())),
      static_cast<const ElementAccumulator**>(const_cast<void*>(a_scales_ptrs.const_data_ptr())),
      reinterpret_cast<LayoutSFA*>(const_cast<void*>(layout_sfa.const_data_ptr())),
      static_cast<const ElementAccumulator**>(const_cast<void*>(b_scales_ptrs.const_data_ptr())),
      reinterpret_cast<LayoutSFB*>(const_cast<void*>(layout_sfb.const_data_ptr()))};

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = torch::stable::accelerator::getCurrentDeviceIndex();
  hw_info.sm_count = sm_count;

  typename GemmKernel::EpilogueArguments epilogue_args{
      {},
      nullptr,
      nullptr,
      static_cast<ElementD**>(const_cast<void*>(out_ptrs.const_data_ptr())),
      static_cast<StrideD*>(const_cast<void*>(stride_d.const_data_ptr()))};

  UnderlyingProblemShape* problem_sizes_as_shapes =
      static_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.const_data_ptr()));
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes_as_shapes, nullptr},
      mainloop_args,
      epilogue_args,
      hw_info};

  auto can_implement_status = gemm_op.can_implement(args);
  STD_TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  auto status = gemm_op.initialize(args, workspace.mutable_data_ptr(), stream);
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm_op.run(stream, nullptr);
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void es_sm90_fp8_blockwise_scaled_group_mm_distpatch_out_dtype(
    torch::stable::Tensor& out_ptrs,
    const torch::stable::Tensor& a_ptrs,
    const torch::stable::Tensor& b_ptrs,
    const torch::stable::Tensor& a_scales_ptrs,
    const torch::stable::Tensor& b_scales_ptrs,
    const torch::stable::Tensor& stride_a,
    const torch::stable::Tensor& stride_b,
    const torch::stable::Tensor& stride_d,
    const torch::stable::Tensor& layout_sfa,
    const torch::stable::Tensor& layout_sfb,
    const torch::stable::Tensor& lm_problem_sizes,
    const torch::stable::Tensor& mm_problem_sizes,
    const torch::stable::Tensor& hm_problem_sizes,
    const torch::stable::Tensor& workspace,
    const torch::stable::Tensor& backup_workspace_0,
    const torch::stable::Tensor& backup_workspace_1,
    bool is_h20_device,
    cudaStream_t stream,
    cudaStream_t backup_stream_0,
    cudaStream_t backup_stream_1) {
  using LowMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::ColumnMajor, PerfConfigLowMH20>;
  using LowMGemmHx00Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::ColumnMajor, PerfConfigLowMHx00>;
  using MiddleMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigMiddleMH20>;
  using MiddleMGemmHx00Traits = ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<
      OutType,
      cutlass::layout::ColumnMajor,
      PerfConfigMiddleMHx00>;
  using HighMGemmH20Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigHighMH20>;
  using HighMGemmHx00Traits =
      ExpertSpecializationSm90FP8BlockwiseGroupedGemmTraits<OutType, cutlass::layout::RowMajor, PerfConfigHighMHx00>;

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmHx00Traits>(
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
        hm_problem_sizes,
        workspace,
        stream,
        132);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<HighMGemmH20Traits>(
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
        hm_problem_sizes,
        workspace,
        stream,
        78);
  }

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<LowMGemmHx00Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        lm_problem_sizes,
        backup_workspace_1,
        backup_stream_1,
        132);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<LowMGemmH20Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        lm_problem_sizes,
        backup_workspace_1,
        backup_stream_1,
        78);
  }

  if (!is_h20_device) {
    launch_sm90_fp8_blockwise_scaled_group_mm<MiddleMGemmHx00Traits>(
        out_ptrs,
        b_ptrs,
        a_ptrs,
        b_scales_ptrs,
        a_scales_ptrs,
        stride_b,
        stride_a,
        stride_d,
        layout_sfb,
        layout_sfa,
        mm_problem_sizes,
        backup_workspace_0,
        backup_stream_0,
        132);
  } else {
    launch_sm90_fp8_blockwise_scaled_group_mm<MiddleMGemmH20Traits>(
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
        mm_problem_sizes,
        backup_workspace_0,
        backup_stream_0,
        78);
  }
}

}  // namespace expert_specialization
