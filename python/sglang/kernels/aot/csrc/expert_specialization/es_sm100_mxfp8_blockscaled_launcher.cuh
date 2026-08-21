#pragma once

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cassert>
#include <iostream>
#include <string>

#include "cute/tensor.hpp"
#include "es_sm100_mxfp8_blockscaled_functor.cuh"
#include "es_sm100_mxfp8_blockscaled_traits.cuh"
#include "expert_specialization_utils.h"
#include "sgl_kernel_cuda_device.h"

namespace expert_specialization {

template <typename GemmTraits>
void es_sm100_mxfp8_blockscaled_group_mm_pre_compute(
    torch::stable::Tensor& a_ptrs,
    torch::stable::Tensor& b_ptrs,
    torch::stable::Tensor& sfa_ptrs,
    torch::stable::Tensor& sfb_ptrs,
    torch::stable::Tensor& d_ptrs,
    torch::stable::Tensor& stride_a,
    torch::stable::Tensor& stride_b,
    torch::stable::Tensor& stride_d,
    torch::stable::Tensor& layout_sfa,
    torch::stable::Tensor& layout_sfb,
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& sfa,
    const torch::stable::Tensor& sfb,
    const torch::stable::Tensor& d,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets,
    cudaStream_t stream) {
  using OffsetFunctor = Sm100Mxfp8BlockScaledOffsetFunctor<GemmTraits>;
  using ElementA = typename OffsetFunctor::ElementA;
  using ElementB = typename OffsetFunctor::ElementB;
  using ElementSF = typename OffsetFunctor::ElementSF;
  using ElementD = typename OffsetFunctor::ElementD;

  using LayoutFunctor = Sm100Mxfp8BlockScaledLayoutFunctor<GemmTraits>;
  using LayoutSFA = typename LayoutFunctor::LayoutSFA;
  using LayoutSFB = typename LayoutFunctor::LayoutSFB;

  using StrideFunctor = Sm100Mxfp8BlockScaledStrideFunctor<GemmTraits>;
  using StrideA = typename StrideFunctor::StrideA;
  using StrideB = typename StrideFunctor::StrideB;
  using StrideD = typename StrideFunctor::StrideD;

  int num_experts = (int)expert_offsets.size(0);
  STD_TORCH_CHECK(
      num_experts <= 1024, "Number of experts cannot exceed 1024, the maximum number of threads per block.");

  OffsetFunctor offset_functor(
      reinterpret_cast<int*>(const_cast<void*>(expert_offsets.const_data_ptr())),
      reinterpret_cast<int*>(const_cast<void*>(blockscale_offsets.const_data_ptr())),
      reinterpret_cast<ElementA*>(const_cast<void*>(a.const_data_ptr())),
      reinterpret_cast<ElementB*>(const_cast<void*>(b.const_data_ptr())),
      reinterpret_cast<ElementSF*>(const_cast<void*>(sfa.const_data_ptr())),
      reinterpret_cast<ElementSF*>(const_cast<void*>(sfb.const_data_ptr())),
      reinterpret_cast<ElementD*>(d.mutable_data_ptr()),
      reinterpret_cast<ElementA**>(a_ptrs.mutable_data_ptr()),
      reinterpret_cast<ElementB**>(b_ptrs.mutable_data_ptr()),
      reinterpret_cast<ElementSF**>(sfa_ptrs.mutable_data_ptr()),
      reinterpret_cast<ElementSF**>(sfb_ptrs.mutable_data_ptr()),
      reinterpret_cast<ElementD**>(d_ptrs.mutable_data_ptr()));
  LayoutFunctor layout_functor(
      reinterpret_cast<LayoutSFA*>(layout_sfa.mutable_data_ptr()),
      reinterpret_cast<LayoutSFB*>(layout_sfb.mutable_data_ptr()));
  StrideFunctor stride_functor(
      reinterpret_cast<StrideA*>(stride_a.mutable_data_ptr()),
      reinterpret_cast<StrideB*>(stride_b.mutable_data_ptr()),
      reinterpret_cast<StrideD*>(stride_d.mutable_data_ptr()));
  sm100Mxfp8BlockscaledGroupedGemmPreComputeKernel<<<1, num_experts, 0, stream>>>(
      static_cast<int*>(const_cast<void*>(problem_sizes.const_data_ptr())),
      offset_functor,
      layout_functor,
      stride_functor);
}

template <typename GemmTraits>
void es_sm100_mxfp8_blockscaled_group_mm(
    const torch::stable::Tensor& a_ptrs,
    const torch::stable::Tensor& b_ptrs,
    const torch::stable::Tensor& sfa_ptrs,
    const torch::stable::Tensor& sfb_ptrs,
    const torch::stable::Tensor& d_ptrs,
    const torch::stable::Tensor& stride_a,
    const torch::stable::Tensor& stride_b,
    const torch::stable::Tensor& stride_d,
    const torch::stable::Tensor& layout_sfa,
    const torch::stable::Tensor& layout_sfb,
    const torch::stable::Tensor& problem_sizes,
    cudaStream_t stream) {
  using Gemm = typename GemmTraits::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementOutput;
  using StrideA = typename GemmTraits::StrideA;
  using StrideB = typename GemmTraits::StrideB;
  using StrideD = typename GemmTraits::StrideD;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  using UnderlyingProblemShape = typename GemmTraits::ProblemShape::UnderlyingProblemShape;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = torch::stable::accelerator::getCurrentDeviceIndex();
  hw_info.sm_count = sgl_kernel::stable::get_cached_device_properties().multiProcessorCount;
  hw_info.cluster_shape = GemmTraits::MMAConfig::preferred_cluster;
  hw_info.cluster_shape_fallback = GemmTraits::MMAConfig::fallback_cluster;

  int num_experts = (int)problem_sizes.size(0);

  UnderlyingProblemShape* underlying_problem_shape =
      reinterpret_cast<UnderlyingProblemShape*>(const_cast<void*>(problem_sizes.const_data_ptr()));

  typename Gemm::Arguments arguments = {
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, underlying_problem_shape, nullptr},
      {reinterpret_cast<const ElementA**>(const_cast<void*>(a_ptrs.const_data_ptr())),
       reinterpret_cast<StrideA*>(const_cast<void*>(stride_a.const_data_ptr())),
       reinterpret_cast<const ElementB**>(const_cast<void*>(b_ptrs.const_data_ptr())),
       reinterpret_cast<StrideB*>(const_cast<void*>(stride_b.const_data_ptr())),
       reinterpret_cast<const ElementSF**>(const_cast<void*>(sfa_ptrs.const_data_ptr())),
       reinterpret_cast<LayoutSFA*>(const_cast<void*>(layout_sfa.const_data_ptr())),
       reinterpret_cast<const ElementSF**>(const_cast<void*>(sfb_ptrs.const_data_ptr())),
       reinterpret_cast<LayoutSFB*>(const_cast<void*>(layout_sfb.const_data_ptr()))},
      {{},
       nullptr,
       nullptr,
       reinterpret_cast<ElementD**>(const_cast<void*>(d_ptrs.const_data_ptr())),
       reinterpret_cast<StrideD*>(const_cast<void*>(stride_d.const_data_ptr()))},
      hw_info,
      {}  // Scheduler
  };

  Gemm gemm;

  auto can_implement_status = gemm.can_implement(arguments);
  STD_TORCH_CHECK(can_implement_status == cutlass::Status::kSuccess, "Failed to implement GEMM");

  size_t workspace_size = gemm.get_workspace_size(arguments);
  auto workspace = expert_specialization::empty(
      d_ptrs.device(), {static_cast<int64_t>(workspace_size)}, torch::headeronly::ScalarType::Byte);

  auto status = gemm.initialize(arguments, workspace.mutable_data_ptr(), stream);
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize GEMM");

  status = gemm.run(stream, nullptr, true);  // Enable PDL
  STD_TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run GEMM");
}

template <typename OutType>
void es_sm100_mxfp8_blockscaled_group_mm_dispatch_out_dtype(
    const torch::stable::Tensor& a,
    const torch::stable::Tensor& b,
    const torch::stable::Tensor& sfa,
    const torch::stable::Tensor& sfb,
    torch::stable::Tensor& d,
    const torch::stable::Tensor& problem_sizes,
    const torch::stable::Tensor& expert_offsets,
    const torch::stable::Tensor& blockscale_offsets,
    cudaStream_t stream) {
  int num_experts = (int)problem_sizes.size(0);
  const auto device = a.device();
  auto a_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto b_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto sfa_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto sfb_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto d_ptrs = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);

  auto stride_a = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto stride_b = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto stride_d = expert_specialization::empty(device, {num_experts}, torch::headeronly::ScalarType::Long);
  auto layout_sfa = expert_specialization::empty(device, {num_experts, 5}, torch::headeronly::ScalarType::Int);
  auto layout_sfb = expert_specialization::empty(device, {num_experts, 5}, torch::headeronly::ScalarType::Int);

  using GemmTraits = ExpertSpecializationSm100MXFP8BlockscaledGroupedGemmTraits<MMA1SMConfig, OutType>;
  es_sm100_mxfp8_blockscaled_group_mm_pre_compute<GemmTraits>(
      a_ptrs,
      b_ptrs,
      sfa_ptrs,
      sfb_ptrs,
      d_ptrs,
      stride_a,
      stride_b,
      stride_d,
      layout_sfa,
      layout_sfb,
      a,
      b,
      sfa,
      sfb,
      d,
      problem_sizes,
      expert_offsets,
      blockscale_offsets,
      stream);
  es_sm100_mxfp8_blockscaled_group_mm<GemmTraits>(
      a_ptrs,
      b_ptrs,
      sfa_ptrs,
      sfb_ptrs,
      d_ptrs,
      stride_a,
      stride_b,
      stride_d,
      layout_sfa,
      layout_sfb,
      problem_sizes,
      stream);
}

}  // namespace expert_specialization
