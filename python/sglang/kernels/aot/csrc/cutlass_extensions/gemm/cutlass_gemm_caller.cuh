// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh

#pragma once

// clang-format will break include orders
// clang-format off
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "sgl_kernel_cuda_device.h"
#include "sgl_kernel_cuda_stream.h"

// clang-format on

/**
 * Helper function for checking CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                           \
  {                                                                                     \
    cutlass::Status error = status;                                                     \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

template <typename GemmKernel>
void cutlass_gemm_caller(
    torch::stable::Device device,
    cute::Shape<int, int, int, int> prob_shape,
    typename GemmKernel::MainloopArguments mainloop_args,
    typename GemmKernel::EpilogueArguments epilogue_args,
    typename GemmKernel::TileSchedulerArguments scheduler = {}) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = torch::stable::accelerator::getCurrentDeviceIndex();
  hw_info.sm_count = sgl_kernel::stable::get_cached_device_properties().multiProcessorCount;
  typename GemmKernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, scheduler};

  // Launch the CUTLASS GEMM kernel.
  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  GemmOp gemm_op;
  CUTLASS_CHECK(gemm_op.can_implement(args));

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto workspace = torch::stable::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::headeronly::ScalarType::Byte,
      torch::headeronly::Layout::Strided,
      device);

  auto stream = sgl_kernel::stable::get_current_cuda_stream(device.index());

  cutlass::Status status = gemm_op.run(args, workspace.mutable_data_ptr(), stream);
  CUTLASS_CHECK(status);
}
