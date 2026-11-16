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

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Layout.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <optional>
#include <type_traits>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"
#include "gemm/gemm_ops.h"
#include "sgl_kernel_cuda_device.h"
#include "sgl_kernel_cuda_stream.h"

using TorchTensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Dtype = torch::headeronly::ScalarType;
using Half = torch::headeronly::Half;
using BFloat16 = torch::headeronly::BFloat16;
using OptionalTensor = std::optional<TorchTensor>;

#define SGL_CURRENT_CUDA_STREAM(tensor_) sgl_kernel::stable::get_current_cuda_stream((tensor_).get_device_index())
#define SGL_INPUT_DATA_PTR(tensor_, type_) const_cast<type_*>((tensor_).const_data_ptr<type_>())
#define SGL_MUTABLE_BYTE_PTR(tensor_) (tensor_).mutable_data_ptr<uint8_t>()
#define SGL_NEW_EMPTY_1(self_, size_, dtype_) \
  torch::stable::empty({static_cast<int64_t>(size_)}, (dtype_), torch::headeronly::Layout::Strided, (self_).device())
#define SGL_NEW_EMPTY_2(self_, size0_, size1_, dtype_) torch::stable::new_empty((self_), {(size0_), (size1_)}, (dtype_))

inline int get_current_sm_version() {
  const auto& device_prop = sgl_kernel::stable::get_cached_device_properties();
  return device_prop.major * 10 + device_prop.minor;
}

#define SGL_GET_SM_VERSION() get_current_sm_version()

template <typename ElementOutput>
ElementOutput* output_data_ptr(TorchTensor& tensor) {
  if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
    return reinterpret_cast<ElementOutput*>(tensor.mutable_data_ptr<Half>());
  } else {
    static_assert(std::is_same_v<ElementOutput, cutlass::bfloat16_t>);
    return reinterpret_cast<ElementOutput*>(tensor.mutable_data_ptr<BFloat16>());
  }
}

template <typename ElementOutput>
ElementOutput* bias_data_ptr(const TorchTensor& tensor) {
  if constexpr (std::is_same_v<ElementOutput, cutlass::half_t>) {
    return const_cast<ElementOutput*>(reinterpret_cast<const ElementOutput*>(tensor.const_data_ptr<Half>()));
  } else {
    static_assert(std::is_same_v<ElementOutput, cutlass::bfloat16_t>);
    return const_cast<ElementOutput*>(reinterpret_cast<const ElementOutput*>(tensor.const_data_ptr<BFloat16>()));
  }
}

#define SGL_OUTPUT_DATA_PTR(tensor_, type_) output_data_ptr<type_>(tensor_)
#define SGL_BIAS_DATA_PTR(tensor_, type_) bias_data_ptr<type_>(tensor_)

using namespace cute;

template <
    typename ElementOutput,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages>
void cutlass_int8_scaled_mm(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf = cutlass::gemm::device::
      DefaultGemmConfiguration<OperatorClass, ArchTag, ElementInputA, ElementInputB, ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA,
      cutlass::layout::RowMajor,
      DefaultGemmConf::kAlignmentA,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      DefaultGemmConf::kAlignmentB,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      NumStages,
      true,
      typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator = cutlass::epilogue::threadblock::PredicatedTileIterator<
      cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
          typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
          GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
          GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
          cutlass::sizeof_bits<ElementOutput>::value>,
      ElementCompute>;

  using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
      ThreadblockShape,
      GemmKernel_::kThreadCount,
      AlphaColTileIterator,
      typename GemmKernel_::Epilogue::OutputTileIterator,
      ElementAccumulator,
      ElementCompute,
      EpilogueOutputOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  using GemmKernel =
      cutlass::gemm::kernel::GemmWithEpilogueVisitor<typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = SGL_INPUT_DATA_PTR(mat_a, ElementInputA);
  auto b_ptr = SGL_INPUT_DATA_PTR(mat_b, ElementInputB);
  auto o_ptr = SGL_OUTPUT_DATA_PTR(out, ElementOutput);

  auto a_s_ptr = SGL_INPUT_DATA_PTR(scales_a, ElementCompute);
  auto b_s_ptr = SGL_INPUT_DATA_PTR(scales_b, ElementCompute);

  int64_t lda = mat_a.stride(0);
  int64_t ldb = mat_b.stride(1);
  int64_t ldd = out.stride(0);

  ElementOutput* bias_ptr = nullptr;
  int64_t ldc = 0;
  if (bias) {
    bias_ptr = SGL_BIAS_DATA_PTR(*bias, ElementOutput);
  }

  typename EpilogueOutputOp::Params linearScalingParams;
  typename EpilogueVisitor::Arguments visitor_args{linearScalingParams};

  typename Gemm::Arguments args{
      {m, n, k}, {a_ptr, lda}, {b_ptr, ldb}, {b_s_ptr, 0}, {a_s_ptr, 0}, {bias_ptr, ldc}, {o_ptr, ldd}, visitor_args};

  auto workspace = SGL_NEW_EMPTY_1(mat_a, gemm_op.get_workspace_size(args), ScalarType::Byte);

  auto stream = SGL_CURRENT_CUDA_STREAM(mat_a);

  auto can_implement = gemm_op.can_implement(args);
  STD_TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "gemm cannot implement, error: ",
      cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, SGL_MUTABLE_BYTE_PTR(workspace), stream);
  STD_TORCH_CHECK(
      status == cutlass::Status::kSuccess, "gemm executioin failed, error: ", cutlassGetStatusString(status));
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm75_dispatch_shape(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  int m = mat_a.size(0);
  if (m <= 32) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<32, 128, 64>,
        cutlass::gemm::GemmShape<32, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 64) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (m <= 256) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        2>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm80_dispatch_shape(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128 && n < 8192) {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

// Dispatch shape for sm89 (L40S, L20, RTX 4090), according to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm89_int8_dispatch.cuh
template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm89_dispatch_shape(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<16, 128, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<32, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 8192) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<256, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<
          ElementOutput,
          ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape,
          5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    cutlass_int8_scaled_mm<
        ElementOutput,
        ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape,
        5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <
    typename ElementOutput,
    typename TileShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    bool WithBias>
void cutlass_int8_scaled_mm_sm90(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  using ArchTag = cutlass::arch::Sm90;

  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementOutput>::value;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;
  using TileSchedulerType = cutlass::gemm::PersistentScheduler;

  using XScale = cutlass::epilogue::fusion::
      Sm90ColBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<1>, Int<0>, Int<0>>>;

  using WScale = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementCompute, ElementCompute, Stride<Int<0>, Int<1>, Int<0>>>;

  using Bias = cutlass::epilogue::fusion::
      Sm90RowBroadcast<0, TileShape, ElementOutput, ElementOutput, Stride<Int<0>, Int<1>, Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  // Scale
  using Compute0 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementCompute, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = cutlass::epilogue::fusion::
      Sm90Compute<cutlass::multiply_add, ElementOutput, ElementCompute, cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementOutput,
      cutlass::layout::RowMajor,
      AlignmentC,
      ElementOutput,
      cutlass::layout::RowMajor,
      AlignmentOutput,
      EpilogueScheduleType,
      EpilogueEVT>::CollectiveOp;

  using Stages = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
      sizeof(typename CollectiveEpilogue::SharedStorage))>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementInputA,
      cutlass::layout::RowMajor,
      AlignmentA,
      ElementInputB,
      cutlass::layout::ColumnMajor,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      Stages,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = SGL_INPUT_DATA_PTR(mat_a, ElementInputA);
  auto b_ptr = SGL_INPUT_DATA_PTR(mat_b, ElementInputB);
  auto o_ptr = SGL_OUTPUT_DATA_PTR(out, ElementOutput);

  auto a_s_ptr = SGL_INPUT_DATA_PTR(scales_a, ElementCompute);
  auto b_s_ptr = SGL_INPUT_DATA_PTR(scales_b, ElementCompute);

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));

  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {a_ptr, stride_a, b_ptr, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       o_ptr,
       stride_d}};

  if constexpr (WithBias) {
    ElementOutput* bias_ptr = SGL_BIAS_DATA_PTR(*bias, ElementOutput);
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {bias_ptr},
        {},
    };
  } else {
    args.epilogue.thread = {
        {a_s_ptr},
        {{b_s_ptr}, {}, {}},
        {},
    };
  }

  auto workspace = SGL_NEW_EMPTY_1(mat_a, gemm_op.get_workspace_size(args), ScalarType::Byte);

  auto stream = SGL_CURRENT_CUDA_STREAM(mat_a);

  auto can_implement = gemm_op.can_implement(args);
  STD_TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "gemm cannot implement, error: ",
      cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, SGL_MUTABLE_BYTE_PTR(workspace), stream);
  STD_TORCH_CHECK(
      status == cutlass::Status::kSuccess, "gemm executioin failed, error: ", cutlassGetStatusString(status));
}

template <typename ElementOutput, typename TileShape, typename ClusterShape, typename MainloopScheduleType>
void sm90_dispatch_bias(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  if (bias) {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, true>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm_sm90<ElementOutput, TileShape, ClusterShape, MainloopScheduleType, false>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

template <typename ElementOutput>
void sm90_dispatch_shape(
    TorchTensor& out,
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const OptionalTensor& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 32) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _8, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_1, _8, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n < 8192) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_1, _4, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _256>,
          Shape<_1, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 4096) {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _64, _128>,
          Shape<_2, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      return sm90_dispatch_bias<
          ElementOutput,
          Shape<_64, _128, _128>,
          Shape<_2, _1, _1>,
          cutlass::gemm::KernelTmaWarpSpecialized>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    return sm90_dispatch_bias<
        ElementOutput,
        Shape<_128, _128, _128>,
        Shape<_2, _1, _1>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

TorchTensor int8_scaled_mm(
    const TorchTensor& mat_a,
    const TorchTensor& mat_b,
    const TorchTensor& scales_a,
    const TorchTensor& scales_b,
    const Dtype& out_dtype,
    const OptionalTensor& bias) {
  STD_TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  STD_TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  STD_TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  STD_TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  STD_TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  STD_TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be a column major tensor");
  STD_TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  STD_TORCH_CHECK(mat_a.size(1) % 16 == 0, "mat_a.size(1) must be multiple of 16 for memory alignment");
  STD_TORCH_CHECK(mat_b.size(0) % 16 == 0, "mat_b.size(0) must be multiple of 16 for memory alignment");
  STD_TORCH_CHECK(mat_b.size(1) % 8 == 0, "mat_b.size(1) must be multiple of 8 for memory alignment");  // out.stride(0)
  STD_TORCH_CHECK(mat_a.scalar_type() == ScalarType::Char, "mat_a must be Int8");
  STD_TORCH_CHECK(mat_b.scalar_type() == ScalarType::Char, "mat_b must be Int8");
  STD_TORCH_CHECK(
      out_dtype == ScalarType::Half || out_dtype == ScalarType::BFloat16, "out_dtype must be Half or BFloat16");

  STD_TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  STD_TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  STD_TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  STD_TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  STD_TORCH_CHECK(scales_a.scalar_type() == ScalarType::Float, "scales_a must be Float32");
  STD_TORCH_CHECK(scales_b.scalar_type() == ScalarType::Float, "scales_b must be Float32");

  if (bias) {
    STD_TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    STD_TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    STD_TORCH_CHECK(bias->scalar_type() == out_dtype, "bias dtype must match output dtype");
  }

  TorchTensor out = SGL_NEW_EMPTY_2(mat_a, mat_a.size(0), mat_b.size(1), out_dtype);

  auto sm_version = SGL_GET_SM_VERSION();

  if (sm_version >= 75 && sm_version < 80) {
    STD_TORCH_CHECK(out_dtype == ScalarType::Half, "out_dtype must be Half for SM75");
    sm75_dispatch_shape<cutlass::half_t, cutlass::arch::Sm75, cutlass::gemm::GemmShape<8, 8, 16>>(
        out, mat_a, mat_b, scales_a, scales_b, bias);
  } else if (sm_version >= 80 && sm_version < 90) {
    // sm86/sm89 has a much smaller shared memory size (100K) than sm80 (160K)
    if (sm_version == 86 || sm_version == 89) {
      if (out_dtype == ScalarType::BFloat16) {
        sm89_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm89_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    } else {
      if (out_dtype == ScalarType::BFloat16) {
        sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    }
  } else if (sm_version == 90) {
#if defined CUDA_VERSION && CUDA_VERSION >= 12000
    // cutlass 3.x
    if (out_dtype == ScalarType::BFloat16) {
      sm90_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm90_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#else
    // fallback to cutlass 2.x
    if (out_dtype == ScalarType::BFloat16) {
      sm80_dispatch_shape<cutlass::bfloat16_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm80_dispatch_shape<cutlass::half_t, cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 8, 32>>(
          out, mat_a, mat_b, scales_a, scales_b, bias);
    }
#endif
  } else {
    STD_TORCH_CHECK(false, "NotImplementedError: No implemented int8_scaled_mm for current compute capability.");
  }

  return out;
}

#undef SGL_CURRENT_CUDA_STREAM
#undef SGL_GET_SM_VERSION
#undef SGL_INPUT_DATA_PTR
#undef SGL_OUTPUT_DATA_PTR
#undef SGL_BIAS_DATA_PTR
#undef SGL_MUTABLE_BYTE_PTR
#undef SGL_NEW_EMPTY_1
#undef SGL_NEW_EMPTY_2
