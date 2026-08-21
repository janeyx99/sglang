/*
Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
Copyright 2025 SGLang Team. All Rights Reserved.

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

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/kernel_hardware_info.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cute/tensor.hpp>
#include <iostream>

#include "attention/attention_ops.h"
#include "cutlass_sm100_mla/device/sm100_mla.hpp"
#include "cutlass_sm100_mla/kernel/sm100_mla_tile_scheduler.hpp"
#include "sgl_kernel_cuda_stream.h"

// clang-format off
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
void cutlass_mla_decode(
    SglTensor const& out,
    SglTensor const& q_nope,
    SglTensor const& q_pe,
    SglTensor const& kv_c_and_k_pe_cache,
    SglTensor const& seq_lens,
    SglTensor const& page_table,
    SglTensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  STD_TORCH_CHECK(false, "CUDA version must be >= 12.4 for cutlass_mla_decode");
}
int64_t cutlass_mla_get_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count, int64_t num_kv_splits) {
  STD_TORCH_CHECK(false, "CUDA version must be >= 12.4 for cutlass_mla_get_workspace_size");
}
#else

#define CUTLASS_CHECK(status)                                                       \
  {                                                                                 \
    cutlass::Status error = status;                                                 \
    STD_TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  }

namespace {

int GetSMVersion() {
  int device = -1;
  STD_CUDA_CHECK(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  STD_CUDA_CHECK(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  STD_CUDA_CHECK(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

}  // namespace

using namespace cute;
using namespace cutlass::fmha::kernel;

template <bool v>
struct IsPersistent {
  static const bool value = v;
};

template <typename T, bool IsPaged128, typename PersistenceOption = IsPersistent<true>>
struct MlaSm100 {
  using Element = T;
  using ElementAcc = float;
  using ElementOut = T;

  using TileShape = Shape<_128, _128, Shape<_512, _64>>;
  using TileShapeH = cute::tuple_element_t<0, TileShape>;
  using TileShapeD = cute::tuple_element_t<2, TileShape>;

  // H K (D_latent D_rope) B
  using ProblemShape = cute::tuple<TileShapeH, int, TileShapeD, int>;

  using StrideQ = cute::tuple<int64_t, _1, int64_t>;  // H D B
  using StrideK = cute::tuple<int64_t, _1, int64_t>;  // K D B
  using StrideO = StrideK;                            // H D B
  using StrideLSE = cute::tuple<_1, int>;             // H B

  using TileScheduler =
      std::conditional_t<PersistenceOption::value, Sm100MlaPersistentTileScheduler, Sm100MlaIndividualTileScheduler>;

  using FmhaKernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
      TileShape,
      Element,
      ElementAcc,
      ElementOut,
      ElementAcc,
      TileScheduler,
      /*kIsCpAsync=*/!IsPaged128>;
  using Fmha = cutlass::fmha::device::MLA<FmhaKernel>;
};

template <typename T>
typename T::Fmha::Arguments args_from_options(
    SglTensor const& out,
    SglTensor const& q_nope,
    SglTensor const& q_pe,
    SglTensor const& kv_c_and_k_pe_cache,
    SglTensor const& seq_lens,
    SglTensor const& page_table,
    double sm_scale,
    int64_t num_kv_splits) {
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope.get_device_index();
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int batches = q_nope.size(0);
  int page_count_per_seq = page_table.size(1);
  int page_count_total = kv_c_and_k_pe_cache.size(0);
  int page_size = kv_c_and_k_pe_cache.size(1);
  int max_seq_len = page_size * page_count_per_seq;
  using TileShapeH = typename T::TileShapeH;
  using TileShapeD = typename T::TileShapeD;
  auto problem_shape = cute::make_tuple(TileShapeH{}, max_seq_len, TileShapeD{}, batches);

  auto [H, K, D, B] = problem_shape;
  auto [D_latent, D_rope] = D;

  float scale = float(sm_scale);

  using StrideQ = typename T::StrideQ;
  using StrideK = typename T::StrideK;
  using StrideO = typename T::StrideO;
  using StrideLSE = typename T::StrideLSE;

  StrideQ stride_Q_nope = cute::make_tuple(
      static_cast<int64_t>(q_nope.stride(1)), _1{}, static_cast<int64_t>(q_nope.stride(0)));
  StrideQ stride_Q_pe = cute::make_tuple(
      static_cast<int64_t>(q_pe.stride(1)), _1{}, static_cast<int64_t>(q_pe.stride(0)));

  StrideK stride_C = cute::make_tuple(
      static_cast<int64_t>(0 + D_latent + D_rope), _1{}, static_cast<int64_t>(page_size * (D_latent + D_rope)));
  StrideLSE stride_PT = cute::make_stride(_1{}, page_count_per_seq);
  StrideLSE stride_LSE = cute::make_tuple(_1{}, 0 + H);
  StrideO stride_O = cute::make_tuple(static_cast<int64_t>(0 + D_latent), _1{}, static_cast<int64_t>(0 + H * D_latent));

  using Element = typename T::Element;
  using ElementOut = typename T::ElementOut;
  using ElementAcc = typename T::ElementAcc;
  auto Q_nope_ptr = const_cast<Element*>(static_cast<const Element*>(q_nope.const_data_ptr()));
  auto Q_pe_ptr = const_cast<Element*>(static_cast<const Element*>(q_pe.const_data_ptr()));
  auto C_ptr = const_cast<Element*>(static_cast<const Element*>(kv_c_and_k_pe_cache.const_data_ptr()));
  typename T::Fmha::Arguments arguments{
      problem_shape,
      {scale,
       Q_nope_ptr,
       stride_Q_nope,
       Q_pe_ptr,
       stride_Q_pe,
       C_ptr,
       stride_C,
       C_ptr + D_latent,
       stride_C,
       const_cast<int*>(static_cast<const int*>(seq_lens.const_data_ptr())),
       const_cast<int*>(static_cast<const int*>(page_table.const_data_ptr())),
       stride_PT,
       page_count_total,
       page_size},
      {static_cast<ElementOut*>(out.mutable_data_ptr()), stride_O, static_cast<ElementAcc*>(nullptr), stride_LSE},
      hw_info,
      // TODO(trevor-m): Change split_kv back to -1 when
      // https://github.com/NVIDIA/cutlass/issues/2274 is fixed. Split_kv=1 will
      // perform worse with larger context length and smaller batch sizes.
      static_cast<int>(num_kv_splits), // split_kv
      nullptr,       // is_var_split_kv
  };
  // TODO(kaixih@nvidia): When split_kv=-1 and is_var_split_kv=false, we compute
  // split_kv automatically based on batch size and sequence length to balance
  // workload across available SMs. Consider using var_split_kv for manual
  // control if needed.
  T::Fmha::set_split_kv(arguments);
  return arguments;
}

template <typename Element, bool IsPaged128, typename PersistenceOption>
void runMla(
    SglTensor const& out,
    SglTensor const& q_nope,
    SglTensor const& q_pe,
    SglTensor const& kv_c_and_k_pe_cache,
    SglTensor const& seq_lens,
    SglTensor const& page_table,
    SglTensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits,
    cudaStream_t stream) {
  using MlaSm100Type = MlaSm100<Element, IsPaged128, PersistenceOption>;
  typename MlaSm100Type::Fmha fmha;
  auto arguments = args_from_options<MlaSm100Type>(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, sm_scale, num_kv_splits);

  CUTLASS_CHECK(fmha.can_implement(arguments));

  CUTLASS_CHECK(fmha.initialize(arguments, workspace.mutable_data_ptr(), stream));

  CUTLASS_CHECK(fmha.run(arguments, workspace.mutable_data_ptr(), stream));
}

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()

void cutlass_mla_decode(
    SglTensor const& out,
    SglTensor const& q_nope,
    SglTensor const& q_pe,
    SglTensor const& kv_c_and_k_pe_cache,
    SglTensor const& seq_lens,
    SglTensor const& page_table,
    SglTensor const& workspace,
    double sm_scale,
    int64_t num_kv_splits) {
  auto sm_version = GetSMVersion();
  // On SM103a, half of the accuracy tests are failing.
  STD_TORCH_CHECK(sm_version == 100, "cutlass_mla_decode is only supported on compute capability 10.0, but found sm version ", sm_version);

  auto in_dtype = q_nope.scalar_type();
  STD_TORCH_CHECK(q_nope.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const auto device_index = q_nope.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream(device_index);
  const int page_size = kv_c_and_k_pe_cache.size(1);

  // NOTE(alcanderian): IsPersistent has bug with manual split_kv.
  // Kernel will hang if batch is too large with large num_kv_splits. (for example bs=8, num_kv_splits=8)
  // Maybe per batch split kv will fix this.
  DISPATCH_BOOL(page_size == 128, IsPaged128, [&] {
    DISPATCH_BOOL(num_kv_splits <= 1, NotManualSplitKV, [&] {
      if (in_dtype == torch::headeronly::ScalarType::Half) {
        runMla<cutlass::half_t, IsPaged128, IsPersistent<NotManualSplitKV>>(
          out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits, stream);
      } else if (in_dtype == torch::headeronly::ScalarType::BFloat16) {
        runMla<cutlass::bfloat16_t, IsPaged128, IsPersistent<NotManualSplitKV>>(
          out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits, stream);
      } else if (in_dtype == torch::headeronly::ScalarType::Float8_e4m3fn) {
        runMla<cutlass::float_e4m3_t, IsPaged128, IsPersistent<NotManualSplitKV>>(
          out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, sm_scale, num_kv_splits, stream);
      } else {
        STD_TORCH_CHECK(false, "Unsupported input data type of MLA");
      }
      return true;
    });
    return true;
  });
}

int64_t cutlass_mla_get_workspace_size(int64_t max_seq_len, int64_t num_batches, int64_t sm_count, int64_t num_kv_splits) {
  // Workspace size depends on ElementAcc and ElementLSE (same as ElementAcc)
  // which are float, so Element type here doesn't matter.
  using MlaSm100Type = MlaSm100<cutlass::half_t, true>;

  // Get split kv. Requires problem shape and sm_count only.
  typename MlaSm100Type::Fmha::Arguments arguments;
  using TileShapeH = typename MlaSm100Type::TileShapeH;
  using TileShapeD = typename MlaSm100Type::TileShapeD;
  arguments.problem_shape =
      cute::make_tuple(TileShapeH{}, static_cast<int>(max_seq_len), TileShapeD{}, static_cast<int>(num_batches));
  // Assumes device 0 when getting sm_count.
  arguments.hw_info.sm_count =
      sm_count <= 0 ? cutlass::KernelHardwareInfo::query_device_multiprocessor_count(/*device_id=*/0) : sm_count;
  arguments.split_kv = static_cast<int>(num_kv_splits);
  MlaSm100Type::Fmha::set_split_kv(arguments);

  return MlaSm100Type::Fmha::get_workspace_size(arguments);
}

#endif
// clang-format on
