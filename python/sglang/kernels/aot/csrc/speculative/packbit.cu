// This is only a plugin used for flashinfer 0.1.6. The new version does not need it.
/*
 * Copyright (c) 2025 by SGLang team.
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <flashinfer/quantization.cuh>

#include "speculative/speculative_ops.h"

using Tensor = SglTensor;

#ifdef TORCH_TARGET_VERSION
#include <torch/headeronly/util/Exception.h>

#include <string>

#define SGL_CHECK_INPUT(x)                                   \
  STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor"); \
  STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

std::string non_cuda_device_string(const Tensor& tensor) {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(tensor.get(), &device_type));
  if (device_type == aoti_torch_device_type_cpu()) {
    return "cpu";
  }
  if (device_type == aoti_torch_device_type_meta()) {
    return "meta";
  }

  const auto device_index = tensor.get_device_index();
  if (device_type == aoti_torch_device_type_xpu()) {
    return "xpu:" + std::to_string(device_index);
  }
  if (device_type == aoti_torch_device_type_mps()) {
    return "mps:" + std::to_string(device_index);
  }
  if (device_type == aoti_torch_device_type_privateuse1()) {
    return "privateuseone:" + std::to_string(device_index);
  }
  return "device_type(" + std::to_string(device_type) + "):" + std::to_string(device_index);
}

}  // namespace

#else
#include "pytorch_extension_utils.h"

#define SGL_CHECK_INPUT(x) CHECK_INPUT(x)
#endif

using namespace flashinfer;

// bitorder = "little"
void segment_packbits(
    Tensor x, Tensor input_indptr, Tensor output_indptr, Tensor y, int64_t batch_size, int64_t cuda_stream) {
  SGL_CHECK_INPUT(x);
  SGL_CHECK_INPUT(input_indptr);
  SGL_CHECK_INPUT(output_indptr);
#ifdef TORCH_TARGET_VERSION
  const auto device = x.get_device_index();
  const auto input_indptr_device = input_indptr.get_device_index();
  const auto output_indptr_device = output_indptr.get_device_index();
  STD_TORCH_CHECK(
      input_indptr_device == device,
      "CHECK_EQ(input_indptr.device(), device) failed. cuda:",
      input_indptr_device,
      " vs cuda:",
      device);
  STD_TORCH_CHECK(
      output_indptr_device == device,
      "CHECK_EQ(output_indptr.device(), device) failed. cuda:",
      output_indptr_device,
      " vs cuda:",
      device);
  if (y.is_cuda()) {
    STD_TORCH_CHECK(
        y.get_device_index() == device,
        "CHECK_EQ(y.device(), device) failed. cuda:",
        y.get_device_index(),
        " vs cuda:",
        device);
  } else {
    STD_TORCH_CHECK(false, "CHECK_EQ(y.device(), device) failed. ", non_cuda_device_string(y), " vs cuda:", device);
  }
  STD_TORCH_CHECK(
      output_indptr.size(0) >= batch_size + 1,
      "CHECK_GE(output_indptr.size(0), batch_size + 1) failed. ",
      output_indptr.size(0),
      " vs ",
      batch_size + 1);
#else
  auto device = x.device();
  CHECK_EQ(input_indptr.device(), device);
  CHECK_EQ(output_indptr.device(), device);
  CHECK_EQ(y.device(), device);
  CHECK_GE(output_indptr.size(0), batch_size + 1);
#endif

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
#ifdef TORCH_TARGET_VERSION
  cudaError_t status = quantization::SegmentPackBits(
      const_cast<bool*>(static_cast<const bool*>(SGL_CONST_DATA_PTR(x))),
      static_cast<uint8_t*>(SGL_MUTABLE_DATA_PTR(y)),
      const_cast<int32_t*>(static_cast<const int32_t*>(SGL_CONST_DATA_PTR(input_indptr))),
      const_cast<int32_t*>(static_cast<const int32_t*>(SGL_CONST_DATA_PTR(output_indptr))),
      batch_size,
      quantization::BitOrder::kLittle,
      stream);
#else
  cudaError_t status = quantization::SegmentPackBits(
      static_cast<bool*>(x.data_ptr()),
      static_cast<uint8_t*>(y.data_ptr()),
      static_cast<int32_t*>(input_indptr.data_ptr()),
      static_cast<int32_t*>(output_indptr.data_ptr()),
      batch_size,
      quantization::BitOrder::kLittle,
      stream);
#endif
}

#undef SGL_CHECK_INPUT
