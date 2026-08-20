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
#include "speculative/speculative_ops.h"

#ifdef TORCH_TARGET_VERSION
#include "sgl_kernel_cuda_stream.h"

#define CHECK_CUDA(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_DIM(d, x) STD_TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")
#define CHECK_EQ(a, b) STD_TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)
#define CHECK_GE(a, b) STD_TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)
#define SGL_KERNEL_DTYPE_CHECK(condition, message) STD_TORCH_CHECK(condition, message)
#define SGL_KERNEL_CHECK(...) STD_TORCH_CHECK(__VA_ARGS__)
#else
#include "pytorch_extension_utils.h"

#define SGL_KERNEL_DTYPE_CHECK(condition, message) \
  do {                                             \
    if (!(condition)) {                            \
      throw std::runtime_error(message);           \
    }                                              \
  } while (false)
#define SGL_KERNEL_CHECK(...) TORCH_CHECK(__VA_ARGS__)
#endif

using Tensor = SglTensor;
using ScalarType = SglScalarType;

#define SPEC_MUTABLE_DATA_PTR(tensor, type) static_cast<type*>(SGL_MUTABLE_DATA_PTR(tensor))
#define SPEC_READ_DATA_PTR(tensor, type) const_cast<type*>(static_cast<const type*>(SGL_CONST_DATA_PTR(tensor)))

#include <string>

#include "speculative_sampling.cuh"

using namespace flashinfer;

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// uniform_samples: [bs, num_draft_tokens]
// target_probs: [bs, num_draft_tokens, vocab_size]
void tree_speculative_sampling_target_only(
    Tensor predicts,
    Tensor accept_index,
    Tensor accept_token_num,  // mutable
    Tensor candidates,
    Tensor retrive_index,
    Tensor retrive_next_token,
    Tensor retrive_next_sibling,
    Tensor uniform_samples,
    Tensor uniform_samples_for_final_sampling,
    Tensor target_probs,
    Tensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic
#ifndef TORCH_TARGET_VERSION
    = true
#endif
) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(uniform_samples);
  CHECK_INPUT(uniform_samples_for_final_sampling);
  CHECK_INPUT(target_probs);
#ifdef TORCH_TARGET_VERSION
  const auto device_index = target_probs.get_device_index();
  const auto check_device = [device_index](const Tensor& tensor, const char* expression) {
    const auto tensor_device_index = tensor.get_device_index();
    STD_TORCH_CHECK(
        tensor_device_index == device_index,
        "CHECK_EQ(",
        expression,
        ") failed. cuda:",
        tensor_device_index,
        " vs cuda:",
        device_index);
  };
  check_device(candidates, "candidates.device(), device");
  check_device(retrive_index, "retrive_index.device(), device");
  check_device(retrive_next_token, "retrive_next_token.device(), device");
  check_device(retrive_next_sibling, "retrive_next_sibling.device(), device");
  check_device(uniform_samples, "uniform_samples.device(), device");
  check_device(uniform_samples_for_final_sampling, "uniform_samples_for_final_sampling.device(), device");
  check_device(target_probs, "target_probs.device(), device");
#else
  auto device = target_probs.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_EQ(uniform_samples_for_final_sampling.device(), device);
  CHECK_EQ(target_probs.device(), device);
#endif
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, uniform_samples);
  CHECK_DIM(3, target_probs);
  CHECK_DIM(3, draft_probs);
  unsigned int batch_size = uniform_samples.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  unsigned int vocab_size = target_probs.size(2);
  CHECK_EQ(batch_size, candidates.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_probs.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, uniform_samples.size(1));
  CHECK_EQ(num_draft_tokens, target_probs.size(1));
  CHECK_EQ(vocab_size, target_probs.size(2));
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  SGL_KERNEL_DTYPE_CHECK(
      predicts.scalar_type() == ScalarType::Int, "Expected 'predicts' to be of type int (torch.int32).");
  SGL_KERNEL_DTYPE_CHECK(
      accept_index.scalar_type() == ScalarType::Int, "Expected 'accept_index' to be of type int (torch.int32).");
  SGL_KERNEL_DTYPE_CHECK(
      accept_token_num.scalar_type() == ScalarType::Int,
      "Expected 'accept_token_num' to be of type int (torch.int32).");
  SGL_KERNEL_DTYPE_CHECK(
      candidates.scalar_type() == ScalarType::Long, "Expected 'candidates' to be of type long (torch.int64).");
  SGL_KERNEL_DTYPE_CHECK(
      retrive_index.scalar_type() == ScalarType::Long, "Expected 'retrive_index' to be of type long (torch.int64).");
  SGL_KERNEL_DTYPE_CHECK(
      retrive_next_token.scalar_type() == ScalarType::Long,
      "Expected 'retrive_next_token' to be of type long (torch.int64).");
  SGL_KERNEL_DTYPE_CHECK(
      retrive_next_sibling.scalar_type() == ScalarType::Long,
      "Expected 'retrive_next_sibling' to be of type long (torch.int64).");
  SGL_KERNEL_DTYPE_CHECK(
      uniform_samples.scalar_type() == ScalarType::Float,
      "Expected 'uniform_samples' to be of type float (torch.float32).");
  SGL_KERNEL_DTYPE_CHECK(
      uniform_samples_for_final_sampling.scalar_type() == ScalarType::Float,
      "Expected 'uniform_samples_for_final_sampling' to be of type float (torch.float32).");
  SGL_KERNEL_DTYPE_CHECK(
      target_probs.scalar_type() == ScalarType::Float, "Expected 'target_probs' to be of type float (torch.float32).");
  SGL_KERNEL_DTYPE_CHECK(
      draft_probs.scalar_type() == ScalarType::Float, "Expected 'target_probs' to be of type float (torch.float32).");
  CHECK_GE(threshold_single, 0);
  CHECK_GE(1, threshold_single);
  CHECK_GE(threshold_acc, 0);
  CHECK_GE(1, threshold_acc);

#ifdef TORCH_TARGET_VERSION
  cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();
#else
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
#endif
  cudaError_t status = sampling::TreeSpeculativeSamplingTargetOnly<float, int32_t, int64_t>(
      SPEC_MUTABLE_DATA_PTR(predicts, int32_t),
      SPEC_MUTABLE_DATA_PTR(accept_index, int32_t),
      SPEC_MUTABLE_DATA_PTR(accept_token_num, int32_t),
      SPEC_READ_DATA_PTR(candidates, int64_t),
      SPEC_READ_DATA_PTR(retrive_index, int64_t),
      SPEC_READ_DATA_PTR(retrive_next_token, int64_t),
      SPEC_READ_DATA_PTR(retrive_next_sibling, int64_t),
      SPEC_READ_DATA_PTR(uniform_samples, float),
      SPEC_READ_DATA_PTR(uniform_samples_for_final_sampling, float),
      SPEC_READ_DATA_PTR(target_probs, float),
      SPEC_MUTABLE_DATA_PTR(draft_probs, float),
      batch_size,
      num_spec_step,
      num_draft_tokens,
      vocab_size,
      static_cast<float>(threshold_single),
      static_cast<float>(threshold_acc),
      deterministic,
      stream);

  SGL_KERNEL_CHECK(
      status == cudaSuccess,
      "TreeSpeculativeSamplingTargetOnly failed with error code " + std::string(cudaGetErrorString(status)));
}

#ifdef TORCH_TARGET_VERSION
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef CHECK_DIM
#undef CHECK_EQ
#undef CHECK_GE
#endif
#undef SGL_KERNEL_DTYPE_CHECK
#undef SGL_KERNEL_CHECK
#undef SPEC_MUTABLE_DATA_PTR
#undef SPEC_READ_DATA_PTR
