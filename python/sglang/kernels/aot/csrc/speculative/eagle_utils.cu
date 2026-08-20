/*
 * Copyright (c) 2025 by SGLang team.
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
#define SGL_KERNEL_DTYPE_CHECK(condition, message) STD_TORCH_CHECK(condition, message)
#define SGL_CURRENT_CUDA_STREAM() sgl_kernel::stable::get_current_cuda_stream()
#else
#include <ATen/cuda/CUDAContext.h>

#if !defined(USE_ROCM) && !defined(USE_MUSA)
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

#define SGL_KERNEL_DTYPE_CHECK(condition, message) \
  do {                                             \
    if (!(condition)) {                            \
      throw std::runtime_error(message);           \
    }                                              \
  } while (false)
#define SGL_CURRENT_CUDA_STREAM() at::cuda::getCurrentCUDAStream()
#endif

using Tensor = SglTensor;
using ScalarType = SglScalarType;

#define SPEC_MUTABLE_DATA_PTR(tensor, type) static_cast<type*>(SGL_MUTABLE_DATA_PTR(tensor))
#define SPEC_READ_DATA_PTR(tensor, type) const_cast<type*>(static_cast<const type*>(SGL_CONST_DATA_PTR(tensor)))

typedef enum { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 } TreeMaskMode;

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] =
// [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token] positions [bs * draft_token] retrive_index [b,
// draft_token] retrive_next_token [b, draft_token] retrive_next_sibling [b, draft_token]
__global__ void build_tree_efficient(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask: [draft_token*num_bytes_per_item | .. ] = [bs*draft_token*num_bytes_per_item]
// positions [bs * draft_token]
// retrive_index [bs, draft_token]
// retrive_next_token [bs, draft_token]
// retrive_next_sibling [bs, draft_token]
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    uint8_t* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    size_t num_bytes_per_item) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
  tree_mask[token_tree_idx] = 1;  // little endian

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      int byte_idx = (cur_position + 1) / 8;
      int bit_idx = (cur_position + 1) % 8;
      tree_mask[token_tree_idx + byte_idx] |= (1 << bit_idx);
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

void build_tree_kernel_efficient(
    Tensor parent_list,
    Tensor selected_index,
    Tensor verified_seq_len,
    Tensor tree_mask,
    Tensor positions,
    Tensor retrive_index,
    Tensor retrive_next_token,
    Tensor retrive_next_sibling,
    int64_t topk,
    int64_t depth,
    int64_t draft_token_num,
    int64_t tree_mask_mode) {
  // TODO (ying) check shape
  // TODO (ying) check type
  int bs = parent_list.size(0);
  dim3 grid(bs);
  dim3 block(draft_token_num);
  const cudaStream_t stream = SGL_CURRENT_CUDA_STREAM();

  if (tree_mask_mode == QLEN_ONLY_BITPACKING) {
    size_t num_bytes_per_item = 1;
    if (draft_token_num > 16) {
      num_bytes_per_item = 4;
    } else if (draft_token_num > 8) {
      num_bytes_per_item = 2;
    }
    build_tree_efficient_partial_packed<<<grid, block, 0, stream>>>(
        SPEC_READ_DATA_PTR(parent_list, int64_t),
        SPEC_READ_DATA_PTR(selected_index, int64_t),
        SPEC_READ_DATA_PTR(verified_seq_len, int64_t),
        SPEC_MUTABLE_DATA_PTR(tree_mask, uint8_t),
        SPEC_MUTABLE_DATA_PTR(positions, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_index, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_next_token, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_next_sibling, int64_t),
        int32_t(topk),
        int32_t(depth),
        int32_t(draft_token_num),
        num_bytes_per_item);
  } else {
    build_tree_efficient<<<grid, block, 0, stream>>>(
        SPEC_READ_DATA_PTR(parent_list, int64_t),
        SPEC_READ_DATA_PTR(selected_index, int64_t),
        SPEC_READ_DATA_PTR(verified_seq_len, int64_t),
        SPEC_MUTABLE_DATA_PTR(tree_mask, bool),
        SPEC_MUTABLE_DATA_PTR(positions, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_index, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_next_token, int64_t),
        SPEC_MUTABLE_DATA_PTR(retrive_next_sibling, int64_t),
        int32_t(topk),
        int32_t(depth),
        int32_t(draft_token_num),
        int32_t(tree_mask_mode));
  }
}

template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
void verify_tree_greedy(
    Tensor predicts,
    Tensor accept_index,
    Tensor accept_token_num,  // mutable
    Tensor candidates,
    Tensor retrive_index,
    Tensor retrive_next_token,
    Tensor retrive_next_sibling,
    Tensor target_predict) {
  CHECK_INPUT(candidates);
  CHECK_INPUT(retrive_index);
  CHECK_INPUT(retrive_next_token);
  CHECK_INPUT(retrive_next_sibling);
  CHECK_INPUT(target_predict);
#ifdef TORCH_TARGET_VERSION
  const auto device_index = target_predict.get_device_index();
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
  check_device(target_predict, "target_predict.device(), device");
#else
  auto device = target_predict.device();
  CHECK_EQ(candidates.device(), device);
  CHECK_EQ(retrive_index.device(), device);
  CHECK_EQ(retrive_next_token.device(), device);
  CHECK_EQ(retrive_next_sibling.device(), device);
  CHECK_EQ(target_predict.device(), device);
#endif
  CHECK_DIM(1, predicts);
  CHECK_DIM(2, accept_index);
  CHECK_DIM(1, accept_token_num);
  CHECK_DIM(2, candidates);
  CHECK_DIM(2, retrive_index);
  CHECK_DIM(2, retrive_next_token);
  CHECK_DIM(2, retrive_next_sibling);
  CHECK_DIM(2, target_predict);
  unsigned int batch_size = candidates.size(0);
  unsigned int num_spec_step = accept_index.size(1);
  unsigned int num_draft_tokens = candidates.size(1);
  CHECK_EQ(batch_size, accept_index.size(0));
  CHECK_EQ(batch_size, accept_token_num.size(0));
  CHECK_EQ(batch_size, retrive_index.size(0));
  CHECK_EQ(batch_size, retrive_next_token.size(0));
  CHECK_EQ(batch_size, retrive_next_sibling.size(0));
  CHECK_EQ(batch_size, target_predict.size(0));
  CHECK_EQ(num_draft_tokens, retrive_index.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_token.size(1));
  CHECK_EQ(num_draft_tokens, retrive_next_sibling.size(1));
  CHECK_EQ(num_draft_tokens, target_predict.size(1));
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
      target_predict.scalar_type() == ScalarType::Long, "Expected 'target_predict' to be of type long (torch.int64).");

  cudaStream_t stream = SGL_CURRENT_CUDA_STREAM();
  dim3 grid(batch_size);
  dim3 block(1);

  VerifyTreeGreedy<int32_t, int64_t><<<grid, block, 0, stream>>>(
      SPEC_MUTABLE_DATA_PTR(predicts, int32_t),
      SPEC_MUTABLE_DATA_PTR(accept_index, int32_t),
      SPEC_MUTABLE_DATA_PTR(accept_token_num, int32_t),
      SPEC_READ_DATA_PTR(candidates, int64_t),
      SPEC_READ_DATA_PTR(retrive_index, int64_t),
      SPEC_READ_DATA_PTR(retrive_next_token, int64_t),
      SPEC_READ_DATA_PTR(retrive_next_sibling, int64_t),
      SPEC_READ_DATA_PTR(target_predict, int64_t),
      batch_size,
      num_spec_step,
      num_draft_tokens);
}

#ifdef TORCH_TARGET_VERSION
#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef CHECK_DIM
#undef CHECK_EQ
#endif
#undef SGL_KERNEL_DTYPE_CHECK
#undef SPEC_MUTABLE_DATA_PTR
#undef SPEC_READ_DATA_PTR
#undef SGL_CURRENT_CUDA_STREAM
