#include <cuda_runtime.h>

#include "speculative/speculative_ops.h"

#ifdef TORCH_TARGET_VERSION
#include "sgl_kernel_cuda_stream.h"
#else
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef USE_ROCM
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#endif

#endif

using Tensor = SglTensor;

namespace {

inline const void* ConstDataPtr(const Tensor& tensor) {
  return SGL_CONST_DATA_PTR(tensor);
}

inline void* MutableDataPtr(const Tensor& tensor) {
  return SGL_MUTABLE_DATA_PTR(tensor);
}

inline cudaStream_t GetCurrentCUDAStream() {
#ifdef TORCH_TARGET_VERSION
  return sgl_kernel::stable::get_current_cuda_stream();
#else
  return at::cuda::getCurrentCUDAStream();
#endif
}

}  // namespace

// tree_mask: [bs * draft_token_num * draft_token_num]
// verified_seq_len: [bs]
// positions: [bs * draft_token_num]
// retrive_index: [bs, draft_token_num]
// retrive_next_token: [bs, draft_token_num]
// retrive_next_sibling: [bs, draft_token_num]
__global__ void reconstructIndicesFromTreeMask(
    bool* tree_mask,
    int64_t* verified_seq_len,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int batch_size,
    int draft_token_num) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= batch_size || tid >= draft_token_num) {
    return;
  }
  int base_offset = draft_token_num * draft_token_num;
  // token_idx: [bid * draft_token_num, (bid + 1) * draft_token_num)
  int token_idx = bid * draft_token_num;
  // tree_mask_idx: [bid * base_offset, (bid + 1) * base_offset)
  int tree_mask_offset = bid * base_offset;

  int depth = 0;
  int parent_idx = -1;

  for (int i = tid - 1, start_idx = tree_mask_offset + tid * draft_token_num; i >= 0; i--) {
    if (tree_mask[start_idx + i]) {
      depth++;
      if (parent_idx == -1) {
        parent_idx = i;
      }
    }
  }
  retrive_index[token_idx + tid] = token_idx + tid;
  positions[token_idx + tid] = depth + verified_seq_len[bid];

  int next_token_idx = -1;
  for (int i = tid + 1; i < draft_token_num; i++) {
    if (tree_mask[tree_mask_offset + i * draft_token_num + tid]) {
      next_token_idx = i;
      break;
    }
  }
  retrive_next_token[token_idx + tid] = next_token_idx;

  int next_sibling_idx = -1;
  if (parent_idx != -1) {
    for (int i = tid + 1; i < draft_token_num; i++) {
      int start_idx = tree_mask_offset + i * draft_token_num + parent_idx;
      if (tree_mask[start_idx]) {
        bool is_sibling = true;
        int end_idx = tree_mask_offset + i * draft_token_num + i;
        for (int j = start_idx + 1; j < end_idx; ++j) {
          if (tree_mask[j]) {
            is_sibling = false;
            break;
          }
        }
        if (is_sibling) {
          next_sibling_idx = i;
          break;
        }
      }
    }
  }
  retrive_next_sibling[token_idx + tid] = next_sibling_idx;
}

void reconstruct_indices_from_tree_mask(
    Tensor tree_mask,
    Tensor verified_seq_len,
    Tensor positions,
    Tensor retrive_index,
    Tensor retrive_next_token,
    Tensor retrive_next_sibling,
    int64_t batch_size,
    int64_t draft_token_num) {
  dim3 grid(batch_size);
  dim3 block(draft_token_num);
  const cudaStream_t stream = GetCurrentCUDAStream();

  reconstructIndicesFromTreeMask<<<grid, block, 0, stream>>>(
      const_cast<bool*>(static_cast<const bool*>(ConstDataPtr(tree_mask))),
      const_cast<int64_t*>(static_cast<const int64_t*>(ConstDataPtr(verified_seq_len))),
      static_cast<int64_t*>(MutableDataPtr(positions)),
      static_cast<int64_t*>(MutableDataPtr(retrive_index)),
      static_cast<int64_t*>(MutableDataPtr(retrive_next_token)),
      static_cast<int64_t*>(MutableDataPtr(retrive_next_sibling)),
      int(batch_size),
      int(draft_token_num));
}
