// Adapted from: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "allreduce/allreduce_ops.h"

#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/macros.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include "sgl_kernel_cuda_stream.h"

using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;

#define SGL_CHECK_EQ(val1, val2) \
  STD_TORCH_CHECK((val1) == (val2), "Check failed: " #val1 " == " #val2 " (", (val1), " vs. ", (val2), "). ")
#define SGL_CHECK_LE(val1, val2) \
  STD_TORCH_CHECK((val1) <= (val2), "Check failed: " #val1 " <= " #val2 " (", (val1), " vs. ", (val2), "). ")
#define SGL_CHECK_NO_MSG(condition)                                                                   \
  STD_TORCH_CHECK(                                                                                    \
      condition,                                                                                      \
      "Expected " #condition                                                                          \
      " to be true, but got false.  (Could this error message be improved?  If so, please report an " \
      "enhancement request to PyTorch.)")
#else
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

using Tensor = torch::Tensor;
using ScalarType = at::ScalarType;

#define SGL_CHECK_EQ TORCH_CHECK_EQ
#define SGL_CHECK_LE TORCH_CHECK_LE
#define SGL_CHECK_NO_MSG(condition) TORCH_CHECK(condition)
#endif

#include "custom_all_reduce.cuh"

static_assert(sizeof(void*) == sizeof(fptr_t));

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, Tensor& rank_data, int64_t rank, bool full_nvlink) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");

  sglang::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<sglang::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new sglang::CustomAllreduce(
      ipc_ptrs, SGL_MUTABLE_DATA_PTR(rank_data), rank_data.numel(), rank, world_size, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(Tensor& t) {
#ifdef TORCH_TARGET_VERSION
  if (t.is_contiguous()) {
    return true;
  }
  int64_t storage_nbytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(t.get(), &storage_nbytes));
  return storage_nbytes - t.storage_offset() * t.element_size() == static_cast<int64_t>(t.numel() * t.element_size());
#else
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
#endif
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, Tensor& inp, Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
#ifdef TORCH_TARGET_VERSION
  STD_TORCH_CHECK(inp.is_cuda(), "CUDAGuardImpl initialized with non-CUDA DeviceType: cpu");
  const auto device_index = inp.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream(device_index);
#else
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
#endif

  SGL_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  SGL_CHECK_EQ(inp.numel(), out.numel());
  SGL_CHECK_NO_MSG(_is_weak_contiguous(out));
  SGL_CHECK_NO_MSG(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    SGL_CHECK_LE(input_size, reg_buffer_sz_bytes);
#ifdef TORCH_TARGET_VERSION
    STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
#else
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
#endif
  } else {
    reg_buffer = const_cast<void*>(SGL_CONST_DATA_PTR(inp));
  }
  switch (out.scalar_type()) {
    case ScalarType::Float: {
      fa->allreduce<float>(
          stream,
          reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(SGL_MUTABLE_DATA_PTR(out)),
          out.numel());
      break;
    }
    case ScalarType::Half: {
      fa->allreduce<half>(
          stream, reinterpret_cast<half*>(reg_buffer), reinterpret_cast<half*>(SGL_MUTABLE_DATA_PTR(out)), out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream,
          reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(SGL_MUTABLE_DATA_PTR(out)),
          out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<sglang::CustomAllreduce*>(_fa);
}

int64_t meta_size() {
  return sizeof(sglang::Signal);
}

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  SGL_CHECK_NO_MSG(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<sglang::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

#undef SGL_CHECK_NO_MSG
#undef SGL_CHECK_LE
#undef SGL_CHECK_EQ
