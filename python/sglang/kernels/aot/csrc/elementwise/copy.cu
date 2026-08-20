#include <cuda_runtime.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <vector>

#include "elementwise/elementwise_ops.h"
#include "sgl_kernel_cuda_stream.h"

template <int N>
struct InputArray {
  int values[N];
};

template <int N>
__global__ void copy_to_gpu_no_ce_kernel(const InputArray<N> input_array, int* output) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    output[idx] = input_array.values[idx];
  }
}

template <int N>
void copy_to_gpu_no_ce_impl(const torch::stable::Tensor& input, torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(input.dim() == 1, "input must be 1-D");
  STD_TORCH_CHECK(static_cast<int>(input.numel()) == N, "input numel must equal template N");
  STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Int, "input dtype must be int32");

  STD_TORCH_CHECK(output.dim() == 1, "output dim");
  STD_TORCH_CHECK(static_cast<int>(output.numel()) == N, "output size");
  STD_TORCH_CHECK(output.is_contiguous(), "output contiguous");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Int, "output dtype");

  STD_TORCH_CHECK(input.is_cpu(), "input must be a CPU tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");

  InputArray<N> input_array;
  const int* input_ptr = input.const_data_ptr<int>();
  for (int i = 0; i < N; ++i)
    input_array.values[i] = input_ptr[i];

  // may use multi thread blocks if performance bottleneck
  dim3 grid(1);
  dim3 block(static_cast<int>(input.numel()));
  cudaStream_t stream = sgl_kernel::stable::get_current_cuda_stream();
  copy_to_gpu_no_ce_kernel<<<grid, block, 0, stream>>>(input_array, output.mutable_data_ptr<int>());
  STD_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_to_gpu_no_ce(const torch::stable::Tensor& input, torch::stable::Tensor& output) {
  int N = static_cast<int>(input.numel());
  // Can use macro if there are more N needed
  if (N == 72) {
    copy_to_gpu_no_ce_impl<72>(input, output);
  } else if (N == 64) {
    copy_to_gpu_no_ce_impl<64>(input, output);
  } else if (N == 32) {
    copy_to_gpu_no_ce_impl<32>(input, output);
  } else {
    STD_TORCH_CHECK(false, "unexpected N");
  }
}
