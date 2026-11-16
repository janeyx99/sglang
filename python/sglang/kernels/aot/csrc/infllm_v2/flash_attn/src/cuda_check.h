#pragma once

#include <cuda_runtime.h>
#include <torch/headeronly/util/Exception.h>

#define INFLLM_CUDA_CHECK(call)                                                         \
  do {                                                                                  \
    const cudaError_t status = (call);                                                  \
    STD_TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status)); \
  } while (0)
