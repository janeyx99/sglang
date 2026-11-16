#include <ATen/ATen.h>
#include <torch/library.h>

void apply_gguf_moe_weights(at::Tensor& outputs, const at::Tensor& weights, const at::Tensor& ids);

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  m.def("apply_gguf_moe_weights(Tensor! outputs, Tensor weights, Tensor ids) -> ()");
}

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
  m.impl("apply_gguf_moe_weights", &apply_gguf_moe_weights);
}
