"""Native regression checks usable with one binary across Torch runtimes.

Set SGL_KERNEL_STABLE_LIBRARY to load only the migrated library, independently
of the remaining legacy extensions. Use pytest --noconftest for this mode.
"""

import os

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    library = os.environ.get("SGL_KERNEL_STABLE_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        import sgl_kernel  # noqa: F401


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox", [False, True])
def test_rotary_fp32_cache_on_nondefault_stream(dtype, is_neox):
    torch.manual_seed(1)
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        q = torch.randn(7, 4, 64, device=device, dtype=dtype)
        k = torch.randn(7, 2, 64, device=device, dtype=dtype)
        positions = torch.arange(7, device=device)
        cache = torch.randn(7, 64, device=device, dtype=torch.float32)
        # The public FP32-cache path rounds coefficients to the query dtype.
        cos, sin = cache.to(dtype).chunk(2, dim=-1)

        def reference(x):
            if is_neox:
                a, b = x.chunk(2, dim=-1)
            else:
                a, b = x[..., ::2], x[..., 1::2]
            first = a * cos[:, None] - b * sin[:, None]
            second = a * sin[:, None] + b * cos[:, None]
            if is_neox:
                return torch.cat((first, second), dim=-1).to(dtype)
            return torch.stack((first, second), dim=-1).flatten(-2).to(dtype)

        q_ref, k_ref = reference(q), reference(k)
        torch.ops.sgl_kernel.rotary_embedding(positions, q, k, 64, cache, is_neox)
    stream.synchronize()
    torch.testing.assert_close(q, q_ref)
    torch.testing.assert_close(k, k_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_accessible_pointer_checks_target_device():
    target = torch.cuda.device_count() - 1
    tensor = torch.empty(16, device=f"cuda:{target}")
    assert (
        torch.ops.sgl_kernel.get_device_accessible_ptr(tensor, target)
        == tensor.data_ptr()
    )
    with pytest.raises(RuntimeError, match="non-negative"):
        torch.ops.sgl_kernel.get_device_accessible_ptr(tensor, -1)
    if target != 0:
        with pytest.raises(RuntimeError, match="target device"):
            torch.ops.sgl_kernel.get_device_accessible_ptr(tensor, 0)
    host = torch.empty(16, pin_memory=True)
    assert torch.ops.sgl_kernel.get_device_accessible_ptr(host, target) != 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gguf_q4_dequantization(dtype):
    torch.manual_seed(2)
    rows, columns = 4, 64
    blocks = rows * columns // 32
    scales = torch.rand(blocks, dtype=torch.float16)
    quants = torch.randint(0, 256, (blocks, 16), dtype=torch.uint8)
    packed = torch.cat((scales.view(torch.uint8).reshape(blocks, 2), quants), dim=1)
    expected = torch.cat((quants & 15, quants >> 4), dim=1).float() - 8
    # GGUF Q4_0 dequantization computes in FP16 before converting the output.
    expected = (
        (expected * scales[:, None].float()).half().reshape(rows, columns).to(dtype)
    )
    actual = torch.ops.sgl_kernel.ggml_dequantize(
        packed.cuda(), 2, rows, columns, dtype
    )
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [17, 129, 257])
@pytest.mark.parametrize("hidden", [64, 68])
@pytest.mark.parametrize("topk", [3, 8])
def test_moe_sum_reduce_dispatch_paths(tokens, hidden, topk):
    torch.manual_seed(3)
    x = torch.randn(tokens, topk, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(tokens, hidden, device="cuda", dtype=x.dtype)
    expected = (x.float().sum(dim=1) * 1.5).to(x.dtype)
    torch.ops.sgl_kernel.moe_sum_reduce(x, output, 1.5)
    torch.testing.assert_close(output, expected)
