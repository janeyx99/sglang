"""Check the installed InfLLM library against its pre-migration outputs.

On the baseline runtime, set SGL_KERNEL_INFLLM_REFERENCE_LIBRARY and optionally
SGL_KERNEL_INFLLM_WRITE_REFERENCE to save inputs and outputs. On newer runtimes,
set SGL_KERNEL_INFLLM_REFERENCE_DATA to that directory. This avoids loading a
version-specific reference binary on a different Torch runtime.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def attention():
    from sgl_kernel.infllm_v2 import attention

    return attention


@pytest.fixture(scope="module")
def reference():
    data = os.environ.get("SGL_KERNEL_INFLLM_REFERENCE_DATA")
    if data:
        return Path(data)
    path = os.environ.get("SGL_KERNEL_INFLLM_REFERENCE_LIBRARY")
    if path is None:
        pytest.skip("set SGL_KERNEL_INFLLM_REFERENCE_LIBRARY for binary equivalence")
    spec = importlib.util.spec_from_file_location("infllm_ops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Reference:
        @staticmethod
        def varlen_fwd_stage1(*args):
            return module.varlen_fwd_stage1(*args, None)

    return Reference()


def inputs(head_dim=64, seqlen_q=64, seqlen_k=17):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    q = torch.randn(seqlen_q, 32, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seqlen_k, 2, head_dim, device=device, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, seqlen_q], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, seqlen_k], device=device, dtype=torch.int32)
    return (q, k, k, cu_q, cu_k, cu_k, seqlen_q, seqlen_k)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lengths", [(256, 16), (64, 17)])
def test_matches_original_binary(
    attention, reference, monkeypatch, head_dim, causal, lengths
):
    name = f"hd{head_dim}-causal{int(causal)}-q{lengths[0]}-k{lengths[1]}.pt"
    if isinstance(reference, Path):
        case = torch.load(reference / name, map_location="cpu", weights_only=True)
        device = torch.device("cuda", torch.cuda.device_count() - 1)
        args = tuple(
            value.to(device) if isinstance(value, torch.Tensor) else value
            for value in case["args"]
        )
        expected = case["expected"].to(device)
    else:
        torch.manual_seed(23)
        args = inputs(head_dim, *lengths)
    stream = torch.cuda.Stream(device=args[0].device)
    stream.wait_stream(torch.cuda.current_stream(args[0].device))
    with torch.cuda.stream(stream):
        actual = attention.infllmv2_attn_stage1(*args, causal=causal)
        if not isinstance(reference, Path):
            with monkeypatch.context() as patch:
                patch.setattr(attention, "load_infllm_ops", lambda: reference)
                expected = attention.infllmv2_attn_stage1(*args, causal=causal)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    output = os.environ.get("SGL_KERNEL_INFLLM_WRITE_REFERENCE")
    if output and not isinstance(reference, Path):
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "args": tuple(
                    value.cpu() if isinstance(value, torch.Tensor) else value
                    for value in args
                ),
                "expected": expected.cpu(),
            },
            directory / name,
        )


def test_empty_keys(attention):
    result = attention.infllmv2_attn_stage1(*inputs(seqlen_k=0))
    assert result.numel() == 0


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"dropout_p": 0.1}, "does not support dropout"),
        ({"softcap": 1.0}, "does not support softcap"),
        ({"window_size": (1, 1)}, "does not support local attention"),
    ],
)
def test_disabled_variants_fail_before_launch(attention, kwargs, message):
    with pytest.raises(RuntimeError, match=message):
        attention.infllmv2_attn_stage1(*inputs(), **kwargs)
