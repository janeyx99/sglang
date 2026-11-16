# Stable ABI migration refresh

This refresh ports `janeyx99/sglang:agent-migration` (`d06121a2881c`) onto
SGLang `f6de13a0c266` (2026-11-16). The previous branches remain intact.

## Current scope

The CUDA common library exports the same 72 operator schemas as this upstream
revision. Of these, 71 live in `common_ops_stable.abi3.so`, built with
`TORCH_TARGET_VERSION=0x020B000000000000` and `USE_CUDA`. The existing loader
selects the SM90 or SM100 math variant and loads both companion libraries.

`apply_gguf_moe_weights`, added since the original migration, stays in
`common_ops.abi3.so`: its ATen overlap checks have no direct stable 2.11
equivalent. Its validation and kernel remain unchanged. The new
`get_device_accessible_ptr` utility is migrated, including pinned host memory
handling and target-device validation. The refresh also preserves upstream's
FP32 rotary-cache support, H200 W4A8 dispatch, and relocated GGUF helpers.

InfLLM-V2 max pooling is migrated. Its separate `infllm_ops` attention extension
still uses pybind, ATen, and `torch_python`. Consequently, the overall
`sglang-kernel` wheel retains its upstream Torch dependency pin; the stable
common library can be tested independently across Torch versions.

## Upstream changes affecting the old branches

- Sparse FA2, cutlass MLA, vertical/slash indexing, AWQ, non-Marlin GPTQ, and
  spatial operators were removed upstream. Their old migrations are not
  resurrected here.
- CUDA speculative operators now use JIT. Their current sources for other
  backends are preserved.
- FlashMLA is an external package.
- FA3 is now the standalone `sgl_fa3/` package on the
  `sgl-project/sgl-flash-attn:sgl-kernel` branch.

The companion FA3 refresh starts from `4d03021a0ffc`, selects
`hopper/flash_api_stable.cpp`, preserves the November variant-pruning behavior,
and carries Dao-AILab fixes `fc6bed978518` (KV batch-index length) and
`aae6770da7df` (empty paged KV caches). Its build uses Torch 2.11; its runtime
dependency is `torch>=2.11`. The SGLang FA3 Docker image and release validation
are updated together with it. The source-package change must land before
using that updated release configuration.

## Remaining InfLLM attention work

1. Replace the `varlen_fwd_stage1` pybind binding with a stable dispatcher schema
   while preserving `infllmv2_attn_stage1`'s Python interface.
2. Convert the approximately 485-line host wrapper's tensor operations,
   allocation, guards, streams, and supporting headers.
3. Preserve the disabled-dropout contract when handling the legacy generator
   argument; the current Python wrapper passes `None`.
4. Remove `torch_python` linkage and load the library through
   `torch.ops.load_library`.
5. Compare numerical behavior and validation with the existing implementation,
   audit the binary, and run it on multiple Torch versions without rebuilding.

The CUDA attention algorithm does not need to change.

## Validation

Validated locally on 2026-11-16 using an aarch64 GB300 host, CUDA 13.0, and
PyTorch 2.11.0+cu130 for compilation:

- Common SM100 precise-math library, compiled for SM100f: 81 stable shim
  references, zero unstable Torch references.
- All 71 migrated runtime schemas exactly match the unchanged upstream
  baseline, including argument names and alias annotations.
- The same common binary passes all 20 focused regressions under both
  PyTorch 2.11.0+cu130 and 2.13.0+cu132.
- A representative selection of the existing activation, normalization,
  copy, MoE top-k, InfLLM pooling, and GEMM tests reports 456 passed and
  25 skipped. The selection samples up to 24 parameter combinations per
  test function; it is not the full kernel suite.
- Standalone FA3, compiled for SM90a: 65 stable shim references, zero
  unstable Torch references. Both dispatcher-contract tests pass on both
  Torch runtimes using the same binary and the new Python loader.

The FA3 numerical/pruned-variant suite has 27 skips on this GB300 host.
SM8x/SM90 numerical validation, the common SM90 fast-math build, x86_64 release
wheels, and the updated release workflow were not exercised locally.

`tests/test_stable_abi.py` exercises the new rotary-cache and device-pointer
paths, MoE reduction dispatch, and relocated GGUF dequantization code. To test a single migrated
binary independently of the legacy wheel:

```bash
SGL_KERNEL_STABLE_LIBRARY=/absolute/path/common_ops_stable.abi3.so \
  python -m pytest --noconftest tests/test_stable_abi.py -q
torch-abi-audit /absolute/path/common_ops_stable.abi3.so
```

Run the same commands in another Torch environment without rebuilding.
FA3's `tests/test_stable_abi.py` checks the dispatcher contract; its attention
and pruning regressions require supported SM8x/SM90 hardware.
