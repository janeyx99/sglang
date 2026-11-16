# Stable ABI migration refresh

This refresh ports `janeyx99/sglang:agent-migration` (`d06121a2881c`) onto
SGLang `f6de13a0c266` (2026-11-16). The previous branches remain intact.

## Current scope

All 72 CUDA common operators and the separate InfLLM attention library use
`TORCH_TARGET_VERSION=0x020B000000000000` and `USE_CUDA`. The wheel contains
`sm90/common_ops.abi3.so`, `sm100/common_ops.abi3.so`, and `infllm_ops.abi3.so`.
There is no legacy companion library or `torch_python` linkage. Architecture
selection still chooses fast math for SM90 and precise math otherwise. The
CUDA package builds with Torch 2.11 and requires `torch>=2.11`.

The refresh preserves pinned-host-memory device pointers, FP32 rotary caches,
H200 W4A8 dispatch, and relocated GGUF helpers. InfLLM keeps its Python
`infllmv2_attn_stage1` interface and CUDA algorithm; the internal dispatcher
omits the unused generator argument. Its inference-only feature flags now
apply to the C++ host wrapper as well as the CUDA sources.

GGUF's kernel is unchanged. Its overlap validation uses dense byte intervals,
including transposed dense inputs, because Torch 2.11 has no stable overlap
shim. As with ATen, non-dense input layouts are not rejected by that check.
Overlapping addresses wrapped by separate Storage objects are now rejected too;
zero-element outputs are no-ops. Tests cover mixed dtypes, strided/broadcast
inputs, negative expert IDs, overlapping views, and disjoint views.

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
are updated together with it. Builds consume the dependency PR commits directly.

## Pinned dependency builds

`scripts/abi_cuda_dependencies.json` is the source of truth for:

- [FA3 PR #59](https://github.com/sgl-project/sgl-flash-attn/pull/59):
  `janeyx99/flash-attention@248ad15bee531ebdad643da607d2b4fd515635ce`.
- [FlashMLA PR #37](https://github.com/sgl-project/FlashMLA/pull/37):
  `janeyx99/FlashMLA-ABI-Stable@a1dbfda44aa803ae138dd458756bafd38e63f294`.
  Its current and frozen kernels are already ABI-stable upstream; this PR
  makes the CUDA 13 package installable on Torch 2.11+.

Both dependency release workflows read this manifest for their default source
checkout. The combined build checks out and verifies these exact commits,
builds dependency wheels at the versions required by SGLang, then builds the
kernel from this checkout. FA3 is included on x86_64 only, matching SGLang's
existing package dependency marker.

On a native Linux host with Docker/buildx and Python 3.11+:

```bash
python -m pip install packaging torch-abi-audit
python scripts/build_abi_cuda_wheels.py --plan --arch x86_64
python scripts/build_abi_cuda_wheels.py --arch x86_64 --output-dir /tmp/abi-cu130
python -m pip install /tmp/abi-cu130/*.whl
```

Use `--arch aarch64` on an ARM build host. The output includes source commits,
wheel hashes, and ABI audit reports. These are separate dependency and kernel
wheels, as in current upstream packaging; installing the built set satisfies
SGLang's existing package/version requirements.

The `Build ABI CUDA wheels` workflow runs this command and uploads artifacts
without publishing them. Its runner input must name a build runner available
to the fork. The CUDA kernel release workflow also audits the whole wheel
before publication. Audit reports are checked explicitly: the audit tool's
exit code alone does not guarantee success.

## Validation

Validated locally on 2026-11-16 using an aarch64 GB300 host, CUDA 13.0, and
PyTorch 2.11.0+cu130 for compilation:

- Both common math variants, compiled for SM100f: 81 stable shim references
  each, zero unstable Torch references.
- InfLLM attention: 62 stable shim references, zero unstable Torch references.
  All three libraries were audited inside the built
  `sglang_kernel-0.4.11-cp311-abi3-linux_aarch64.whl`.
- The original common library passes all 97 focused regressions, including
  the new GGUF coverage. The migrated wheel passes 109 tests on each of
  Torch 2.11 and 2.13, including bitwise InfLLM comparisons against the
  original outputs. The fast-math common variant also passes all 97 tests.
- All 72 common dispatcher schemas, including alias annotations, exactly
  match the original library on both Torch runtimes.
- A representative selection of existing kernel tests reports 456 passed,
  25 skipped, and 4,303 deselected using the installed wheel. It samples up
  to 24 parameter combinations per function, rather than the full suite.
- Standalone FA3, compiled for SM90a: 65 stable shim references, zero
  unstable Torch references. Both dispatcher-contract tests pass on both
  Torch runtimes using the same binary and the new Python loader.
- Both libraries in the published FlashMLA 1.0.5 aarch64 wheel audit clean:
  64 stable shims for current kernels and 68 for frozen kernels. The same
  wheel passes all 144 sparse-prefill statistics/CUDA-graph cases on each
  of Torch 2.11 and 2.13.

The FA3 numerical/pruned-variant suite has 27 skips on this GB300 host.
SM8x/SM90 numerical validation, x86_64 release wheels, the complete Docker
wheel-set build, and the updated workflows were not exercised locally. The
local kernel wheel targets SM100f; it is a native development wheel, not a
manylinux release artifact. The rest of SGLang retains its upstream Torch
2.14.1 serving-runtime requirement.

To build the native development wheel in a prepared Torch 2.11/CUDA 13.0
environment:

```bash
CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
CMAKE_BUILD_PARALLEL_LEVEL=8 \
python -m pip wheel python/sglang/kernels/aot --no-build-isolation --no-deps \
  -Ccmake.define.SGL_KERNEL_CUDA_ARCHITECTURES=100f \
  -Ccmake.define.SGL_KERNEL_COMPILE_THREADS=1 -w dist
python scripts/audit_cuda_wheels.py dist/sglang_kernel-*.whl
```

`test_infllm_stable_abi.py` can save reference inputs and outputs with
`SGL_KERNEL_INFLLM_REFERENCE_LIBRARY` and
`SGL_KERNEL_INFLLM_WRITE_REFERENCE`. On another Torch runtime, use
`SGL_KERNEL_INFLLM_REFERENCE_DATA` to compare with those saved outputs.
The old version-specific binary cannot be loaded safely on the new runtime.

`test_stable_abi.py` can also load one common library through
`SGL_KERNEL_STABLE_LIBRARY`. Run the same tests on another Torch runtime
without rebuilding.
