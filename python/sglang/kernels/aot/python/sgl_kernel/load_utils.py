import ctypes
import glob
import logging
import os
import re
import shutil
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)

_MIN_STABLE_TORCH_VERSION = (2, 11)


def _check_stable_torch_runtime():
    """Require the minimum runtime that exports the stable APIs we build against."""
    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if match is None:
        raise RuntimeError(
            f"sgl-kernel could not determine the PyTorch runtime version from {torch.__version__!r}"
        )

    runtime_version = tuple(map(int, match.groups()))
    if runtime_version < _MIN_STABLE_TORCH_VERSION:
        raise RuntimeError(
            "sgl-kernel's stable CUDA operators require PyTorch >= 2.11 at "
            f"runtime; found {torch.__version__}"
        )


def _get_compute_capability():
    """Get the compute capability of the current GPU."""
    if not torch.cuda.is_available():
        return None

    # Get the current device
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)

    # Return as integer (major * 10 + minor)
    return properties.major * 10 + properties.minor


def _filter_compiled_extensions(file_list):
    """Filter and prioritize compiled extensions over Python source files."""
    compiled_extensions = [".so", ".pyd", ".dll"]  # Common compiled extension suffixes
    compiled_files = []
    other_files = []

    for file_path in file_list:
        path = Path(file_path)
        # Check if it's a compiled extension (including complex names like .abi3.so, .cpython-312.so)
        if any(
            str(path).endswith(ext) or ext in str(path) for ext in compiled_extensions
        ):
            compiled_files.append(file_path)
        else:
            other_files.append(file_path)

    # Return compiled files first, then others
    return compiled_files + other_files


def _load_architecture_specific_ops():
    """Load the appropriate common_ops library based on GPU architecture."""
    compute_capability = _get_compute_capability()
    logger.debug(
        f"[sgl_kernel] GPU Detection: compute_capability = {compute_capability}"
    )

    # Get the directory where sgl_kernel is installed
    sgl_kernel_dir = Path(__file__).parent
    logger.debug(f"[sgl_kernel] sgl_kernel directory: {sgl_kernel_dir}")

    # Determine which version to load based on GPU architecture
    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
    else:
        ops_subdir = "sm100"
        variant_name = "CPU/No GPU detected (using precise math)"

    # Look for the compiled module with any valid extension

    ops_pattern = str(sgl_kernel_dir / ops_subdir / "common_ops.*")
    raw_matching_files = glob.glob(ops_pattern)
    matching_files = _filter_compiled_extensions(raw_matching_files)

    logger.debug(f"[sgl_kernel] Attempting to load {variant_name}")
    logger.debug(f"[sgl_kernel] Looking for library matching pattern: {ops_pattern}")
    logger.debug(f"[sgl_kernel] Found files: {raw_matching_files}")
    logger.debug(f"[sgl_kernel] Prioritized files: {matching_files}")

    previous_import_errors: List[Exception] = []

    # Try to load from the architecture-specific directory
    if matching_files:
        ops_path = Path(matching_files[0])  # Use the first prioritized file
        logger.debug(f"[sgl_kernel] Found architecture-specific library: {ops_path}")
        try:
            logger.debug(f"[sgl_kernel] Loading ops library from {ops_path}...")
            torch.ops.load_library(str(ops_path))
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded {variant_name}")
            logger.debug(f"[sgl_kernel] ✓ Library file: {ops_path}")
            return ops_path

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load from {ops_path}: {type(e).__name__}: {e}"
            )
            # Continue to fallback
    else:
        logger.debug(
            f"[sgl_kernel] ✗ Architecture-specific library not found matching pattern: {ops_pattern}"
        )

    # Try alternative directory (in case installation structure differs)
    alt_pattern = str(sgl_kernel_dir / "common_ops.*")
    raw_alt_files = glob.glob(alt_pattern)
    alt_matching_files = _filter_compiled_extensions(raw_alt_files)
    logger.debug(f"[sgl_kernel] Attempting fallback: looking for pattern {alt_pattern}")
    logger.debug(f"[sgl_kernel] Found fallback files: {raw_alt_files}")
    logger.debug(f"[sgl_kernel] Prioritized fallback files: {alt_matching_files}")

    if alt_matching_files:
        alt_path = Path(alt_matching_files[0])  # Use the first prioritized file
        logger.debug(f"[sgl_kernel] Found fallback library: {alt_path}")
        try:
            logger.debug(f"[sgl_kernel] Loading fallback library from {alt_path}...")
            torch.ops.load_library(str(alt_path))
            logger.debug("[sgl_kernel] ✓ Successfully loaded fallback library")
            logger.debug(f"[sgl_kernel] ✓ Library file: {alt_path}")
            return alt_path

        except Exception as e:
            previous_import_errors.append(e)
            logger.debug(
                f"[sgl_kernel] ✗ Failed to load fallback from {alt_path}: {type(e).__name__}: {e}"
            )
    else:
        logger.debug(
            f"[sgl_kernel] ✗ Fallback library not found matching pattern: {alt_pattern}"
        )

    attempt_error_msg = "\n".join(
        f"- {type(err).__name__}: {err}" for err in previous_import_errors
    )

    # All attempts failed
    cuda_version = torch.version.cuda
    if cuda_version and cuda_version.startswith("12"):
        install_hint = (
            "pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/"
        )
    else:
        install_hint = "pip install --upgrade sglang-kernel"

    error_msg = f"""
[sgl_kernel] CRITICAL: Could not load any common_ops library!

Attempted locations:
1. Architecture-specific pattern: {ops_pattern} - found files: {matching_files}
2. Fallback pattern: {alt_pattern} - found files: {alt_matching_files}

GPU Info:
- Compute capability: {compute_capability}
- Expected variant: {variant_name}
- CUDA version: {cuda_version}

Please ensure sgl_kernel is properly installed with:
{install_hint}

Error details from previous import attempts:
{attempt_error_msg}
"""
    logger.debug(error_msg)
    raise ImportError(error_msg)


def _load_stable_ops(common_ops_path: Path):
    """Load the stable common ops library next to the selected legacy library."""
    ops_pattern = str(common_ops_path.parent / "common_ops_stable.*")
    raw_matching_files = glob.glob(ops_pattern)
    matching_files = _filter_compiled_extensions(raw_matching_files)

    logger.debug(
        f"[sgl_kernel] Looking for stable library matching pattern: {ops_pattern}"
    )
    logger.debug(f"[sgl_kernel] Found stable files: {raw_matching_files}")

    if not matching_files:
        raise ImportError(
            f"[sgl_kernel] Could not find common_ops_stable next to {common_ops_path}"
        )

    ops_path = Path(matching_files[0])
    logger.debug(f"[sgl_kernel] Loading stable ops library from {ops_path}...")
    torch.ops.load_library(str(ops_path))
    logger.debug("[sgl_kernel] ✓ Successfully loaded stable common ops")
    logger.debug(f"[sgl_kernel] ✓ Library file: {ops_path}")


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        else:
            # Guess #3
            cuda_home = "/usr/local/cuda"
    return cuda_home


def _preload_cuda_library():
    """Preload the CUDA runtime library to help avoid 'libcudart.so not found' issues."""
    cuda_home = Path(_find_cuda_home())

    candidate_dirs = [
        cuda_home / "lib",
        cuda_home / "lib64",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ]

    # Determine CUDA major version to try the matching library first.
    # On CUDA 13 systems (e.g., DGX Spark), only libcudart.so.13 exists.
    cuda_major = torch.version.cuda.split(".")[0] if torch.version.cuda else "12"
    lib_versions = list(dict.fromkeys([cuda_major, "13", "12"]))

    for base in candidate_dirs:
        for lib_version in lib_versions:
            candidate = base / f"libcudart.so.{lib_version}"
            if candidate.exists():
                try:
                    cuda_runtime_lib = candidate.resolve()
                    ctypes.CDLL(str(cuda_runtime_lib), mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Preloaded CUDA runtime under {cuda_runtime_lib}")
                    return
                except Exception as e:
                    logger.debug(f"Failed to load {candidate}: {e}")
                    continue

    logger.debug("[sgl_kernel] Could not preload CUDA runtime library")
