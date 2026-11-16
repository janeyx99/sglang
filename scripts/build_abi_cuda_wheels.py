"""Build CUDA wheels from the pinned dependency PRs and this SGLang checkout.

Requires a native x86_64/aarch64 Linux host with Docker (including buildx), git,
Python 3.11+, packaging, and torch-abi-audit. Produces local artifacts only.
"""

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

from audit_cuda_wheels import audit_wheel
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("abi_cuda_dependencies.json")


def dependencies():
    pins = json.loads(MANIFEST.read_text())
    project = tomllib.loads((ROOT / "python/pyproject.toml").read_text())["project"]
    requirements = {
        requirement.name: requirement
        for requirement in map(Requirement, project["dependencies"])
    }
    for name, pin in pins.items():
        if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]):
            raise ValueError(f"{name} must pin a full commit SHA")
        if pin["version"] not in requirements[name].specifier:
            raise ValueError(
                f"{name} build version does not satisfy SGLang's dependency"
            )
    return pins


def run(*args, cwd=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)


def checkout(name, pin, work):
    source = work / name
    run("git", "init", source)
    run(
        "git",
        "remote",
        "add",
        "origin",
        f"https://github.com/{pin['repository']}.git",
        cwd=source,
    )
    run("git", "fetch", "--depth=1", "origin", pin["commit"], cwd=source)
    run("git", "checkout", "--detach", "FETCH_HEAD", cwd=source)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if actual != pin["commit"]:
        raise RuntimeError(f"{name}: checked out {actual}, expected {pin['commit']}")
    if name == "sgl-flash-mla":
        run("git", "submodule", "update", "--init", "--recursive", cwd=source)
    return source


def main():
    pins = dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", action="store_true", help="Print the build inputs without building"
    )
    parser.add_argument(
        "--arch", choices=["x86_64", "aarch64"], default=platform.machine()
    )
    parser.add_argument(
        "--work-dir", type=Path, help="An empty directory for dependency checkouts"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/abi-cu130")
    args = parser.parse_args()
    selected = {
        name: pin for name, pin in pins.items() if args.arch in pin["architectures"]
    }
    if args.plan:
        print(
            json.dumps(
                {
                    "architecture": args.arch,
                    "dependencies": selected,
                    "kernel_source": str(ROOT),
                },
                indent=2,
            )
        )
        return
    if platform.machine() != args.arch:
        parser.error(
            "Build on the selected architecture; cross-compilation is not supported"
        )
    for command in ("docker", "git", "torch-abi-audit"):
        if shutil.which(command) is None:
            parser.error(f"Required executable is missing: {command}")
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="sglang-abi-build-"))
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        parser.error(f"Use an empty work directory: {work}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Use an empty output directory: {output}")
    artifacts = {}
    for name, pin in selected.items():
        source = checkout(name, pin, work)
        if name == "sgl-fa3":
            package = source / "sgl_fa3"
            (package / "python/sgl_fa3/_version.py").write_text(
                f'__version__ = "{pin["version"]}"\n'
            )
            run("bash", package / "scripts/build.sh", "3.10", "13.0", ROOT)
        else:
            package = source
            (package / "flash_mla/VERSION").write_text(pin["version"])
            run(
                "bash",
                ROOT / "scripts/build_sgl_flash_mla.sh",
                "3.12",
                "13.4",
                source,
                args.arch,
            )
        wheels = list((package / "dist").glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one {name} wheel, found {wheels}")
        destination = output / wheels[0].name
        shutil.copy2(wheels[0], destination)
        artifacts[destination.name] = {"source": pin, "audit": audit_wheel(destination)}

    # Build the kernel in a fresh copy so stale build/dist files in a developer
    # checkout cannot enter the artifact. Build products are omitted.
    kernel = work / "sglang-kernel"
    source = ROOT / "python/sglang/kernels/aot"
    shutil.copytree(
        source,
        kernel,
        ignore=shutil.ignore_patterns(
            "build", "dist", "dist-pypi", "__pycache__", "*.so", "*.egg-info"
        ),
    )
    run("bash", "./build.sh", "3.11", "13.0", args.arch, cwd=kernel)
    wheels = list((kernel / "dist").glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one sglang-kernel wheel, found {wheels}")
    destination = output / wheels[0].name
    shutil.copy2(wheels[0], destination)
    artifacts[destination.name] = {
        "source": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "modified_files": subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(source)], cwd=ROOT, text=True
            ).splitlines(),
        },
        "audit": audit_wheel(destination),
    }
    for name, artifact in artifacts.items():
        with (output / name).open("rb") as wheel:
            artifact["sha256"] = hashlib.file_digest(wheel, "sha256").hexdigest()
    (output / "build-manifest.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    print(f"Audited CUDA wheels and build-manifest.json: {output}")


if __name__ == "__main__":
    main()
