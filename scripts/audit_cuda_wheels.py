"""Fail unless every native library in each CUDA wheel passes its ABI audit."""

import argparse
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from packaging.utils import parse_wheel_filename


def check_report(report):
    libraries = report.get("extensions", []) + report.get("bundled_libs", [])
    if report.get("error") or not libraries:
        raise RuntimeError(f"ABI audit did not inspect a library: {report}")
    for library in libraries:
        torch = library.get("torch") or {}
        cpython = library.get("cpython") or {}
        if (
            library.get("error")
            or not torch
            or (torch.get("uses_torch") and torch.get("stable") is not True)
            or cpython.get("violations")
        ):
            raise RuntimeError(f"Native library failed its ABI audit: {library}")


def audit_wheel(wheel):
    wheel = Path(wheel).resolve()
    _, _, _, tags = parse_wheel_filename(wheel.name)
    if not all(tag.abi == "abi3" for tag in tags):
        raise RuntimeError(f"CUDA wheel must use CPython abi3: {wheel.name}")
    reports = {}
    with zipfile.ZipFile(wheel) as archive, tempfile.TemporaryDirectory() as directory:
        members = [
            name for name in archive.namelist() if re.search(r"\.so(?:\.\d+)*$", name)
        ]
        if not members:
            raise RuntimeError(f"No native libraries found in {wheel}")
        for index, member in enumerate(members):
            # Extract only the library bytes to a generated path. Do not trust
            # archive paths or import the package while auditing its binary.
            library = Path(directory) / f"{index}-{Path(member).name}"
            library.write_bytes(archive.read(member))
            result = subprocess.run(
                ["torch-abi-audit", "--json", str(library)],
                check=True,
                text=True,
                capture_output=True,
            )
            report = json.loads(result.stdout)
            # torch-abi-audit can exit zero even when its report contains an
            # error or unstable symbols. Inspect the report, not just the exit.
            check_report(report)
            for record in report["extensions"] + report["bundled_libs"]:
                record["path"] = member
            report["root"] = wheel.name
            reports[member] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    for wheel in args.wheels:
        print(json.dumps({wheel.name: audit_wheel(wheel)}, indent=2))


if __name__ == "__main__":
    main()
