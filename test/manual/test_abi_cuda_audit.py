"""Reject audit reports that can accompany a successful CLI exit."""

import importlib.util
from pathlib import Path

import pytest

_script = Path(__file__).resolve().parents[2] / "scripts/audit_cuda_wheels.py"
_spec = importlib.util.spec_from_file_location("audit_cuda_wheels", _script)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


def report(*libraries, error=None):
    return {"error": error, "extensions": [], "bundled_libs": list(libraries)}


def library(stable=True, error=None, violations=None):
    return {
        "error": error,
        "torch": {"uses_torch": True, "stable": stable},
        "cpython": {"violations": violations or []},
    }


def test_all_stable_libraries_pass():
    _audit.check_report(report(library(), library()))


@pytest.mark.parametrize(
    "audit_report",
    [
        report(error="not a compiled library"),
        report(),
        report(library(), library(stable=False)),
        report(library(error="could not inspect symbols")),
        report(library(violations=["_PyObject_PrivateSymbol"])),
    ],
)
def test_report_errors_and_unstable_companions_fail(audit_report):
    with pytest.raises(RuntimeError):
        _audit.check_report(audit_report)
