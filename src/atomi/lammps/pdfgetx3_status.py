"""Status checks for external PDFGetX3 installations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _find_spec(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except ModuleNotFoundError:
        return False


def _package_status(import_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    available = _find_spec(import_name)
    version = None
    if available:
        try:
            version = metadata.version(distribution_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


def _venv_bin(env_dir: str | Path) -> Path:
    root = Path(env_dir).expanduser()
    return root / ("Scripts" if os.name == "nt" else "bin")


def configured_pdfgetx3_executable() -> str | None:
    direct = os.environ.get("ATOMI_PDFGETX3_EXE", "").strip()
    if direct:
        return direct
    env_dir = os.environ.get("ATOMI_PDFGETX3_ENV", "").strip()
    if env_dir:
        candidate = _venv_bin(env_dir) / "pdfgetx3"
        if candidate.exists():
            return str(candidate)
    return None


def configured_pdfgetx3_python() -> str | None:
    direct = os.environ.get("ATOMI_PDFGETX3_PYTHON", "").strip()
    if direct:
        return direct
    env_dir = os.environ.get("ATOMI_PDFGETX3_ENV", "").strip()
    if env_dir:
        for candidate in (_venv_bin(env_dir) / "python", _venv_bin(env_dir) / "python.exe"):
            if candidate.exists():
                return str(candidate)
    return None


def _resolve_executable(command: str | None) -> dict[str, Any]:
    requested = command or "pdfgetx3"
    path = Path(requested).expanduser()
    explicit = path.is_absolute() or path.parent != Path(".")
    info: dict[str, Any] = {
        "requested": requested,
        "available": False,
        "source": "explicit-path" if explicit else "PATH",
        "resolved": None,
    }
    if explicit:
        resolved = path.resolve() if path.exists() else path
        info["resolved"] = str(resolved)
        info["available"] = resolved.exists()
        if not resolved.exists():
            info["error"] = "explicit executable path does not exist"
        return info
    found = shutil.which(requested)
    info["resolved"] = found
    info["available"] = found is not None
    return info


def _run_version(executable: str | None, timeout: float = 10.0) -> dict[str, Any]:
    if not executable:
        return {"attempted": False}
    command = [executable, "--version"]
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except OSError as exc:
        return {"attempted": True, "command": command, "returncode": None, "output": str(exc)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {"attempted": True, "command": command, "returncode": None, "timed_out": True, "output": output.strip()}
    return {"attempted": True, "command": command, "returncode": proc.returncode, "output": proc.stdout.strip()}


PROBE_SCRIPT = """
import importlib.util, json, shutil, sys
from importlib import metadata

def pkg(import_name, dist_name=None):
    try:
        ok = importlib.util.find_spec(import_name) is not None
    except ModuleNotFoundError:
        ok = False
    version = None
    if ok:
        try:
            version = metadata.version(dist_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": ok, "version": version}

print(json.dumps({
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "diffpy_pdfgetx": pkg("diffpy.pdfgetx", "diffpy.pdfgetx"),
    "pdfgetx3_on_path": shutil.which("pdfgetx3"),
}))
"""


def probe_pdfgetx3_python(python_executable: str | Path | None, timeout: float = 20.0) -> dict[str, Any]:
    if not python_executable:
        return {"configured": False, "available": False}
    python = str(Path(python_executable).expanduser())
    try:
        proc = subprocess.run(
            [python, "-c", PROBE_SCRIPT],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except OSError as exc:
        return {"configured": True, "requested_python": python, "available": False, "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"configured": True, "requested_python": python, "available": False, "error": f"probe timed out: {exc}"}
    if proc.returncode != 0:
        return {
            "configured": True,
            "requested_python": python,
            "available": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"configured": True, "requested_python": python, "available": False, "stdout": proc.stdout.strip()}
    data["configured"] = True
    data["requested_python"] = python
    data["available"] = bool(data.get("diffpy_pdfgetx", {}).get("available") or data.get("pdfgetx3_on_path"))
    return data


def inspect_pdfgetx3_environment(
    executable: str | None = None,
    python_executable: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    configured_exe = executable or configured_pdfgetx3_executable()
    resolution = _resolve_executable(configured_exe)
    external = probe_pdfgetx3_python(python_executable or configured_pdfgetx3_python(), timeout=timeout)
    active = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "diffpy_pdfgetx": _package_status("diffpy.pdfgetx", "diffpy.pdfgetx"),
    }
    version_probe = _run_version(resolution.get("resolved") if resolution.get("available") else None, timeout=min(timeout, 10.0))

    suggestions: list[str] = []
    if resolution.get("available"):
        mode = "executable"
        suggestions.append("PDFGetX3 executable is available from PATH or local Atomi config.")
    elif external.get("available"):
        mode = "external-python"
        suggestions.append("PDFGetX3 Python package/environment is visible through ATOMI_PDFGETX3_PYTHON/ATOMI_PDFGETX3_ENV.")
    else:
        mode = "missing"
        suggestions.append("Keep PDFGetX3 in a separate environment and export ATOMI_PDFGETX3_ENV or ATOMI_PDFGETX3_EXE.")
        suggestions.append("For raw .chi workflows, pass --pdfgetx3 /path/to/pdfgetx3 or configure profiles.pdfgetx3.")

    return {
        "schema_version": 1,
        "module": "pdfgetx3",
        "active_environment": active,
        "external_environment": external,
        "executable": resolution,
        "version_probe": version_probe,
        "pdfgetx3_mode": mode,
        "ready_for_pdfgetx3": mode in {"executable", "external-python"},
        "atomi_environment": {
            "ATOMI_PDFGETX3_EXE": os.environ.get("ATOMI_PDFGETX3_EXE", ""),
            "ATOMI_PDFGETX3_ENV": os.environ.get("ATOMI_PDFGETX3_ENV", ""),
            "ATOMI_PDFGETX3_PYTHON": os.environ.get("ATOMI_PDFGETX3_PYTHON", ""),
            "ATOMI_PDFGETX3_WHEELHOUSE": os.environ.get("ATOMI_PDFGETX3_WHEELHOUSE", ""),
        },
        "suggestions": suggestions,
        "notes": [
            "PDFGetX3 may live in a separate environment from the main Atomi MD/DFT environment.",
            "pdf_md_compare can use the configured executable for raw experimental .chi reduction.",
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Atomi PDFGetX3 status")
    active = report["active_environment"]
    active_pkg = active["diffpy_pdfgetx"]["version"] if active["diffpy_pdfgetx"]["available"] else "missing"
    print(f"  active python: {active['python']} ({active['python_version']})")
    print(f"  active diffpy.pdfgetx: {active_pkg}")
    exe = report["executable"]
    print(f"  executable requested: {exe['requested']}")
    print(f"  executable resolved: {exe.get('resolved') or 'missing'}")
    print(f"  executable available: {exe['available']}")
    external = report["external_environment"]
    if external.get("configured"):
        external_pkg = external.get("diffpy_pdfgetx", {}).get("version") if external.get("diffpy_pdfgetx", {}).get("available") else "missing"
        print(f"  external PDFGetX3 python: {external.get('requested_python')}")
        print(f"  external diffpy.pdfgetx: {external_pkg}")
        if external.get("pdfgetx3_on_path"):
            print(f"  external pdfgetx3 on PATH: {external['pdfgetx3_on_path']}")
        if external.get("error") or external.get("stderr"):
            print(f"    reason: {external.get('error') or external.get('stderr')}")
    else:
        print("  external PDFGetX3 python: not configured")
    if report["version_probe"].get("attempted"):
        print(f"  version probe: {report['version_probe'].get('output') or 'no output'}")
    print(f"  selected PDFGetX3 mode: {report['pdfgetx3_mode']}")
    env = {key: value for key, value in report["atomi_environment"].items() if value}
    if env:
        print("  Atomi PDFGetX3 config variables:")
        for key, value in env.items():
            print(f"    {key}={value}")
    print("  suggestions:")
    for item in report["suggestions"]:
        print(f"    - {item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pdfgetx3_status",
        description="Inspect PDFGetX3 executable and configured external environment availability.",
    )
    parser.add_argument("--pdfgetx3", help="PDFGetX3 executable name/path to probe.")
    parser.add_argument("--python", dest="python_executable", help="External Python executable containing diffpy.pdfgetx.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Probe timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the report to a JSON file.")
    args = parser.parse_args(argv)

    report = inspect_pdfgetx3_environment(
        executable=args.pdfgetx3,
        python_executable=args.python_executable,
        timeout=args.timeout,
    )
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.write}")
    elif args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()
