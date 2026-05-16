"""XAFS optional dependency status helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any


def _package_status(import_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    available = importlib.util.find_spec(import_name) is not None
    version = None
    if available:
        try:
            version = metadata.version(distribution_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


def configured_larch_python() -> str | None:
    """Return the configured external Larch Python, if any."""
    direct = os.environ.get("ATOMI_XAFS_LARCH_PYTHON", "").strip()
    if direct:
        return direct
    env_dir = os.environ.get("ATOMI_XAFS_LARCH_ENV", "").strip()
    if env_dir:
        root = Path(env_dir).expanduser()
        for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate)
    return None


def probe_larch_python(python_executable: str | Path | None) -> dict[str, Any]:
    """Probe whether a Python executable can import Larch and xraydb."""
    if not python_executable:
        return {"configured": False, "available": False}
    python = str(Path(python_executable).expanduser())
    script = (
        "import importlib.util, json, sys\n"
        "from importlib import metadata\n"
        "def pkg(import_name, dist_name=None):\n"
        "    ok = importlib.util.find_spec(import_name) is not None\n"
        "    version = None\n"
        "    if ok:\n"
        "        try:\n"
        "            version = metadata.version(dist_name or import_name)\n"
        "        except metadata.PackageNotFoundError:\n"
        "            version = 'unknown'\n"
        "    return {'available': ok, 'version': version}\n"
        "data = {\n"
        "    'python': sys.executable,\n"
        "    'python_version': sys.version.split()[0],\n"
        "    'larch': pkg('larch', 'xraylarch'),\n"
        "    'xraydb': pkg('xraydb'),\n"
        "}\n"
        "try:\n"
        "    from larch import Group\n"
        "    from larch.xafs import xftf\n"
        "    data['xftf_available'] = True\n"
        "except Exception as exc:\n"
        "    data['xftf_available'] = False\n"
        "    data['xftf_error'] = str(exc)\n"
        "print(json.dumps(data))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", script],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except OSError as exc:
        return {"configured": True, "python": python, "available": False, "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {"configured": True, "python": python, "available": False, "error": f"probe timed out: {exc}"}
    if proc.returncode != 0:
        return {
            "configured": True,
            "python": python,
            "available": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
        }
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"configured": True, "python": python, "available": False, "stdout": proc.stdout.strip()}
    data["configured"] = True
    data["requested_python"] = python
    data["available"] = bool(data.get("xftf_available"))
    return data


def build_xafs_status() -> dict[str, Any]:
    active = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "larch": _package_status("larch", "xraylarch"),
        "xraydb": _package_status("xraydb"),
    }
    try:
        from larch import Group  # noqa: F401
        from larch.xafs import xftf  # noqa: F401

        active["xftf_available"] = True
    except Exception as exc:
        active["xftf_available"] = False
        active["xftf_error"] = str(exc)

    external_python = configured_larch_python()
    external = probe_larch_python(external_python)
    feff_exe = os.environ.get("ATOMI_XAFS_FEFF_EXE") or os.environ.get("ATOMI_FEFF_EXE") or ""

    suggestions: list[str] = []
    if active["xftf_available"]:
        larch_mode = "active-python"
        suggestions.append("Larch is importable in the active Atomi Python environment.")
    elif external.get("available"):
        larch_mode = "external-python"
        suggestions.append("Larch is available through ATOMI_XAFS_LARCH_PYTHON/ATOMI_XAFS_LARCH_ENV.")
    else:
        larch_mode = "missing"
        suggestions.append(
            'Install optional XAFS support with: python -m pip install --upgrade --upgrade-strategy only-if-needed "atomi[xafs] @ git+https://github.com/xiaofenglab/atomi.git@main"'
        )
        suggestions.append(
            "Or keep Larch in a separate environment and export ATOMI_XAFS_LARCH_PYTHON=/path/to/larch_env/bin/python."
        )
    if not active["xraydb"]["available"]:
        suggestions.append("Install xraydb in the active Atomi environment if edge metadata is needed there.")

    return {
        "active_environment": active,
        "external_larch_environment": external,
        "feff_executable": feff_exe,
        "larch_mode": larch_mode,
        "ready_for_larch_transform": larch_mode in {"active-python", "external-python"},
        "suggestions": suggestions,
    }


def print_xafs_status(status: dict[str, Any]) -> None:
    active = status["active_environment"]
    external = status["external_larch_environment"]
    print("Atomi XAFS/Larch status")
    print(f"  active python: {active['python']} ({active['python_version']})")
    print(f"  active xraylarch/larch: {active['larch']['version'] if active['larch']['available'] else 'missing'}")
    print(f"  active xraydb: {active['xraydb']['version'] if active['xraydb']['available'] else 'missing'}")
    print(f"  active Larch xftf: {'yes' if active.get('xftf_available') else 'no'}")
    if not active.get("xftf_available") and active.get("xftf_error"):
        print(f"    reason: {active['xftf_error']}")
    if external.get("configured"):
        print(f"  external Larch python: {external.get('requested_python') or external.get('python')}")
        print(f"  external xraylarch/larch: {external.get('larch', {}).get('version') if external.get('larch', {}).get('available') else 'missing'}")
        print(f"  external xraydb: {external.get('xraydb', {}).get('version') if external.get('xraydb', {}).get('available') else 'missing'}")
        print(f"  external Larch xftf: {'yes' if external.get('xftf_available') else 'no'}")
        if external.get("error") or external.get("stderr") or external.get("xftf_error"):
            print(f"    reason: {external.get('error') or external.get('stderr') or external.get('xftf_error')}")
    else:
        print("  external Larch python: not configured")
    print(f"  FEFF executable env: {status['feff_executable'] or 'not configured'}")
    print(f"  selected Larch mode: {status['larch_mode']}")
    print("  suggestions:")
    for item in status["suggestions"]:
        print(f"    - {item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="xafs_status",
        description="Report active and configured optional Larch/xraydb availability for Atomi XAFS workflows.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    status = build_xafs_status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_xafs_status(status)

