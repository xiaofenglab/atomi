"""Optional zentropy runtime status helpers."""

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


def _package_status(import_name: str, distribution_name: str | None = None) -> dict[str, Any]:
    available = importlib.util.find_spec(import_name) is not None
    version = None
    if available:
        try:
            version = metadata.version(distribution_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


def configured_zentropy_python() -> str | None:
    """Return a configured external zentropy Python, if one exists."""
    direct = os.environ.get("ATOMI_ZENTROPY_PYTHON", "").strip()
    if direct:
        return direct
    env_dir = os.environ.get("ATOMI_ZENTROPY_ENV", "").strip()
    if env_dir:
        root = Path(env_dir).expanduser()
        for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate)
    return None


def configured_zentropy_executable() -> str | None:
    """Return a configured zentropy executable or a PATH match."""
    direct = os.environ.get("ATOMI_ZENTROPY_EXE", "").strip()
    if direct:
        return direct
    return shutil.which("pyzentropy") or shutil.which("zentropy")


PROBE_SCRIPT = """
import importlib.util, json, sys
from importlib import metadata

def pkg(import_name, dist_name=None):
    ok = importlib.util.find_spec(import_name) is not None
    version = None
    if ok:
        try:
            version = metadata.version(dist_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": ok, "version": version}

data = {
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "pyzentropy": pkg("pyzentropy", "pyzentropy"),
    "zentropy": pkg("zentropy", "zentropy"),
}
print(json.dumps(data))
"""


def probe_zentropy_python(python_executable: str | Path | None, timeout: float = 20.0) -> dict[str, Any]:
    """Probe whether an external Python can import pyzentropy."""
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
    data["available"] = bool(data.get("pyzentropy", {}).get("available"))
    return data


def build_zentropy_status(
    external_python: str | None = None,
    executable: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Return active/external zentropy runtime availability."""
    active = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "pyzentropy": _package_status("pyzentropy", "pyzentropy"),
        "zentropy": _package_status("zentropy", "zentropy"),
    }
    external = probe_zentropy_python(external_python or configured_zentropy_python(), timeout=timeout)
    exe = executable or configured_zentropy_executable() or ""

    suggestions: list[str] = []
    warnings: list[str] = []
    if active["pyzentropy"]["available"]:
        mode = "active-python"
        suggestions.append("pyzentropy is importable in the active Atomi Python environment.")
    elif external.get("available"):
        mode = "external-python"
        suggestions.append("pyzentropy is available through ATOMI_ZENTROPY_PYTHON/ATOMI_ZENTROPY_ENV.")
    elif exe:
        mode = "executable"
        suggestions.append("A zentropy executable is configured or visible on PATH.")
    else:
        mode = "missing"
        suggestions.append(
            "Install optional support with: python -m pip install --upgrade "
            '--upgrade-strategy only-if-needed "atomi[zentropy] @ '
            'git+https://github.com/xiaofenglab/atomi.git@main"'
        )
        suggestions.append(
            "Or keep pyzentropy in a separate environment and export "
            "ATOMI_ZENTROPY_PYTHON=/path/to/zentropy_env/bin/python."
        )

    if active["zentropy"]["available"] and not active["pyzentropy"]["available"]:
        warnings.append(
            "The active Python has a package named zentropy, but Atomi expects "
            "pyzentropy for materials zentropy workflows."
        )
    if external.get("zentropy", {}).get("available") and not external.get("pyzentropy", {}).get("available"):
        warnings.append(
            "The external Python has zentropy but not pyzentropy; verify it is "
            "the intended materials runtime."
        )

    return {
        "active_environment": active,
        "external_environment": external,
        "zentropy_executable": exe,
        "zentropy_mode": mode,
        "ready_for_zentropy_runtime": mode in {"active-python", "external-python", "executable"},
        "warnings": warnings,
        "suggestions": suggestions,
    }


def print_zentropy_status(status: dict[str, Any]) -> None:
    active = status["active_environment"]
    external = status["external_environment"]
    print("Atomi zentropy status")
    print(f"  active python: {active['python']} ({active['python_version']})")
    print(
        "  active pyzentropy: "
        f"{active['pyzentropy']['version'] if active['pyzentropy']['available'] else 'missing'}"
    )
    print(
        "  active zentropy package: "
        f"{active['zentropy']['version'] if active['zentropy']['available'] else 'missing'}"
    )
    if external.get("configured"):
        print(f"  external zentropy python: {external.get('requested_python') or external.get('python')}")
        external_pyzentropy = external.get("pyzentropy", {})
        external_pyzentropy_version = (
            external_pyzentropy.get("version") if external_pyzentropy.get("available") else "missing"
        )
        print(
            "  external pyzentropy: "
            f"{external_pyzentropy_version}"
        )
        if external.get("error") or external.get("stderr"):
            print(f"    reason: {external.get('error') or external.get('stderr')}")
    else:
        print("  external zentropy python: not configured")
    print(f"  zentropy executable env/PATH: {status['zentropy_executable'] or 'not configured'}")
    print(f"  selected zentropy mode: {status['zentropy_mode']}")
    if status["warnings"]:
        print("  warnings:")
        for item in status["warnings"]:
            print(f"    - {item}")
    print("  suggestions:")
    for item in status["suggestions"]:
        print(f"    - {item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="zentropy_status",
        description="Report active and configured optional pyzentropy availability.",
    )
    parser.add_argument("--python", dest="external_python", help="Probe this external Python executable.")
    parser.add_argument("--executable", help="Record this zentropy executable path.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="External Python probe timeout in seconds.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    status = build_zentropy_status(
        external_python=args.external_python,
        executable=args.executable,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_zentropy_status(status)


if __name__ == "__main__":
    main()
