"""Environment and database inspection helpers for pycalphad workflows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from importlib import metadata, util
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


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def configured_calphad_python() -> str | None:
    direct = os.environ.get("ATOMI_CALPHAD_PYTHON", "").strip()
    if direct:
        return direct
    env_dir = os.environ.get("ATOMI_CALPHAD_ENV", "").strip()
    if env_dir:
        root = Path(env_dir).expanduser()
        for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
            if candidate.exists():
                return str(candidate)
    return None


def configured_databases() -> list[str]:
    return _parse_csv(os.environ.get("ATOMI_CALPHAD_DATABASES", ""))


PROBE_SCRIPT = """
import importlib.util, json, sys
from importlib import metadata
from pathlib import Path

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
    "pycalphad": pkg("pycalphad", "pycalphad"),
}
db_path = sys.argv[1] if len(sys.argv) > 1 else ""
if db_path:
    db = {"path": db_path, "exists": Path(db_path).is_file(), "parsed": False, "elements": [], "phases": [], "error": None}
    if db["exists"] and data["pycalphad"]["available"]:
        try:
            from pycalphad import Database
            dbf = Database(db_path)
            db["parsed"] = True
            db["elements"] = sorted(str(item) for item in getattr(dbf, "elements", []))
            db["phases"] = sorted(str(item) for item in getattr(dbf, "phases", {}).keys())
        except Exception as exc:
            db["error"] = str(exc)
    elif db["exists"]:
        db["error"] = "pycalphad is not installed in this Python environment"
    data["database"] = db
print(json.dumps(data))
"""


def probe_calphad_python(
    python_executable: str | Path | None,
    database: Path | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    if not python_executable:
        return {"configured": False, "available": False}
    python = str(Path(python_executable).expanduser())
    command = [python, "-c", PROBE_SCRIPT]
    if database is not None:
        command.append(str(database.expanduser()))
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except OSError as exc:
        return {"configured": True, "python": python, "available": False, "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "configured": True,
            "python": python,
            "available": False,
            "error": f"probe timed out: {exc}",
        }
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
    data["available"] = bool(data.get("pycalphad", {}).get("available"))
    return data


def inspect_calphad_environment(
    database: Path | None = None,
    components: list[str] | None = None,
    phases: list[str] | None = None,
    external_python: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Build a pycalphad/CALPHAD environment report."""
    available = util.find_spec("pycalphad") is not None
    active = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "pycalphad": _package_status("pycalphad", "pycalphad"),
    }
    external = probe_calphad_python(
        external_python or configured_calphad_python(),
        database=database,
        timeout=timeout,
    )
    env_databases = configured_databases()
    suggestions: list[str] = []
    if active["pycalphad"]["available"]:
        mode = "active-python"
        suggestions.append("pycalphad is importable in the active Atomi Python environment.")
    elif external.get("available"):
        mode = "external-python"
        suggestions.append("pycalphad is available through ATOMI_CALPHAD_PYTHON/ATOMI_CALPHAD_ENV.")
    else:
        mode = "missing"
        suggestions.append("Keep pycalphad in a separate environment and export ATOMI_CALPHAD_PYTHON=/path/to/env/bin/python.")
        suggestions.append("Or install pycalphad into the active environment only when direct CALPHAD sampling is needed there.")
    report: dict[str, Any] = {
        "schema_version": 1,
        "module": "calphad",
        "pycalphad": {
            "available": available,
            "version": active["pycalphad"]["version"] if available else None,
        },
        "active_environment": active,
        "external_environment": external,
        "calphad_mode": mode,
        "ready_for_pycalphad": mode in {"active-python", "external-python"},
        "database": None,
        "configured_databases": env_databases,
        "requested_components": components or [],
        "requested_phases": phases or [],
        "suggestions": suggestions,
        "notes": [
            "pycalphad may live in a separate MOOSE/CALPHAD environment.",
            "Keep thermodynamic databases project-local or reference them with explicit paths.",
        ],
    }

    if database is None:
        return report

    db_path = database.expanduser()
    db_report: dict[str, Any] = {
        "path": str(db_path),
        "exists": db_path.is_file(),
        "parsed": False,
        "elements": [],
        "phases": [],
        "error": None,
    }
    if db_path.is_file() and available:
        try:
            from pycalphad import Database

            dbf = Database(str(db_path))
            db_report["parsed"] = True
            db_report["elements"] = sorted(str(item) for item in getattr(dbf, "elements", []))
            db_report["phases"] = sorted(str(item) for item in getattr(dbf, "phases", {}).keys())
        except Exception as exc:  # pycalphad raises parser-specific exceptions.
            db_report["error"] = str(exc)
    elif db_path.is_file() and external.get("database"):
        db_report.update(external["database"])
        db_report["parsed_by"] = "external-python"
    elif db_path.is_file() and not available:
        db_report["error"] = "pycalphad is not installed in the active or configured external Python environment"
    report["database"] = db_report
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Atomi CALPHAD/pycalphad status")
    active = report["active_environment"]
    external = report["external_environment"]
    active_status = active["pycalphad"]["version"] if active["pycalphad"]["available"] else "missing"
    print(f"  active python: {active['python']} ({active['python_version']})")
    print(f"  active pycalphad: {active_status}")
    if external.get("configured"):
        external_status = (
            external.get("pycalphad", {}).get("version")
            if external.get("pycalphad", {}).get("available")
            else "missing"
        )
        print(f"  external CALPHAD python: {external.get('requested_python') or external.get('python')}")
        print(f"  external pycalphad: {external_status}")
        if external.get("error") or external.get("stderr"):
            print(f"    reason: {external.get('error') or external.get('stderr')}")
    else:
        print("  external CALPHAD python: not configured")
    print(f"  selected CALPHAD mode: {report['calphad_mode']}")
    if report.get("configured_databases"):
        print(f"  configured databases: {', '.join(report['configured_databases'])}")
    if report.get("database"):
        db = report["database"]
        print(f"Database: {db['path']}")
        print(f"  exists: {db['exists']}")
        print(f"  parsed: {db['parsed']}")
        if db["parsed"]:
            print(f"  elements: {', '.join(db['elements']) if db['elements'] else 'none'}")
            print(f"  phases: {', '.join(db['phases']) if db['phases'] else 'none'}")
        if db["error"]:
            print(f"  error: {db['error']}")
    if report["requested_components"]:
        print(f"Requested components: {', '.join(report['requested_components'])}")
    if report["requested_phases"]:
        print(f"Requested phases: {', '.join(report['requested_phases'])}")
    print("  suggestions:")
    for item in report["suggestions"]:
        print(f"    - {item}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="calphad_status",
        description="Inspect active/configured pycalphad availability and optionally parse a TDB database.",
    )
    parser.add_argument("--database", type=Path, help="Thermodynamic database, usually a .tdb file.")
    parser.add_argument("--components", help="Comma-separated component list, e.g. U,O,VA.")
    parser.add_argument("--phases", help="Comma-separated phase list to record for this project.")
    parser.add_argument("--python", dest="external_python", help="Probe this external pycalphad Python executable.")
    parser.add_argument("--timeout", type=float, default=20.0, help="External Python probe timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the report to a JSON file.")
    args = parser.parse_args(argv)

    report = inspect_calphad_environment(
        database=args.database,
        components=_parse_csv(args.components),
        phases=_parse_csv(args.phases),
        external_python=args.external_python,
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
