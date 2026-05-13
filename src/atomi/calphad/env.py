"""Environment and database inspection helpers for pycalphad workflows."""

from __future__ import annotations

import argparse
import json
from importlib import metadata, util
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def inspect_calphad_environment(
    database: Path | None = None,
    components: list[str] | None = None,
    phases: list[str] | None = None,
) -> dict[str, Any]:
    """Build a pycalphad/CALPHAD environment report."""
    available = util.find_spec("pycalphad") is not None
    report: dict[str, Any] = {
        "schema_version": 1,
        "module": "calphad",
        "pycalphad": {
            "available": available,
            "version": _package_version("pycalphad") if available else None,
        },
        "database": None,
        "requested_components": components or [],
        "requested_phases": phases or [],
        "notes": [
            "Install pycalphad in the active Python environment for CALPHAD calculations.",
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
    elif db_path.is_file() and not available:
        db_report["error"] = "pycalphad is not installed in this Python environment"
    report["database"] = db_report
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("Atomi CALPHAD environment")
    pyc = report["pycalphad"]
    status = pyc["version"] if pyc["available"] else "missing"
    print(f"pycalphad: {status}")
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="calphad-doctor",
        description="Inspect pycalphad availability and optionally parse a TDB database.",
    )
    parser.add_argument("--database", type=Path, help="Thermodynamic database, usually a .tdb file.")
    parser.add_argument("--components", help="Comma-separated component list, e.g. U,O,VA.")
    parser.add_argument("--phases", help="Comma-separated phase list to record for this project.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the report to a JSON file.")
    args = parser.parse_args(argv)

    report = inspect_calphad_environment(
        database=args.database,
        components=_parse_csv(args.components),
        phases=_parse_csv(args.phases),
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
