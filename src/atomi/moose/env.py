"""Environment inspection helpers for MOOSE-based workflows."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


COMMON_MOOSE_EXECUTABLES = [
    "moose-opt",
    "moose-dbg",
    "moose-devel",
    "moose_test-opt",
]

MOOSE_ENV_VARS = [
    "MOOSE_DIR",
    "METHOD",
    "PETSC_DIR",
    "LIBMESH_DIR",
    "LIBMESH_METHOD",
    "CC",
    "CXX",
    "FC",
]


def _resolve_executable(name_or_path: str) -> str | None:
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return str(path)
    return shutil.which(name_or_path)


def _run_command(command: list[str], timeout: int = 5) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        return {"command": command, "returncode": None, "output": str(exc)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {"command": command, "returncode": None, "timed_out": True, "output": output.strip()}
    return {"command": command, "returncode": result.returncode, "output": result.stdout.strip()}


def inspect_moose_environment(app: str | None = None) -> dict[str, Any]:
    """Build a portable MOOSE environment report without assuming a specific app name."""
    candidates = list(COMMON_MOOSE_EXECUTABLES)
    if app:
        candidates.insert(0, app)

    executables: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate in executables:
            continue
        resolved = _resolve_executable(candidate)
        item: dict[str, Any] = {"path": resolved, "available": resolved is not None}
        if resolved is not None and candidate == app:
            item["version_probe"] = _run_command([resolved, "--version"])
        executables[candidate] = item

    return {
        "schema_version": 1,
        "module": "moose",
        "cwd": str(Path.cwd()),
        "app": app,
        "executables": executables,
        "environment": {name: os.environ.get(name, "") for name in MOOSE_ENV_VARS},
        "notes": [
            "MOOSE applications are usually project-specific executables such as app-opt.",
            "Use --app /path/to/app-opt when the executable is not on PATH.",
            "Record required compiler/MPI/PETSc modules in the project config for each HPC.",
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    print("Atomi MOOSE environment")
    if report.get("app"):
        print(f"Requested app: {report['app']}")
    found = [
        f"{name}={item['path']}"
        for name, item in report["executables"].items()
        if item["available"]
    ]
    missing = [name for name, item in report["executables"].items() if not item["available"]]
    print(f"Found executables: {', '.join(found) if found else 'none'}")
    if missing:
        print(f"Missing candidates: {', '.join(missing)}")
    env = {key: value for key, value in report["environment"].items() if value}
    print("Environment variables:")
    if env:
        for key, value in env.items():
            print(f"  {key}={value}")
    else:
        print("  none of the common MOOSE variables are set")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-doctor",
        description="Inspect MOOSE application executables and common MOOSE environment variables.",
    )
    parser.add_argument("--app", help="MOOSE application executable name or path, e.g. app-opt.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the report to a JSON file.")
    args = parser.parse_args(argv)

    report = inspect_moose_environment(app=args.app)
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
