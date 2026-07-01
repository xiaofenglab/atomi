"""Bridge Atomi/OpenMolcas outputs to Pegamoid orbital inspection.

Pegamoid is a GUI orbital/density viewer for OpenMolcas HDF5, InpOrb,
Molden, GRID_IT, and cube files.  This module deliberately keeps it optional:
Atomi can prepare reproducible launch wrappers and status probes without
forcing Qt/VTK GUI dependencies into the main HPC environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_STATUS = "atomi.pegamoid_status.v1"
SCHEMA_PLAN = "atomi.pegamoid_install_plan.v1"
SCHEMA_PROJECT = "atomi.pegamoid_bridge_project.v1"


def _run_probe(cmd: list[str], timeout: float = 8.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - platform/runtime specific
        return {"ok": False, "error": str(exc), "cmd": cmd}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "cmd": cmd,
    }


def probe_pegamoid(
    *,
    python: str | None = None,
    script: str | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    """Return a non-GUI Pegamoid availability report."""

    env_python = python or os.environ.get("ATOMI_PEGAMOID_PYTHON") or ""
    env_script = script or os.environ.get("ATOMI_PEGAMOID_SCRIPT") or ""
    env_exe = executable or os.environ.get("ATOMI_PEGAMOID_EXE") or "pegamoid.py"
    resolved_exe = shutil.which(env_exe) if env_exe else None
    resolved_python = shutil.which(env_python) if env_python else None

    python_probe: dict[str, Any] | None = None
    if resolved_python:
        python_probe = _run_probe(
            [
                resolved_python,
                "-c",
                "import importlib.metadata as m; "
                "print(m.version('Pegamoid'))",
            ]
        )

    script_path = Path(env_script).expanduser() if env_script else None
    script_exists = bool(script_path and script_path.exists())
    available = bool(resolved_exe or (resolved_python and script_exists) or (python_probe and python_probe.get("ok")))
    return {
        "schema": SCHEMA_STATUS,
        "pegamoid": {
            "available": available,
            "executable": env_exe,
            "resolved_executable": resolved_exe or "",
            "python": env_python,
            "resolved_python": resolved_python or "",
            "script": str(script_path) if script_path else "",
            "script_exists": script_exists,
            "pip_probe": python_probe or {},
            "notes": [
                "Pegamoid is a GUI viewer; Atomi only probes/writes launch wrappers.",
                "Use ThinLinc, VNC, or trusted X forwarding for interactive viewing on HPC.",
            ],
        },
    }


def status_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check configured Pegamoid runtime for OpenMolcas orbital viewing.")
    parser.add_argument("--python", help="Python executable for a separate Pegamoid environment.")
    parser.add_argument("--script", help="pegamoid.py script path.")
    parser.add_argument("--executable", help="Pegamoid console script, usually pegamoid.py.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = probe_pegamoid(python=args.python, script=args.script, executable=args.executable)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        p = report["pegamoid"]
        print("Pegamoid / Atomi status")
        print(f"  available : {'yes' if p['available'] else 'no'}")
        print(f"  executable: {p['resolved_executable'] or p['executable'] or '(not set)'}")
        print(f"  python    : {p['resolved_python'] or p['python'] or '(not set)'}")
        print(f"  script    : {p['script'] or '(not set)'}")
    return 0


def install_plan_main(args: argparse.Namespace) -> dict[str, Any]:
    plan = {
        "schema": SCHEMA_PLAN,
        "recommendation": "Install Pegamoid in a separate GUI/runtime environment, not in m_lammps_env.",
        "reason": (
            "Pegamoid is an OpenMolcas orbital/density viewer and pulls GUI/VTK/Qt dependencies; "
            "keeping it external protects the main Atomi/LAMMPS/VASP environment."
        ),
        "hpc_profile": {
            "pegamoid": {
                "env_path": "$HOME/atomi_hpc/pegamoid_env",
                "python": "$HOME/atomi_hpc/pegamoid_env/bin/python",
                "script": "$HOME/atomi_hpc/pegamoid_env/bin/pegamoid.py",
                "environment": {
                    "ATOMI_PEGAMOID_PYTHON": "$HOME/atomi_hpc/pegamoid_env/bin/python",
                    "ATOMI_PEGAMOID_SCRIPT": "$HOME/atomi_hpc/pegamoid_env/bin/pegamoid.py",
                    "PEGAMOID_MAXSCRATCH": "2GB",
                },
            }
        },
        "commands": [
            "python3 -m venv $HOME/atomi_hpc/pegamoid_env",
            "$HOME/atomi_hpc/pegamoid_env/bin/python -m pip install --upgrade pip",
            "$HOME/atomi_hpc/pegamoid_env/bin/python -m pip install Pegamoid",
            "eval \"$(confighpc --config $HOME/atomi_hpc/atomi_hpc_config.kit.local.json --shell)\"",
            "pegamoid-status",
        ],
        "viewer_notes": [
            "Prefer ThinLinc/VNC for the GUI; plain ssh -X may fail for VTK/OpenGL.",
            "Open HDF5/InpOrb together when using InpOrb orbital selections.",
            "OpenMolcas RASSCF/RASSI should write HDF5/TDM/TRD1 when transition densities are needed.",
        ],
    }
    if getattr(args, "json", False):
        print(json.dumps(plan, indent=2, sort_keys=True))
    else:
        print("Pegamoid / Atomi HPC install plan")
        print(plan["recommendation"])
        print("Suggested commands:")
        for cmd in plan["commands"]:
            print(f"  {cmd}")
    return plan


def install_plan_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print recommended Pegamoid setup for Atomi/OpenMolcas.")
    parser.add_argument("--json", action="store_true")
    install_plan_main(parser.parse_args(argv))
    return 0


def _quote_path(path: Path) -> str:
    return shlex.quote(str(path))


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    files = [Path(p).expanduser().resolve() for p in args.file]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Pegamoid input file(s) not found: " + ", ".join(missing))

    label = args.label
    run_script = outdir / "run_pegamoid.sh"
    file_args = " ".join(_quote_path(p) for p in files)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'export PEGAMOID_MAXSCRATCH="${{PEGAMOID_MAXSCRATCH:-{args.maxscratch}}}"',
    ]
    if args.module:
        lines.append(f'module load "{args.module}"')
    lines.extend(
        [
            'if [[ -n "${ATOMI_PEGAMOID_SCRIPT:-}" ]]; then',
            f'  exec "${{ATOMI_PEGAMOID_PYTHON:-python}}" "${{ATOMI_PEGAMOID_SCRIPT}}" {file_args}',
            'elif command -v "${ATOMI_PEGAMOID_EXE:-pegamoid.py}" >/dev/null 2>&1; then',
            f'  exec "${{ATOMI_PEGAMOID_EXE:-pegamoid.py}}" {file_args}',
            "else",
            '  echo "Pegamoid is not configured. Run pegamoid-install-plan." >&2',
            "  exit 2",
            "fi",
            "",
        ]
    )
    run_script.write_text("\n".join(lines), encoding="utf-8")
    run_script.chmod(0o755)

    metadata = {
        "schema": SCHEMA_PROJECT,
        "label": label,
        "files": [str(p) for p in files],
        "run_script": str(run_script),
        "module": args.module or "",
        "maxscratch": args.maxscratch,
        "notes": [
            "This wrapper launches Pegamoid interactively; it is not a batch compute job.",
            "Use RASSCF/RASSI HDF5 plus InpOrb/RasOrb/Molden as available.",
        ],
    }
    project = outdir / "pegamoid_bridge_project.json"
    project.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote Pegamoid launcher: {run_script}")
    print(f"Wrote project metadata: {project}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge OpenMolcas orbital files to Pegamoid launch wrappers.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status")
    p.add_argument("--python")
    p.add_argument("--script")
    p.add_argument("--executable")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lambda ns: status_cli(_namespace_to_status_args(ns)))

    p = sub.add_parser("install-plan")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=lambda ns: install_plan_main(ns))

    p = sub.add_parser("prepare")
    p.add_argument("--file", action="append", required=True, help="OpenMolcas HDF5/InpOrb/Molden/cube file. Repeatable.")
    p.add_argument("--outdir", type=Path, default=Path("pegamoid_view"))
    p.add_argument("--label", default="pegamoid_view")
    p.add_argument("--module", default=os.environ.get("ATOMI_PEGAMOID_MODULE", ""))
    p.add_argument("--maxscratch", default="2GB")
    p.set_defaults(func=prepare_main)
    return parser


def _namespace_to_status_args(ns: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for key in ("python", "script", "executable"):
        value = getattr(ns, key, None)
        if value:
            out.extend([f"--{key}", value])
    if getattr(ns, "json", False):
        out.append("--json")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
