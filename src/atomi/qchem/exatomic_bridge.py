"""Optional eXatomic/NBO bridge for OpenMolcas postanalysis.

eXatomic is useful for parsing OpenMolcas orbital data and exporting NBO-style
input files, but it is deliberately optional. Atomi must keep the standard
Molcas postanalysis path usable when eXatomic or the external NBO program is not
installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA_STATUS = "atomi.molcas_exatomic_status.v1"
SCHEMA_PLAN = "atomi.molcas_exatomic_install_plan.v1"
SCHEMA_NBO_EXPORT = "atomi.molcas_exatomic_nbo_export.v1"
EXATOMIC_VERSION = "0.6.0"


def _json_dump(payload: dict[str, Any], path: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _python_probe_code() -> str:
    return (
        "import json, os, sys\n"
        "os.environ.setdefault('NUMBA_DISABLE_JIT', '1')\n"
        "payload = {'python': sys.executable}\n"
        "try:\n"
        "    import exatomic\n"
        "    payload.update({\n"
        "        'available': True,\n"
        "        'version': getattr(exatomic, '__version__', ''),\n"
        "        'module_file': getattr(exatomic, '__file__', ''),\n"
        "    })\n"
        "    try:\n"
        "        import exatomic.molcas.output as molcas_output\n"
        "        import exatomic.nbo.inputs as nbo_inputs\n"
        "        payload['molcas_parser'] = getattr(molcas_output, '__file__', '')\n"
        "        payload['nbo_input_module'] = getattr(nbo_inputs, '__file__', '')\n"
        "    except Exception as exc:\n"
        "        payload['available'] = False\n"
        "        payload['bridge_import_error'] = repr(exc)\n"
        "except Exception as exc:\n"
        "    payload.update({'available': False, 'import_error': repr(exc)})\n"
        "print(json.dumps(payload, sort_keys=True))\n"
    )


def _runtime_note(returncode: int) -> str:
    if returncode in (-11, 139):
        return (
            "eXatomic import crashed in numba/llvmlite. Keep this environment for core Atomi work "
            "and use a sidecar eXatomic runtime or patched eXatomic build for NBO export."
        )
    if returncode != 0:
        return (
            "eXatomic is installed but not importable in this Python. This commonly reflects old "
            "eXatomic code meeting a newer pandas/numba/llvmlite stack."
        )
    return ""


def probe_exatomic(python: str | None = None) -> dict[str, Any]:
    """Probe eXatomic availability without importing it at module import time."""

    requested_python = python or os.environ.get("ATOMI_EXATOMIC_PYTHON") or sys.executable
    expanded = Path(requested_python).expanduser()
    resolved = shutil.which(requested_python) or (str(expanded) if expanded.exists() else "")
    if not resolved:
        return {
            "schema": SCHEMA_STATUS,
            "available": False,
            "python": requested_python,
            "resolved_python": "",
            "error": "Python executable not found.",
        }
    try:
        proc = subprocess.run(
            [resolved, "-c", _python_probe_code()],
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - platform/runtime specific
        return {
            "schema": SCHEMA_STATUS,
            "available": False,
            "python": requested_python,
            "resolved_python": resolved,
            "error": str(exc),
        }
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
    except Exception:
        parsed = {}
    if proc.returncode != 0:
        parsed.setdefault("available", False)
        parsed.setdefault("runtime_note", _runtime_note(proc.returncode))
    return {
        "schema": SCHEMA_STATUS,
        "python": requested_python,
        "resolved_python": resolved,
        "returncode": proc.returncode,
        "stdout_head": "\n".join(proc.stdout.splitlines()[:6]),
        "stderr_head": "\n".join(proc.stderr.splitlines()[:8]),
        **parsed,
    }


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    report = probe_exatomic(args.python)
    if args.json:
        _json_dump(report)
    else:
        print("Atomi eXatomic / Molcas-NBO bridge status")
        print(f"  python     : {report.get('resolved_python') or report.get('python')}")
        print(f"  available  : {'yes' if report.get('available') else 'no'}")
        print(f"  version    : {report.get('version') or '(not importable)'}")
        if report.get("returncode") not in (None, 0):
            print(f"  returncode : {report['returncode']}")
        if report.get("module_file"):
            print(f"  module     : {report['module_file']}")
        if report.get("runtime_note"):
            print(f"  runtime    : {report['runtime_note']}")
        if report.get("import_error"):
            print(f"  error      : {report['import_error']}")
        if report.get("bridge_import_error"):
            print(f"  bridge err : {report['bridge_import_error']}")
    return report


def install_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA_PLAN,
        "recommendation": "Add eXatomic as an optional MOLCAS postanalysis bridge, not as a required Atomi dependency.",
        "why_optional": [
            "The current PyPI release is exatomic 0.6.0; keep this old/large parser isolated from core Atomi.",
            "Atomi already parses RASSI transitions, XANES broadening, root audits, and AO composition directly.",
            "eXatomic adds value for Molcas universe parsing and NBO-style input export, not for choosing active spaces or roots.",
            "The licensed/standalone NBO program is separate; eXatomic can prepare input, but it is not the NBO engine.",
            "A clean pip install is not enough: eXatomic 0.6.0 must also survive the status import probe.",
        ],
        "roles": {
            "atomi_core": "RASSCF/CASPT2/RASSI input preparation, root counting, XANES plots, AO/MO coefficient summaries.",
            "exatomic_bridge": "Optional Molcas output/orbital parser and GENNBO input exporter.",
            "nbo_program": "External NBO execution and Natural Population/Bond Orbital analysis.",
        },
        "commands": {
            "kit_justus2_m_lammps_env": [
                f"~/m_lammps_env/bin/python -m pip install 'exatomic=={EXATOMIC_VERSION}' 'pandas<3,>=1.5' 'ipywidgets<8,>=7' PyYAML seaborn tables",
                "NUMBA_DISABLE_JIT=1 ~/m_lammps_env/bin/python -c 'import exatomic; print(exatomic.__version__)'",
                "molcas-exatomic-status",
            ],
            "wsu_kamiak_private_molcas_tools": [
                "source ~/atomi_private/activate_molcas_tools.sh",
                f"python -m pip install 'exatomic=={EXATOMIC_VERSION}' 'pandas<3,>=1.5' 'ipywidgets<8,>=7' PyYAML seaborn tables",
                "NUMBA_DISABLE_JIT=1 python -c 'import exatomic; print(exatomic.__version__)'",
                "python -m atomi.qchem.exatomic_bridge status",
            ],
        },
        "example_export": [
            "molcas-exatomic-bridge nbo-export --molcas-out RUN.out --momatrix RUN.RasOrb --overlap RUN.OneInt --out RUN.gennbo",
            "Run the external NBO program on RUN.gennbo only if NBO is licensed/installed for that machine.",
        ],
        "guardrails": [
            "Do not run this from $HOME root on HPC; run inside the project run/postanalysis folder.",
            "Do not make eXatomic mandatory for standard molcas-postanalysis.",
            "Do not infer Ce/O active spaces from NBO; use NBO/NPA only as a post-hoc chemical guard.",
            "If status reports a numba/llvmlite crash, do not downgrade shared production environments blindly; use a sidecar runtime.",
        ],
    }


def install_plan_main(args: argparse.Namespace) -> dict[str, Any]:
    plan = install_plan()
    if args.json:
        _json_dump(plan)
    else:
        print("Atomi eXatomic / NBO optional bridge plan")
        print(plan["recommendation"])
        print("\nKIT/JUSTUS2:")
        for cmd in plan["commands"]["kit_justus2_m_lammps_env"]:
            print(f"  {cmd}")
        print("\nWSU/Kamiak:")
        for cmd in plan["commands"]["wsu_kamiak_private_molcas_tools"]:
            print(f"  {cmd}")
    return plan


def _editor_to_text(editor: Any) -> str:
    """Best-effort text extraction from an eXatomic Editor-like object."""

    for attr in ("_lines", "lines"):
        lines = getattr(editor, attr, None)
        if isinstance(lines, list):
            return "\n".join(str(line) for line in lines) + "\n"
    if isinstance(editor, str):
        return editor if editor.endswith("\n") else editor + "\n"
    text = str(editor)
    return text if text.endswith("\n") else text + "\n"


def _summarize_universe(uni: Any) -> dict[str, Any]:
    attrs = [
        "atom",
        "basis_set",
        "basis_set_order",
        "overlap",
        "momatrix",
        "orbital",
        "density",
        "sf_energy",
        "so_energy",
        "sf_oscillator",
        "so_oscillator",
        "natural_occ",
        "caspt2_energy",
    ]
    out: dict[str, Any] = {}
    for attr in attrs:
        value = getattr(uni, attr, None)
        if value is None:
            out[attr] = {"present": False}
            continue
        shape = getattr(value, "shape", None)
        out[attr] = {"present": True, "shape": list(shape) if shape is not None else ""}
    return out


def nbo_export_main(args: argparse.Namespace) -> dict[str, Any]:
    status = probe_exatomic(args.python)
    if not status.get("available"):
        summary = {
            "schema": SCHEMA_NBO_EXPORT,
            "ok": False,
            "reason": "eXatomic is not available in the selected Python environment.",
            "status": status,
            "install_plan": install_plan(),
        }
        if args.summary:
            _json_dump(summary, args.summary)
        raise RuntimeError("eXatomic is not importable; run molcas-exatomic-install-plan.")

    try:
        from exatomic.molcas.output import parse_molcas  # type: ignore[import-not-found]
        from exatomic.nbo.inputs import Input as NboInput  # type: ignore[import-not-found]
    except Exception as exc:
        summary = {
            "schema": SCHEMA_NBO_EXPORT,
            "ok": False,
            "reason": "eXatomic is importable, but Molcas/NBO bridge modules failed to import.",
            "error": repr(exc),
            "status": status,
        }
        if args.summary:
            _json_dump(summary, args.summary)
        raise

    molcas_out = args.molcas_out.resolve()
    out_path = args.out.resolve()
    kwargs: dict[str, Any] = {}
    if args.momatrix:
        kwargs["momatrix"] = args.momatrix
    if args.overlap:
        kwargs["overlap"] = args.overlap
    if args.occvec:
        kwargs["occvec"] = args.occvec
    try:
        uni = parse_molcas(str(molcas_out), **kwargs)
        nbo_input = NboInput.from_universe(
            uni,
            mocoefs=args.mocoefs,
            orbocc=args.orbocc,
            name=args.name or molcas_out.stem,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_editor_to_text(nbo_input), encoding="utf-8")
        summary = {
            "schema": SCHEMA_NBO_EXPORT,
            "ok": True,
            "molcas_out": str(molcas_out),
            "nbo_input": str(out_path),
            "status": status,
            "universe": _summarize_universe(uni),
            "notes": [
                "This is an NBO/GENNBO input export, not an NBO calculation.",
                "Run the external NBO program separately if licensed and installed.",
            ],
        }
    except Exception as exc:
        summary = {
            "schema": SCHEMA_NBO_EXPORT,
            "ok": False,
            "molcas_out": str(molcas_out),
            "nbo_input": str(out_path),
            "error": repr(exc),
            "status": status,
            "notes": [
                "NBO export usually requires atom, basis, overlap, MO coefficient, and occupation data.",
                "For OpenMolcas, provide the matching RasOrb/InpOrb and overlap file if the output alone is insufficient.",
            ],
        }
        if args.summary:
            _json_dump(summary, args.summary)
        raise
    if args.summary:
        _json_dump(summary, args.summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Wrote NBO input: {out_path}")
        if args.summary:
            print(f"Wrote summary: {args.summary}")
    return summary


def status_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check eXatomic availability for Atomi MOLCAS postanalysis.")
    parser.add_argument("--python")
    parser.add_argument("--json", action="store_true")
    status_main(parser.parse_args(argv))
    return 0


def install_plan_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print optional eXatomic setup for Atomi MOLCAS postanalysis.")
    parser.add_argument("--json", action="store_true")
    install_plan_main(parser.parse_args(argv))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optional eXatomic/NBO bridge for OpenMolcas postanalysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Check whether eXatomic is importable.")
    p.add_argument("--python")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=status_main)

    p = sub.add_parser("install-plan", help="Print recommended eXatomic setup commands.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=install_plan_main)

    p = sub.add_parser("nbo-export", help="Export a Molcas result to an NBO/GENNBO input using eXatomic.")
    p.add_argument("--molcas-out", type=Path, required=True)
    p.add_argument("--momatrix", help="MO coefficient file name, usually relative to the Molcas output directory.")
    p.add_argument("--overlap", help="Overlap file name, usually relative to the Molcas output directory.")
    p.add_argument("--occvec", help="Optional eXatomic occupation-vector selector.")
    p.add_argument("--mocoefs", help="Optional eXatomic MO coefficient column name.")
    p.add_argument("--orbocc", help="Optional eXatomic orbital occupation column name.")
    p.add_argument("--name", default="")
    p.add_argument("--out", type=Path, default=Path("molcas_nbo_export.gennbo"))
    p.add_argument("--summary", type=Path, default=Path("molcas_nbo_export_summary.json"))
    p.add_argument("--python", help="Probe this Python for eXatomic before export.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=nbo_export_main)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
