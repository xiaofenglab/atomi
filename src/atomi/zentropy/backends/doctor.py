"""Backend availability diagnostics for the defect thermodynamic engine."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .registry import backend_names, backend_registry


def build_backend_doctor_report(backend: str | None = None) -> dict[str, Any]:
    registry = backend_registry()
    names = [backend] if backend else sorted(registry)
    rows = []
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown backend {name!r}. Choices: {', '.join(sorted(registry))}")
        rows.append(registry[name].capability_report())
    return {
        "schema": "atomi.zentropy.defect_backends.doctor.v1",
        "backend_count": len(rows),
        "backends": rows,
        "notes": [
            "smol and CASM are optional lazy backends; missing packages should not break base Atomi.",
            "Use built-in pocc_gqca_population_vector and pocc_motif_mc for schema/prototype work before CE-MC packages are installed.",
        ],
    }


def _print_text(report: dict[str, Any]) -> None:
    for row in report["backends"]:
        print(f"backend: {row['backend']}")
        print(f"available: {str(row.get('available', False)).lower()}")
        if row.get("version"):
            print(f"version: {row['version']}")
        for key in ("casm_cli", "ccasm_cli"):
            if key in row:
                print(f"{key}: {row.get(key) or 'not_found'}")
        install_hint = row.get("install_hint")
        if install_hint:
            print(f"install_hint: {install_hint}")
        capabilities = row.get("capabilities") or {}
        if capabilities:
            print("capabilities:")
            for name, value in sorted(capabilities.items()):
                print(f"  {name}: {value}")
        print("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atomi-defects backend doctor")
    parser.add_argument("--backend", choices=backend_names(), help="Only report one backend.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    report = build_backend_doctor_report(args.backend)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    return report
