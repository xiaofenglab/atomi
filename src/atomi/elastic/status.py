"""Status checks for optional elastic visualization backends."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from importlib import metadata
from typing import Any


def package_status(import_name: str, dist_name: str | None = None) -> dict[str, Any]:
    available = importlib.util.find_spec(import_name) is not None
    version = None
    if available:
        try:
            version = metadata.version(dist_name or import_name)
        except metadata.PackageNotFoundError:
            version = "unknown"
    return {"available": available, "version": version}


def probe() -> dict[str, Any]:
    data: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "elate": package_status("ELATE", "ELATE"),
        "matplotlib": package_status("matplotlib"),
        "native_backend_available": True,
    }
    try:
        from ELATE.elastic import Elastic  # noqa: F401

        data["elate"]["elastic_class_available"] = True
    except Exception as exc:
        data["elate"]["elastic_class_available"] = False
        data["elate"]["error"] = str(exc)
    data["selected_default_backend"] = "elate" if data["elate"]["elastic_class_available"] else "native"
    return data


def format_status(data: dict[str, Any]) -> str:
    lines = [
        "Atomi Elastic Visualization status",
        f"  active python: {data['python']} ({data['python_version']})",
        f"  ELATE: {data['elate']['version'] if data['elate']['available'] else 'missing'}",
        f"  ELATE Elastic class: {'yes' if data['elate'].get('elastic_class_available') else 'no'}",
        f"  native backend: {'yes' if data['native_backend_available'] else 'no'}",
        f"  matplotlib: {data['matplotlib']['version'] if data['matplotlib']['available'] else 'missing'}",
        f"  default elastic_viz backend: {data['selected_default_backend']}",
    ]
    suggestions: list[str] = []
    if not data["elate"].get("elastic_class_available"):
        suggestions.extend(
            [
                "ELATE is optional; elastic_viz --backend auto will use native formulas.",
                'To try ELATE directly: python -m pip install "ELATE @ git+https://github.com/coudertlab/elate.git@master"',
            ]
        )
    if not data["matplotlib"]["available"]:
        suggestions.append("Install matplotlib if you want summary PNG plots.")
    if suggestions:
        lines.append("  suggestions:")
        lines.extend(f"    - {item}" for item in suggestions)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elate_status",
        description="Check optional ELATE/native elastic visualization availability.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    data = probe()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(format_status(data))
    return data


if __name__ == "__main__":
    main()
