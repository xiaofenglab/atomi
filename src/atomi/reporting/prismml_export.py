"""Export Atomi result artifacts as PrismML prompt manifests."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".yaml", ".yml"}
STRUCTURE_SUFFIXES = {".xyz", ".extxyz", ".cif", ".vasp"}
NOTE_SUFFIXES = {".txt", ".md", ".rst", ".log", ".out"}
DEFAULT_PROMPT_FAMILIES = (
    "title_visual",
    "workflow_overview",
    "atomistic_mechanism",
    "thermo_story",
    "presentation_background",
)


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    role: str


def classify_artifact(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    if suffix in PLOT_SUFFIXES:
        return "plot"
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in STRUCTURE_SUFFIXES or name in {"POSCAR", "CONTCAR", "XDATCAR"}:
        return "structure"
    if suffix in NOTE_SUFFIXES:
        return "note"
    return "artifact"


def artifact_role(kind: str) -> str:
    return {
        "plot": "scientific_evidence",
        "table": "data_context",
        "structure": "atomistic_context",
        "note": "workflow_context",
    }.get(kind, "supporting_context")


def _matches_any(path: Path, patterns: list[str]) -> bool:
    text = path.as_posix()
    return any(fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def scan_artifacts(
    paths: list[Path],
    *,
    limit: int = 80,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Artifact]:
    include = include or []
    exclude = exclude or []
    found: list[Artifact] = []
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(f"Atomi run path does not exist: {root}")
        candidates = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
        for path in candidates:
            if include and not _matches_any(path, include):
                continue
            if exclude and _matches_any(path, exclude):
                continue
            kind = classify_artifact(path)
            if kind == "artifact":
                continue
            found.append(Artifact(path=path.resolve(), kind=kind, role=artifact_role(kind)))
            if len(found) >= limit:
                return found
    return found


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return text or "atomi"


def _artifact_summary(artifacts: list[Artifact]) -> str:
    if not artifacts:
        return "No Atomi artifacts were found; create a general scientific visual without fake data."
    names = ", ".join(artifact.path.name for artifact in artifacts[:12])
    if len(artifacts) > 12:
        names += f", and {len(artifacts) - 12} more"
    kinds = sorted({artifact.kind for artifact in artifacts})
    return f"Available Atomi artifact kinds: {', '.join(kinds)}. Representative files: {names}."


def _family_prompt(
    family: str,
    *,
    project_title: str,
    material: str,
    style: str,
    artifact_summary: str,
) -> str:
    shared = (
        f"Project: {project_title}. Material/topic: {material}. "
        f"Style: {style}. Use the Atomi artifacts only as scientific context. "
        "Do not invent quantitative axes, labels, phase boundaries, spectra, tables, or atom coordinates. "
        f"{artifact_summary}"
    )
    prompts = {
        "title_visual": (
            "A high-impact scientific conference title image for a computational materials talk. "
            "Show an elegant atomistic-to-thermodynamics visual metaphor with clean lighting, realistic depth, "
            "and no text. "
            + shared
        ),
        "workflow_overview": (
            "A polished workflow overview visual connecting atomistic simulation, thermodynamic analysis, "
            "transport prediction, and presentation-ready figures. Avoid readable text; use abstract panels "
            "and directional composition instead. "
            + shared
        ),
        "atomistic_mechanism": (
            "A scientifically restrained atomistic mechanism illustration suggesting defects, cation disorder, "
            "lattice vibrations, and heat transport in an oxide crystal. No labels or fake annotations. "
            + shared
        ),
        "thermo_story": (
            "A presentation background visual suggesting free energy landscapes, temperature dependence, "
            "and uncertainty-aware computational thermodynamics, without fake plots or numbers. "
            + shared
        ),
        "presentation_background": (
            "A subtle wide scientific presentation background with enough quiet space for real Atomi plots "
            "to be overlaid later. No text, no fake axes, no watermark. "
            + shared
        ),
    }
    return prompts[family]


def build_prompt_records(
    artifacts: list[Artifact],
    *,
    project_title: str,
    material: str,
    style: str,
    size: str,
    steps: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    summary = _artifact_summary(artifacts)
    project_slug = _slug(project_title)
    artifact_records = [
        {"path": str(artifact.path), "kind": artifact.kind, "role": artifact.role} for artifact in artifacts
    ]
    created_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for index, family in enumerate(DEFAULT_PROMPT_FAMILIES):
        records.append(
            {
                "schema_version": 1,
                "producer": "atomi-prismml-export",
                "created_at": created_at,
                "name": family,
                "prompt": _family_prompt(
                    family,
                    project_title=project_title,
                    material=material,
                    style=style,
                    artifact_summary=summary,
                ),
                "negative_prompt": "text, watermark, fake plot axes, fake labels, fake numbers, unreadable tables",
                "size": size,
                "steps": int(steps),
                "seed": int(seed_start) + index,
                "output": f"outputs/atomi_bridge/{project_slug}_{family}.png",
                "project_title": project_title,
                "material": material,
                "atomi_artifacts": artifact_records,
                "warnings": [
                    "Generated images are illustrative only; use Atomi plots as quantitative evidence.",
                    "Do not use generated images as replacements for real simulation outputs.",
                ],
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_prismml_batch(path: Path, records: list[dict[str, Any]], prismml_dir: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/bin/sh", "set -eu", ""]
    if prismml_dir is not None:
        lines.append(f"cd {shlex.quote(str(prismml_dir))}")
        lines.append("")
    lines.extend(
        [
            'BACKEND_PORT="${BACKEND_PORT:-8800}"',
            'SEND_SCRIPT="${SEND_SCRIPT:-./scripts/send_request.sh}"',
            "",
        ]
    )
    for record in records:
        lines.extend(
            [
                f"mkdir -p {shlex.quote(str(Path(record['output']).parent))}",
                '"${SEND_SCRIPT}" \\',
                f"  -p {shlex.quote(record['prompt'])} \\",
                f"  --size {shlex.quote(str(record['size']))} \\",
                f"  --seed {shlex.quote(str(record['seed']))} \\",
                f"  --steps {shlex.quote(str(record['steps']))} \\",
                f"  --output {shlex.quote(record['output'])}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True, help="Atomi result folder or file. Repeatable.")
    parser.add_argument("--project-title", required=True, help="Human-facing talk/project title.")
    parser.add_argument("--material", required=True, help="Material/topic phrase to include in prompts.")
    parser.add_argument(
        "--style",
        default="presentation-quality scientific visualization, restrained, high contrast, no fake text",
        help="Presentation style guidance.",
    )
    parser.add_argument("--size", default="1024x1024", help="Default image size. Default: 1024x1024.")
    parser.add_argument("--steps", type=int, default=4, help="PrismML generation steps. Default: 4.")
    parser.add_argument("--seed-start", type=int, default=4201, help="First deterministic seed. Default: 4201.")
    parser.add_argument("--limit-artifacts", type=int, default=80, help="Maximum Atomi artifacts to reference. Default: 80.")
    parser.add_argument("--include", action="append", default=[], help="Optional glob filter. Repeatable.")
    parser.add_argument("--exclude", action="append", default=[], help="Optional glob exclusion. Repeatable.")
    parser.add_argument("--out", type=Path, default=Path("prismml_prompts.jsonl"), help="JSONL manifest output.")
    parser.add_argument("--write-batch", type=Path, help="Optional shell batch that calls PrismML send_request.sh.")
    parser.add_argument("--prismml-dir", type=Path, help="PrismML checkout path for --write-batch.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    artifacts = scan_artifacts(args.run, limit=args.limit_artifacts, include=args.include, exclude=args.exclude)
    records = build_prompt_records(
        artifacts,
        project_title=args.project_title,
        material=args.material,
        style=args.style,
        size=args.size,
        steps=args.steps,
        seed_start=args.seed_start,
    )
    write_jsonl(args.out, records)
    print(f"Wrote PrismML prompt manifest: {args.out}")
    print(f"Referenced Atomi artifacts: {len(artifacts)}")
    if args.write_batch:
        write_prismml_batch(args.write_batch, records, args.prismml_dir)
        print(f"Wrote PrismML batch runner: {args.write_batch}")


if __name__ == "__main__":  # pragma: no cover
    main()
