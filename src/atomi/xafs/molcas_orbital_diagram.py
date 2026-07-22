"""Figure-style Molcas MO/orbital transition diagrams for XANES analysis.

The 2021 Ce L3-edge paper combines orbital pictures with a compact
metal-ligand orbital splitting diagram.  This module keeps that idea
reproducible for Atomi: numeric MO/transition data come from standard CSVs,
while orbital isosurface images can be supplied later from eXatomic, Grid_it,
Pegamoid, Molden, or another viewer through a small manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "atomi.xafs.molcas_orbital_diagram.v1"


@dataclass(frozen=True)
class Level:
    label: str
    block: str
    energy_ev: float
    occupation: float | None
    character: str
    color: str


@dataclass(frozen=True)
class TransitionArrow:
    label: str
    source_label: str
    target_label: str
    energy_ev: float | None
    oscillator_strength: float
    state_from: str
    state_to: str


@dataclass(frozen=True)
class OrbitalImage:
    label: str
    image: str
    description: str


def _svg_escape(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _short_label(text: object, limit: int = 58) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 1)].rstrip() + "..."


def _as_float(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _row_value(row: dict[str, str], *candidates: str) -> str:
    lookup = {key.strip().lower(): key for key in row}
    for candidate in candidates:
        found = lookup.get(candidate.lower())
        if found is not None:
            return row.get(found, "")
    return ""


def _edge_matches(level: Level, edge: str) -> bool:
    edge_norm = edge.strip().lower()
    if not edge_norm or edge_norm in {"all", "l23", "both"}:
        return True
    if level.block.strip().lower() == edge_norm:
        return True
    if level.block.strip().lower() == "core" and edge_norm in level.label.lower():
        return True
    return False


def read_levels(path: Path, *, edge: str = "L3") -> list[Level]:
    levels: list[Level] = []
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = _row_value(row, "label", "name", "orbital", "state")
            energy = _as_float(_row_value(row, "energy_ev", "energy", "relative_energy_ev"))
            if not label or energy is None:
                continue
            level = Level(
                label=label,
                block=_row_value(row, "block", "group", "edge") or "levels",
                energy_ev=energy,
                occupation=_as_float(_row_value(row, "occupation", "occ", "occupancy")),
                character=_row_value(row, "character", "assignment", "description", "note"),
                color=_row_value(row, "color", "colour") or "#3b6ea8",
            )
            if _edge_matches(level, edge):
                levels.append(level)
    if not levels:
        raise ValueError(f"No MO levels were read from {path} for edge={edge!r}")
    return levels


def read_arrows(path: Path | None) -> list[TransitionArrow]:
    if path is None:
        return []
    arrows: list[TransitionArrow] = []
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = _row_value(row, "source_label", "from_label", "source", "initial_label")
            target = _row_value(row, "target_label", "to_label", "target", "final_label")
            strength = _as_float(_row_value(row, "oscillator_strength", "strength", "intensity", "fosc"), 0.0)
            if not source or not target or strength is None or strength <= 0:
                continue
            arrows.append(
                TransitionArrow(
                    label=_row_value(row, "label", "transition", "state_label") or target,
                    source_label=source,
                    target_label=target,
                    energy_ev=_as_float(_row_value(row, "energy_ev", "transition_energy_ev", "energy")),
                    oscillator_strength=strength,
                    state_from=_row_value(row, "state_from", "from") or "",
                    state_to=_row_value(row, "state_to", "to") or "",
                )
            )
    return arrows


def read_orbital_manifest(path: Path | None) -> list[OrbitalImage]:
    if path is None:
        return []
    path = path.expanduser()
    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = list(payload.get("orbitals") or payload.get("images") or [])
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    else:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    orbitals: list[OrbitalImage] = []
    for idx, row in enumerate(rows, start=1):
        image = str(row.get("image") or row.get("path") or row.get("file") or "").strip()
        if image and not Path(image).expanduser().is_absolute():
            image = str((path.parent / image).resolve())
        label = str(row.get("label") or row.get("orbital") or row.get("name") or f"orbital {idx}").strip()
        description = str(row.get("description") or row.get("assignment") or row.get("character") or "").strip()
        orbitals.append(OrbitalImage(label=label, image=image, description=description))
    return orbitals


def _block_order(levels: list[Level]) -> list[str]:
    order: list[str] = []
    for preferred in ("core", "L3", "L2"):
        for level in levels:
            if level.block == preferred and preferred not in order:
                order.append(preferred)
    for level in levels:
        if level.block not in order:
            order.append(level.block)
    return order


def _filtered_arrows(levels: list[Level], arrows: list[TransitionArrow], max_transitions: int) -> list[TransitionArrow]:
    labels = {level.label for level in levels}
    keep = [arrow for arrow in arrows if arrow.source_label in labels and arrow.target_label in labels]
    keep.sort(key=lambda item: item.oscillator_strength, reverse=True)
    if max_transitions > 0:
        keep = keep[:max_transitions]
    return keep


def _level_coordinates(
    levels: list[Level],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> tuple[dict[str, tuple[float, float]], list[str], tuple[float, float]]:
    blocks = _block_order(levels)
    energies = [level.energy_ev for level in levels]
    emin, emax = min(energies), max(energies)
    if math.isclose(emin, emax):
        emin -= 1.0
        emax += 1.0

    def sy(energy: float) -> float:
        return y1 - (energy - emin) / (emax - emin) * (y1 - y0)

    coords: dict[str, tuple[float, float]] = {}
    col_w = (x1 - x0) / max(1, len(blocks))
    for block_idx, block in enumerate(blocks):
        block_levels = [level for level in levels if level.block == block]
        x = x0 + block_w_center(col_w, block_idx)
        spread = min(36.0, max(0.0, (col_w - 122.0) / max(1, len(block_levels) - 1)))
        for idx, level in enumerate(block_levels):
            offset = (idx - (len(block_levels) - 1) / 2.0) * spread
            coords[level.label] = (x + offset, sy(level.energy_ev))
    return coords, blocks, (emin, emax)


def block_w_center(col_w: float, block_idx: int) -> float:
    return col_w * (block_idx + 0.5)


def write_diagram_svg(
    path: Path,
    *,
    levels: list[Level],
    arrows: list[TransitionArrow],
    orbitals: list[OrbitalImage],
    title: str,
    edge: str,
    footnote: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1560, 930
    orbital_x, orbital_y = 42, 84
    orbital_w, orbital_h = 470, 730
    plot_x0, plot_x1 = 590, 1168
    plot_y0, plot_y1 = 128, 790
    key_x, key_y = 1214, 164
    coords, blocks, energy_bounds = _level_coordinates(levels, x0=plot_x0, x1=plot_x1, y0=plot_y0, y1=plot_y1)
    arrows = _filtered_arrows(levels, arrows, max_transitions=len(arrows))
    max_strength = max((arrow.oscillator_strength for arrow in arrows), default=1.0)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 z" fill="#7b3030"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="42" y="42" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="700" fill="#252b33">{_svg_escape(title)}</text>',
        f'<text x="42" y="68" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#59636f">Edge/filter: {_svg_escape(edge)}. Energies are relative within the supplied postanalysis diagram table.</text>',
        f'<rect x="{orbital_x}" y="{orbital_y}" width="{orbital_w}" height="{orbital_h}" rx="10" fill="#f7f8fa" stroke="#d4dbe5"/>',
        f'<text x="{orbital_x + 18}" y="{orbital_y + 32}" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#252b33">Orbital / NTO image panel</text>',
    ]
    slots = orbitals if orbitals else [
        OrbitalImage("Ce 4f / ligand-hole active pair", "", "Supply eXatomic/Grid_it/Pegamoid image via --orbital-manifest."),
        OrbitalImage("Ce 5d/6s acceptor manifold", "", "Use Molden/Cube/Grid/RasOrb exports from the matching MOLCAS block."),
        OrbitalImage("Localized O 2p ligand combination", "", "The current r7 local bundle has CSV/JSON diagrams but no orbital grid files."),
    ]
    slot_h = min(202, (orbital_h - 76) / max(1, len(slots)))
    for idx, orbital in enumerate(slots[:4]):
        y = orbital_y + 54 + idx * slot_h
        lines.append(f'<rect x="{orbital_x + 18}" y="{y:.2f}" width="{orbital_w - 36}" height="{slot_h - 14:.2f}" rx="8" fill="#ffffff" stroke="#dce2ea"/>')
        if orbital.image:
            lines.append(
                f'<image href="{_svg_escape(orbital.image)}" x="{orbital_x + 30}" y="{y + 12:.2f}" '
                f'width="156" height="{slot_h - 48:.2f}" preserveAspectRatio="xMidYMid meet"/>'
            )
            label_x = orbital_x + 204
        else:
            lines.append(f'<rect x="{orbital_x + 30}" y="{y + 16:.2f}" width="150" height="{slot_h - 54:.2f}" rx="8" fill="#f1f4f8" stroke="#cfd7e2" stroke-dasharray="6 5"/>')
            lines.append(f'<text x="{orbital_x + 105}" y="{y + slot_h / 2 - 6:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="12" text-anchor="middle" fill="#657282">image slot</text>')
            label_x = orbital_x + 204
        lines.append(f'<text x="{label_x}" y="{y + 30:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#2f3640">{_svg_escape(_short_label(orbital.label, 34))}</text>')
        lines.append(f'<text x="{label_x}" y="{y + 52:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="11.5" fill="#59636f">{_svg_escape(_short_label(orbital.description, 44))}</text>')

    lines.extend(
        [
            f'<text x="{plot_x0}" y="98" font-family="Arial, Helvetica, sans-serif" font-size="16" font-weight="700" fill="#252b33">MO / ligand-field and transition diagram</text>',
            f'<line x1="{plot_x0}" x2="{plot_x1}" y1="{plot_y1}" y2="{plot_y1}" stroke="#313640" stroke-width="1.2"/>',
            f'<line x1="{plot_x0}" x2="{plot_x0}" y1="{plot_y0}" y2="{plot_y1}" stroke="#313640" stroke-width="1.2"/>',
            f'<text x="{plot_x0 - 48}" y="{(plot_y0 + plot_y1) / 2:.1f}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="600" text-anchor="middle" fill="#2f3640" transform="rotate(-90 {plot_x0 - 48} {(plot_y0 + plot_y1) / 2:.1f})">Relative energy (eV)</text>',
        ]
    )
    emin, emax = energy_bounds
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y1 - frac * (plot_y1 - plot_y0)
        value = emin + frac * (emax - emin)
        lines.append(f'<line x1="{plot_x0}" x2="{plot_x1}" y1="{y:.2f}" y2="{y:.2f}" stroke="#edf0f3" stroke-width="1"/>')
        lines.append(f'<text x="{plot_x0 - 10}" y="{y + 4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="11" text-anchor="end" fill="#59636f">{value:.3g}</text>')
    col_w = (plot_x1 - plot_x0) / max(1, len(blocks))
    for block_idx, block in enumerate(blocks):
        x = plot_x0 + block_w_center(col_w, block_idx)
        lines.append(f'<text x="{x:.2f}" y="{plot_y1 + 34}" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" text-anchor="middle" fill="#2f3640">{_svg_escape(block)}</text>')
        lines.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_y0}" y2="{plot_y1}" stroke="#f3f5f7" stroke-width="1"/>')
    for level in levels:
        x, y = coords[level.label]
        occ = level.occupation if level.occupation is not None else 0.0
        half = 45 if level.block == "core" else 52
        lines.append(f'<line x1="{x - half:.2f}" x2="{x + half:.2f}" y1="{y:.2f}" y2="{y:.2f}" stroke="{_svg_escape(level.color)}" stroke-width="3.4" stroke-linecap="round"/>')
        if occ >= 1.5:
            lines.append(f'<circle cx="{x - 10:.2f}" cy="{y - 8:.2f}" r="3.5" fill="{_svg_escape(level.color)}"/>')
            lines.append(f'<circle cx="{x + 10:.2f}" cy="{y - 8:.2f}" r="3.5" fill="{_svg_escape(level.color)}"/>')
        elif occ >= 0.5:
            lines.append(f'<circle cx="{x:.2f}" cy="{y - 8:.2f}" r="3.5" fill="{_svg_escape(level.color)}"/>')
        label_anchor = "end" if level.block == "core" else "start"
        label_x = x - half - 8 if level.block == "core" else x + half + 8
        lines.append(f'<text x="{label_x:.2f}" y="{y + 4:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="11.5" text-anchor="{label_anchor}" fill="#2f3640">{_svg_escape(_short_label(level.label, 30))}</text>')
    selected = _filtered_arrows(levels, arrows, max_transitions=len(arrows))
    lines.append(f'<text x="{key_x}" y="{key_y - 30}" font-family="Arial, Helvetica, sans-serif" font-size="14" font-weight="700" fill="#252b33">Transition key</text>')
    for idx, arrow in enumerate(selected, start=1):
        x0, y0 = coords[arrow.source_label]
        x1, y1 = coords[arrow.target_label]
        width_scale = 1.2 + 4.2 * arrow.oscillator_strength / max_strength
        mid_x = (x0 + x1) / 2.0 + 24
        points = f"M{x0 + 44:.2f},{y0:.2f} C{mid_x:.2f},{y0:.2f} {mid_x:.2f},{y1:.2f} {x1 - 50:.2f},{y1:.2f}"
        lines.append(f'<path d="{points}" fill="none" stroke="#7b3030" stroke-width="{width_scale:.2f}" opacity="0.65" marker-end="url(#arrowhead)"/>')
        tag_x = x1 - 64
        tag_y = y1 - 12
        lines.append(f'<circle cx="{tag_x:.2f}" cy="{tag_y:.2f}" r="9" fill="#ffffff" stroke="#7b3030" stroke-width="1.2"/>')
        lines.append(f'<text x="{tag_x:.2f}" y="{tag_y + 3.5:.2f}" font-family="Arial, Helvetica, sans-serif" font-size="10" font-weight="700" text-anchor="middle" fill="#7b3030">{idx}</text>')
        e_text = "" if arrow.energy_ev is None else f", {arrow.energy_ev:.2f} eV"
        state_text = f"SO {arrow.state_from}->{arrow.state_to}" if arrow.state_from or arrow.state_to else arrow.label
        lines.append(f'<text x="{key_x}" y="{key_y + 18 * (idx - 1):.2f}" font-family="Arial, Helvetica, sans-serif" font-size="11.2" fill="#2f3640"><tspan font-weight="700">{idx}.</tspan> {_svg_escape(_short_label(state_text, 20))}: f={arrow.oscillator_strength:.3g}{_svg_escape(e_text)}</text>')
    if not selected:
        lines.append(f'<text x="{key_x}" y="{key_y}" font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#59636f">No transition arrows matched the supplied levels.</text>')
    if footnote:
        lines.append(f'<text x="42" y="874" font-family="Arial, Helvetica, sans-serif" font-size="11.5" fill="#59636f">{_svg_escape(_short_label(footnote, 210))}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_write_png(path: Path, *, levels: list[Level], arrows: list[TransitionArrow], title: str, edge: str, dpi: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    coords, blocks, _ = _level_coordinates(levels, x0=0.0, x1=float(max(1, len(_block_order(levels)))), y0=0.0, y1=1.0)
    fig, (ax_left, ax) = plt.subplots(1, 2, figsize=(12.5, 6.8), gridspec_kw={"width_ratios": [1.0, 1.55]})
    ax_left.axis("off")
    ax_left.set_title("Orbital/NTO images", loc="left", fontsize=12, fontweight="bold")
    for idx, label in enumerate(("active pair", "acceptor", "ligand orbital")):
        y = 0.78 - idx * 0.28
        ax_left.add_patch(plt.Rectangle((0.05, y), 0.9, 0.20, fill=False, ls="--", ec="#9aa7b5"))
        ax_left.text(0.5, y + 0.10, f"{label}\nimage slot", ha="center", va="center", fontsize=9, color="#59636f")
    ax.set_title(f"MO/transition diagram ({edge})", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative energy (eV)")
    block_to_x = {block: idx for idx, block in enumerate(blocks)}
    for level in levels:
        x = block_to_x[level.block]
        ax.hlines(level.energy_ev, x - 0.28, x + 0.28, colors=level.color, linewidth=2.6)
        ax.text(x + 0.31, level.energy_ev, _short_label(level.label, 20), fontsize=8, va="center")
    plotted_arrows = _filtered_arrows(levels, arrows, max_transitions=12)
    max_strength = max((arrow.oscillator_strength for arrow in plotted_arrows), default=1.0)
    label_to_level = {level.label: level for level in levels}
    for idx, arrow in enumerate(plotted_arrows, start=1):
        source = label_to_level[arrow.source_label]
        target = label_to_level[arrow.target_label]
        x0, x1 = block_to_x[source.block] + 0.25, block_to_x[target.block] - 0.25
        ax.annotate(
            "",
            xy=(x1, target.energy_ev),
            xytext=(x0, source.energy_ev),
            arrowprops={
                "arrowstyle": "->",
                "lw": 0.8 + 3.0 * arrow.oscillator_strength / max_strength,
                "color": "#7b3030",
                "alpha": 0.65,
            },
        )
        ax.text(x1, target.energy_ev, str(idx), ha="right", va="bottom", fontsize=8, color="#7b3030")
    ax.set_xticks(range(len(blocks)), blocks)
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    levels = read_levels(args.levels_csv, edge=args.edge)
    arrows = _filtered_arrows(levels, read_arrows(args.arrows_csv), args.max_transitions)
    if args.min_relative_strength > 0 and arrows:
        max_strength = max(arrow.oscillator_strength for arrow in arrows)
        arrows = [arrow for arrow in arrows if arrow.oscillator_strength >= args.min_relative_strength * max_strength]
    orbitals = read_orbital_manifest(args.orbital_manifest)
    title = args.title or "Molcas XANES orbital/MO transition diagram"
    footnote = args.footnote
    if not footnote and not orbitals:
        footnote = (
            "Orbital isosurfaces were not embedded because no orbital-image manifest was supplied; "
            "export Molden/Cube/Grid_it/eXatomic/Pegamoid images from the matching MOLCAS block and pass --orbital-manifest."
        )
    write_diagram_svg(args.out_svg, levels=levels, arrows=arrows, orbitals=orbitals, title=title, edge=args.edge, footnote=footnote)
    png_written = False
    if args.out_png:
        png_written = maybe_write_png(args.out_png, levels=levels, arrows=arrows, title=title, edge=args.edge, dpi=args.dpi)
    summary = {
        "schema": SCHEMA,
        "levels_csv": str(args.levels_csv),
        "arrows_csv": str(args.arrows_csv) if args.arrows_csv else None,
        "orbital_manifest": str(args.orbital_manifest) if args.orbital_manifest else None,
        "edge": args.edge,
        "n_levels": len(levels),
        "n_arrows": len(arrows),
        "n_orbital_images": len(orbitals),
        "out_svg": str(args.out_svg),
        "out_png": str(args.out_png) if png_written else None,
        "notes": [
            "MO/transition diagram is generated from CSV postanalysis data.",
            "Orbital panels are populated only when a manifest of rendered orbital images is supplied.",
        ],
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels-csv", type=Path, required=True, help="CSV with label, block, energy_ev, occupation, character, color columns.")
    parser.add_argument("--arrows-csv", type=Path, help="Optional CSV with source_label, target_label, oscillator_strength columns.")
    parser.add_argument("--orbital-manifest", type=Path, help="Optional CSV/JSON list of rendered orbital images.")
    parser.add_argument("--edge", default="L3", help="Edge/block filter, e.g. L3, L2, L23, all.")
    parser.add_argument("--max-transitions", type=int, default=12)
    parser.add_argument("--min-relative-strength", type=float, default=0.0)
    parser.add_argument("--out-svg", type=Path, required=True)
    parser.add_argument("--out-png", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--footnote", default="")
    parser.add_argument("--dpi", type=int, default=260)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
