from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path


ELEMENT_COLORS = {
    "H": "#d9d9d9",
    "Li": "#be8af0",
    "C": "#4d4d4d",
    "N": "#2f5fb3",
    "O": "#d73027",
    "F": "#66bd63",
    "Na": "#2c7fb8",
    "Mg": "#9ecae1",
    "Al": "#bdbdbd",
    "Si": "#fdae61",
    "P": "#e78ac3",
    "S": "#ffd92f",
    "Cl": "#35a66a",
    "K": "#7651bf",
    "Ca": "#80cdc1",
    "Ti": "#8da0cb",
    "Ga": "#a6761d",
    "La": "#1b9e77",
    "Gd": "#7570b3",
    "U": "#1b1b1b",
}

FALLBACK_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#9c755f",
    "#76b7b2",
    "#edc948",
]


@dataclass
class Atom:
    label: str
    x: float
    y: float
    z: float


@dataclass
class CellFrame:
    atoms: list[Atom]
    cell: list[list[float]]
    source: str
    frame_index: int | None = None


def parse_type_map(values: list[str] | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Type-map entries must be TYPE=ELEMENT, got {value!r}")
        left, right = value.split("=", 1)
        mapping[int(left)] = right.strip()
    return mapping


def cell_from_bounds(bounds: dict[str, tuple[float, float]]) -> list[list[float]]:
    return [
        [bounds["x"][1] - bounds["x"][0], 0.0, 0.0],
        [0.0, bounds["y"][1] - bounds["y"][0], 0.0],
        [0.0, 0.0, bounds["z"][1] - bounds["z"][0]],
    ]


def parse_cell_lengths(value: str | None) -> list[list[float]] | None:
    if not value:
        return None
    parts = [float(v) for v in re.split(r"[,\s]+", value.strip()) if v]
    if len(parts) != 3:
        raise ValueError("--cell expects three lengths, e.g. 12.5,12.5,18.8")
    return [[parts[0], 0.0, 0.0], [0.0, parts[1], 0.0], [0.0, 0.0, parts[2]]]


def parse_cell_vectors(values: list[str] | None) -> list[list[float]] | None:
    if not values:
        return None
    if len(values) != 3:
        raise ValueError("--cell-vector must be provided exactly three times")
    vectors = []
    for value in values:
        parts = [float(v) for v in re.split(r"[,\s]+", value.strip()) if v]
        if len(parts) != 3:
            raise ValueError(f"Cell vector must have three values, got {value!r}")
        vectors.append(parts)
    return vectors


def read_cp2k_cell(inp: Path | None) -> list[list[float]] | None:
    if inp is None or not inp.exists():
        return None
    text = inp.read_text(encoding="utf-8", errors="ignore")
    abc = re.search(r"^\s*ABC\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)", text, re.MULTILINE)
    if abc:
        a, b, c = (float(abc.group(i)) for i in range(1, 4))
        return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]
    vectors = []
    for axis in ("A", "B", "C"):
        match = re.search(
            rf"^\s*{axis}\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([0-9.Ee+-]+)",
            text,
            re.MULTILINE,
        )
        if not match:
            return None
        vectors.append([float(match.group(i)) for i in range(1, 4)])
    return vectors


def lattice_from_extxyz(comment: str) -> list[list[float]] | None:
    match = re.search(r'Lattice="([^"]+)"', comment)
    if not match:
        return None
    parts = [float(v) for v in match.group(1).split()]
    if len(parts) != 9:
        return None
    return [parts[0:3], parts[3:6], parts[6:9]]


def infer_cell_from_atoms(atoms: list[Atom], padding: float = 2.0) -> list[list[float]]:
    if not atoms:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    return [
        [max(xs) - min(xs) + padding, 0.0, 0.0],
        [0.0, max(ys) - min(ys) + padding, 0.0],
        [0.0, 0.0, max(zs) - min(zs) + padding],
    ]


def read_lammps_data(path: Path, type_map: dict[int, str]) -> CellFrame:
    bounds: dict[str, tuple[float, float]] = {}
    masses_comments: dict[int, str] = {}
    atoms_started = False
    masses_started = False
    atoms: list[Atom] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2] in {"xlo", "ylo", "zlo"}:
            bounds[parts[2][0]] = (float(parts[0]), float(parts[1]))
            continue
        if line.startswith("Masses"):
            masses_started = True
            atoms_started = False
            continue
        if line.startswith("Atoms"):
            atoms_started = True
            masses_started = False
            continue
        if masses_started and parts[0].isdigit() and "#" in raw:
            masses_comments[int(parts[0])] = raw.split("#", 1)[1].strip().split()[0]
            continue
        if atoms_started and parts[0].lstrip("-").isdigit() and len(parts) >= 5:
            typ = int(parts[1])
            label = type_map.get(typ) or masses_comments.get(typ) or f"T{typ}"
            atoms.append(Atom(label, float(parts[2]), float(parts[3]), float(parts[4])))
    if set(bounds) != {"x", "y", "z"}:
        raise ValueError(f"LAMMPS data file lacks orthorhombic x/y/z bounds: {path}")
    return CellFrame(atoms=atoms, cell=cell_from_bounds(bounds), source=str(path))


def split_lammps_dump_frames(lines: list[str]) -> list[tuple[dict[str, tuple[float, float]], list[str], list[list[str]]]]:
    frames = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError("Malformed LAMMPS dump: missing NUMBER OF ATOMS")
        natoms = int(lines[i + 1].strip())
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Malformed LAMMPS dump: missing BOX BOUNDS")
        bounds: dict[str, tuple[float, float]] = {}
        for axis in "xyz":
            lohi = lines[i + 1].split()
            bounds[axis] = (float(lohi[0]), float(lohi[1]))
            i += 1
        i += 1
        if i >= len(lines) or not lines[i].startswith("ITEM: ATOMS"):
            raise ValueError("Malformed LAMMPS dump: missing ATOMS")
        columns = lines[i].split()[2:]
        rows = [lines[i + 1 + j].split() for j in range(natoms)]
        frames.append((bounds, columns, rows))
        i += 1 + natoms
    return frames


def read_lammps_dump(path: Path, type_map: dict[int, str], frame_index: int) -> CellFrame:
    frames = split_lammps_dump_frames(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    if not frames:
        raise ValueError(f"No frames found in LAMMPS dump: {path}")
    idx = frame_index if frame_index >= 0 else len(frames) + frame_index
    bounds, columns, rows = frames[idx]
    col = {name: i for i, name in enumerate(columns)}
    for needed in ("type", "x", "y", "z"):
        if needed not in col:
            raise ValueError(f"LAMMPS dump frame lacks column {needed!r}; columns={columns}")
    atoms = [
        Atom(type_map.get(int(row[col["type"]]), f"T{row[col['type']]}"), float(row[col["x"]]), float(row[col["y"]]), float(row[col["z"]]))
        for row in rows
    ]
    return CellFrame(atoms=atoms, cell=cell_from_bounds(bounds), source=str(path), frame_index=idx)


def read_xyz(path: Path, frame_index: int, fallback_cell: list[list[float]] | None) -> CellFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: list[tuple[str, list[Atom]]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        natoms = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atoms = []
        for row in lines[i + 2 : i + 2 + natoms]:
            parts = row.split()
            if len(parts) >= 4:
                atoms.append(Atom(parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
        frames.append((comment, atoms))
        i += 2 + natoms
    if not frames:
        raise ValueError(f"No XYZ frames found: {path}")
    idx = frame_index if frame_index >= 0 else len(frames) + frame_index
    comment, atoms = frames[idx]
    cell = fallback_cell or lattice_from_extxyz(comment) or infer_cell_from_atoms(atoms)
    return CellFrame(atoms=atoms, cell=cell, source=str(path), frame_index=idx)


def cell_corners(cell: list[list[float]]) -> list[tuple[float, float, float]]:
    a, b, c = cell
    corners = []
    for ia, ib, ic in (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ):
        corners.append(
            (
                ia * a[0] + ib * b[0] + ic * c[0],
                ia * a[1] + ib * b[1] + ic * c[1],
                ia * a[2] + ib * b[2] + ic * c[2],
            )
        )
    return corners


def center_and_span(frame: CellFrame) -> tuple[tuple[float, float, float], float]:
    corners = cell_corners(frame.cell)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    zs = [p[2] for p in corners]
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    return center, span


def project(point: tuple[float, float, float], center: tuple[float, float, float], scale: float, origin: tuple[float, float]):
    x, y, z = point
    cx, cy, cz = center
    x -= cx
    y -= cy
    z -= cz
    px = (x - y) * 0.866
    py = (x + y) * 0.36 - z * 0.82
    depth = x + y + z
    return origin[0] + px * scale, origin[1] + py * scale, depth


def color_for(label: str, palette: dict[str, str]) -> str:
    if label in ELEMENT_COLORS:
        return ELEMENT_COLORS[label]
    if label not in palette:
        palette[label] = FALLBACK_COLORS[len(palette) % len(FALLBACK_COLORS)]
    return palette[label]


def render_panel(
    frame: CellFrame,
    title: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    global_span: float | None = None,
) -> str:
    center, span = center_and_span(frame)
    scale = min(width, height) / ((global_span or span) * 1.85)
    origin = (x0 + width / 2.0, y0 + height * 0.56)
    segments = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    corners = cell_corners(frame.cell)
    palette: dict[str, str] = {}
    parts = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 26:.1f}" text-anchor="middle" '
        f'font-size="21" font-family="Arial, sans-serif" font-weight="600">{escape(title)}</text>'
    ]
    for i, j in segments:
        p1 = project(corners[i], center, scale, origin)
        p2 = project(corners[j], center, scale, origin)
        parts.append(
            f'<line x1="{p1[0]:.2f}" y1="{p1[1]:.2f}" x2="{p2[0]:.2f}" y2="{p2[1]:.2f}" '
            'stroke="#222" stroke-width="1.4" opacity="0.82"/>'
        )
    projected = [
        (project((atom.x, atom.y, atom.z), center, scale, origin), atom)
        for atom in frame.atoms
    ]
    for (px, py, depth), atom in sorted(projected, key=lambda item: item[0][2]):
        radius = 3.5 if len(frame.atoms) > 300 else 4.3
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color_for(atom.label, palette)}" '
            'stroke="#ffffff" stroke-width="0.45" opacity="0.93"/>'
        )
    parts.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height - 12:.1f}" text-anchor="middle" '
        f'font-size="14" font-family="Arial, sans-serif" fill="#333">{len(frame.atoms)} atoms'
        f'{f", frame {frame.frame_index}" if frame.frame_index is not None else ""}</text>'
    )
    return "\n".join(parts)


def render_svg(frames: list[CellFrame], labels: list[str], title: str, output: Path) -> dict:
    panel_w = 640
    panel_h = 510
    margin = 40
    gap = 40
    width = margin * 2 + len(frames) * panel_w + (len(frames) - 1) * gap
    height = 680
    spans = [center_and_span(frame)[1] for frame in frames]
    global_span = max(spans)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-size="30" '
        f'font-family="Arial, sans-serif" font-weight="700">{escape(title)}</text>',
    ]
    all_labels = sorted({atom.label for frame in frames for atom in frame.atoms})
    for idx, frame in enumerate(frames):
        x0 = margin + idx * (panel_w + gap)
        parts.append(render_panel(frame, labels[idx], x0, 76, panel_w, panel_h, global_span=global_span))
    legend_x = width / 2 - min(len(all_labels), 8) * 42
    y = 622
    palette: dict[str, str] = {}
    for idx, label in enumerate(all_labels[:16]):
        x = legend_x + idx * 84
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{color_for(label, palette)}" stroke="#fff" stroke-width="0.6"/>')
        parts.append(f'<text x="{x + 16:.1f}" y="{y + 5}" font-size="16" font-family="Arial, sans-serif">{escape(label)}</text>')
    parts.append(
        f'<text x="{width / 2:.1f}" y="656" text-anchor="middle" font-size="14" '
        'font-family="Arial, sans-serif" fill="#333">Same projection and scaling are used across panels; black edges show periodic cell bounds.</text>'
    )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return {
        "output": str(output),
        "title": title,
        "sources": [frame.source for frame in frames],
        "natoms": [len(frame.atoms) for frame in frames],
        "labels": labels,
    }


def load_frame(args: argparse.Namespace, source: Path, cell_override: list[list[float]] | None) -> CellFrame:
    if args.format == "lammps-data":
        return read_lammps_data(source, parse_type_map(args.type_map))
    if args.format == "lammps-dump":
        return read_lammps_dump(source, parse_type_map(args.type_map), args.frame)
    if args.format == "xyz":
        return read_xyz(source, args.frame, cell_override)
    suffix = source.suffix.lower()
    if suffix in {".data", ".lmp"}:
        return read_lammps_data(source, parse_type_map(args.type_map))
    if suffix in {".dump", ".lammpstrj"}:
        return read_lammps_dump(source, parse_type_map(args.type_map), args.frame)
    if suffix in {".xyz", ".extxyz"}:
        return read_xyz(source, args.frame, cell_override)
    raise ValueError(f"Could not infer format for {source}; use --format.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render generic LAMMPS/AIMD cells or trajectory frames as an SVG.")
    parser.add_argument("sources", type=Path, nargs="+", help="LAMMPS data/dump or XYZ/extxyz sources.")
    parser.add_argument("--format", choices=("auto", "lammps-data", "lammps-dump", "xyz"), default="auto")
    parser.add_argument("--label", action="append", help="Panel label. Repeat once per source.")
    parser.add_argument("--title", default="Simulation Cell Visualization")
    parser.add_argument("--out", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--summary-json", type=Path, help="Optional JSON summary output.")
    parser.add_argument("--frame", type=int, default=-1, help="Frame index for trajectory inputs; negative counts from end.")
    parser.add_argument("--type-map", action="append", help="LAMMPS type map entry, e.g. --type-map 1=K --type-map 2=Cl.")
    parser.add_argument("--cell", help="Cell lengths for XYZ/AIMD input, e.g. 12.58,12.58,18.87.")
    parser.add_argument("--cell-vector", action="append", help="Cell vector for XYZ/AIMD input. Repeat three times.")
    parser.add_argument("--cp2k-inp", type=Path, help="Read &CELL from a CP2K input for XYZ/AIMD input.")
    args = parser.parse_args(argv)

    cell_override = parse_cell_vectors(args.cell_vector) or parse_cell_lengths(args.cell) or read_cp2k_cell(args.cp2k_inp)
    frames = [load_frame(args, source, cell_override) for source in args.sources]
    labels = args.label or [Path(frame.source).stem for frame in frames]
    if len(labels) != len(frames):
        raise ValueError("--label must be repeated once per source")
    summary = render_svg(frames, labels, args.title, args.out)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
