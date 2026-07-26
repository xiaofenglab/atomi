"""Read and render OpenMolcas GRID_IT ASCII orbital grids.

The GRID_IT values are treated as the scientific source of truth. Rendering
uses only NumPy and Matplotlib, so batch/headless postanalysis does not require
the Qt/VTK stack used by interactive viewers such as Pegamoid.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np


SCHEMA_ORBITAL_GRID = "atomi.molcas_orbital_grid.v1"
GRID_LABEL_RE = re.compile(
    r"GridName=\s+(?P<symmetry>\d+)\s+(?P<orbital>\d+)\s+"
    r"(?P<energy>[-+0-9.EeDd]+)\s+\((?P<occupation>[-+0-9.EeDd]+)\)"
)


@dataclass(frozen=True)
class OrbitalLabel:
    grid_index: int
    symmetry: int | None
    orbital: int | None
    energy: float | None
    occupation: float | None
    label: str

    @property
    def key(self) -> tuple[int, int] | None:
        if self.symmetry is None or self.orbital is None:
            return None
        return self.symmetry, self.orbital


@dataclass(frozen=True)
class GridMetadata:
    path: Path
    title: str
    atoms: tuple[dict[str, Any], ...]
    n_grids: int
    n_points: int
    block_size: int
    dimensions: tuple[int, int, int]
    origin: np.ndarray
    axes: np.ndarray
    labels: tuple[OrbitalLabel, ...]


def _fortran_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _open_text(path: Path) -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _after_equals(line: str) -> str:
    return line.split("=", 1)[1].strip()


def read_grid_metadata(path: Path) -> GridMetadata:
    """Read a GRID_IT ASCII header without loading the volumetric data."""

    path = path.expanduser().resolve()
    with _open_text(path) as handle:
        if not handle.readline():
            raise ValueError(f"Empty OpenMolcas grid file: {path}")
        title = handle.readline().strip()
        natom_line = handle.readline()
        if "Natom=" not in natom_line:
            raise ValueError(f"Expected Natom header in {path}")
        natom = int(_after_equals(natom_line))
        atoms: list[dict[str, Any]] = []
        for _ in range(natom):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"Invalid atom row in {path}: {' '.join(fields)}")
            atoms.append(
                {
                    "label": fields[0],
                    "xyz_bohr": [_fortran_float(value) for value in fields[1:4]],
                }
            )

        n_grids = 0
        n_points = 0
        block_size = 0
        dimensions = (0, 0, 0)
        origin = np.zeros(3, dtype=float)
        axes = np.zeros((3, 3), dtype=float)
        labels: list[OrbitalLabel] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"GRID_IT data section is missing from {path}")
            stripped = line.strip()
            if stripped.startswith("Title="):
                break
            if stripped.startswith("N_of_Grids="):
                n_grids = int(_after_equals(stripped))
            elif stripped.startswith("N_of_Points="):
                n_points = int(_after_equals(stripped))
            elif stripped.startswith("Block_Size="):
                block_size = int(_after_equals(stripped))
            elif stripped.startswith("Net="):
                net = tuple(int(value) for value in _after_equals(stripped).split())
                dimensions = tuple(value + 1 for value in net)
            elif stripped.startswith("Origin="):
                origin = np.asarray([_fortran_float(value) for value in _after_equals(stripped).split()], dtype=float)
            elif stripped.startswith("Axis_"):
                axis_index = int(stripped.split("=", 1)[0].split("_")[1]) - 1
                axes[axis_index] = [_fortran_float(value) for value in _after_equals(stripped).split()]
            elif stripped.startswith("GridName="):
                match = GRID_LABEL_RE.search(stripped)
                if match:
                    symmetry = int(match.group("symmetry"))
                    orbital = int(match.group("orbital"))
                    energy = _fortran_float(match.group("energy"))
                    occupation = _fortran_float(match.group("occupation"))
                    label = f"sym{symmetry}_orb{orbital}"
                else:
                    symmetry = None
                    orbital = None
                    energy = None
                    occupation = None
                    label = stripped.split("=", 1)[1].strip().replace(" ", "_").lower()
                labels.append(
                    OrbitalLabel(
                        grid_index=len(labels),
                        symmetry=symmetry,
                        orbital=orbital,
                        energy=energy,
                        occupation=occupation,
                        label=label,
                    )
                )

    expected_points = int(np.prod(dimensions))
    if n_grids <= 0 or len(labels) != n_grids:
        raise ValueError(f"GRID_IT header reports {n_grids} grids but defines {len(labels)} labels")
    if expected_points <= 0 or expected_points != n_points:
        raise ValueError(f"GRID_IT dimensions {dimensions} imply {expected_points} points, not {n_points}")
    if block_size <= 0:
        raise ValueError("GRID_IT Block_Size must be positive")
    return GridMetadata(
        path=path,
        title=title,
        atoms=tuple(atoms),
        n_grids=n_grids,
        n_points=n_points,
        block_size=block_size,
        dimensions=dimensions,
        origin=origin,
        axes=axes,
        labels=tuple(labels),
    )


def _seek_data(handle: TextIO) -> str:
    while True:
        line = handle.readline()
        if not line:
            raise ValueError("GRID_IT data section is missing")
        if line.strip().startswith("Title="):
            return line


def read_grid_volumes(
    metadata: GridMetadata,
    selected_indices: set[int],
) -> dict[int, np.ndarray]:
    """Load selected GRID_IT volumes by zero-based grid index."""

    invalid = sorted(index for index in selected_indices if index < 0 or index >= metadata.n_grids)
    if invalid:
        raise ValueError(f"GRID_IT grid indices out of range: {invalid}")
    flat = {index: np.empty(metadata.n_points, dtype=float) for index in selected_indices}
    with _open_text(metadata.path) as handle:
        pending_title = _seek_data(handle)
        offset = 0
        while offset < metadata.n_points:
            block_length = min(metadata.block_size, metadata.n_points - offset)
            for grid_index in range(metadata.n_grids):
                title_line = pending_title if pending_title is not None else handle.readline()
                pending_title = None
                if not title_line or not title_line.strip().startswith("Title="):
                    raise ValueError(
                        f"Expected GRID_IT Title row at point offset {offset}, grid {grid_index}"
                    )
                target = flat.get(grid_index)
                if target is None:
                    for _ in range(block_length):
                        if not handle.readline():
                            raise ValueError("Unexpected end of GRID_IT data")
                else:
                    values = np.fromiter(
                        (_fortran_float(handle.readline().strip()) for _ in range(block_length)),
                        dtype=float,
                        count=block_length,
                    )
                    if values.size != block_length:
                        raise ValueError("Unexpected end of selected GRID_IT orbital data")
                    target[offset : offset + block_length] = values
            offset += block_length
    return {index: values.reshape(metadata.dimensions) for index, values in flat.items()}


def parse_orbital_specs(specs: list[str], labels: tuple[OrbitalLabel, ...]) -> list[int]:
    """Resolve repeatable ``SYM:ORB`` or ``SYM:FIRST-LAST`` selections."""

    available = {label.key: label.grid_index for label in labels if label.key is not None}
    if not specs:
        return [label.grid_index for label in labels if label.key is not None]
    requested: list[tuple[int, int]] = []
    for spec in specs:
        for token in spec.replace(",", " ").split():
            if ":" not in token:
                raise ValueError(f"Orbital selection must be SYM:ORB or SYM:FIRST-LAST, not {token!r}")
            symmetry_text, orbital_text = token.split(":", 1)
            symmetry = int(symmetry_text)
            if "-" in orbital_text:
                first_text, last_text = orbital_text.split("-", 1)
                first, last = int(first_text), int(last_text)
                if last < first:
                    raise ValueError(f"Reversed orbital range: {token!r}")
                requested.extend((symmetry, orbital) for orbital in range(first, last + 1))
            else:
                requested.append((symmetry, int(orbital_text)))
    missing = [f"{sym}:{orb}" for sym, orb in requested if (sym, orb) not in available]
    if missing:
        choices = ", ".join(f"{sym}:{orb}" for sym, orb in sorted(available))
        raise ValueError(f"Requested orbital(s) not present: {', '.join(missing)}. Available: {choices}")
    out: list[int] = []
    for key in requested:
        index = available[key]
        if index not in out:
            out.append(index)
    return out


def _element(label: str) -> str:
    match = re.match(r"([A-Za-z]+)", label)
    return match.group(1).capitalize() if match else label


def _grid_coordinates(metadata: GridMetadata, indices: np.ndarray) -> np.ndarray:
    denominators = np.maximum(np.asarray(metadata.dimensions, dtype=float) - 1.0, 1.0)
    fractions = indices / denominators
    return metadata.origin + fractions @ metadata.axes


def _phase_surface_points(
    values: np.ndarray,
    isovalue: float,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(values)
    band = (absolute >= isovalue) & (absolute <= isovalue * 1.12)
    if np.count_nonzero(band) < 200:
        band = absolute >= isovalue
    indices = np.argwhere(band)
    if indices.shape[0] > max_points:
        take = np.linspace(0, indices.shape[0] - 1, max_points, dtype=int)
        indices = indices[take]
    phases = np.sign(values[tuple(indices.T)])
    return indices, phases


def _draw_cluster(ax: Any, metadata: GridMetadata) -> None:
    xyz = np.asarray([atom["xyz_bohr"] for atom in metadata.atoms], dtype=float)
    for i, atom in enumerate(metadata.atoms):
        element = _element(atom["label"])
        color = {"U": "#25283d", "O": "#b8c4ce"}.get(element, "#7b8794")
        size = 62 if element == "U" else 25
        ax.scatter(*xyz[i], s=size, c=color, edgecolors="white", linewidths=0.45, depthshade=True)
    uranium = [i for i, atom in enumerate(metadata.atoms) if _element(atom["label"]) == "U"]
    oxygen = [i for i, atom in enumerate(metadata.atoms) if _element(atom["label"]) == "O"]
    for i in uranium:
        for j in oxygen:
            if np.linalg.norm(xyz[i] - xyz[j]) <= 5.2:
                ax.plot(*zip(xyz[i], xyz[j]), color="#88929c", lw=0.45, alpha=0.55)


def _set_axes(ax: Any, metadata: GridMetadata) -> None:
    corners = np.asarray(
        [
            metadata.origin,
            metadata.origin + metadata.axes[0],
            metadata.origin + metadata.axes[1],
            metadata.origin + metadata.axes[2],
            metadata.origin + metadata.axes.sum(axis=0),
        ]
    )
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    center = (lower + upper) / 2
    radius = max(upper - lower) / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=40)


def _draw_orbital(
    ax: Any,
    metadata: GridMetadata,
    label: OrbitalLabel,
    values: np.ndarray,
    *,
    isofraction: float,
    absolute_isovalue: float | None,
    max_points: int,
    center_mask: np.ndarray,
    min_center_fraction: float,
) -> dict[str, Any]:
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    max_abs = float(np.max(np.abs(values)))
    if not np.isfinite(values).all() or max_abs <= 0.0:
        raise ValueError(f"Orbital {label.label} contains non-finite or zero-only values")
    isovalue = float(absolute_isovalue) if absolute_isovalue is not None else isofraction * max_abs
    if isovalue <= 0.0 or isovalue >= max_abs:
        raise ValueError(f"Invalid isovalue {isovalue:g} for orbital {label.label} with max_abs {max_abs:g}")
    indices, phases = _phase_surface_points(values, isovalue, max_points=max_points)
    coordinates = _grid_coordinates(metadata, indices)
    positive = phases > 0
    negative = phases < 0
    if not positive.any() or not negative.any():
        raise ValueError(f"Orbital {label.label} does not show both phase signs at isovalue {isovalue:g}")
    squared = np.square(values)
    total_weight = float(np.sum(squared))
    center_fraction = float(np.sum(squared[center_mask]) / total_weight)
    localization_guard_pass = center_fraction >= min_center_fraction
    ax.scatter(
        coordinates[negative, 0],
        coordinates[negative, 1],
        coordinates[negative, 2],
        s=2.2,
        c="#2b59c3",
        alpha=0.64,
        linewidths=0,
        depthshade=False,
    )
    ax.scatter(
        coordinates[positive, 0],
        coordinates[positive, 1],
        coordinates[positive, 2],
        s=2.2,
        c="#d1495b",
        alpha=0.64,
        linewidths=0,
        depthshade=False,
    )
    _draw_cluster(ax, metadata)
    _set_axes(ax, metadata)
    occupation = "n/a" if label.occupation is None else f"{label.occupation:.2f}"
    ax.set_title(
        f"sym {label.symmetry}, orbital {label.orbital}\nocc {occupation} | iso {isovalue:.3g}",
        fontsize=8.5,
        pad=0,
    )
    return {
        "grid_index": label.grid_index,
        "symmetry": label.symmetry,
        "orbital": label.orbital,
        "energy": label.energy,
        "occupation": label.occupation,
        "min": minimum,
        "max": maximum,
        "max_abs": max_abs,
        "isovalue": isovalue,
        "positive_voxels": int(np.count_nonzero(values >= isovalue)),
        "negative_voxels": int(np.count_nonzero(values <= -isovalue)),
        "surface_points_rendered": int(indices.shape[0]),
        "center_weight_fraction": center_fraction,
        "min_center_fraction": min_center_fraction,
        "localization_guard_pass": localization_guard_pass,
        "phase_guard_pass": True,
        "finite_guard_pass": True,
    }


def render_orbital_grids(args: argparse.Namespace) -> int:
    metadata = read_grid_metadata(args.grid)
    selected = parse_orbital_specs(args.orbital, metadata.labels)
    if len(selected) > args.max_orbitals:
        raise ValueError(
            f"Selection has {len(selected)} orbitals; --max-orbitals is {args.max_orbitals}"
        )
    if args.metadata_only:
        record = _metadata_record(metadata)
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0

    volumes = read_grid_volumes(metadata, set(selected))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for rendering; install atomi[molcas-viz] "
            "or use --metadata-only"
        ) from exc

    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.center_atom_index < 1 or args.center_atom_index > len(metadata.atoms):
        raise ValueError(
            f"--center-atom-index {args.center_atom_index} is outside 1-{len(metadata.atoms)}"
        )
    all_indices = np.indices(metadata.dimensions, dtype=float).reshape(3, -1).T
    all_coordinates = _grid_coordinates(metadata, all_indices)
    center_xyz = np.asarray(
        metadata.atoms[args.center_atom_index - 1]["xyz_bohr"],
        dtype=float,
    )
    center_mask = (
        np.sum(np.square(all_coordinates - center_xyz), axis=1)
        <= args.center_radius_bohr**2
    ).reshape(metadata.dimensions)
    ncols = min(args.columns, len(selected))
    nrows = math.ceil(len(selected) / ncols)
    figure = plt.figure(figsize=(3.55 * ncols, 3.45 * nrows), facecolor="white")
    rows: list[dict[str, Any]] = []
    for panel, grid_index in enumerate(selected, start=1):
        label = metadata.labels[grid_index]
        values = volumes[grid_index]
        ax = figure.add_subplot(nrows, ncols, panel, projection="3d")
        row = _draw_orbital(
            ax,
            metadata,
            label,
            values,
            isofraction=args.isofraction,
            absolute_isovalue=args.isovalue,
            max_points=args.max_surface_points,
            center_mask=center_mask,
            min_center_fraction=args.min_center_fraction,
        )
        rows.append(row)

        single = plt.figure(figsize=(5.2, 5.0), facecolor="white")
        single_ax = single.add_subplot(111, projection="3d")
        _draw_orbital(
            single_ax,
            metadata,
            label,
            values,
            isofraction=args.isofraction,
            absolute_isovalue=args.isovalue,
            max_points=args.max_surface_points,
            center_mask=center_mask,
            min_center_fraction=args.min_center_fraction,
        )
        single.suptitle(metadata.title.strip() or "OpenMolcas GRID_IT orbital", fontsize=11)
        single.tight_layout()
        single.savefig(
            outdir / f"orbital_sym{label.symmetry}_{label.orbital}.png",
            dpi=args.dpi,
            bbox_inches="tight",
        )
        plt.close(single)

    for panel in range(len(selected) + 1, nrows * ncols + 1):
        ax = figure.add_subplot(nrows, ncols, panel)
        ax.axis("off")
    figure.suptitle(
        args.title or metadata.title.strip() or "OpenMolcas GRID_IT orbital manifold",
        fontsize=14,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "OpenMolcas GRID_IT amplitudes | red/blue = opposite orbital phases | coordinates in bohr",
        ha="center",
        fontsize=8,
        color="#59636e",
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.97))
    montage_path = outdir / args.montage_name
    figure.savefig(montage_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    csv_path = outdir / args.csv_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = _metadata_record(metadata)
    summary.update(
        {
            "selected_orbitals": rows,
            "n_selected": len(rows),
            "all_numerical_guards_pass": all(
                row["phase_guard_pass"] and row["finite_guard_pass"] for row in rows
            ),
            "all_localization_guards_pass": all(
                row["localization_guard_pass"] for row in rows
            ),
            "all_selected_guards_pass": all(
                row["phase_guard_pass"]
                and row["finite_guard_pass"]
                and row["localization_guard_pass"]
                for row in rows
            ),
            "center_atom_index": args.center_atom_index,
            "center_atom_label": metadata.atoms[args.center_atom_index - 1]["label"],
            "center_radius_bohr": args.center_radius_bohr,
            "min_center_fraction": args.min_center_fraction,
            "montage": str(montage_path),
            "csv": str(csv_path),
            "renderer": "matplotlib_headless_phase_surface_points",
            "interactive_viewer": "Pegamoid remains optional; use it for interactive HDF5/InpOrb/grid inspection.",
        }
    )
    summary_path = outdir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote orbital montage: {montage_path}")
    print(f"Wrote orbital QA table: {csv_path}")
    print(f"Wrote orbital summary: {summary_path}")
    return 0


def _metadata_record(metadata: GridMetadata) -> dict[str, Any]:
    return {
        "schema": SCHEMA_ORBITAL_GRID,
        "source": str(metadata.path),
        "title": metadata.title,
        "n_grids": metadata.n_grids,
        "n_points": metadata.n_points,
        "dimensions": list(metadata.dimensions),
        "origin_bohr": metadata.origin.tolist(),
        "axes_bohr": metadata.axes.tolist(),
        "atoms": list(metadata.atoms),
        "orbitals": [
            {
                "grid_index": label.grid_index,
                "symmetry": label.symmetry,
                "orbital": label.orbital,
                "energy": label.energy,
                "occupation": label.occupation,
                "label": label.label,
            }
            for label in metadata.labels
        ],
    }


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "orbital-grid-render",
        help="Render OpenMolcas GRID_IT ASCII orbital grids headlessly with phase/finite guards.",
    )
    parser.add_argument("--grid", type=Path, required=True, help="GRID_IT .grid or .grid.gz file.")
    parser.add_argument(
        "--orbital",
        action="append",
        default=[],
        help="Orbital selector SYM:ORB or SYM:FIRST-LAST; repeatable. Default: all orbital grids.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("molcas_orbital_grid"))
    parser.add_argument("--title", default="")
    parser.add_argument("--isofraction", type=float, default=0.08)
    parser.add_argument("--isovalue", type=float)
    parser.add_argument("--max-orbitals", type=int, default=16)
    parser.add_argument("--max-surface-points", type=int, default=16000)
    parser.add_argument("--center-atom-index", type=int, default=1)
    parser.add_argument("--center-radius-bohr", type=float, default=3.0)
    parser.add_argument(
        "--min-center-fraction",
        type=float,
        default=0.0,
        help="Minimum integral fraction of |orbital|^2 inside the center sphere; 0 reports without rejecting.",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--montage-name", default="molcas_orbital_grid_montage.png")
    parser.add_argument("--csv-name", default="molcas_orbital_grid_qa.csv")
    parser.add_argument("--summary-name", default="molcas_orbital_grid_summary.json")
    parser.add_argument("--metadata-only", action="store_true")
    parser.set_defaults(func=render_orbital_grids)
    return parser
