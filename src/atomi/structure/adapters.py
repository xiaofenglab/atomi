"""Reusable structure and trajectory readers for Atomi workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StructureFrame:
    """A small backend-neutral structure/trajectory frame."""

    symbols: list[str]
    cell: list[list[float]] | None = None
    frac: list[list[float]] | None = None
    coords: list[list[float]] | None = None
    comment: str = ""
    index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": self.symbols,
            "cell": self.cell,
            "frac": self.frac,
            "coords": self.coords,
            "comment": self.comment,
            "index": self.index,
            "metadata": self.metadata,
        }


def parse_cell_abc(value: str | None) -> list[list[float]] | None:
    if not value:
        return None
    parts = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("Cell ABC must have exactly three lengths, e.g. '12.58,12.58,18.87'.")
    a, b, c = (float(item) for item in parts)
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]]


def cell_from_cp2k_input(path: Path | None) -> list[list[float]] | None:
    """Read an orthogonal ABC or explicit A/B/C cell from a CP2K input."""

    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    abc_match = re.search(r"(?im)^\s*ABC\s+([0-9eE.+\-]+)\s+([0-9eE.+\-]+)\s+([0-9eE.+\-]+)", text)
    if abc_match:
        return parse_cell_abc(",".join(abc_match.groups()))
    vectors: dict[str, list[float]] = {}
    for key in ("A", "B", "C"):
        match = re.search(rf"(?im)^\s*{key}\s+([0-9eE.+\-]+)\s+([0-9eE.+\-]+)\s+([0-9eE.+\-]+)", text)
        if match:
            vectors[key] = [float(item) for item in match.groups()]
    if {"A", "B", "C"} <= set(vectors):
        return [vectors["A"], vectors["B"], vectors["C"]]
    return None


def cell_from_xyz_comment(comment: str) -> list[list[float]] | None:
    """Read an extxyz Lattice comment as three row cell vectors."""

    match = re.search(r'Lattice="([^"]+)"', comment)
    if not match:
        return None
    values = [float(item) for item in match.group(1).split()]
    if len(values) != 9:
        return None
    return [values[0:3], values[3:6], values[6:9]]


def read_cp2k_xyz_frames(path: Path) -> list[dict[str, Any]]:
    """Read CP2K/XYZ trajectory frames as dictionaries.

    The dictionary shape is kept compatible with the historical SLUSCHI bridge
    parser: ``comment``, ``symbols``, and Cartesian ``coords``.
    """

    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            natoms = int(line.strip())
            comment = handle.readline().rstrip("\n")
            symbols: list[str] = []
            coords: list[list[float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {path}")
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append({"comment": comment, "symbols": symbols, "coords": coords})
    return frames


def read_vasp_poscar_basis(path: Path) -> dict[str, Any]:
    """Read VASP5 POSCAR lattice/species/count metadata."""

    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {path}")
    scale = float(lines[1].split()[0])
    lattice = [[float(value) * scale for value in lines[idx].split()[:3]] for idx in range(2, 5)]
    token_line = lines[5].split()
    if all(re.fullmatch(r"[-+]?\d+", token) for token in token_line):
        raise ValueError("VASP4-style POSCAR without element symbols is not supported; pass a VASP5 POSCAR/XDATCAR header.")
    elements = token_line
    counts = [int(token) for token in lines[6].split()]
    if len(elements) != len(counts):
        raise ValueError(f"POSCAR element/count length mismatch in {path}")
    symbols: list[str] = []
    for element, count in zip(elements, counts):
        symbols.extend([element] * count)
    return {"lattice": lattice, "elements": elements, "counts": counts, "symbols": symbols, "natoms": sum(counts)}


def read_vasp_xdatcar_frames(path: Path, natoms: int) -> list[list[list[float]]]:
    """Read fractional frames from a VASP XDATCAR."""

    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    frames: list[list[list[float]]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].lower().startswith("direct configuration"):
            coords: list[list[float]] = []
            for row in lines[idx + 1 : idx + 1 + natoms]:
                parts = row.split()
                if len(parts) < 3:
                    break
                coords.append([float(parts[0]) % 1.0, float(parts[1]) % 1.0, float(parts[2]) % 1.0])
            if len(coords) == natoms:
                frames.append(coords)
            idx += natoms + 1
        else:
            idx += 1
    return frames


def vasp_xdatcar_structure_frames(poscar: Path, xdatcar: Path) -> list[StructureFrame]:
    """Read VASP POSCAR/XDATCAR as backend-neutral ``StructureFrame`` rows."""

    basis = read_vasp_poscar_basis(poscar)
    frames = read_vasp_xdatcar_frames(xdatcar, int(basis["natoms"]))
    return [
        StructureFrame(
            symbols=list(basis["symbols"]),
            cell=[list(row) for row in basis["lattice"]],
            frac=frame,
            index=index,
            metadata={
                "engine": "vasp",
                "poscar": str(poscar),
                "xdatcar": str(xdatcar),
                "elements": list(basis["elements"]),
                "counts": list(basis["counts"]),
            },
        )
        for index, frame in enumerate(frames)
    ]
