"""Local-structure and cluster analysis helpers backed by ASE/Pymatgen.

The module keeps ASE as the broad default reader because it handles common
VASP, CP2K XYZ, and LAMMPS files. Pymatgen is optional and useful for richer
crystal-file parsing when installed.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class NeighborRecord:
    """One neighbor around a selected center atom."""

    center_index: int
    center_label: str
    center_symbol: str
    atom_index: int
    atom_label: str
    symbol: str
    distance: float
    vector: tuple[float, float, float]


@dataclass(frozen=True)
class LocalSummary:
    """Compact radial summary for one local environment."""

    center_index: int
    center_label: str
    center_symbol: str
    n_neighbors: int
    first_shell_n: int
    distances: tuple[float, ...]
    d1: float | None
    d_first_shell: float | None
    d_next: float | None
    first_shell_mean: float | None
    first_shell_std: float | None
    first_shell_span: float | None
    first_to_next_gap: float | None
    short_lt_1p9: int
    short_lt_2p0: int
    approximate_self_symmetry: str


def _optional_module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _import_ase():
    try:
        from ase import Atoms  # noqa: F401
        from ase.io import read, write  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised on lean installs
        raise RuntimeError(
            "ASE is required for local-structure analysis. Install Atomi with "
            "`pip install atomi[materials]` or install `ase` in this environment."
        ) from exc
    from ase.io import read, write

    return read, write


def _read_with_pymatgen(path: Path):
    try:
        from pymatgen.core import Molecule, Structure
        from pymatgen.io.ase import AseAtomsAdaptor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Pymatgen backend requested, but pymatgen is not importable. "
            "Use `--backend ase` or install Atomi with the `materials` extra."
        ) from exc

    errors: list[str] = []
    for cls in (Structure, Molecule):
        try:
            obj = cls.from_file(str(path))
            return AseAtomsAdaptor.get_atoms(obj)
        except Exception as exc:  # pragma: no cover - depends on input format
            errors.append(f"{cls.__name__}: {exc}")
    raise RuntimeError(f"Pymatgen could not read {path}: {'; '.join(errors)}")


def read_atoms(path: str | Path, *, backend: str = "ase", fmt: str | None = None, index: str | int | None = None):
    """Read a structure or selected trajectory frame as an ASE Atoms object."""

    path = Path(path)
    if backend == "pymatgen":
        if fmt is not None or index is not None:
            raise ValueError("The pymatgen backend currently reads single structure files only.")
        return _read_with_pymatgen(path)
    if backend != "ase":
        raise ValueError(f"Unknown backend {backend!r}; expected 'ase' or 'pymatgen'.")

    read, _ = _import_ase()
    kwargs = {}
    if fmt:
        kwargs["format"] = fmt
    if index is not None:
        try:
            index_value: str | int = int(index)
        except (TypeError, ValueError):
            index_value = str(index)
        kwargs["index"] = index_value
    return read(str(path), **kwargs)


def _label_counts(symbols: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels


def parse_center_selector(selector: str, symbols: Sequence[str]) -> list[int]:
    """Parse global or element-relative 1-based center selectors.

    Examples:
    - ``3,8,12`` selects global atom indices 3, 8, and 12.
    - ``U:3-8`` selects the 3rd through 8th uranium atoms in the file.
    - ``U:3-5,O:1`` can mix element-relative selectors.
    """

    selected: list[int] = []
    by_element: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        by_element.setdefault(symbol, []).append(idx)

    for raw_token in selector.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" in token:
            symbol, raw_range = token.split(":", 1)
            symbol = symbol.strip()
            if symbol not in by_element:
                raise ValueError(f"No atoms with element {symbol!r} are present.")
            pool = by_element[symbol]
            for one_based in _expand_ranges(raw_range):
                if one_based < 1 or one_based > len(pool):
                    raise ValueError(f"{symbol}:{one_based} is outside 1..{len(pool)}.")
                selected.append(pool[one_based - 1])
        else:
            for one_based in _expand_ranges(token):
                if one_based < 1 or one_based > len(symbols):
                    raise ValueError(f"Global atom index {one_based} is outside 1..{len(symbols)}.")
                selected.append(one_based - 1)

    deduped: list[int] = []
    seen: set[int] = set()
    for idx in selected:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


def _expand_ranges(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split("+"):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(part))
    return values


def collect_neighbors(
    atoms,
    center_index: int,
    *,
    radius: float,
    neighbor_elements: set[str] | None = None,
) -> list[NeighborRecord]:
    """Collect sorted neighbors around a 0-based center atom index."""

    symbols = atoms.get_chemical_symbols()
    labels = _label_counts(symbols)
    indices = [idx for idx in range(len(symbols)) if idx != center_index]
    vectors = atoms.get_distances(center_index, indices, mic=bool(any(atoms.pbc)), vector=True)
    distances = [float(math.sqrt(float((vec * vec).sum()))) for vec in vectors]
    records: list[NeighborRecord] = []
    for idx, vec, distance in zip(indices, vectors, distances):
        if distance > radius:
            continue
        if neighbor_elements is not None and symbols[idx] not in neighbor_elements:
            continue
        records.append(
            NeighborRecord(
                center_index=center_index,
                center_label=labels[center_index],
                center_symbol=symbols[center_index],
                atom_index=idx,
                atom_label=labels[idx],
                symbol=symbols[idx],
                distance=distance,
                vector=(float(vec[0]), float(vec[1]), float(vec[2])),
            )
        )
    return sorted(records, key=lambda rec: (rec.distance, rec.atom_index))


def summarize_environment(records: Sequence[NeighborRecord], *, first_shell_n: int = 9, symmetry_tol: float = 0.05) -> LocalSummary:
    """Summarize one center environment from sorted neighbor records."""

    if not records:
        raise ValueError("Cannot summarize an empty neighbor list.")
    distances = tuple(float(rec.distance) for rec in records)
    shell = distances[:first_shell_n]
    mean = sum(shell) / len(shell) if shell else None
    std = _std(shell) if shell else None
    span = (max(shell) - min(shell)) if shell else None
    d_next = distances[first_shell_n] if len(distances) > first_shell_n else None
    gap = (d_next - shell[-1]) if d_next is not None and shell else None
    return LocalSummary(
        center_index=records[0].center_index,
        center_label=records[0].center_label,
        center_symbol=records[0].center_symbol,
        n_neighbors=len(records),
        first_shell_n=len(shell),
        distances=distances,
        d1=distances[0] if distances else None,
        d_first_shell=shell[-1] if shell else None,
        d_next=d_next,
        first_shell_mean=mean,
        first_shell_std=std,
        first_shell_span=span,
        first_to_next_gap=gap,
        short_lt_1p9=sum(1 for d in shell if d < 1.9),
        short_lt_2p0=sum(1 for d in shell if d < 2.0),
        approximate_self_symmetry=approximate_self_symmetry(records[:first_shell_n], tol=symmetry_tol),
    )


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def radial_fingerprint(records: Sequence[NeighborRecord], *, first_shell_n: int | None = None) -> tuple[float, ...]:
    distances = [rec.distance for rec in records[:first_shell_n]]
    return tuple(round(float(value), 6) for value in distances)


def pair_distance_fingerprint(records: Sequence[NeighborRecord], *, first_shell_n: int | None = None) -> tuple[float, ...]:
    shell = list(records[:first_shell_n])
    pairs: list[float] = []
    for i, left in enumerate(shell):
        lx, ly, lz = left.vector
        for right in shell[i + 1 :]:
            rx, ry, rz = right.vector
            pairs.append(math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2))
    return tuple(round(value, 6) for value in sorted(pairs))


def compare_environments(
    environments: dict[str, Sequence[NeighborRecord]],
    *,
    first_shell_n: int = 9,
) -> list[dict[str, float | str]]:
    """Compare local cages with radial and neighbor-neighbor fingerprints."""

    labels = list(environments)
    rows: list[dict[str, float | str]] = []
    for i, left_label in enumerate(labels):
        for right_label in labels[i + 1 :]:
            left = environments[left_label]
            right = environments[right_label]
            left_radial = radial_fingerprint(left, first_shell_n=first_shell_n)
            right_radial = radial_fingerprint(right, first_shell_n=first_shell_n)
            left_pair = pair_distance_fingerprint(left, first_shell_n=first_shell_n)
            right_pair = pair_distance_fingerprint(right, first_shell_n=first_shell_n)
            rows.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "radial_max_abs_delta_A": _max_abs_delta(left_radial, right_radial),
                    "radial_rms_delta_A": _rms_delta(left_radial, right_radial),
                    "neighbor_pair_max_abs_delta_A": _max_abs_delta(left_pair, right_pair),
                    "neighbor_pair_rms_delta_A": _rms_delta(left_pair, right_pair),
                }
            )
    return rows


def _max_abs_delta(left: Sequence[float], right: Sequence[float]) -> float:
    n = min(len(left), len(right))
    if n == 0:
        return float("nan")
    return max(abs(left[i] - right[i]) for i in range(n))


def _rms_delta(left: Sequence[float], right: Sequence[float]) -> float:
    n = min(len(left), len(right))
    if n == 0:
        return float("nan")
    return math.sqrt(sum((left[i] - right[i]) ** 2 for i in range(n)) / n)


def approximate_self_symmetry(records: Sequence[NeighborRecord], *, tol: float = 0.05) -> str:
    """Return a conservative approximate cage self-symmetry label.

    This is intentionally modest: it detects whether simple proper/improper
    signed-axis permutations preserve the neighbor set within ``tol`` after
    alignment to principal axes. Anything ambiguous is labeled C1.
    """

    if len(records) < 3:
        return "C1"
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return "unknown-no-scipy"

    coords = np.array([rec.vector for rec in records], dtype=float)
    coords -= coords.mean(axis=0)
    inertia = coords.T @ coords
    _, eigvec = np.linalg.eigh(inertia)
    aligned = coords @ eigvec

    operations = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        base = np.eye(3)[:, perm]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    operations.append(base @ np.diag([sx, sy, sz]))

    matched = 0
    for op in operations:
        transformed = aligned @ op
        cost = np.linalg.norm(transformed[:, None, :] - aligned[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost)
        if float(cost[row_ind, col_ind].max()) <= tol:
            matched += 1
    if matched <= 1:
        return "C1"
    return f"approx-{matched}-ops"


def write_cluster_xyz(records: Sequence[NeighborRecord], path: Path, *, include_center: bool = True) -> None:
    """Write one center-origin cluster XYZ file."""

    if not records:
        raise ValueError("Cannot write an empty cluster.")
    rows: list[tuple[str, tuple[float, float, float], str]] = []
    if include_center:
        rows.append((records[0].center_symbol, (0.0, 0.0, 0.0), records[0].center_label))
    for rec in records:
        rows.append((rec.symbol, rec.vector, rec.atom_label))
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(rows)}\n")
        handle.write(f"center={records[0].center_label}; coordinates are center-origin Angstrom vectors\n")
        for symbol, xyz, label in rows:
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}  # {label}\n")


def _summary_to_row(summary: LocalSummary) -> dict[str, str | int | float | None]:
    return {
        "center_label": summary.center_label,
        "center_global_index_1based": summary.center_index + 1,
        "center_element": summary.center_symbol,
        "n_neighbors_in_radius": summary.n_neighbors,
        "first_shell_n": summary.first_shell_n,
        "d1_A": summary.d1,
        "d_first_shell_A": summary.d_first_shell,
        "d_next_A": summary.d_next,
        "first_shell_mean_A": summary.first_shell_mean,
        "first_shell_std_A": summary.first_shell_std,
        "first_shell_span_A": summary.first_shell_span,
        "first_to_next_gap_A": summary.first_to_next_gap,
        "short_bonds_lt_1p9_A": summary.short_lt_1p9,
        "short_bonds_lt_2p0_A": summary.short_lt_2p0,
        "approximate_self_symmetry": summary.approximate_self_symmetry,
        "distances_A": " ".join(f"{value:.6f}" for value in summary.distances),
    }


def analyze_file(
    input_path: str | Path,
    *,
    centers: str,
    outdir: str | Path,
    backend: str = "ase",
    fmt: str | None = None,
    index: str | int | None = None,
    radius: float = 3.0,
    neighbor_elements: Iterable[str] | None = None,
    first_shell_n: int = 9,
    write_clusters: bool = False,
    compare: bool = True,
    symmetry_tol: float = 0.05,
    quiet: bool = False,
) -> dict[str, object]:
    """Analyze selected local environments and write CSV/JSON artifacts."""

    atoms = read_atoms(input_path, backend=backend, fmt=fmt, index=index)
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError("ASE returned an empty frame list.")
        atoms = atoms[-1]
    symbols = atoms.get_chemical_symbols()
    center_indices = parse_center_selector(centers, symbols)
    neighbor_set = set(neighbor_elements) if neighbor_elements else None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    environments: dict[str, list[NeighborRecord]] = {}
    summaries: list[LocalSummary] = []
    for center_index in center_indices:
        records = collect_neighbors(atoms, center_index, radius=radius, neighbor_elements=neighbor_set)
        if not records:
            raise ValueError(f"No neighbors found for center {center_index + 1} within {radius} A.")
        summary = summarize_environment(records, first_shell_n=first_shell_n, symmetry_tol=symmetry_tol)
        environments[summary.center_label] = records
        summaries.append(summary)

    summary_csv = outdir / "local_structure_summary.csv"
    summary_json = outdir / "local_structure_summary.json"
    rows = [_summary_to_row(summary) for summary in summaries]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "input": str(input_path),
                "backend": backend,
                "format": fmt,
                "index": index,
                "radius_A": radius,
                "neighbor_elements": sorted(neighbor_set) if neighbor_set else None,
                "first_shell_n": first_shell_n,
                "summaries": rows,
            },
            handle,
            indent=2,
        )

    cluster_paths: list[str] = []
    if write_clusters:
        cluster_dir = outdir / "clusters"
        cluster_dir.mkdir(exist_ok=True)
        for label, records in environments.items():
            cluster_path = cluster_dir / f"{label}_cluster.xyz"
            write_cluster_xyz(records[:first_shell_n], cluster_path)
            cluster_paths.append(str(cluster_path))

    compare_rows: list[dict[str, float | str]] = []
    compare_csv = None
    if compare and len(environments) > 1:
        compare_rows = compare_environments(environments, first_shell_n=first_shell_n)
        compare_csv = outdir / "pairwise_cage_fingerprint.csv"
        with compare_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0]))
            writer.writeheader()
            writer.writerows(compare_rows)

    result = {
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "compare_csv": str(compare_csv) if compare_csv else None,
        "cluster_paths": cluster_paths,
        "summaries": rows,
        "comparisons": compare_rows,
    }
    if not quiet:
        _print_result(result)
    return result


def _print_result(result: dict[str, object]) -> None:
    print(f"Wrote {result['summary_csv']}")
    if result.get("compare_csv"):
        print(f"Wrote {result['compare_csv']}")
    cluster_paths = result.get("cluster_paths") or []
    if cluster_paths:
        print(f"Wrote {len(cluster_paths)} cluster XYZ files")
    for row in result["summaries"]:  # type: ignore[index]
        print(
            "{center_label}: CN={n_neighbors_in_radius}, first-shell d1={d1_A:.4f} A, "
            "dN={d_first_shell_A:.4f} A, next={d_next_A}, symmetry={approximate_self_symmetry}".format(
                **_format_print_row(row)
            )
        )


def _format_print_row(row: dict[str, object]) -> dict[str, object]:
    formatted = dict(row)
    if formatted["d_next_A"] is None:
        formatted["d_next_A"] = "none"
    elif isinstance(formatted["d_next_A"], float):
        formatted["d_next_A"] = f"{formatted['d_next_A']:.4f} A"
    return formatted


def _split_elements(text: str | None) -> list[str] | None:
    if text is None:
        return None
    values = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    return values or None


def doctor() -> int:
    print("Atomi local-structure backends")
    print(f"  ASE: {'available' if _optional_module_available('ase') else 'missing'}")
    print(f"  Pymatgen: {'available' if _optional_module_available('pymatgen') else 'missing'}")
    print(f"  SciPy symmetry helper: {'available' if _optional_module_available('scipy') else 'missing'}")
    print("Common ASE formats: vasp, extxyz, xyz, lammps-data, lammps-dump-text")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local-structure",
        description="Extract and compare local clusters from VASP, CP2K, and LAMMPS structures.",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze selected local environments.")
    analyze.add_argument("--input", required=True, help="Structure or trajectory file.")
    analyze.add_argument("--backend", choices=("ase", "pymatgen"), default="ase")
    analyze.add_argument("--format", dest="fmt", default=None, help="ASE input format, e.g. vasp or lammps-dump-text.")
    analyze.add_argument("--index", default=None, help="ASE frame index, e.g. -1 for final CP2K/LAMMPS frame.")
    analyze.add_argument("--centers", required=True, help="Centers such as U:3-8 or 12,15,19.")
    analyze.add_argument("--neighbor-elements", default=None, help="Comma-separated neighbor elements, e.g. O or O,Cl.")
    analyze.add_argument("--radius", type=float, default=3.0)
    analyze.add_argument("--first-shell-n", type=int, default=9)
    analyze.add_argument("--symmetry-tol", type=float, default=0.05)
    analyze.add_argument("--outdir", default="local_structure_analysis")
    analyze.add_argument("--write-clusters", action="store_true")
    analyze.add_argument("--no-compare", action="store_true")
    analyze.add_argument("--quiet", action="store_true")

    subparsers.add_parser("doctor", help="Report optional backend availability.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "doctor"):
        return doctor()
    if args.command == "analyze":
        analyze_file(
            args.input,
            centers=args.centers,
            outdir=args.outdir,
            backend=args.backend,
            fmt=args.fmt,
            index=args.index,
            radius=args.radius,
            neighbor_elements=_split_elements(args.neighbor_elements),
            first_shell_n=args.first_shell_n,
            write_clusters=args.write_clusters,
            compare=not args.no_compare,
            symmetry_tol=args.symmetry_tol,
            quiet=args.quiet,
        )
        return 0
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
