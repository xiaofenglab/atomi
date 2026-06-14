"""Crystal graph dataset export for GNN/MLIP thermodynamic bridges.

This module keeps the first graph-learning interface deliberately small and
backend-neutral.  It turns ASE-readable structures, or Atomi CETrainingSet
records, into JSONL graph rows that can be consumed by later GNN, MACE, CHGNet,
or uncertainty-learning layers without making those packages required here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised through public functions when ASE exists
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers, covalent_radii
    from ase.io import read as ase_read
    from ase.neighborlist import neighbor_list
except Exception:  # pragma: no cover - keeps importable in minimal environments
    Atoms = Any  # type: ignore[misc,assignment]
    atomic_masses = None
    atomic_numbers = None
    covalent_radii = None
    ase_read = None
    neighbor_list = None

from atomi.zentropy.backends.base import CETrainingSet, read_ce_training_jsonl

SCHEMA = "atomi.ml.crystal_graph_dataset.v1"


@dataclass
class GraphDatasetSummary:
    """Summary for a graph JSONL export or validation run."""

    schema: str = SCHEMA
    output: str = ""
    n_records: int = 0
    n_skipped: int = 0
    n_edges_total: int = 0
    records_missing_edges: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_ase() -> None:
    if ase_read is None or neighbor_list is None or atomic_numbers is None:
        raise RuntimeError(
            "ASE is required for crystal graph dataset export. "
            "Install atomi with the materials extras or install ase>=3.23."
        )


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    number = _finite_float(value)
    if number is not None:
        return number
    return str(value)


def _float_list(values: Iterable[Any], ndigits: int = 12) -> list[float]:
    out: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            raise ValueError(f"Cannot convert {value!r} to a finite float.")
        out.append(round(number, ndigits))
    return out


def _read_structure(path: Path) -> Atoms:
    _ensure_ase()
    try:
        return ase_read(str(path))  # type: ignore[misc]
    except Exception as exc:
        raise ValueError(f"Could not read structure with ASE: {path}") from exc


def _species_counts(atoms: Atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def _cell_payload(atoms: Atoms) -> dict[str, Any]:
    cell = atoms.cell
    lengths_angles = cell.cellpar()
    return {
        "matrix_A": [_float_list(row) for row in cell.array],
        "lengths_A": _float_list(lengths_angles[:3]),
        "angles_deg": _float_list(lengths_angles[3:]),
        "volume_A3": round(float(atoms.get_volume()), 12),
        "pbc": [bool(item) for item in atoms.pbc],
    }


def _node_features(atoms: Atoms) -> list[dict[str, Any]]:
    _ensure_ase()
    scaled = atoms.get_scaled_positions(wrap=False)
    cart = atoms.get_positions()
    rows: list[dict[str, Any]] = []
    for idx, symbol in enumerate(atoms.get_chemical_symbols()):
        z = int(atomic_numbers[symbol])  # type: ignore[index]
        rows.append(
            {
                "index": idx,
                "element": symbol,
                "atomic_number": z,
                "mass_amu": round(float(atomic_masses[z]), 12),  # type: ignore[index]
                "covalent_radius_A": round(float(covalent_radii[z]), 12),  # type: ignore[index]
                "frac_coords": _float_list(scaled[idx]),
                "cart_coords_A": _float_list(cart[idx]),
            }
        )
    return rows


def _edge_features(atoms: Atoms, cutoff: float) -> list[dict[str, Any]]:
    _ensure_ase()
    if len(atoms) == 0:
        return []
    src, dst, dist, vec, shift = neighbor_list("ijdDS", atoms, cutoff)  # type: ignore[misc]
    edges: list[dict[str, Any]] = []
    for i, j, d, vector, image in zip(src, dst, dist, vec, shift):
        edges.append(
            {
                "src": int(i),
                "dst": int(j),
                "distance_A": round(float(d), 12),
                "vector_A": _float_list(vector),
                "cell_shift": [int(x) for x in image],
            }
        )
    return edges


def atoms_to_graph_record(
    atoms: Atoms,
    *,
    record_id: str,
    structure_path: str = "",
    cutoff: float = 5.0,
    labels: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one ASE Atoms object into an Atomi graph JSON row."""

    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive.")
    edges = _edge_features(atoms, cutoff)
    volume = float(atoms.get_volume()) if atoms.get_volume() > 0 else None
    global_features = {
        "cutoff_A": float(cutoff),
        "volume_A3": volume,
        "density_atoms_per_A3": (len(atoms) / volume) if volume else None,
        "n_edges": len(edges),
    }
    return {
        "schema": SCHEMA,
        "record_id": str(record_id),
        "structure_path": str(structure_path),
        "formula": atoms.get_chemical_formula(),
        "natoms": len(atoms),
        "species_counts": _species_counts(atoms),
        "cell": _cell_payload(atoms),
        "node_features": _node_features(atoms),
        "edges": edges,
        "global_features": _jsonable(global_features),
        "labels": _jsonable(labels or {}),
        "metadata": _jsonable(metadata or {}),
    }


def read_label_csv(
    path: Path,
    *,
    record_id_column: str = "record_id",
    path_column: str = "structure_path",
) -> dict[str, dict[str, Any]]:
    """Read simple labels keyed by record id and, when present, structure path."""

    labels: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            payload: dict[str, Any] = {}
            for key, value in row.items():
                if key in {record_id_column, path_column}:
                    continue
                number = _finite_float(value)
                payload[key] = number if number is not None else value
            record_id = str(row.get(record_id_column) or "").strip()
            structure_path = str(row.get(path_column) or "").strip()
            if record_id:
                labels[record_id] = payload
            if structure_path:
                labels[structure_path] = payload
                labels[str(Path(structure_path).resolve())] = payload
    return labels


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> GraphDatasetSummary:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = GraphDatasetSummary(output=str(path))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")
            summary.n_records += 1
            n_edges = len(row.get("edges") or [])
            summary.n_edges_total += n_edges
            if n_edges == 0:
                summary.records_missing_edges.append(str(row.get("record_id") or ""))
    return summary


def _write_summary(path: Path, summary: GraphDatasetSummary) -> None:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_graph_dataset(
    structure_paths: Sequence[Path],
    output: Path,
    *,
    cutoff: float = 5.0,
    labels: Mapping[str, Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GraphDatasetSummary:
    """Build a graph JSONL dataset from ASE-readable structure paths."""

    rows: list[dict[str, Any]] = []
    summary = GraphDatasetSummary(output=str(output))
    label_map = labels or {}
    for path in structure_paths:
        try:
            atoms = _read_structure(path)
        except Exception as exc:
            summary.n_skipped += 1
            summary.skipped.append({"path": str(path), "reason": str(exc)})
            continue
        record_id = path.stem
        row_labels = label_map.get(record_id) or label_map.get(str(path)) or label_map.get(str(path.resolve())) or {}
        rows.append(
            atoms_to_graph_record(
                atoms,
                record_id=record_id,
                structure_path=str(path),
                cutoff=cutoff,
                labels=row_labels,
                metadata=metadata,
            )
        )
    written = _write_jsonl(output, rows)
    written.n_skipped = summary.n_skipped
    written.skipped = summary.skipped
    _write_summary(output, written)
    return written


def _ce_record_labels(record: Any) -> dict[str, Any]:
    labels: dict[str, Any] = {
        "composition": record.composition,
        "motif_features": record.motif_features,
        "energy_eV": record.energy_eV,
        "free_energy_terms": record.free_energy_terms,
        "weight": record.weight,
        "uncertainty_eV": record.uncertainty_eV,
    }
    return {key: value for key, value in labels.items() if value is not None}


def build_graph_dataset_from_ce_training_set(
    training_set: CETrainingSet,
    output: Path,
    *,
    base_dir: Path | None = None,
    cutoff: float = 5.0,
) -> GraphDatasetSummary:
    """Export graph rows from a CETrainingSet with structure-backed records."""

    rows: list[dict[str, Any]] = []
    summary = GraphDatasetSummary(output=str(output))
    base = base_dir or Path.cwd()
    for record in training_set.records:
        if not record.structure_path:
            summary.n_skipped += 1
            summary.skipped.append({"record_id": record.record_id, "reason": "empty structure_path"})
            continue
        structure_path = Path(record.structure_path)
        if not structure_path.is_absolute():
            structure_path = base / structure_path
        try:
            atoms = _read_structure(structure_path)
        except Exception as exc:
            summary.n_skipped += 1
            summary.skipped.append({"record_id": record.record_id, "path": str(structure_path), "reason": str(exc)})
            continue
        rows.append(
            atoms_to_graph_record(
                atoms,
                record_id=record.record_id,
                structure_path=str(structure_path),
                cutoff=cutoff,
                labels=_ce_record_labels(record),
                metadata={
                    "system_name": training_set.system_name,
                    "parent_structure_path": training_set.parent_structure_path,
                    "source": record.source,
                    "record_metadata": record.metadata,
                    "training_set_metadata": training_set.metadata,
                },
            )
        )
    written = _write_jsonl(output, rows)
    written.n_skipped = summary.n_skipped
    written.skipped = summary.skipped
    _write_summary(output, written)
    return written


def build_graph_dataset_from_ce_training_jsonl(
    training_jsonl: Path,
    output: Path,
    *,
    cutoff: float = 5.0,
) -> GraphDatasetSummary:
    training_set = read_ce_training_jsonl(training_jsonl)
    return build_graph_dataset_from_ce_training_set(
        training_set,
        output,
        base_dir=training_jsonl.resolve().parent,
        cutoff=cutoff,
    )


def validate_graph_jsonl(path: Path) -> GraphDatasetSummary:
    """Validate the lightweight graph JSONL contract."""

    summary = GraphDatasetSummary(output=str(path))
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") != SCHEMA:
                summary.n_skipped += 1
                summary.skipped.append({"line": str(lineno), "reason": "schema mismatch"})
                continue
            if "node_features" not in row or "edges" not in row or "labels" not in row:
                summary.n_skipped += 1
                summary.skipped.append({"line": str(lineno), "reason": "missing graph fields"})
                continue
            summary.n_records += 1
            n_edges = len(row.get("edges") or [])
            summary.n_edges_total += n_edges
            if n_edges == 0:
                summary.records_missing_edges.append(str(row.get("record_id") or lineno))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystal-graph-dataset",
        description="Export ASE-readable structures or Atomi CETrainingSet rows as graph JSONL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build graph JSONL from structure files.")
    build.add_argument("structures", nargs="+", type=Path)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--cutoff", type=float, default=5.0)
    build.add_argument("--label-csv", type=Path)
    build.add_argument("--record-id-column", default="record_id")
    build.add_argument("--path-column", default="structure_path")
    build.add_argument("--metadata-json", type=Path)

    ce = sub.add_parser("from-ce-training", help="Build graph JSONL from CETrainingSet JSONL.")
    ce.add_argument("--training-jsonl", type=Path, required=True)
    ce.add_argument("--out", type=Path, required=True)
    ce.add_argument("--cutoff", type=float, default=5.0)

    validate = sub.add_parser("validate", help="Validate an Atomi graph JSONL dataset.")
    validate.add_argument("graph_jsonl", type=Path)

    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        labels = (
            read_label_csv(
                args.label_csv,
                record_id_column=args.record_id_column,
                path_column=args.path_column,
            )
            if args.label_csv
            else None
        )
        metadata = json.loads(args.metadata_json.read_text(encoding="utf-8")) if args.metadata_json else None
        summary = build_graph_dataset(args.structures, args.out, cutoff=args.cutoff, labels=labels, metadata=metadata)
    elif args.command == "from-ce-training":
        summary = build_graph_dataset_from_ce_training_jsonl(args.training_jsonl, args.out, cutoff=args.cutoff)
    elif args.command == "validate":
        summary = validate_graph_jsonl(args.graph_jsonl)
    else:  # pragma: no cover - argparse enforces this.
        raise ValueError(f"Unsupported command: {args.command}")

    payload = summary.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
