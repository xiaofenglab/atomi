from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

from atomi.vasp.magmom import existing_magmom_values
from atomi.vasp.qha_summary import parse_outcar, parse_vasprun


SCHEMA = "atomi.zentropy.defect_motif_db.v1"


def read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_formula(formula: str) -> dict[str, float]:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: dict[str, float] = {}
    for element, raw_count in tokens:
        count = float(raw_count) if raw_count else 1.0
        counts[element] = counts.get(element, 0.0) + count
    return counts


def parse_key_values(items: list[str] | None, cast=float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE item, got: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = cast(value.strip())
    return result


def parse_metadata_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("run") or row.get("run_dir") or row.get("motif_id") or "").strip()
            if not key:
                continue
            rows[key] = {k: v for k, v in row.items() if v not in (None, "")}
    return rows


def parse_site_state_csv(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("run") or row.get("run_dir") or row.get("motif_id") or "").strip()
            if not key:
                continue
            clean: dict[str, Any] = {k: v for k, v in row.items() if v not in (None, "")}
            for int_key in ("atom_index_1based", "atom_index"):
                if int_key in clean:
                    clean[int_key] = int(clean[int_key])
            for float_key in ("valence", "magmom", "charge"):
                if float_key in clean:
                    clean[float_key] = float(clean[float_key])
            grouped.setdefault(key, []).append(clean)
    return grouped


def candidate_keys(run_dir: Path, motif_id: str) -> set[str]:
    resolved = run_dir.resolve()
    return {
        str(run_dir),
        str(resolved),
        run_dir.name,
        motif_id,
        resolved.name,
    }


def metadata_for_run(metadata: dict[str, dict[str, str]], run_dir: Path, motif_id: str) -> dict[str, str]:
    for key in candidate_keys(run_dir, motif_id):
        if key in metadata:
            return metadata[key]
    return {}


def site_states_for_run(site_states: dict[str, list[dict[str, Any]]], run_dir: Path, motif_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in candidate_keys(run_dir, motif_id):
        rows.extend(site_states.get(key, []))
    return rows


def structure_file_for_run(run_dir: Path) -> Path | None:
    for name in ("CONTCAR", "POSCAR", "vasprun.xml", "OUTCAR", "OUTCAR.gz"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def read_structure(run_dir: Path) -> tuple[Atoms, Path]:
    source = structure_file_for_run(run_dir)
    if source is None:
        raise FileNotFoundError(f"No CONTCAR/POSCAR/vasprun.xml/OUTCAR found in {run_dir}")
    return read(source, index=-1), source


def count_symbols(atoms: Atoms) -> dict[str, int]:
    counts: dict[str, int] = {}
    for symbol in atoms.get_chemical_symbols():
        counts[symbol] = counts.get(symbol, 0) + 1
    return dict(sorted(counts.items()))


def reduced_formula(counts: dict[str, int]) -> str:
    gcd = 0
    for count in counts.values():
        gcd = count if gcd == 0 else math.gcd(gcd, count)
    if gcd <= 0:
        return ""
    pieces = []
    for element, count in counts.items():
        n = count // gcd
        pieces.append(element if n == 1 else f"{element}{n}")
    return "".join(pieces)


def structure_hash(atoms: Atoms) -> str:
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "cell": np.round(atoms.cell.array, 8).tolist(),
        "scaled_positions": np.round(atoms.get_scaled_positions(wrap=True), 8).tolist(),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def calc_record_for_run(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "vasprun.xml",
        run_dir / "vasprun.xml.gz",
        run_dir / "OUTCAR",
        run_dir / "OUTCAR.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            record = parse_vasprun(path) if "vasprun.xml" in path.name else parse_outcar(path)
        except Exception as exc:
            return {"parser_used": "failed", "source_file": str(path), "error": str(exc)}
        if finite(record.get("energy_eV")) or finite(record.get("volume_A3")):
            return record
    return {"parser_used": "missing", "source_file": None}


def parse_final_outcar_magmoms(path: Path, expected_atoms: int) -> list[float] | None:
    if not path.exists():
        return None
    lines = read_text(path).splitlines()
    block_start = None
    for index, line in enumerate(lines):
        if "magnetization" in line.lower() and "(x)" in line.lower():
            block_start = index
    if block_start is None:
        return None
    moments: list[float] = []
    for line in lines[block_start + 1 :]:
        parts = line.split()
        if not parts:
            if moments:
                break
            continue
        if parts[0].lower().startswith("tot"):
            break
        try:
            int(parts[0])
        except ValueError:
            continue
        try:
            moments.append(float(parts[-1]))
        except ValueError:
            continue
        if len(moments) == expected_atoms:
            break
    return moments if len(moments) == expected_atoms else None


def magmom_for_run(run_dir: Path, atoms: Atoms) -> dict[str, Any]:
    natoms = len(atoms)
    source = None
    values = parse_final_outcar_magmoms(run_dir / "OUTCAR", natoms)
    if values is not None:
        source = "OUTCAR magnetization (x)"
    if values is None:
        values = parse_final_outcar_magmoms(run_dir / "OUTCAR.gz", natoms)
        if values is not None:
            source = "OUTCAR.gz magnetization (x)"
    if values is None and (run_dir / "INCAR").exists():
        values = existing_magmom_values(run_dir / "INCAR", natoms)
        if values is not None:
            source = "INCAR MAGMOM"
    if values is None:
        return {"source": "missing", "values": [], "by_element": {}}

    by_element: dict[str, dict[str, float]] = {}
    symbols = atoms.get_chemical_symbols()
    for symbol in sorted(set(symbols)):
        arr = np.asarray([value for value, sp in zip(values, symbols) if sp == symbol], dtype=float)
        by_element[symbol] = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "abs_mean": float(np.mean(np.abs(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return {"source": source, "values": values, "by_element": by_element}


def size_normalization(
    counts: dict[str, int],
    parent_counts: dict[str, float],
    host_cation: str | None,
    guest_cations: list[str],
    oxygen: str,
) -> dict[str, Any]:
    parent_cation_count = sum(v for k, v in parent_counts.items() if k != oxygen)
    if parent_cation_count <= 0:
        parent_cation_count = 1.0
    if host_cation:
        cation_count = counts.get(host_cation, 0) + sum(counts.get(sp, 0) for sp in guest_cations)
    else:
        cation_count = sum(count for element, count in counts.items() if element != oxygen)
    formula_units = float(cation_count) / parent_cation_count if cation_count else math.nan
    parent_oxygen = parent_counts.get(oxygen, 0.0)
    oxygen_count = counts.get(oxygen, 0)
    expected_oxygen = formula_units * parent_oxygen if finite(formula_units) else math.nan
    delta = (oxygen_count - expected_oxygen) / formula_units if formula_units else math.nan
    guest_total = sum(counts.get(sp, 0) for sp in guest_cations)
    x_guest = guest_total / cation_count if cation_count else 0.0
    per_fu = {
        element: count / formula_units
        for element, count in counts.items()
        if finite(formula_units) and formula_units > 0
    }
    return {
        "formula_units": formula_units,
        "cation_count": cation_count,
        "guest_cation_count": guest_total,
        "guest_cation_fraction": x_guest,
        "oxygen_delta_per_formula_unit": delta,
        "oxygen_count": oxygen_count,
        "expected_oxygen_count": expected_oxygen,
        "composition_per_formula_unit": per_fu,
        "normalization_note": (
            "Formula units are derived from the cation sublattice so DFT motifs "
            "from different supercell sizes can be compared by x_guest, delta, "
            "and energy per parent formula unit."
        ),
    }


def charge_metadata(counts: dict[str, int], formula_units: float, valence: dict[str, float]) -> dict[str, Any]:
    if not valence or not formula_units or not finite(formula_units):
        return {"valence_model": valence, "nominal_charge_per_formula_unit": None}
    total = 0.0
    missing = []
    for element, count in counts.items():
        if element not in valence:
            missing.append(element)
            continue
        total += count * valence[element]
    return {
        "valence_model": valence,
        "missing_valence_elements": missing,
        "nominal_charge_total": total,
        "nominal_charge_per_formula_unit": total / formula_units,
    }


def split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in re.split(r"[,;]", raw) if item.strip()]


def case_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "case_info.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_record(
    run_dir: Path,
    args: argparse.Namespace,
    parent_counts: dict[str, float],
    metadata_rows: dict[str, dict[str, str]],
    site_state_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    atoms, source_structure = read_structure(run_dir)
    counts = count_symbols(atoms)
    calc = calc_record_for_run(run_dir)
    info = case_info(run_dir)
    provisional_id = run_dir.name
    meta = metadata_for_run(metadata_rows, run_dir, provisional_id)
    motif_id = meta.get("motif_id") or info.get("case_name") or run_dir.name
    meta = metadata_for_run(metadata_rows, run_dir, motif_id) or meta
    normalization = size_normalization(
        counts,
        parent_counts,
        args.host_cation,
        args.guest_cation,
        args.oxygen,
    )
    formula_units = normalization["formula_units"]
    energy = float(calc["energy_eV"]) if finite(calc.get("energy_eV")) else None
    volume = float(atoms.get_volume())
    if finite(calc.get("volume_A3")):
        volume = float(calc["volume_A3"])

    magmom = magmom_for_run(run_dir, atoms)
    tags = set(args.tag or [])
    tags.update(split_tags(meta.get("tags")))
    for key in ("family", "motif_family"):
        if info.get(key):
            tags.add(str(info[key]))
    motif_family = meta.get("motif_family") or meta.get("family") or info.get("family") or args.motif_family
    motif_type = meta.get("motif_type") or info.get("family") or args.motif_type

    record = {
        "motif_id": motif_id,
        "material": args.material,
        "phase": args.phase,
        "parent_formula": args.parent_formula,
        "run_dir": str(run_dir.resolve()),
        "source_structure_file": str(source_structure.resolve()),
        "structure_hash": structure_hash(atoms),
        "motif_family": motif_family,
        "motif_type": motif_type,
        "defect_label": meta.get("defect_label") or args.defect_label,
        "charge_state": meta.get("charge_state") or args.charge_state,
        "degeneracy": float(meta.get("degeneracy", args.degeneracy)),
        "tags": sorted(tags),
        "notes": meta.get("notes") or args.notes,
        "counts": counts,
        "formula": reduced_formula(counts),
        "natoms": len(atoms),
        "volume_A3": volume,
        "volume_per_formula_unit_A3": volume / formula_units if formula_units else None,
        "energy_eV": energy,
        "energy_per_formula_unit_eV": energy / formula_units if energy is not None and formula_units else None,
        "calculation": calc,
        "size_normalization": normalization,
        "charge": charge_metadata(counts, formula_units, args.valence),
        "magmom": magmom,
        "site_states": site_states_for_run(site_state_rows, run_dir, motif_id),
        "case_info": info,
        "zentropy_role": (
            "microstate record; later zentropy stages should combine records at "
            "fixed thermodynamic constraints using degeneracy and G_i(T)"
        ),
    }
    return record


def discover_runs(args: argparse.Namespace) -> list[Path]:
    runs = [path.resolve() for path in args.run]
    if args.root:
        for path in sorted(args.root.resolve().glob(args.glob)):
            if path.is_dir():
                runs.append(path)
    unique: list[Path] = []
    seen = set()
    for path in runs:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise ValueError("No run directories were provided. Use --run or --root/--glob.")
    return unique


def write_record_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "motif_id",
        "material",
        "phase",
        "motif_family",
        "motif_type",
        "defect_label",
        "charge_state",
        "degeneracy",
        "formula",
        "natoms",
        "formula_units",
        "guest_cation_fraction",
        "oxygen_delta_per_formula_unit",
        "energy_eV",
        "energy_per_formula_unit_eV",
        "volume_A3",
        "volume_per_formula_unit_A3",
        "magmom_source",
        "run_dir",
        "source_structure_file",
        "tags",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            norm = rec["size_normalization"]
            row = {
                "motif_id": rec["motif_id"],
                "material": rec["material"],
                "phase": rec["phase"],
                "motif_family": rec["motif_family"],
                "motif_type": rec["motif_type"],
                "defect_label": rec["defect_label"],
                "charge_state": rec["charge_state"],
                "degeneracy": rec["degeneracy"],
                "formula": rec["formula"],
                "natoms": rec["natoms"],
                "formula_units": norm["formula_units"],
                "guest_cation_fraction": norm["guest_cation_fraction"],
                "oxygen_delta_per_formula_unit": norm["oxygen_delta_per_formula_unit"],
                "energy_eV": rec["energy_eV"],
                "energy_per_formula_unit_eV": rec["energy_per_formula_unit_eV"],
                "volume_A3": rec["volume_A3"],
                "volume_per_formula_unit_A3": rec["volume_per_formula_unit_A3"],
                "magmom_source": rec["magmom"]["source"],
                "run_dir": rec["run_dir"],
                "source_structure_file": rec["source_structure_file"],
                "tags": ",".join(rec["tags"]),
            }
            writer.writerow(row)


def index_main(argv: list[str] | None = None) -> None:
    parser = build_index_parser()
    args = parser.parse_args(argv)
    args.guest_cation = args.guest_cation or []
    args.valence = parse_key_values(args.valence, float)
    parent_counts = parse_formula(args.parent_formula)
    metadata_rows = parse_metadata_csv(args.metadata_csv)
    site_state_rows = parse_site_state_csv(args.site_state_csv)
    records = [
        build_record(run, args, parent_counts, metadata_rows, site_state_rows)
        for run in discover_runs(args)
    ]
    payload = {
        "schema": SCHEMA,
        "material": args.material,
        "phase": args.phase,
        "parent_formula": args.parent_formula,
        "host_cation": args.host_cation,
        "guest_cations": args.guest_cation,
        "oxygen": args.oxygen,
        "valence_model": args.valence,
        "records": records,
        "stage": "defect motif database; zentropy stage 1",
    }
    write_json(args.db, payload)
    write_record_csv(args.csv, records)
    print(f"Indexed motifs : {len(records)}")
    print(f"Wrote DB       : {args.db.resolve()}")
    print(f"Wrote CSV      : {args.csv.resolve()}")


def load_db(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected motif DB schema in {path}: {data.get('schema')}")
    return data


def record_matches(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.motif_id and record["motif_id"] not in set(args.motif_id):
        return False
    if args.family and record.get("motif_family") not in set(args.family):
        return False
    if args.tag and not set(args.tag).issubset(set(record.get("tags", []))):
        return False
    return True


def selected_records(db: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = [record for record in db["records"] if record_matches(record, args)]
    if args.max_records is not None:
        rows = rows[: args.max_records]
    if not rows:
        raise ValueError("No motif records matched the selection.")
    return rows


def format_magmom_line(values: list[float], decimals: int = 3) -> str:
    pieces = []
    for value in values:
        text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
        pieces.append("0" if text in ("", "-0") else text)
    return "MAGMOM = " + " ".join(pieces) + "\n"


def repeat_magmoms(values: list[float], repeat: tuple[int, int, int]) -> list[float]:
    nrepeat = repeat[0] * repeat[1] * repeat[2]
    return list(values) * nrepeat


def copy_template_files(template: Path | None, outdir: Path) -> None:
    if template is None:
        return
    for item in template.iterdir():
        if item.name == "POSCAR":
            continue
        target = outdir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def export_mlip_main(argv: list[str] | None = None) -> None:
    parser = build_export_parser()
    args = parser.parse_args(argv)
    db = load_db(args.db)
    records = selected_records(db, args)
    repeat = tuple(args.repeat)
    manifest = []
    args.outdir.mkdir(parents=True, exist_ok=True)
    for record in records:
        atoms = read(Path(record["source_structure_file"]), index=-1)
        values = record.get("magmom", {}).get("values", [])
        if values and len(values) == len(atoms):
            atoms.set_initial_magnetic_moments(values)
        out = args.outdir / record["motif_id"]
        out.mkdir(parents=True, exist_ok=True)
        copy_template_files(args.vasp_template, out)
        expanded = atoms.repeat(repeat)
        write(out / "POSCAR", expanded, format="vasp", direct=True, vasp5=True, sort=False)
        magmoms = repeat_magmoms(values, repeat) if values else []
        if args.write_magmom and magmoms:
            (out / "INCAR.magmom").write_text(format_magmom_line(magmoms), encoding="utf-8")
        if args.write_extxyz:
            write(out / f"{record['motif_id']}.extxyz", expanded, format="extxyz")
        meta = {
            "motif_id": record["motif_id"],
            "source_record": record,
            "repeat": repeat,
            "natoms": len(expanded),
            "purpose": "MLIP defect-motif input structure exported from zentropy motif DB",
        }
        write_json(out / "mlip_export_metadata.json", meta)
        manifest.append(
            {
                "motif_id": record["motif_id"],
                "out_dir": str(out.resolve()),
                "natoms": len(expanded),
                "repeat": " ".join(str(x) for x in repeat),
                "guest_cation_fraction": record["size_normalization"]["guest_cation_fraction"],
                "oxygen_delta_per_formula_unit": record["size_normalization"][
                    "oxygen_delta_per_formula_unit"
                ],
                "source_structure_file": record["source_structure_file"],
            }
        )
    write_rows(args.outdir / "mlip_export_manifest.csv", manifest)
    print(f"Exported structures : {len(manifest)}")
    print(f"Output directory    : {args.outdir.resolve()}")
    print(f"Manifest            : {(args.outdir / 'mlip_export_manifest.csv').resolve()}")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_main(argv: list[str] | None = None) -> None:
    parser = build_summarize_parser()
    args = parser.parse_args(argv)
    db = load_db(args.db)
    rows = selected_records(db, args)
    print(f"Motif DB: {args.db.resolve()}")
    print(f"Records : {len(rows)} / {len(db['records'])}")
    header = ["motif_id", "family", "x_guest", "delta", "E/fu_eV", "magmom", "tags"]
    print("  ".join(f"{name:>12s}" for name in header))
    for rec in rows:
        norm = rec["size_normalization"]
        e_fu = rec.get("energy_per_formula_unit_eV")
        mag = rec.get("magmom", {}).get("by_element", {})
        mag_text = ",".join(f"{k}:{v['mean']:.2f}" for k, v in mag.items()) if mag else ""
        values = [
            rec["motif_id"],
            str(rec.get("motif_family") or ""),
            f"{norm['guest_cation_fraction']:.5f}",
            f"{norm['oxygen_delta_per_formula_unit']:.5f}",
            "" if e_fu is None else f"{e_fu:.8f}",
            mag_text,
            ",".join(rec.get("tags", [])),
        ]
        print("  ".join(f"{value:>12s}" for value in values))


def build_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_motif_db index",
        description="Index refined VASP defect motifs into a zentropy-ready motif database.",
    )
    parser.add_argument("--run", action="append", type=Path, default=[], help="VASP run directory. Repeatable.")
    parser.add_argument("--root", type=Path, help="Root containing motif run directories.")
    parser.add_argument("--glob", default="*", help="Directory glob under --root.")
    parser.add_argument("--db", type=Path, default=Path("defect_motif_db.json"))
    parser.add_argument("--csv", type=Path, default=Path("defect_motif_index.csv"))
    parser.add_argument("--metadata-csv", type=Path, help="Optional per-run motif metadata CSV.")
    parser.add_argument("--site-state-csv", type=Path, help="Optional site valence/spin-state CSV.")
    parser.add_argument("--material", default="(Gd,U)O2")
    parser.add_argument("--phase", default="defect_fluorite")
    parser.add_argument("--parent-formula", default="UO2")
    parser.add_argument("--host-cation", default="U")
    parser.add_argument("--guest-cation", nargs="*", default=["Gd"])
    parser.add_argument("--oxygen", default="O")
    parser.add_argument("--valence", nargs="*", default=["U=4", "Gd=3", "O=-2"])
    parser.add_argument("--motif-family", default="user_refined")
    parser.add_argument("--motif-type", default="defect_motif")
    parser.add_argument("--defect-label", default="")
    parser.add_argument("--charge-state", default="")
    parser.add_argument("--degeneracy", type=float, default=1.0)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--notes", default="")
    return parser


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=Path("defect_motif_db.json"))
    parser.add_argument("--motif-id", action="append", default=[])
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--max-records", type=int)


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_motif_db export-mlip",
        description="Export selected motif records as VASP/MLIP input structures.",
    )
    add_selection_args(parser)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--repeat", type=int, nargs=3, default=[1, 1, 1])
    parser.add_argument("--vasp-template", type=Path, help="Optional template files copied beside POSCAR.")
    parser.add_argument("--write-magmom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-extxyz", action="store_true")
    return parser


def build_summarize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_motif_db summarize",
        description="Print a compact readout of a zentropy defect motif database.",
    )
    add_selection_args(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_motif_db",
        description="Stage-1 defect motif database for zentropy-guided defect thermodynamics.",
    )
    sub = parser.add_subparsers(dest="action")
    sub.add_parser("index", help="Index refined VASP defect motifs.")
    sub.add_parser("summarize", help="Print a compact database readout.")
    sub.add_parser("export-mlip", help="Export selected motifs as MLIP/VASP structures.")
    return parser


def main(argv: list[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    if not raw or raw[0] in ("-h", "--help"):
        build_parser().parse_args(raw)
        return
    action, rest = raw[0], raw[1:]
    if action == "index":
        index_main(rest)
    elif action == "summarize":
        summarize_main(rest)
    elif action == "export-mlip":
        export_mlip_main(rest)
    else:
        build_parser().error(f"unknown action: {action}")


if __name__ == "__main__":
    main()
