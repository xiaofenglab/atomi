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


def parse_moment_state_specs(items: list[str] | None) -> dict[str, list[tuple[float, str]]]:
    """Parse Element:magnitude=label mappings for spin-index site labels."""
    result: dict[str, list[tuple[float, str]]] = {}
    for item in items or []:
        if "=" not in item or ":" not in item.split("=", 1)[0]:
            raise ValueError(f"Expected Element:MAG=LABEL item, got: {item}")
        left, label = item.split("=", 1)
        element, raw_magnitude = left.split(":", 1)
        result.setdefault(element.strip(), []).append((abs(float(raw_magnitude)), label.strip()))
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


def merge_metadata_rows(
    base: dict[str, dict[str, str]],
    override: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    merged = {key: dict(value) for key, value in base.items()}
    for key, row in override.items():
        merged.setdefault(key, {}).update(row)
    return merged


def merge_site_state_rows(
    base: dict[str, list[dict[str, Any]]],
    extra: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged = {key: list(value) for key, value in base.items()}
    for key, rows in extra.items():
        merged.setdefault(key, []).extend(rows)
    return merged


def spin_label_for_moment(
    element: str,
    magmom: float,
    moment_states: dict[str, list[tuple[float, str]]],
    tolerance: float,
) -> str:
    sign = "up" if magmom > 0 else "down" if magmom < 0 else "zero"
    magnitude = abs(float(magmom))
    best_label = None
    best_delta = None
    for ref, label in moment_states.get(element, []):
        delta = abs(magnitude - ref)
        if delta <= tolerance and (best_delta is None or delta < best_delta):
            best_label = label
            best_delta = delta
    if best_label:
        return f"{best_label}_{sign}"
    return f"{element}|m={magnitude:.3f}|{sign}"


def classify_spin_order(values: list[float], threshold: float) -> dict[str, Any]:
    """Classify a set of collinear moments as a lightweight diagnostic."""
    active = [float(value) for value in values if abs(float(value)) >= threshold]
    zero_count = len(values) - len(active)
    up_count = sum(1 for value in active if value > 0)
    down_count = sum(1 for value in active if value < 0)
    net_moment = sum(active)
    abs_moment_sum = sum(abs(value) for value in active)
    if not active:
        label = "nonmagnetic"
    elif len(active) == 1:
        label = "single_spin"
    elif up_count == len(active) or down_count == len(active):
        label = "FM"
    elif up_count == down_count and abs_moment_sum and abs(net_moment) / abs_moment_sum <= 0.25:
        label = "AFM"
    else:
        label = "AFM-like"
    return {
        "label": label,
        "active": len(active),
        "up": up_count,
        "down": down_count,
        "zero": zero_count,
        "net_moment": net_moment,
        "abs_moment_sum": abs_moment_sum,
    }


def format_spin_order(info: dict[str, Any]) -> str:
    label = str(info.get("label", "unknown"))
    return (
        f"{label}(active={int(info.get('active', 0))},"
        f"up={int(info.get('up', 0))},down={int(info.get('down', 0))},"
        f"zero={int(info.get('zero', 0))},net={float(info.get('net_moment', 0.0)):.6g})"
    )


def spin_order_tag(label: str) -> str:
    clean = re.sub(r"[^0-9a-z]+", "_", label.lower()).strip("_")
    return clean or "unknown"


def spin_index_aliases(run_dir: str, motif_id: str) -> set[str]:
    aliases = {run_dir, motif_id}
    if run_dir:
        path = Path(run_dir).expanduser()
        aliases.add(path.name)
        try:
            aliases.add(str(path.resolve()))
        except OSError:
            pass
    return {item for item in aliases if item}


def parse_spin_index_csv(
    path: Path | None,
    moment_states: dict[str, list[tuple[float, str]]],
    tolerance: float,
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, Any]]]]:
    """Convert magit enum spin_index.csv into motif metadata and site-state rows."""
    if path is None:
        return {}, {}
    metadata: dict[str, dict[str, str]] = {}
    site_states: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_dir = (row.get("run_dir") or "").strip()
            motif_id = (row.get("name") or Path(run_dir).name or "spin_variant").strip()
            dopant_mode = (row.get("dopant_mode") or "").strip()
            host_mode = (row.get("host_mode") or "").strip()
            tags = ["magit", "spin_enum"]
            if dopant_mode:
                tags.append(f"dopant_{dopant_mode}")
            if host_mode:
                tags.append(f"host_{host_mode}")
            metadata_row = {
                "motif_id": motif_id,
                "motif_family": "magit_spin_variant",
                "motif_type": "magnetic_spin_configuration",
                "defect_label": row.get("defect_label", ""),
                "tags": ";".join(tags),
                "notes": f"Imported from magit spin index: {path}",
            }
            states: list[dict[str, Any]] = []
            try:
                moments = json.loads(row.get("moments_by_atom") or "[]")
            except json.JSONDecodeError:
                moments = []
            for item in moments:
                try:
                    atom_index = int(item.get("atom"))
                    element = str(item.get("element", "")).strip()
                    magmom = float(item.get("magmom", 0.0))
                except (TypeError, ValueError, AttributeError):
                    continue
                if not element:
                    continue
                states.append(
                    {
                        "atom_index_1based": atom_index,
                        "element": element,
                        "magmom": magmom,
                        "spin_label": spin_label_for_moment(
                            element,
                            magmom,
                            moment_states,
                            tolerance,
                        ),
                        "role": "magit_spin_variant_site",
                    }
                )
            for key in spin_index_aliases(run_dir, motif_id):
                metadata[key] = dict(metadata_row)
                site_states[key] = [dict(state) for state in states]
    return metadata, site_states


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
        "motif_metadata": meta,
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


def resolve_scan_path(raw: str | None, base: Path) -> Path | None:
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def first_existing_path(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def scan_structure_for_run(run_dir: Path) -> Path | None:
    return first_existing_path(
        [
            run_dir / "CONTCAR",
            run_dir / "POSCAR",
            run_dir / "vasprun.xml",
            run_dir / "OUTCAR",
            run_dir / "OUTCAR.gz",
        ]
    )


def scan_outcar_for_run(run_dir: Path) -> Path | None:
    return first_existing_path([run_dir / "OUTCAR", run_dir / "OUTCAR.gz"])


def scan_incar_for_run(run_dir: Path) -> Path | None:
    return first_existing_path([run_dir / "INCAR"])


def scan_entries_from_csv(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    base = path.resolve().parent
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            run = resolve_scan_path(row.get("run_dir") or row.get("run"), base)
            structure = first_existing_path(
                [
                    resolve_scan_path(row.get("structure"), base),
                    resolve_scan_path(row.get("contcar"), base),
                    resolve_scan_path(row.get("poscar"), base),
                ]
            )
            outcar = first_existing_path(
                [
                    resolve_scan_path(row.get("outcar"), base),
                    resolve_scan_path(row.get("OUTCAR"), base),
                ]
            )
            incar = first_existing_path(
                [
                    resolve_scan_path(row.get("incar"), base),
                    resolve_scan_path(row.get("INCAR"), base),
                ]
            )
            if run is not None:
                structure = structure or scan_structure_for_run(run)
                outcar = outcar or scan_outcar_for_run(run)
                incar = incar or scan_incar_for_run(run)
            elif structure is not None:
                run = structure.parent
            else:
                run = (base / f"motif_row_{index:04d}").resolve()
            entries.append(
                {
                    "run_dir": run,
                    "structure": structure,
                    "outcar": outcar,
                    "incar": incar,
                    "row": {k: v for k, v in row.items() if v not in (None, "")},
                }
            )
    return entries


def discover_scan_entries(args: argparse.Namespace) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if args.input_csv:
        entries.extend(scan_entries_from_csv(args.input_csv))
    if args.run or args.root:
        for run in discover_runs(args):
            entries.append(
                {
                    "run_dir": run,
                    "structure": scan_structure_for_run(run),
                    "outcar": scan_outcar_for_run(run),
                    "incar": scan_incar_for_run(run),
                    "row": {},
                }
            )
    if not entries:
        raise ValueError("No scan entries were provided. Use --run, --root/--glob, or --input-csv.")
    return entries


def safe_motif_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_.+-]+", "_", value.strip()).strip("_")
    return slug or "motif"


def magmom_values_from_sources(
    outcar: Path | None,
    incar: Path | None,
    natoms: int,
) -> tuple[list[float], str]:
    if outcar is not None and outcar.exists():
        values = parse_final_outcar_magmoms(outcar, natoms)
        if values is not None:
            return values, outcar.name
    if incar is not None and incar.exists():
        values = existing_magmom_values(incar, natoms)
        if values is not None:
            return values, incar.name
    return [], "missing"


def valence_from_spin_label(label: str) -> float | None:
    base = label.rsplit("_", 1)[0]
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([+-])", base)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2) == "+" else -value


def infer_defect_metadata(
    counts: dict[str, int],
    parent_counts: dict[str, float],
    host_cation: str | None,
    guest_cations: list[str],
    oxygen: str,
) -> dict[str, Any]:
    norm = size_normalization(counts, parent_counts, host_cation, guest_cations, oxygen)
    guest_parts = [f"{element}{counts[element]}" for element in guest_cations if counts.get(element, 0)]
    oxygen_delta_count = 0
    if finite(norm.get("expected_oxygen_count")):
        oxygen_delta_count = int(round(float(norm["expected_oxygen_count"]) - counts.get(oxygen, 0)))

    labels: list[str] = []
    tags = ["auto_metadata"]
    if guest_parts:
        labels.extend(guest_parts)
        tags.extend(["dopant_substitution", *[part.lower() for part in guest_parts]])
    if oxygen_delta_count > 0:
        labels.append(f"O_V{oxygen_delta_count}")
        tags.append("oxygen_vacancy")
    elif oxygen_delta_count < 0:
        labels.append(f"O_i{abs(oxygen_delta_count)}")
        tags.append("oxygen_interstitial")

    if guest_parts and oxygen_delta_count > 0:
        family = "dopant_oxygen_vacancy_complex"
    elif guest_parts:
        family = "dopant_substitution"
    elif oxygen_delta_count > 0:
        family = "oxygen_vacancy"
    elif oxygen_delta_count < 0:
        family = "oxygen_interstitial"
    else:
        family = "stoichiometric_or_unknown"
        labels.append("stoichiometric")

    tags.extend(
        [
            f"x_guest={norm['guest_cation_fraction']:.6g}",
            f"oxygen_delta={norm['oxygen_delta_per_formula_unit']:.6g}",
            f"formula={reduced_formula(counts)}",
        ]
    )
    return {
        "motif_family": family,
        "motif_type": "defect_motif",
        "defect_label": "_".join(labels),
        "tags": tags,
        "normalization": norm,
        "oxygen_defect_count": oxygen_delta_count,
    }


def write_auto_metadata_csvs(
    metadata_path: Path,
    site_state_path: Path,
    report_path: Path,
    metadata_rows: list[dict[str, Any]],
    site_state_rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
) -> None:
    metadata_fields = [
        "run",
        "motif_id",
        "motif_family",
        "motif_type",
        "defect_label",
        "tags",
        "degeneracy",
        "charge_state",
        "notes",
        "source_structure",
        "source_outcar",
        "source_incar",
        "formula",
        "natoms",
        "formula_units",
        "guest_cation_fraction",
        "oxygen_delta_per_formula_unit",
        "magmom_source",
        "spin_order_host",
        "spin_order_host_detail",
        "spin_order_all",
        "spin_order_all_detail",
    ]
    site_state_fields = [
        "run",
        "motif_id",
        "atom_index_1based",
        "element",
        "valence",
        "magmom",
        "spin_label",
        "role",
    ]
    report_fields = [
        "run",
        "motif_id",
        "status",
        "formula",
        "natoms",
        "motif_family",
        "defect_label",
        "formula_units",
        "guest_cation_fraction",
        "oxygen_delta_per_formula_unit",
        "magmom_source",
        "spin_order_host",
        "spin_order_host_detail",
        "spin_order_all",
        "spin_order_all_detail",
        "site_states",
        "message",
    ]
    write_rows_with_fields(metadata_path, metadata_rows, metadata_fields)
    write_rows_with_fields(site_state_path, site_state_rows, site_state_fields)
    write_rows_with_fields(report_path, report_rows, report_fields)


def write_rows_with_fields(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def materialize_scan_run(
    root: Path | None,
    motif_id: str,
    atoms: Atoms,
    structure: Path,
    outcar: Path | None,
    incar: Path | None,
    case_info_payload: dict[str, Any],
) -> Path | None:
    if root is None:
        return None
    outdir = root / safe_motif_slug(motif_id)
    suffix = 2
    while outdir.exists():
        outdir = root / f"{safe_motif_slug(motif_id)}_{suffix:03d}"
        suffix += 1
    outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / "POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=False)
    if outcar is not None and outcar.exists():
        shutil.copy2(outcar, outdir / "OUTCAR")
    if incar is not None and incar.exists():
        shutil.copy2(incar, outdir / "INCAR")
    write_json(outdir / "case_info.json", case_info_payload)
    return outdir.resolve()


def auto_metadata_main(argv: list[str] | None = None) -> None:
    parser = build_auto_metadata_parser()
    args = parser.parse_args(argv)
    args.guest_cation = args.guest_cation or []
    args.valence = parse_key_values(args.valence, float)
    args.moment_state = parse_moment_state_specs(args.moment_state)
    parent_counts = parse_formula(args.parent_formula)
    materialize_root = args.materialize_root.resolve() if args.materialize_root else None
    if materialize_root is not None:
        materialize_root.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, Any]] = []
    site_state_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for entry in discover_scan_entries(args):
        run_dir = Path(entry["run_dir"]).resolve()
        structure = entry.get("structure")
        row = entry.get("row", {})
        if structure is None or not Path(structure).exists():
            report_rows.append(
                {
                    "run": str(run_dir),
                    "motif_id": row.get("motif_id") or run_dir.name,
                    "status": "skipped",
                    "message": "No CONTCAR/POSCAR/structure path found.",
                }
            )
            continue
        structure = Path(structure).resolve()
        outcar = Path(entry["outcar"]).resolve() if entry.get("outcar") else None
        incar = Path(entry["incar"]).resolve() if entry.get("incar") else None
        try:
            atoms = read(structure, index=-1)
        except Exception as exc:
            report_rows.append(
                {
                    "run": str(run_dir),
                    "motif_id": row.get("motif_id") or run_dir.name,
                    "status": "skipped",
                    "message": f"Could not read structure {structure}: {exc}",
                }
            )
            continue

        counts = count_symbols(atoms)
        inferred = infer_defect_metadata(
            counts,
            parent_counts,
            args.host_cation,
            args.guest_cation,
            args.oxygen,
        )
        norm = inferred["normalization"]
        motif_id = row.get("motif_id") or row.get("name") or run_dir.name
        motif_id = safe_motif_slug(str(motif_id))
        magmoms, magmom_source = magmom_values_from_sources(outcar, incar, len(atoms))
        source_run = run_dir
        case_payload = {
            "case_name": motif_id,
            "family": row.get("motif_family") or inferred["motif_family"],
            "source_run_dir": str(run_dir),
            "source_structure": str(structure),
            "source_outcar": str(outcar) if outcar else "",
            "source_incar": str(incar) if incar else "",
            "generated_by": "zentropy_motif_db auto-metadata",
        }
        materialized = materialize_scan_run(
            materialize_root,
            motif_id,
            atoms,
            structure,
            outcar,
            incar,
            case_payload,
        )
        if materialized is not None:
            source_run = materialized

        metadata_tags = split_tags(row.get("tags"))
        metadata_tags.extend(inferred["tags"])
        metadata_row = {
            "run": str(source_run.resolve()),
            "motif_id": motif_id,
            "motif_family": row.get("motif_family") or row.get("family") or inferred["motif_family"],
            "motif_type": row.get("motif_type") or inferred["motif_type"],
            "defect_label": row.get("defect_label") or inferred["defect_label"],
            "tags": ";".join(dict.fromkeys(metadata_tags)),
            "degeneracy": row.get("degeneracy") or args.degeneracy,
            "charge_state": row.get("charge_state") or "",
            "notes": row.get("notes") or (
                f"Auto-generated metadata; expected O={norm['expected_oxygen_count']:.6g}, "
                f"actual O={counts.get(args.oxygen, 0)}."
            ),
            "source_structure": str(structure),
            "source_outcar": str(outcar) if outcar else "",
            "source_incar": str(incar) if incar else "",
            "formula": reduced_formula(counts),
            "natoms": len(atoms),
            "formula_units": norm["formula_units"],
            "guest_cation_fraction": norm["guest_cation_fraction"],
            "oxygen_delta_per_formula_unit": norm["oxygen_delta_per_formula_unit"],
            "magmom_source": magmom_source,
        }

        symbols = atoms.get_chemical_symbols()
        magnetic_elements = set(args.moment_state)
        if args.site_element:
            magnetic_elements.update(args.site_element)
        motif_site_rows: list[dict[str, Any]] = []
        for atom_index, (element, magmom) in enumerate(zip(symbols, magmoms), start=1):
            if element not in magnetic_elements and abs(float(magmom)) < args.magmom_min_abs:
                continue
            spin_label = spin_label_for_moment(
                element,
                float(magmom),
                args.moment_state,
                args.moment_state_tolerance,
            )
            if element in args.guest_cation:
                role = "dopant"
            elif element == args.host_cation:
                role = "host"
            else:
                role = "magnetic_site"
            valence = valence_from_spin_label(spin_label)
            site_row = {
                "run": str(source_run.resolve()),
                "motif_id": motif_id,
                "atom_index_1based": atom_index,
                "element": element,
                "valence": "" if valence is None else f"{valence:g}",
                "magmom": f"{float(magmom):.6g}",
                "spin_label": spin_label,
                "role": role,
            }
            site_state_rows.append(site_row)
            motif_site_rows.append(site_row)

        all_moments = [float(row["magmom"]) for row in motif_site_rows]
        host_moments = [
            float(row["magmom"])
            for row in motif_site_rows
            if args.host_cation and row["element"] == args.host_cation
        ]
        all_spin_order = classify_spin_order(all_moments, args.magmom_min_abs)
        host_spin_order = classify_spin_order(host_moments, args.magmom_min_abs)
        metadata_row["spin_order_host"] = host_spin_order["label"]
        metadata_row["spin_order_host_detail"] = format_spin_order(host_spin_order)
        metadata_row["spin_order_all"] = all_spin_order["label"]
        metadata_row["spin_order_all_detail"] = format_spin_order(all_spin_order)
        metadata_tags.extend(
            [
                f"spin_host_{spin_order_tag(str(host_spin_order['label']))}",
                f"spin_all_{spin_order_tag(str(all_spin_order['label']))}",
            ]
        )
        metadata_row["tags"] = ";".join(dict.fromkeys(metadata_tags))
        metadata_row["notes"] = (
            f"{metadata_row['notes']} "
            f"spin_order_host={metadata_row['spin_order_host_detail']}; "
            f"spin_order_all={metadata_row['spin_order_all_detail']}."
        )
        metadata_rows.append(metadata_row)

        report_rows.append(
            {
                "run": str(source_run.resolve()),
                "motif_id": motif_id,
                "status": "ok",
                "formula": metadata_row["formula"],
                "natoms": len(atoms),
                "motif_family": metadata_row["motif_family"],
                "defect_label": metadata_row["defect_label"],
                "formula_units": norm["formula_units"],
                "guest_cation_fraction": norm["guest_cation_fraction"],
                "oxygen_delta_per_formula_unit": norm["oxygen_delta_per_formula_unit"],
                "magmom_source": magmom_source,
                "spin_order_host": metadata_row["spin_order_host"],
                "spin_order_host_detail": metadata_row["spin_order_host_detail"],
                "spin_order_all": metadata_row["spin_order_all"],
                "spin_order_all_detail": metadata_row["spin_order_all_detail"],
                "site_states": len(motif_site_rows),
                "message": "",
            }
        )

    write_auto_metadata_csvs(
        args.metadata_csv,
        args.site_state_csv,
        args.report_csv,
        metadata_rows,
        site_state_rows,
        report_rows,
    )
    print(f"Scanned motifs      : {len(metadata_rows)}")
    print(f"Wrote metadata CSV  : {args.metadata_csv.resolve()}")
    print(f"Wrote site states   : {args.site_state_csv.resolve()}")
    print(f"Wrote scan report   : {args.report_csv.resolve()}")
    if materialize_root is not None:
        print(f"Materialized runs   : {materialize_root.resolve()}")
    print("Next zentropy step:")
    print(
        "  zentropy_motif_db index "
        f"--metadata-csv {args.metadata_csv.resolve()} "
        f"--site-state-csv {args.site_state_csv.resolve()}"
    )


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
        "spin_order_host",
        "spin_order_all",
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
                "spin_order_host": rec.get("motif_metadata", {}).get("spin_order_host", ""),
                "spin_order_all": rec.get("motif_metadata", {}).get("spin_order_all", ""),
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
    args.moment_state = parse_moment_state_specs(args.moment_state)
    parent_counts = parse_formula(args.parent_formula)
    spin_metadata, spin_site_states = parse_spin_index_csv(
        args.spin_index,
        args.moment_state,
        args.moment_state_tolerance,
    )
    metadata_rows = merge_metadata_rows(spin_metadata, parse_metadata_csv(args.metadata_csv))
    site_state_rows = merge_site_state_rows(spin_site_states, parse_site_state_csv(args.site_state_csv))
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
    parser.add_argument(
        "--spin-index",
        type=Path,
        help="Optional magit enum spin_index.csv to annotate spin/MAGMOM variants.",
    )
    parser.add_argument(
        "--moment-state",
        action="append",
        default=[],
        help="Map spin-index magnitudes to labels, e.g. --moment-state U:2=U4+.",
    )
    parser.add_argument(
        "--moment-state-tolerance",
        type=float,
        default=0.25,
        help="Tolerance for matching --moment-state magnitudes.",
    )
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


def add_scan_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", action="append", type=Path, default=[], help="VASP run directory. Repeatable.")
    parser.add_argument("--root", type=Path, help="Root containing motif run directories.")
    parser.add_argument("--glob", default="*", help="Directory glob under --root.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        help=(
            "Optional CSV with run/run_dir and/or structure/poscar/contcar, "
            "outcar, incar columns. Relative paths are resolved from the CSV."
        ),
    )


def build_auto_metadata_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zentropy_motif_db auto-metadata",
        description=(
            "Scan existing VASP defect runs and generate motif_metadata.csv plus "
            "site_states.csv for zentropy_motif_db index."
        ),
    )
    add_scan_source_args(parser)
    parser.add_argument("--metadata-csv", type=Path, default=Path("motif_metadata.csv"))
    parser.add_argument("--site-state-csv", type=Path, default=Path("site_states.csv"))
    parser.add_argument("--report-csv", type=Path, default=Path("motif_auto_metadata_report.csv"))
    parser.add_argument(
        "--materialize-root",
        type=Path,
        help=(
            "Optional folder where POSCAR/OUTCAR/INCAR from arbitrary paths are "
            "copied into index-ready VASP run directories."
        ),
    )
    parser.add_argument("--parent-formula", default="UO2")
    parser.add_argument("--host-cation", default="U")
    parser.add_argument("--guest-cation", nargs="*", default=["Gd"])
    parser.add_argument("--oxygen", default="O")
    parser.add_argument("--valence", nargs="*", default=["U=4", "Gd=3", "O=-2"])
    parser.add_argument(
        "--moment-state",
        action="append",
        default=[],
        help="Map magnetic moment magnitudes to labels, e.g. --moment-state U:2=U4+.",
    )
    parser.add_argument("--moment-state-tolerance", type=float, default=0.35)
    parser.add_argument(
        "--site-element",
        action="append",
        default=[],
        help="Always write site-state rows for this element. Repeatable.",
    )
    parser.add_argument(
        "--magmom-min-abs",
        type=float,
        default=0.2,
        help="Write site-state rows for non-declared elements above this |magmom|.",
    )
    parser.add_argument("--degeneracy", type=float, default=1.0)
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
    sub.add_parser("auto-metadata", help="Scan VASP runs and write metadata CSV inputs.")
    sub.add_parser("generate-spins", help="Generate magit spin-variant VASP folders.")
    sub.add_parser("summarize", help="Print a compact database readout.")
    sub.add_parser("export-mlip", help="Export selected motifs as MLIP/VASP structures.")
    return parser


def generate_spins_main(argv: list[str] | None = None) -> None:
    from atomi.vasp.magmom import build_enum_parser, enumerate_spin_configs

    parser = build_enum_parser()
    parser.prog = "zentropy_motif_db generate-spins"
    parser.description = (
        "Generate physics-informed spin/MAGMOM motif variants using the magit enum engine."
    )
    args = parser.parse_args(argv)
    enumerate_spin_configs(args)
    spin_index = args.index.resolve() if args.index else args.output_root.resolve() / "spin_index.csv"
    print("Next zentropy step:")
    print(f"  zentropy_motif_db index --root {args.output_root.resolve()} --spin-index {spin_index}")


def main(argv: list[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else argv
    if not raw or raw[0] in ("-h", "--help"):
        build_parser().parse_args(raw)
        return
    action, rest = raw[0], raw[1:]
    if action == "index":
        index_main(rest)
    elif action == "auto-metadata":
        auto_metadata_main(rest)
    elif action == "generate-spins":
        generate_spins_main(rest)
    elif action == "summarize":
        summarize_main(rest)
    elif action == "export-mlip":
        export_mlip_main(rest)
    else:
        build_parser().error(f"unknown action: {action}")


if __name__ == "__main__":
    main()
