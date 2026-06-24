"""Static DFT+U magnetic-branch entropy workflow for VASP.

This module treats magnetic/spin arrangements as a discrete microstate ensemble,
closely analogous to POCC/zentropy motif ensembles.  A set of static DFT+U
branch energies is converted to F_mag(T), S_mag(T), U_mag(T), and branch
probabilities through a Boltzmann partition function.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import shutil
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atomi.vasp.checks import ENERGY_PATTERNS
from atomi.vasp.magmom import PoscarStructure, read_poscar_structure

SCHEMA = "atomi.vasp.magnetic_entropy.v1"
KB_EV_PER_K = 8.617333262145e-5
EV_PER_MOL_TO_J_PER_MOL = 96485.33212331002
R_J_PER_MOL_K = 8.31446261815324

ENERGY_ALIASES = {
    "e0": "e0",
    "zero_temp": "e0",
    "without_entropy": "without_entropy",
    "sigma0": "without_entropy",
    "toten": "toten",
    "free_energy": "toten",
    "f": "f",
    "dav": "dav",
}

BRANCH_FIELDS = [
    "label",
    "run_dir",
    "pattern",
    "degeneracy",
    "energy_eV_cell",
    "energy_kind",
    "status",
    "source",
    "total_moment_muB",
    "note",
]

THERMO_FIELDS = [
    "T_K",
    "formula_units",
    "n_branches",
    "Z",
    "E_min_eV_cell",
    "U_rel_eV_cell",
    "F_rel_eV_cell",
    "S_eV_cell_K",
    "S_J_mol_cell_K",
    "U_rel_eV_per_fu",
    "F_rel_eV_per_fu",
    "S_eV_fu_K",
    "S_J_mol_fu_K",
    "g_eff_per_cell",
    "g_eff_per_fu",
    "dominant_branch",
    "dominant_probability",
]

PROBABILITY_FIELDS = [
    "T_K",
    "label",
    "probability",
    "degeneracy",
    "delta_E_eV_cell",
    "delta_E_eV_per_fu",
    "run_dir",
]


@dataclass
class BranchInput:
    label: str
    run_dir: str = ""
    pattern: str = ""
    degeneracy: float = 1.0
    energy_eV_cell: float | None = None
    energy_kind: str = ""
    status: str = "PENDING"
    source: str = ""
    total_moment_muB: float | None = None
    note: str = ""


@dataclass
class MagneticThermoPoint:
    T_K: float
    formula_units: float
    n_branches: int
    Z: float
    E_min_eV_cell: float
    U_rel_eV_cell: float
    F_rel_eV_cell: float
    S_eV_cell_K: float
    S_J_mol_cell_K: float
    U_rel_eV_per_fu: float
    F_rel_eV_per_fu: float
    S_eV_fu_K: float
    S_J_mol_fu_K: float
    g_eff_per_cell: float
    g_eff_per_fu: float
    dominant_branch: str
    dominant_probability: float


def parse_temperature_values(values: list[str] | None) -> list[float]:
    if not values:
        return [300.0]
    temps: list[float] = []
    for raw in values:
        for chunk in str(raw).replace(",", " ").split():
            if not chunk:
                continue
            if ":" in chunk:
                parts = [float(x) for x in chunk.split(":")]
                if len(parts) not in {2, 3}:
                    raise ValueError(f"temperature grid must be start:stop[:step], got {chunk}")
                start, stop = parts[0], parts[1]
                step = parts[2] if len(parts) == 3 else 100.0
                if step <= 0:
                    raise ValueError("temperature step must be positive")
                value = start
                while value <= stop + 1.0e-9:
                    temps.append(value)
                    value += step
            else:
                temps.append(float(chunk))
    return sorted(dict.fromkeys(temps))


def parse_key_float_map(values: list[str] | None, default: float = 0.0) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if not values:
        return mapping
    for raw in values:
        for chunk in str(raw).replace(",", " ").split():
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(f"expected ELEMENT=value token, got {chunk}")
            key, value = chunk.split("=", 1)
            mapping[key.strip()] = float(value)
    if not mapping and default:
        mapping["*"] = default
    return mapping


def species_per_atom(structure: PoscarStructure) -> list[str]:
    atoms: list[str] = []
    for symbol, count in zip(structure.species.symbols, structure.species.counts):
        atoms.extend([symbol] * count)
    return atoms


def parse_species_list(values: list[str] | None) -> set[str]:
    result: set[str] = set()
    for raw in values or []:
        for chunk in str(raw).replace(",", " ").split():
            if chunk:
                result.add(chunk)
    return result


def moment_for_symbol(symbol: str, moment_map: dict[str, float], default: float) -> float:
    if symbol in moment_map:
        return moment_map[symbol]
    if "*" in moment_map:
        return moment_map["*"]
    return default


def build_moment_pattern(
    structure: PoscarStructure,
    *,
    magnetic_species: set[str],
    moment_map: dict[str, float],
    default_moment: float,
    pattern: str,
    layer_axis: int = 2,
) -> list[float]:
    atoms = species_per_atom(structure)
    mag_indices = [idx for idx, symbol in enumerate(atoms) if not magnetic_species or symbol in magnetic_species]
    moments = [0.0] * len(atoms)
    normalized = pattern.lower().replace("_", "-")
    if normalized in {"nonmag", "non-magnetic", "nm"}:
        return moments
    if not mag_indices:
        return moments

    if normalized in {"fm", "ferro", "ferromagnetic"}:
        signs = {idx: 1.0 for idx in mag_indices}
    elif normalized in {"afm-alt", "afm-index", "alternating"}:
        signs = {idx: (1.0 if order % 2 == 0 else -1.0) for order, idx in enumerate(mag_indices)}
    elif normalized in {"afm-layer", "layer", "layered"}:
        coords = sorted(
            [(idx, structure.scaled_positions[idx][layer_axis] % 1.0) for idx in mag_indices],
            key=lambda item: (item[1], item[0]),
        )
        half = len(coords) // 2
        signs = {idx: (1.0 if order < half else -1.0) for order, (idx, _) in enumerate(coords)}
        if len(coords) % 2:
            signs[coords[half][0]] = 0.0
        if abs(sum(signs.values())) == len([value for value in signs.values() if value != 0.0]):
            signs = {idx: (1.0 if order % 2 == 0 else -1.0) for order, idx in enumerate(mag_indices)}
    elif normalized in {"afm-block", "block"}:
        half = len(mag_indices) / 2.0
        signs = {idx: (1.0 if order < half else -1.0) for order, idx in enumerate(mag_indices)}
    else:
        raise ValueError(f"unknown magnetic pattern: {pattern}")

    for idx in mag_indices:
        symbol = atoms[idx]
        moments[idx] = signs[idx] * abs(moment_for_symbol(symbol, moment_map, default_moment))
    return moments


def magmom_line(values: list[float]) -> str:
    return " ".join(f"{value:.6g}" for value in values)


def patch_incar(
    template_text: str,
    *,
    moments: list[float],
    nonmagnetic: bool,
    static: bool,
    lorbit: int | None,
    extra_tags: list[str] | None = None,
) -> str:
    drop = {"ispin", "magmom", "lorbit"}
    if static:
        drop.update({"nsw", "ibrion", "isif"})
    out: list[str] = []
    for raw in template_text.splitlines():
        stripped = raw.strip()
        key = stripped.split("=", 1)[0].strip().lower() if "=" in stripped else ""
        if key in drop:
            continue
        out.append(raw.rstrip())
    out.append("")
    out.append("# Atomi magnetic entropy branch tags")
    out.append(f"ISPIN = {1 if nonmagnetic else 2}")
    if lorbit is not None:
        out.append(f"LORBIT = {lorbit}")
    if not nonmagnetic:
        out.append(f"MAGMOM = {magmom_line(moments)}")
    if static:
        out.extend(["NSW = 0", "IBRION = -1", "ISIF = 2"])
    for tag in extra_tags or []:
        tag = tag.strip()
        if tag:
            out.append(tag)
    return "\n".join(out).rstrip() + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_template_files(template_dir: Path | None, run_dir: Path, skip: set[str] | None = None) -> None:
    if template_dir is None:
        return
    skip = {name.upper() for name in (skip or set())}
    for name in ("POTCAR", "KPOINTS", "INCAR", "job.sh", "submit.sh", "run.sh", "sbatch.sh"):
        src = template_dir / name
        if name.upper() in skip or not src.exists():
            continue
        dst = run_dir / name
        if src.is_dir():
            continue
        shutil.copy2(src, dst)


def prepare_branches(args: argparse.Namespace) -> dict[str, Any]:
    structure_path = args.structure.resolve()
    structure = read_poscar_structure(structure_path)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    magnetic_species = parse_species_list(args.mag_species)
    moment_map = parse_key_float_map(args.moment_map)
    patterns = []
    for raw in args.pattern:
        patterns.extend([p for p in raw.replace(",", " ").split() if p])
    if not patterns:
        patterns = ["nonmag", "fm", "afm-alt", "afm-layer"]
    template_dir = args.template_dir.resolve() if args.template_dir else None
    template_incar = ""
    if args.incar_template:
        template_incar = args.incar_template.read_text(encoding="utf-8", errors="replace")
    elif template_dir and (template_dir / "INCAR").is_file():
        template_incar = (template_dir / "INCAR").read_text(encoding="utf-8", errors="replace")

    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        label = args.label_prefix + pattern.lower().replace("_", "-").replace(" ", "-")
        run_dir = outdir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        copy_template_files(template_dir, run_dir, skip={"INCAR", "POSCAR"})
        shutil.copy2(structure_path, run_dir / "POSCAR")
        moments = build_moment_pattern(
            structure,
            magnetic_species=magnetic_species,
            moment_map=moment_map,
            default_moment=args.default_moment,
            pattern=pattern,
            layer_axis=args.layer_axis,
        )
        nonmagnetic = all(abs(value) < 1.0e-12 for value in moments)
        if template_incar:
            incar = patch_incar(
                template_incar,
                moments=moments,
                nonmagnetic=nonmagnetic,
                static=args.static,
                lorbit=args.lorbit,
                extra_tags=args.incar_tag,
            )
            (run_dir / "INCAR").write_text(incar, encoding="utf-8")
        (run_dir / "MAGMOM.atomi.txt").write_text(magmom_line(moments) + "\n", encoding="utf-8")
        row = {
            "label": label,
            "run_dir": str(run_dir.relative_to(outdir)),
            "pattern": pattern,
            "degeneracy": args.degeneracy,
            "energy_eV_cell": "",
            "energy_kind": "",
            "status": "PREPARED",
            "source": str(structure_path),
            "total_moment_muB": sum(moments),
            "note": "static DFT+U magnetic entropy branch",
        }
        rows.append(row)
    case_csv = outdir / "magnetic_branch_manifest.csv"
    write_csv(case_csv, rows, BRANCH_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "stage": "prepare",
        "structure": str(structure_path),
        "outdir": str(outdir),
        "patterns": patterns,
        "magnetic_species": sorted(magnetic_species),
        "moment_map": moment_map,
        "default_moment": args.default_moment,
        "formula_units": args.formula_units,
        "case_manifest": str(case_csv),
        "notes": [
            "Run each prepared VASP static DFT+U branch to convergence, then collect energies.",
            "Branch degeneracy is a microstate weight; revise it if symmetry or POCC-style multiplicities are known.",
        ],
    }
    (outdir / "magnetic_entropy_prepare_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(rows)} magnetic branches in {outdir}")
    print(f"Manifest: {case_csv}")
    return metadata


def open_text_path(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def latest_energy_from_text(text: str, preferred_kind: str = "e0") -> tuple[float | None, str]:
    preferred_kind = ENERGY_ALIASES.get(preferred_kind, preferred_kind)
    order = [preferred_kind] + [kind for kind in ("e0", "without_entropy", "toten", "f", "dav") if kind != preferred_kind]
    for kind in order:
        pattern = ENERGY_PATTERNS.get(kind)
        if pattern is None:
            continue
        matches = pattern.findall(text)
        if matches:
            try:
                return float(matches[-1]), kind
            except ValueError:
                continue
    return None, ""


def extract_total_moment(text: str) -> float | None:
    candidates = re.findall(r"number of electron\s+[-+0-9.Ee]+\s+magnetization\s+([-+0-9.Ee]+)", text)
    if candidates:
        try:
            return float(candidates[-1])
        except ValueError:
            return None
    # Fallback to final magnetization table total row.
    matches = list(re.finditer(r"magnetization\s*\(x\)", text, re.IGNORECASE))
    if not matches:
        return None
    tail = text[matches[-1].end() :]
    for line in tail.splitlines():
        parts = line.split()
        if parts and parts[0].lower().startswith("tot"):
            try:
                return float(parts[-1])
            except ValueError:
                return None
    return None


def candidate_output_files(run_dir: Path) -> list[Path]:
    names = ["OUTCAR", "OUTCAR.gz", "OSZICAR", "OSZICAR.gz", "vasprun.xml", "vasprun.xml.gz"]
    files = [run_dir / name for name in names if (run_dir / name).is_file()]
    files.extend(sorted(run_dir.glob("*.out")))
    files.extend(sorted(run_dir.glob("vasp.out*")))
    return files


def parse_tar_outputs(tar_path: Path, preferred_kind: str) -> tuple[float | None, str, str, float | None]:
    best: tuple[float | None, str, str, float | None] = (None, "", "", None)
    try:
        with tarfile.open(tar_path, "r:*") as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            priority = ["OUTCAR", "OSZICAR", "vasprun.xml"]
            members.sort(key=lambda m: next((i for i, p in enumerate(priority) if m.name.endswith(p)), 99))
            for member in members:
                if not any(member.name.endswith(name) for name in ("OUTCAR", "OSZICAR", "vasprun.xml")):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                text = handle.read().decode("utf-8", errors="replace")
                energy, kind = latest_energy_from_text(text, preferred_kind)
                if energy is not None:
                    moment = extract_total_moment(text)
                    return energy, kind, f"{tar_path}:{member.name}", moment
    except (tarfile.TarError, OSError):
        return best
    return best


def collect_branches(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    manifest = args.manifest.resolve() if args.manifest else root / "magnetic_branch_manifest.csv"
    rows_in = read_csv_rows(manifest)
    rows: list[dict[str, Any]] = []
    for row in rows_in:
        run_dir = Path(row.get("run_dir") or row.get("run") or row.get("path") or "")
        resolved = run_dir if run_dir.is_absolute() else root / run_dir
        energy: float | None = _float_or_none(row.get("energy_eV_cell"))
        energy_kind = str(row.get("energy_kind") or "")
        source = str(row.get("source") or "")
        total_moment = _float_or_none(row.get("total_moment_muB"))
        status = str(row.get("status") or "")
        if args.force or energy is None:
            status = "NO_OUTPUT"
            for path in candidate_output_files(resolved):
                text = open_text_path(path)
                energy, energy_kind = latest_energy_from_text(text, args.energy)
                if energy is not None:
                    source = str(path)
                    total_moment = extract_total_moment(text)
                    status = "OK"
                    break
            if energy is None and args.include_archives:
                for tar_path in sorted(resolved.glob("*.tgz")) + sorted(resolved.glob("*.tar.gz")):
                    energy, energy_kind, source, total_moment = parse_tar_outputs(tar_path, args.energy)
                    if energy is not None:
                        status = "OK"
                        break
        rows.append(
            {
                "label": row.get("label") or resolved.name,
                "run_dir": str(run_dir),
                "pattern": row.get("pattern") or "",
                "degeneracy": _float_or_none(row.get("degeneracy")) or 1.0,
                "energy_eV_cell": "" if energy is None else f"{energy:.12f}",
                "energy_kind": energy_kind,
                "status": status,
                "source": source,
                "total_moment_muB": "" if total_moment is None else f"{total_moment:.8f}",
                "note": row.get("note") or "",
            }
        )
    output = args.output.resolve() if args.output else root / "magnetic_branch_energies.csv"
    write_csv(output, rows, BRANCH_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "stage": "collect",
        "root": str(root),
        "manifest": str(manifest),
        "output": str(output),
        "energy_preference": args.energy,
        "n_ok": sum(1 for row in rows if row["status"] == "OK"),
        "n_total": len(rows),
    }
    (output.with_suffix(".json")).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Collected {metadata['n_ok']} / {metadata['n_total']} branch energies -> {output}")
    return metadata


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_branch_inputs(path: Path, require_ok: bool = True) -> list[BranchInput]:
    branches: list[BranchInput] = []
    for row in read_csv_rows(path):
        status = str(row.get("status") or "").upper()
        energy = _float_or_none(row.get("energy_eV_cell") or row.get("energy_eV") or row.get("E_eV"))
        if require_ok and energy is None:
            continue
        branches.append(
            BranchInput(
                label=str(row.get("label") or row.get("branch") or f"branch_{len(branches)+1}"),
                run_dir=str(row.get("run_dir") or ""),
                pattern=str(row.get("pattern") or ""),
                degeneracy=_float_or_none(row.get("degeneracy")) or 1.0,
                energy_eV_cell=energy,
                energy_kind=str(row.get("energy_kind") or ""),
                status=status or "OK",
                source=str(row.get("source") or ""),
                total_moment_muB=_float_or_none(row.get("total_moment_muB")),
                note=str(row.get("note") or ""),
            )
        )
    if not branches:
        raise ValueError(f"No usable branch energies found in {path}")
    return branches


def solve_magnetic_entropy(
    branches: list[BranchInput],
    temperatures: list[float],
    *,
    formula_units: float = 1.0,
) -> tuple[list[MagneticThermoPoint], list[dict[str, Any]]]:
    usable = [branch for branch in branches if branch.energy_eV_cell is not None]
    if not usable:
        raise ValueError("No branch energies were supplied.")
    energies = [float(branch.energy_eV_cell) for branch in usable]
    e_min = min(energies)
    thermo: list[MagneticThermoPoint] = []
    probs: list[dict[str, Any]] = []
    for temp in temperatures:
        if temp <= 0:
            raise ValueError("Temperatures must be positive for Boltzmann magnetic entropy.")
        beta = 1.0 / (KB_EV_PER_K * temp)
        terms: list[float] = []
        for branch in usable:
            delta = float(branch.energy_eV_cell) - e_min
            terms.append(max(branch.degeneracy, 0.0) * math.exp(-beta * delta))
        Z = sum(terms)
        if Z <= 0:
            raise ValueError(f"Partition function underflow at T={temp:g} K")
        probabilities = [term / Z for term in terms]
        u_rel = sum(p * (float(branch.energy_eV_cell) - e_min) for p, branch in zip(probabilities, usable))
        f_rel = -KB_EV_PER_K * temp * math.log(Z)
        s_cell = (u_rel - f_rel) / temp
        dominant_index = max(range(len(usable)), key=lambda idx: probabilities[idx])
        point = MagneticThermoPoint(
            T_K=temp,
            formula_units=formula_units,
            n_branches=len(usable),
            Z=Z,
            E_min_eV_cell=e_min,
            U_rel_eV_cell=u_rel,
            F_rel_eV_cell=f_rel,
            S_eV_cell_K=s_cell,
            S_J_mol_cell_K=s_cell * EV_PER_MOL_TO_J_PER_MOL,
            U_rel_eV_per_fu=u_rel / formula_units,
            F_rel_eV_per_fu=f_rel / formula_units,
            S_eV_fu_K=s_cell / formula_units,
            S_J_mol_fu_K=s_cell * EV_PER_MOL_TO_J_PER_MOL / formula_units,
            g_eff_per_cell=math.exp(s_cell / KB_EV_PER_K),
            g_eff_per_fu=math.exp((s_cell / formula_units) / KB_EV_PER_K),
            dominant_branch=usable[dominant_index].label,
            dominant_probability=probabilities[dominant_index],
        )
        thermo.append(point)
        for probability, branch in zip(probabilities, usable):
            delta = float(branch.energy_eV_cell) - e_min
            probs.append(
                {
                    "T_K": temp,
                    "label": branch.label,
                    "probability": probability,
                    "degeneracy": branch.degeneracy,
                    "delta_E_eV_cell": delta,
                    "delta_E_eV_per_fu": delta / formula_units,
                    "run_dir": branch.run_dir,
                }
            )
    return thermo, probs


def solve_command(args: argparse.Namespace) -> dict[str, Any]:
    branches = load_branch_inputs(args.branches.resolve(), require_ok=not args.keep_incomplete)
    temperatures = parse_temperature_values(args.temperature)
    thermo, probs = solve_magnetic_entropy(branches, temperatures, formula_units=args.formula_units)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    thermo_rows = [asdict(point) for point in thermo]
    thermo_csv = outdir / "magnetic_entropy_vs_T.csv"
    probs_csv = outdir / "magnetic_branch_probabilities.csv"
    write_csv(thermo_csv, thermo_rows, THERMO_FIELDS)
    write_csv(probs_csv, probs, PROBABILITY_FIELDS)
    metadata = {
        "schema": SCHEMA,
        "stage": "solve",
        "branches": str(args.branches.resolve()),
        "outdir": str(outdir),
        "formula_units": args.formula_units,
        "temperatures_K": temperatures,
        "notes": [
            "Energies are treated as whole-cell microstate energies in the partition function.",
            "Per-formula quantities are obtained by dividing cell U, F, and S by --formula-units.",
            "Use branch degeneracy for symmetry/POCC multiplicity or empirical spin-state degeneracy when justified.",
        ],
    }
    (outdir / "magnetic_entropy_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {thermo_csv}")
    print(f"Wrote {probs_csv}")
    return metadata


def estimate_command(args: argparse.Namespace) -> dict[str, Any]:
    spins = [float(value) for value in args.spin]
    rows: list[dict[str, Any]] = []
    for spin in spins:
        degeneracy = 2.0 * spin + 1.0
        s_j = R_J_PER_MOL_K * math.log(degeneracy) * args.centers_per_formula
        rows.append(
            {
                "S": spin,
                "degeneracy": degeneracy,
                "centers_per_formula": args.centers_per_formula,
                "S_mag_J_mol_formula_K": s_j,
                "S_mag_eV_formula_K": s_j / EV_PER_MOL_TO_J_PER_MOL,
            }
        )
    output = args.output.resolve() if args.output else Path("spin_degeneracy_entropy.csv").resolve()
    fields = ["S", "degeneracy", "centers_per_formula", "S_mag_J_mol_formula_K", "S_mag_eV_formula_K"]
    write_csv(output, rows, fields)
    print(f"Wrote {output}")
    return {"schema": SCHEMA, "stage": "estimate", "output": str(output), "rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-magnetic-entropy",
        description="Prepare, collect, and solve static DFT+U magnetic-branch entropy ensembles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Generate VASP static magnetic branch folders from one structure.")
    prep.add_argument("--structure", type=Path, required=True, help="POSCAR/CONTCAR input structure.")
    prep.add_argument("--outdir", type=Path, required=True, help="Output folder for prepared branches.")
    prep.add_argument("--template-dir", type=Path, help="Folder containing INCAR/KPOINTS/POTCAR/sbatch templates to copy.")
    prep.add_argument("--incar-template", type=Path, help="Explicit INCAR template; overrides template-dir/INCAR.")
    prep.add_argument("--pattern", action="append", default=[], help="Branch pattern(s): nonmag,fm,afm-alt,afm-layer,afm-block.")
    prep.add_argument("--label-prefix", default="", help="Optional prefix for branch folder names.")
    prep.add_argument("--mag-species", action="append", default=[], help="Magnetic species, comma/space separated. Empty means all atoms.")
    prep.add_argument("--moment-map", action="append", default=[], help="Initial moments, e.g. U=2 C=0 or U=1.5.")
    prep.add_argument("--default-moment", type=float, default=1.0, help="Default absolute moment for selected magnetic atoms.")
    prep.add_argument("--layer-axis", type=int, choices=(0, 1, 2), default=2, help="Fractional axis for afm-layer split.")
    prep.add_argument("--degeneracy", type=float, default=1.0, help="Initial branch degeneracy written to manifest.")
    prep.add_argument("--formula-units", type=float, default=1.0, help="Formula units per prepared VASP cell, stored as metadata.")
    prep.add_argument("--static", action="store_true", help="Patch INCAR as static NSW=0, IBRION=-1, ISIF=2.")
    prep.add_argument("--lorbit", type=int, default=11, help="Patch LORBIT for later local spin analysis; use negative to omit.")
    prep.add_argument("--incar-tag", action="append", default=[], help="Extra INCAR tag line to append; repeatable.")
    prep.set_defaults(func=prepare_branches)

    collect = sub.add_parser("collect", help="Collect branch energies from prepared/completed VASP folders.")
    collect.add_argument("--root", type=Path, default=Path("."), help="Root containing magnetic_branch_manifest.csv.")
    collect.add_argument("--manifest", type=Path, help="Branch manifest CSV; default root/magnetic_branch_manifest.csv.")
    collect.add_argument("--output", type=Path, help="Output branch energy CSV; default root/magnetic_branch_energies.csv.")
    collect.add_argument("--energy", choices=sorted(ENERGY_ALIASES), default="e0", help="Preferred VASP energy kind.")
    collect.add_argument("--force", action="store_true", help="Reparse outputs even if manifest has energies.")
    collect.add_argument("--include-archives", action="store_true", help="Also parse OUTCAR/OSZICAR inside *.tgz or *.tar.gz archives.")
    collect.set_defaults(func=collect_branches)

    solve = sub.add_parser("solve", help="Compute T-dependent magnetic entropy from branch energies.")
    solve.add_argument("--branches", type=Path, required=True, help="CSV from collect, or compatible branch energy table.")
    solve.add_argument("--temperature", action="append", help="Temperature(s) in K, comma list, or start:stop:step grid.")
    solve.add_argument("--formula-units", type=float, default=1.0, help="Formula units per DFT branch cell.")
    solve.add_argument("--outdir", type=Path, default=Path("magnetic_entropy_solve"))
    solve.add_argument("--keep-incomplete", action="store_true", help="Keep rows even if status is not OK, as long as energy is present.")
    solve.set_defaults(func=solve_command)

    estimate = sub.add_parser("estimate", help="Quick ideal spin-degeneracy estimate R ln(2S+1).")
    estimate.add_argument("--spin", action="append", required=True, help="Spin S value; repeat for several cases.")
    estimate.add_argument("--centers-per-formula", type=float, default=1.0, help="Magnetic centers per formula unit.")
    estimate.add_argument("--output", type=Path, help="Output CSV path.")
    estimate.set_defaults(func=estimate_command)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "lorbit") and args.lorbit is not None and args.lorbit < 0:
        args.lorbit = None
    return args.func(args)


if __name__ == "__main__":
    main()
