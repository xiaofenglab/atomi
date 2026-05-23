"""Side-by-side diagnostics for two comparable VASP runs."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atomi.vasp.checks import _latest_vasp_energy
from atomi.vasp.magmom import PoscarStructure, read_poscar_structure
from atomi.vasp.spin_report import (
    MagnetizationBlock,
    extract_last_magnetization_block,
    magnetic_order,
)


@dataclass
class RunData:
    label: str
    run_dir: Path
    structure_path: Path | None
    outcar_path: Path | None
    structure: PoscarStructure | None
    energy_eV: float | None
    energy_kind: str
    mag_block: MagnetizationBlock | None
    warnings: list[str]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def det3(m: list[list[float]]) -> float:
    a, b, c = m
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def angle_deg(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom <= 0:
        return float("nan")
    value = max(-1.0, min(1.0, dot(a, b) / denom))
    return math.degrees(math.acos(value))


def mat_vec(frac: list[float], cell: list[list[float]]) -> list[float]:
    return [
        frac[0] * cell[0][j] + frac[1] * cell[1][j] + frac[2] * cell[2][j]
        for j in range(3)
    ]


def cell_metrics(structure: PoscarStructure | None) -> dict[str, float] | None:
    if structure is None:
        return None
    a, b, c = structure.cell
    return {
        "a_A": norm(a),
        "b_A": norm(b),
        "c_A": norm(c),
        "alpha_deg": angle_deg(b, c),
        "beta_deg": angle_deg(a, c),
        "gamma_deg": angle_deg(a, b),
        "volume_A3": abs(det3(structure.cell)),
    }


def expanded_labels(structure: PoscarStructure | None) -> list[str]:
    if structure is None:
        return []
    labels: list[str] = []
    for element, count in zip(structure.species.symbols, structure.species.counts):
        labels.extend([element] * count)
    return labels


def composition(structure: PoscarStructure | None) -> Counter[str]:
    return Counter(expanded_labels(structure))


def choose_structure(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    for name in ("CONTCAR", "POSCAR"):
        path = run_dir / name
        if path.is_file():
            return path.resolve()
    return None


def choose_outcar(run_dir: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = run_dir / name
        if path.is_file():
            return path.resolve()
    candidates = sorted(run_dir.glob("vasp.out*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve() if candidates else None


def load_run(
    run: Path,
    *,
    label: str,
    structure_path: Path | None,
    outcar_path: Path | None,
    energy_kind: str,
) -> RunData:
    run_dir = run.resolve()
    if run_dir.is_file():
        run_dir = run_dir.parent.resolve()
    warnings: list[str] = []
    structure_file = choose_structure(run_dir, structure_path)
    outcar_file = choose_outcar(run_dir, outcar_path)
    structure = None
    if structure_file is None:
        warnings.append("missing CONTCAR/POSCAR")
    else:
        try:
            structure = read_poscar_structure(structure_file)
        except Exception as exc:
            warnings.append(f"could not read structure {structure_file}: {exc}")
            structure = None
    energy = None
    found_kind = ""
    mag_block = None
    if outcar_file is None:
        warnings.append("missing OUTCAR/OUTCAR.gz/vasp.out*")
    else:
        energy, found_kind = _latest_vasp_energy(outcar_file, preferred_kind=energy_kind)
        if energy is None:
            warnings.append(f"no parseable energy in {outcar_file}")
        try:
            natoms = structure.species.total_atoms if structure is not None else None
            mag_block = extract_last_magnetization_block(outcar_file, natoms=natoms)
        except Exception as exc:
            warnings.append(f"no usable magnetization from {outcar_file}: {exc}")
    return RunData(
        label=label,
        run_dir=run_dir,
        structure_path=structure_file,
        outcar_path=outcar_file,
        structure=structure,
        energy_eV=energy,
        energy_kind=found_kind,
        mag_block=mag_block,
        warnings=warnings,
    )


def fmt(value: Any, width: int = 13, precision: int = 6) -> str:
    if value is None:
        return "NA".rjust(width)
    try:
        number = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text[:width].rjust(width)
    if not math.isfinite(number):
        return "NA".rjust(width)
    return f"{number:{width}.{precision}f}"


def fmt_delta(a: float | None, b: float | None, width: int = 13, precision: int = 6) -> str:
    if a is None or b is None:
        return "NA".rjust(width)
    return fmt(b - a, width=width, precision=precision)


def print_comparison_header(a: RunData, b: RunData) -> None:
    print("VASP Run Comparison")
    print("=" * 80)
    print(f"A label          : {a.label}")
    print(f"A run            : {a.run_dir}")
    print(f"A structure      : {a.structure_path or 'missing'}")
    print(f"A output         : {a.outcar_path or 'missing'}")
    print(f"B label          : {b.label}")
    print(f"B run            : {b.run_dir}")
    print(f"B structure      : {b.structure_path or 'missing'}")
    print(f"B output         : {b.outcar_path or 'missing'}")
    print()


def print_energy(a: RunData, b: RunData) -> None:
    print("Energy")
    print("-" * 80)
    print(f"{'quantity':<18} {'A':>15} {'B':>15} {'B-A':>15}")
    print(f"{'energy_eV':<18} {fmt(a.energy_eV, 15, 8)} {fmt(b.energy_eV, 15, 8)} {fmt_delta(a.energy_eV, b.energy_eV, 15, 8)}")
    print(f"{'energy_kind':<18} {a.energy_kind or 'NA':>15} {b.energy_kind or 'NA':>15} {'':>15}")
    if a.energy_eV is not None and b.energy_eV is not None:
        na = a.structure.species.total_atoms if a.structure else None
        nb = b.structure.species.total_atoms if b.structure else None
        if na and nb:
            print(f"{'dE_per_atom_eV':<18} {'':>15} {'':>15} {fmt((b.energy_eV / nb) - (a.energy_eV / na), 15, 8)}")
    print()


def print_cell(a: RunData, b: RunData) -> None:
    print("Unit Cell")
    print("-" * 80)
    ma = cell_metrics(a.structure)
    mb = cell_metrics(b.structure)
    print(f"{'metric':<18} {'A':>15} {'B':>15} {'B-A':>15} {'rel_%':>10}")
    for key in ("a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg", "volume_A3"):
        va = ma.get(key) if ma else None
        vb = mb.get(key) if mb else None
        rel = None
        if va not in (None, 0) and vb is not None:
            rel = 100.0 * (vb - va) / va
        print(f"{key:<18} {fmt(va, 15, 6)} {fmt(vb, 15, 6)} {fmt_delta(va, vb, 15, 6)} {fmt(rel, 10, 4)}")
    print()


def print_composition(a: RunData, b: RunData) -> None:
    print("Composition")
    print("-" * 80)
    ca = composition(a.structure)
    cb = composition(b.structure)
    keys = list(dict.fromkeys(list(ca.keys()) + list(cb.keys())))
    print(f"{'element':<10} {'A_count':>10} {'B_count':>10} {'B-A':>10}")
    for key in keys:
        print(f"{key:<10} {ca.get(key, 0):>10d} {cb.get(key, 0):>10d} {cb.get(key, 0) - ca.get(key, 0):>10d}")
    print()


def element_moment_summary(run: RunData) -> dict[str, list[float]]:
    labels = expanded_labels(run.structure)
    moments = run.mag_block.moments if run.mag_block is not None else []
    out: dict[str, list[float]] = {}
    for element, moment in zip(labels, moments):
        out.setdefault(element, []).append(moment)
    return out


def print_spins(a: RunData, b: RunData, threshold: float) -> None:
    print("Spin / Magnetization")
    print("-" * 80)
    ma = a.mag_block.moments if a.mag_block is not None else []
    mb = b.mag_block.moments if b.mag_block is not None else []
    total_a = sum(ma) if ma else None
    total_b = sum(mb) if mb else None
    max_a = max((abs(value) for value in ma), default=None)
    max_b = max((abs(value) for value in mb), default=None)
    print(f"{'quantity':<18} {'A':>15} {'B':>15} {'B-A':>15}")
    print(f"{'moment_rows':<18} {len(ma):>15d} {len(mb):>15d} {len(mb) - len(ma):>15d}")
    print(f"{'total_moment':<18} {fmt(total_a, 15, 6)} {fmt(total_b, 15, 6)} {fmt_delta(total_a, total_b, 15, 6)}")
    print(f"{'max_abs_moment':<18} {fmt(max_a, 15, 6)} {fmt(max_b, 15, 6)} {fmt_delta(max_a, max_b, 15, 6)}")
    print()
    print(f"{'element':<10} {'A_sum':>13} {'B_sum':>13} {'B-A':>13} {'A_order':>12} {'B_order':>12}")
    ea = element_moment_summary(a)
    eb = element_moment_summary(b)
    keys = list(dict.fromkeys(list(ea.keys()) + list(eb.keys())))
    for key in keys:
        va = ea.get(key, [])
        vb = eb.get(key, [])
        print(
            f"{key:<10} {fmt(sum(va) if va else None, 13, 5)} {fmt(sum(vb) if vb else None, 13, 5)} "
            f"{fmt_delta(sum(va) if va else None, sum(vb) if vb else None, 13, 5)} "
            f"{magnetic_order(va, threshold=threshold) if va else 'NA':>12} "
            f"{magnetic_order(vb, threshold=threshold) if vb else 'NA':>12}"
        )
    print()


def average_cell(a: PoscarStructure, b: PoscarStructure) -> list[list[float]]:
    return [
        [(a.cell[i][j] + b.cell[i][j]) * 0.5 for j in range(3)]
        for i in range(3)
    ]


def atom_difference_rows(a: RunData, b: RunData) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if a.structure is None or b.structure is None:
        return [], ["missing structure; no atom displacement comparison"]
    labels_a = expanded_labels(a.structure)
    labels_b = expanded_labels(b.structure)
    if labels_a != labels_b:
        warnings.append("atom species order/count differs; comparing only matching index prefix")
    n = min(len(labels_a), len(labels_b), len(a.structure.scaled_positions), len(b.structure.scaled_positions))
    cell = average_cell(a.structure, b.structure)
    ma = a.mag_block.moments if a.mag_block is not None else []
    mb = b.mag_block.moments if b.mag_block is not None else []
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        fa = a.structure.scaled_positions[idx]
        fb = b.structure.scaled_positions[idx]
        df = [fb[j] - fa[j] for j in range(3)]
        df_mic = [value - round(value) for value in df]
        dcart = mat_vec(df_mic, cell)
        disp = norm(dcart)
        moment_a = ma[idx] if idx < len(ma) else None
        moment_b = mb[idx] if idx < len(mb) else None
        rows.append(
            {
                "index": idx + 1,
                "element_A": labels_a[idx],
                "element_B": labels_b[idx],
                "frac_A": fa,
                "frac_B": fb,
                "dfrac": df_mic,
                "disp_A": disp,
                "moment_A": moment_a,
                "moment_B": moment_b,
                "dmoment": (moment_b - moment_a) if moment_a is not None and moment_b is not None else None,
            }
        )
    return rows, warnings


def print_atoms(a: RunData, b: RunData, top: int, all_atoms: bool) -> dict[str, Any]:
    print("Atom-By-Atom Structural Difference")
    print("-" * 80)
    rows, warnings = atom_difference_rows(a, b)
    if not rows:
        print("No comparable atom rows.")
        print()
        return {"warnings": warnings}
    rms = math.sqrt(sum(row["disp_A"] ** 2 for row in rows) / len(rows))
    mean = sum(row["disp_A"] for row in rows) / len(rows)
    max_row = max(rows, key=lambda row: row["disp_A"])
    sorted_rows = rows if all_atoms else sorted(rows, key=lambda row: row["disp_A"], reverse=True)[:top]
    print(f"Compared atoms     : {len(rows)}")
    print(f"Mean displacement  : {mean:.6f} A")
    print(f"RMS displacement   : {rms:.6f} A")
    print(f"Max displacement   : {max_row['disp_A']:.6f} A at atom {max_row['index']} ({max_row['element_A']}/{max_row['element_B']})")
    print()
    print(
        f"{'atom':>6} {'el':>6} {'disp_A':>11} {'dmag':>11} "
        f"{'A_frac':>26} {'B_frac':>26}"
    )
    for row in sorted_rows:
        fa = " ".join(f"{value:.4f}" for value in row["frac_A"])
        fb = " ".join(f"{value:.4f}" for value in row["frac_B"])
        element = row["element_A"] if row["element_A"] == row["element_B"] else f"{row['element_A']}/{row['element_B']}"
        print(
            f"{row['index']:6d} {element:>6} {row['disp_A']:11.5f} "
            f"{fmt(row['dmoment'], 11, 5)} {fa:>26} {fb:>26}"
        )
    print()
    return {
        "warnings": warnings,
        "n_atoms": len(rows),
        "mean_disp_A": mean,
        "rms_disp_A": rms,
        "max_disp_A": max_row["disp_A"],
        "max_atom": max_row["index"],
    }


def print_warnings(a: RunData, b: RunData, extra: list[str]) -> None:
    warnings = [(a.label, warning) for warning in a.warnings] + [(b.label, warning) for warning in b.warnings]
    warnings.extend(("comparison", warning) for warning in extra)
    if not warnings:
        return
    print("Warnings")
    print("-" * 80)
    for label, warning in warnings:
        print(f"{label}: {warning}")
    print()


def print_brief_diagnostic(a: RunData, b: RunData, atom_stats: dict[str, Any]) -> None:
    print("Brief Diagnostic")
    print("-" * 80)
    if a.energy_eV is not None and b.energy_eV is not None:
        d_e = b.energy_eV - a.energy_eV
        print(f"Energy difference B-A: {d_e:+.8f} eV.")
    ma = cell_metrics(a.structure)
    mb = cell_metrics(b.structure)
    if ma and mb:
        dvol = mb["volume_A3"] - ma["volume_A3"]
        rel = 100.0 * dvol / ma["volume_A3"] if ma["volume_A3"] else 0.0
        print(f"Cell volume difference B-A: {dvol:+.6f} A^3 ({rel:+.4f}%).")
    if "rms_disp_A" in atom_stats:
        print(
            "Internal structural difference: "
            f"RMS {atom_stats['rms_disp_A']:.5f} A, "
            f"max {atom_stats['max_disp_A']:.5f} A at atom {atom_stats['max_atom']}."
        )
    ma_mom = a.mag_block.moments if a.mag_block is not None else []
    mb_mom = b.mag_block.moments if b.mag_block is not None else []
    if ma_mom and mb_mom:
        print(f"Total moment difference B-A: {sum(mb_mom) - sum(ma_mom):+.6f} mu_B.")
    print()


def compare_runs(args: argparse.Namespace) -> None:
    a = load_run(
        args.run_a,
        label=args.label_a,
        structure_path=args.structure_a,
        outcar_path=args.outcar_a,
        energy_kind=args.energy,
    )
    b = load_run(
        args.run_b,
        label=args.label_b,
        structure_path=args.structure_b,
        outcar_path=args.outcar_b,
        energy_kind=args.energy,
    )
    print_comparison_header(a, b)
    print_energy(a, b)
    print_cell(a, b)
    print_composition(a, b)
    print_spins(a, b, threshold=args.moment_threshold)
    atom_stats = print_atoms(a, b, top=args.top_atoms, all_atoms=args.all_atoms)
    print_warnings(a, b, atom_stats.get("warnings", []))
    print_brief_diagnostic(a, b, atom_stats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-compare-runs",
        description=(
            "Compare two comparable VASP runs side by side: energy, cell, "
            "composition, magnetic moments, and atom displacements."
        ),
    )
    parser.add_argument("run_a", type=Path, help="First VASP run directory.")
    parser.add_argument("run_b", type=Path, help="Second VASP run directory.")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--structure-a", type=Path, help="Override A structure file; default CONTCAR then POSCAR.")
    parser.add_argument("--structure-b", type=Path, help="Override B structure file; default CONTCAR then POSCAR.")
    parser.add_argument("--outcar-a", type=Path, help="Override A OUTCAR/OUTCAR.gz/output file.")
    parser.add_argument("--outcar-b", type=Path, help="Override B OUTCAR/OUTCAR.gz/output file.")
    parser.add_argument(
        "--energy",
        default="toten",
        choices=("toten", "without_entropy", "e0", "f", "dav"),
        help="Preferred energy kind. Falls back like checkeng.",
    )
    parser.add_argument("--moment-threshold", type=float, default=0.2, help="Threshold for FM/AFM/nonmagnetic labels.")
    parser.add_argument("--top-atoms", type=int, default=12, help="Number of largest-displacement atoms to print.")
    parser.add_argument("--all-atoms", action="store_true", help="Print every comparable atom row.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    compare_runs(args)


if __name__ == "__main__":
    main()
