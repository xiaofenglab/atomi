from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from atomi.vasp.prep import (
    DEFAULT_TEMPLATE_FILES,
    resolve_input_poscar,
    species_order_from_atoms,
    summarize_atoms,
    validate_vasp_template,
)

@dataclass
class StressForceRecord:
    run_dir: Path
    case_name: str
    family: str
    source_poscar: Path
    seed: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_vasp_template(template_dir: Path | None, dest_dir: Path, copy_all: bool) -> None:
    if template_dir is None:
        return
    if copy_all:
        for item in sorted(template_dir.iterdir()):
            if item.name == "POSCAR":
                continue
            target = dest_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            elif item.is_file():
                shutil.copy2(item, target)
        return

    for filename in DEFAULT_TEMPLATE_FILES:
        src = template_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing {filename} in template dir: {template_dir}")
        shutil.copy2(src, dest_dir / filename)


def write_case(
    atoms: Atoms,
    out_dir: Path,
    info: dict,
    template_dir: Path | None,
    copy_template_all: bool,
) -> None:
    ensure_dir(out_dir)
    write(out_dir / "POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=False)
    (out_dir / "case_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    copy_vasp_template(template_dir, out_dir, copy_all=copy_template_all)


def apply_deformation(atoms: Atoms, deformation_gradient: np.ndarray) -> Atoms:
    result = atoms.copy()
    result.set_cell(result.cell.array @ deformation_gradient.T, scale_atoms=True)
    return result


def hydrostatic_f(eps: float) -> np.ndarray:
    return np.eye(3) * (1.0 + eps)


def uniaxial_f(axis: int, eps: float) -> np.ndarray:
    deformation = np.eye(3)
    deformation[axis, axis] += eps
    return deformation


def ortho_volume_conserving_f(axis: int, eps: float) -> np.ndarray:
    deformation = np.eye(3)
    stretch = 1.0 + eps
    compression = 1.0 / np.sqrt(stretch)
    for i in range(3):
        deformation[i, i] = compression
    deformation[axis, axis] = stretch
    return deformation


def shear_f(pair: tuple[int, int], gamma: float) -> np.ndarray:
    deformation = np.eye(3)
    i, j = pair
    deformation[i, j] = gamma
    return deformation


def normalized_random_vectors(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=shape)
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0.0] = 1.0
    return vectors / norms[:, None]


def random_displacement(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> Atoms:
    result = atoms.copy()
    disp = normalized_random_vectors(result.positions.shape, rng)
    mags = amplitude * rng.uniform(0.7, 1.3, size=(len(result), 1))
    result.set_positions(result.positions + disp * mags)
    return result


def structured_small_displacement(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> Atoms:
    result = atoms.copy()
    disp = normalized_random_vectors(result.positions.shape, rng)
    mags = amplitude * rng.uniform(0.9, 1.1, size=(len(result), 1))
    result.set_positions(result.positions + disp * mags)
    return result


def species_biased_displacement(
    atoms: Atoms,
    species: str,
    amplitude: float,
    rng: np.random.Generator,
) -> Atoms:
    result = atoms.copy()
    symbols = np.array(result.get_chemical_symbols())
    mask = symbols == species
    nsel = int(mask.sum())
    if nsel == 0:
        return result

    disp = np.zeros_like(result.positions)
    vectors = normalized_random_vectors((nsel, 3), rng)
    mags = amplitude * rng.uniform(0.7, 1.3, size=(nsel, 1))
    disp[mask] = vectors * mags
    result.set_positions(result.positions + disp)
    return result


def fmt_float(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.4f}".replace(".", "p")


def parse_weight_items(items: list[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Bad family weight, expected family=weight: {item}")
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value.strip())
    return weights


def allocate_balanced_targets(
    total_target: int,
    family_to_size: dict[str, int],
    family_weights: dict[str, float],
) -> dict[str, int]:
    families = [family for family, size in family_to_size.items() if size > 0]
    if total_target <= 0 or not families:
        return {family: 0 for family in family_to_size}

    total_target = min(total_target, sum(family_to_size[family] for family in families))
    remaining_families = set(families)
    selected = {family: 0 for family in family_to_size}

    while total_target > sum(selected.values()) and remaining_families:
        remaining_target = total_target - sum(selected.values())
        weight_sum = sum(max(family_weights.get(family, 1.0), 0.0) for family in remaining_families)
        if weight_sum <= 0:
            weight_sum = float(len(remaining_families))
            raw = {family: remaining_target / weight_sum for family in remaining_families}
        else:
            raw = {
                family: remaining_target * max(family_weights.get(family, 1.0), 0.0) / weight_sum
                for family in remaining_families
            }

        base = {family: int(math.floor(raw[family])) for family in remaining_families}
        remainder = remaining_target - sum(base.values())
        frac_order = sorted(
            remaining_families,
            key=lambda family: (raw[family] - base[family], family),
            reverse=True,
        )
        for family in frac_order[:remainder]:
            base[family] += 1

        progressed = False
        full_now = set()
        for family in list(remaining_families):
            capacity = family_to_size[family] - selected[family]
            take = min(base[family], capacity)
            if take > 0:
                selected[family] += take
                progressed = True
            if selected[family] >= family_to_size[family]:
                full_now.add(family)

        remaining_families -= full_now
        if not progressed:
            break

    return selected


def generate_pool(atoms0: Atoms, poscar: Path, args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    pool = []

    def add_candidate(name: str, atoms: Atoms, family: str, info_extra: dict | None = None) -> None:
        info = {
            "case_name": name,
            "family": family,
            "source_poscar": str(poscar),
            "mode": args.mode,
            "seed": args.seed,
        }
        if info_extra:
            info.update(info_extra)
        pool.append({"name": name, "atoms": atoms, "family": family, "info": info})

    if args.mode in ("stress", "both"):
        if not args.no_hydro:
            for eps in args.hydro_strains:
                deformation = hydrostatic_f(eps)
                add_candidate(
                    f"hydro_e_{fmt_float(eps)}",
                    apply_deformation(atoms0, deformation),
                    "hydrostatic",
                    {
                        "strain_value": eps,
                        "deformation_gradient": deformation.tolist(),
                        "linear_scale": 1.0 + eps,
                        "volume_scale": float(np.linalg.det(deformation)),
                    },
                )

        if not args.no_uniaxial:
            for axis, axis_name in enumerate(("x", "y", "z")):
                for eps in args.elastic_strains:
                    deformation = uniaxial_f(axis, eps)
                    add_candidate(
                        f"uniaxial_{axis_name}_e_{fmt_float(eps)}",
                        apply_deformation(atoms0, deformation),
                        "uniaxial",
                        {
                            "axis": axis_name,
                            "strain_value": eps,
                            "deformation_gradient": deformation.tolist(),
                            "volume_scale": float(np.linalg.det(deformation)),
                        },
                    )

        if not args.no_orthorhombic:
            for axis, axis_name in enumerate(("x", "y", "z")):
                for eps in args.elastic_strains:
                    deformation = ortho_volume_conserving_f(axis, eps)
                    add_candidate(
                        f"ortho_vc_{axis_name}_e_{fmt_float(eps)}",
                        apply_deformation(atoms0, deformation),
                        "orthorhombic_volume_conserving",
                        {
                            "stretch_axis": axis_name,
                            "strain_value": eps,
                            "deformation_gradient": deformation.tolist(),
                            "volume_scale": float(np.linalg.det(deformation)),
                        },
                    )

        if not args.no_shear:
            for pair, pair_name in [((0, 1), "xy"), ((0, 2), "xz"), ((1, 2), "yz")]:
                for gamma in args.elastic_strains:
                    deformation = shear_f(pair, gamma)
                    add_candidate(
                        f"shear_{pair_name}_g_{fmt_float(gamma)}",
                        apply_deformation(atoms0, deformation),
                        "shear",
                        {
                            "shear_pair": pair_name,
                            "strain_value": gamma,
                            "deformation_gradient": deformation.tolist(),
                            "volume_scale": float(np.linalg.det(deformation)),
                        },
                    )

    if args.mode in ("force", "both"):
        for index in range(args.n_rd_small):
            add_candidate(
                f"rd_small_{index + 1:03d}",
                random_displacement(atoms0, args.rd_small_amp, rng),
                "rd_small",
                {"amplitude_A": args.rd_small_amp},
            )
        for index in range(args.n_rd_large):
            add_candidate(
                f"rd_large_{index + 1:03d}",
                random_displacement(atoms0, args.rd_large_amp, rng),
                "rd_large",
                {"amplitude_A": args.rd_large_amp},
            )
        for index in range(args.n_disp_small):
            add_candidate(
                f"disp_small_{index + 1:03d}",
                structured_small_displacement(atoms0, args.disp_small_amp, rng),
                "disp_small",
                {"amplitude_A": args.disp_small_amp},
            )
        for index in range(args.n_bias_mixed):
            add_candidate(
                f"bias_mixed_{index + 1:03d}",
                random_displacement(atoms0, args.bias_mixed_amp, rng),
                "bias_mixed",
                {"amplitude_A": args.bias_mixed_amp},
            )
        if args.bias_species:
            present = set(atoms0.get_chemical_symbols())
            if args.bias_species in present:
                for index in range(args.n_bias_species):
                    add_candidate(
                        f"bias_{args.bias_species}_{index + 1:03d}",
                        species_biased_displacement(
                            atoms0,
                            args.bias_species,
                            args.bias_species_amp,
                            rng,
                        ),
                        "bias_species",
                        {"species": args.bias_species, "amplitude_A": args.bias_species_amp},
                    )

    return pool


def select_balanced(pool: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict[str, int], dict[str, int]]:
    if not pool:
        raise RuntimeError("No candidate structures were generated. Check --mode and family toggles.")

    family_to_items: dict[str, list[dict]] = defaultdict(list)
    for item in pool:
        family_to_items[item["family"]].append(item)

    family_to_size = {family: len(items) for family, items in family_to_items.items()}
    targets = allocate_balanced_targets(args.target_size, family_to_size, parse_weight_items(args.family_weight))
    py_rng = random.Random(args.seed)
    selected = []
    for family, items in sorted(family_to_items.items()):
        indices = list(range(len(items)))
        py_rng.shuffle(indices)
        indices = sorted(indices[: min(targets[family], len(items))])
        selected.extend(items[index] for index in indices)
    py_rng.shuffle(selected)
    return selected, family_to_size, targets


def write_runlist(records: list[StressForceRecord], runlist: Path) -> None:
    ensure_dir(runlist.parent)
    base = runlist.parent.resolve()
    lines = []
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(base)))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_index(records: list[StressForceRecord], index_path: Path) -> None:
    ensure_dir(index_path.parent)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_dir", "case_name", "family", "source_poscar", "seed"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "case_name": record.case_name,
                    "family": record.family,
                    "source_poscar": str(record.source_poscar.resolve()),
                    "seed": record.seed,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-stress-force-candidates",
        description="Prepare stress/force VASP candidate runs from an equilibrium POSCAR.",
    )
    parser.add_argument(
        "--poscar",
        type=Path,
        default=None,
        help="Equilibrium POSCAR. Defaults to VASP_TEMPLATE/POSCAR, then ./POSCAR.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--vasp-template", type=Path, default=None)
    parser.add_argument(
        "--copy-template-all",
        action="store_true",
        help="Copy every template file/subdirectory except POSCAR.",
    )
    parser.add_argument("--runlist", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--mode", choices=("stress", "force", "both"), default="both")
    parser.add_argument("--target-size", type=int, default=80)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--hydro-strains",
        type=float,
        nargs="*",
        default=[-0.04, -0.02, -0.01, 0.00, 0.01, 0.02, 0.04],
    )
    parser.add_argument(
        "--elastic-strains",
        type=float,
        nargs="*",
        default=[-0.02, -0.01, -0.005, 0.005, 0.01, 0.02],
    )
    parser.add_argument("--no-hydro", action="store_true")
    parser.add_argument("--no-uniaxial", action="store_true")
    parser.add_argument("--no-orthorhombic", action="store_true")
    parser.add_argument("--no-shear", action="store_true")

    parser.add_argument("--n-rd-small", type=int, default=12)
    parser.add_argument("--n-rd-large", type=int, default=8)
    parser.add_argument("--n-disp-small", type=int, default=10)
    parser.add_argument("--n-bias-mixed", type=int, default=6)
    parser.add_argument("--n-bias-species", type=int, default=6)
    parser.add_argument("--rd-small-amp", type=float, default=0.01)
    parser.add_argument("--rd-large-amp", type=float, default=0.02)
    parser.add_argument("--disp-small-amp", type=float, default=0.01)
    parser.add_argument("--bias-mixed-amp", type=float, default=0.03)
    parser.add_argument("--bias-species-amp", type=float, default=0.04)
    parser.add_argument("--bias-species", default=None, help="Optional element symbol, e.g. O.")
    parser.add_argument(
        "--family-weight",
        nargs="*",
        default=[],
        help="Optional family=weight items, e.g. hydrostatic=1.5 shear=1.2.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = ensure_dir(args.output_root.resolve())
    template_dir = args.vasp_template.resolve() if args.vasp_template else None
    poscar = resolve_input_poscar(args.poscar, template_dir)
    atoms0 = read(poscar)
    validate_vasp_template(template_dir, atoms=atoms0, require_poscar=args.poscar is None)

    pool = generate_pool(atoms0, poscar, args)
    selected, family_to_size, targets = select_balanced(pool, args)

    records = []
    summary = []
    for item in selected:
        run_dir = output_root / item["name"]
        write_case(item["atoms"], run_dir, item["info"], template_dir, args.copy_template_all)
        summary.append(item["info"])
        records.append(
            StressForceRecord(
                run_dir=run_dir,
                case_name=item["name"],
                family=item["family"],
                source_poscar=poscar,
                seed=args.seed,
            )
        )

    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "source_poscar": str(poscar),
        "mode": args.mode,
        "target_size_requested": args.target_size,
        "generated_pool_size": len(pool),
        "selected_size": len(selected),
        "family_sizes_generated": family_to_size,
        "family_targets_selected": targets,
        "family_weights": parse_weight_items(args.family_weight),
    }
    (output_root / "selection_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    runlist = args.runlist.resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.resolve() if args.index else output_root / "candidate_index.csv"
    write_runlist(records, runlist)
    write_index(records, index)

    print(f"Template POSCAR : {poscar}")
    print(f"Structure       : {summarize_atoms(atoms0)}")
    print(f"Species order   : {', '.join(species_order_from_atoms(atoms0))}")
    print(f"Mode            : {args.mode}")
    print(f"Output root     : {output_root}")
    print(f"Generated pool  : {len(pool)}")
    print(f"Selected set    : {len(selected)}")
    print("Family generated sizes:")
    for family, size in sorted(family_to_size.items()):
        print(f"  {family:32s} {size:4d}")
    print("Family selected targets:")
    for family, size in sorted(targets.items()):
        print(f"  {family:32s} {size:4d}")
    print(f"Wrote runlist   : {runlist}")
    print(f"Wrote index     : {index}")
