from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.io import read, write

from atomi.vasp.prefail import copy_vasp_template
from atomi.vasp.prep import (
    resolve_input_poscar,
    species_order_from_atoms,
    summarize_atoms,
    validate_vasp_template,
)


@dataclass
class DefectRecord:
    run_dir: Path
    case_name: str
    family: str
    source_poscar: Path
    seed: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_float(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.4f}".replace(".", "p")


def order_atoms_by_species(atoms: Atoms, species_order: tuple[str, ...]) -> Atoms:
    symbols = np.array(atoms.get_chemical_symbols())
    indices = []
    for species in species_order:
        indices.extend(np.where(symbols == species)[0].tolist())
    indices.extend(index for index, symbol in enumerate(symbols) if symbol not in species_order)
    return atoms[indices].copy()


def write_case(
    atoms: Atoms,
    out_dir: Path,
    info: dict,
    template_dir: Path | None,
    copy_template_all: bool,
    species_order: tuple[str, ...],
) -> None:
    ensure_dir(out_dir)
    ordered = order_atoms_by_species(atoms, species_order)
    write(out_dir / "POSCAR", ordered, format="vasp", direct=True, vasp5=True, sort=False)
    (out_dir / "case_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    copy_vasp_template(template_dir, out_dir, copy_all=copy_template_all)


def remove_indices(atoms: Atoms, indices: list[int]) -> Atoms:
    result = atoms.copy()
    del result[sorted(indices, reverse=True)]
    return result


def random_indices_for_species(
    atoms: Atoms,
    species: str,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    candidates = np.where(np.array(atoms.get_chemical_symbols()) == species)[0]
    if len(candidates) < count:
        raise ValueError(f"Need {count} {species} atoms, but POSCAR has {len(candidates)}.")
    return rng.choice(candidates, size=count, replace=False).astype(int).tolist()


def min_distance_to_atoms(atoms: Atoms, cart: np.ndarray) -> float:
    if len(atoms) == 0:
        return math.inf
    probe = Atom("H", position=cart)
    tmp = atoms.copy()
    tmp.append(probe)
    distances = tmp.get_distances(len(tmp) - 1, list(range(len(tmp) - 1)), mic=True)
    return float(np.min(distances))


def random_interstitial_position(
    atoms: Atoms,
    rng: np.random.Generator,
    min_distance: float,
    max_attempts: int,
) -> tuple[np.ndarray, float, bool]:
    best_frac = None
    best_distance = -1.0
    for _ in range(max_attempts):
        frac = rng.random(3)
        cart = frac @ atoms.cell.array
        distance = min_distance_to_atoms(atoms, cart)
        if distance >= min_distance:
            return frac, distance, True
        if distance > best_distance:
            best_frac = frac
            best_distance = distance
    assert best_frac is not None
    return best_frac, best_distance, False


def add_interstitial(
    atoms: Atoms,
    species: str,
    rng: np.random.Generator,
    min_distance: float,
    max_attempts: int,
) -> tuple[Atoms, dict]:
    frac, distance, accepted = random_interstitial_position(atoms, rng, min_distance, max_attempts)
    result = atoms.copy()
    result.append(Atom(species, position=frac @ result.cell.array))
    return result, {
        "interstitial_species": species,
        "interstitial_scaled_position": frac.tolist(),
        "nearest_neighbor_distance_A": distance,
        "met_min_distance": accepted,
    }


def reduced_formula_counts(atoms: Atoms, species_order: tuple[str, ...]) -> dict[str, int]:
    counts = {
        species: atoms.get_chemical_symbols().count(species)
        for species in species_order
    }
    gcd = 0
    for count in counts.values():
        gcd = count if gcd == 0 else math.gcd(gcd, count)
    if gcd <= 0:
        raise ValueError("Could not derive reduced formula from empty structure.")
    return {species: count // gcd for species, count in counts.items() if count // gcd > 0}


def generate_pool(atoms0: Atoms, poscar: Path, args: argparse.Namespace) -> list[dict]:
    species_order = species_order_from_atoms(atoms0)
    selected_species = tuple(args.species) if args.species else species_order
    interstitial_species = tuple(args.interstitial_species) if args.interstitial_species else selected_species
    rng = np.random.default_rng(args.seed)
    pool = []

    def add_candidate(name: str, atoms: Atoms, family: str, info_extra: dict | None = None) -> None:
        info = {
            "case_name": name,
            "family": family,
            "source_poscar": str(poscar),
            "seed": args.seed,
            "composition": summarize_atoms(atoms),
        }
        if info_extra:
            info.update(info_extra)
        pool.append({"name": name, "atoms": atoms, "family": family, "info": info})

    if not args.no_vacancies:
        for species in selected_species:
            for index in range(args.n_vacancy):
                remove = random_indices_for_species(atoms0, species, args.vacancy_count, rng)
                name = f"vac_{species}_{index + 1:03d}"
                add_candidate(
                    name,
                    remove_indices(atoms0, remove),
                    "vacancy",
                    {"vacancy_species": species, "removed_atom_indices": remove},
                )

    if not args.no_interstitials:
        for species in interstitial_species:
            for index in range(args.n_interstitial):
                candidate, info = add_interstitial(
                    atoms0,
                    species,
                    rng,
                    args.interstitial_min_distance,
                    args.max_insert_attempts,
                )
                name = f"int_{species}_{index + 1:03d}"
                add_candidate(name, candidate, "interstitial", info)

    if not args.no_frenkel:
        for species in selected_species:
            for index in range(args.n_frenkel):
                remove = random_indices_for_species(atoms0, species, 1, rng)
                vacancy = remove_indices(atoms0, remove)
                candidate, insert_info = add_interstitial(
                    vacancy,
                    species,
                    rng,
                    args.interstitial_min_distance,
                    args.max_insert_attempts,
                )
                name = f"frenkel_{species}_{index + 1:03d}"
                add_candidate(
                    name,
                    candidate,
                    "frenkel",
                    {
                        "frenkel_species": species,
                        "removed_atom_index": remove[0],
                        **insert_info,
                    },
                )

    if not args.no_schottky and len(species_order) > 1:
        formula = reduced_formula_counts(atoms0, species_order)
        if sum(formula.values()) >= len(atoms0):
            print(
                "[warning] Skipping Schottky defects because one reduced formula unit "
                "would remove the whole POSCAR."
            )
        else:
            for index in range(args.n_schottky):
                removed = []
                for species, count in formula.items():
                    removed.extend(random_indices_for_species(atoms0, species, count, rng))
                name = f"schottky_{index + 1:03d}"
                add_candidate(
                    name,
                    remove_indices(atoms0, removed),
                    "schottky",
                    {"removed_formula_counts": formula, "removed_atom_indices": removed},
                )

    if not args.no_antisite and len(selected_species) > 1:
        for sp_a, sp_b in itertools.combinations(selected_species, 2):
            for index in range(args.n_antisite):
                idx_a = random_indices_for_species(atoms0, sp_a, 1, rng)[0]
                idx_b = random_indices_for_species(atoms0, sp_b, 1, rng)[0]
                candidate = atoms0.copy()
                candidate[idx_a].symbol = sp_b
                candidate[idx_b].symbol = sp_a
                name = f"antisite_{sp_a}_{sp_b}_{index + 1:03d}"
                add_candidate(
                    name,
                    candidate,
                    "antisite",
                    {
                        "species_a": sp_a,
                        "species_b": sp_b,
                        "swapped_atom_indices": [idx_a, idx_b],
                    },
                )

    if args.n_volume_strain > 0:
        for eps in args.volume_strains[: args.n_volume_strain]:
            candidate = atoms0.copy()
            candidate.set_cell(candidate.cell * (1.0 + eps), scale_atoms=True)
            name = f"defect_free_volume_{fmt_float(eps)}"
            add_candidate(
                name,
                candidate,
                "defect_free_volume",
                {"linear_strain": eps, "volume_scale": float((1.0 + eps) ** 3)},
            )

    return pool


def write_runlist(records: list[DefectRecord], runlist: Path) -> None:
    ensure_dir(runlist.parent)
    base = runlist.parent.resolve()
    lines = []
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(base)))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_index(records: list[DefectRecord], index_path: Path) -> None:
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
        prog="vasp-defect-candidates",
        description="Prepare vacancy/interstitial/Frenkel/Schottky VASP defect candidates.",
    )
    parser.add_argument(
        "--poscar",
        type=Path,
        default=None,
        help="Reference POSCAR. Defaults to VASP_TEMPLATE/POSCAR, then ./POSCAR.",
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
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--interstitial-species", nargs="*", default=None)

    parser.add_argument("--n-vacancy", type=int, default=4)
    parser.add_argument("--vacancy-count", type=int, default=1)
    parser.add_argument("--n-interstitial", type=int, default=4)
    parser.add_argument("--n-frenkel", type=int, default=4)
    parser.add_argument("--n-schottky", type=int, default=4)
    parser.add_argument("--n-antisite", type=int, default=2)
    parser.add_argument("--interstitial-min-distance", type=float, default=1.2)
    parser.add_argument("--max-insert-attempts", type=int, default=2000)
    parser.add_argument(
        "--volume-strains",
        type=float,
        nargs="*",
        default=[-0.04, -0.02, 0.02, 0.04],
        help="Optional defect-free compressed/expanded structures for short-range coverage.",
    )
    parser.add_argument("--n-volume-strain", type=int, default=0)

    parser.add_argument("--no-vacancies", action="store_true")
    parser.add_argument("--no-interstitials", action="store_true")
    parser.add_argument("--no-frenkel", action="store_true")
    parser.add_argument("--no-schottky", action="store_true")
    parser.add_argument("--no-antisite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_root = ensure_dir(args.output_root.resolve())
    template_dir = args.vasp_template.resolve() if args.vasp_template else None
    poscar = resolve_input_poscar(args.poscar, template_dir)
    atoms0 = read(poscar)
    species_order = species_order_from_atoms(atoms0)
    validate_vasp_template(template_dir, atoms=atoms0, require_poscar=args.poscar is None)

    pool = generate_pool(atoms0, poscar, args)
    if not pool:
        raise RuntimeError("No defect candidates were generated. Check species and --no-* options.")

    records = []
    summary = []
    for item in pool:
        run_dir = output_root / item["family"] / item["name"]
        write_case(
            item["atoms"],
            run_dir,
            item["info"],
            template_dir,
            args.copy_template_all,
            species_order,
        )
        summary.append(item["info"])
        records.append(
            DefectRecord(
                run_dir=run_dir,
                case_name=item["name"],
                family=item["family"],
                source_poscar=poscar,
                seed=args.seed,
            )
        )

    runlist = args.runlist.resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.resolve() if args.index else output_root / "candidate_index.csv"
    write_runlist(records, runlist)
    write_index(records, index)
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    by_family: dict[str, int] = {}
    for record in records:
        by_family[record.family] = by_family.get(record.family, 0) + 1

    print(f"Template POSCAR : {poscar}")
    print(f"Structure       : {summarize_atoms(atoms0)}")
    print(f"Species order   : {', '.join(species_order)}")
    print(f"Output root     : {output_root}")
    print(f"Candidate runs  : {len(records)}")
    print("Family counts:")
    for family, count in sorted(by_family.items()):
        print(f"  {family:24s} {count:4d}")
    print(f"Wrote runlist   : {runlist}")
    print(f"Wrote index     : {index}")


if __name__ == "__main__":
    main()
