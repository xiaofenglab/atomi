"""Build randomized alkaline explicit-water CP2K boxes.

This is the alkaline/base-side sibling of :mod:`atomi.cp2k.acid_box`.  It keeps
seed atoms first, then places outer-sphere cations, hydroxide fragments, and
randomly rotated waters with simple hard-sphere rejection.  The generated XYZ is
intended as a chemically sensible starting point, not an equilibrated solvent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from atomi.cp2k.acid_box import (
    CP2K_DATA_DIR,
    DENSITY_PRESETS,
    Restraint,
    auto_box_length,
    center_atoms,
    density_to_water_count,
    detect_restraints,
    find_metal_index,
    overlap_with_existing,
    parse_density,
    place_fragment_random,
    place_outer_sphere_single_atom,
    render_colvars,
    render_constraints,
    render_geoopt_input,
    render_nvt_input,
    require_ase,
)


DEFAULT_CATION_SHELLS = {
    "Na": 7.5,
    "K": 8.0,
    "Ca": 8.5,
}


def build_oh_minus() -> Any:
    Atoms, _, _, _ = require_ase()
    return Atoms(symbols=["O", "H"], positions=[[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])


def place_outer_sphere_fragments(
    system: Any,
    fragment: Any,
    n: int,
    box: float,
    solute_center: np.ndarray,
    shell_radius: float,
    jitter: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> Any:
    placed = system.copy()
    for _ in range(n):
        for _attempt in range(max_attempts):
            frag = fragment.copy()
            if len(frag) > 1:
                from atomi.cp2k.acid_box import rotate_atoms

                frag = rotate_atoms(frag, rng)
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            radius = max(4.0, shell_radius + rng.normal(scale=jitter))
            target = solute_center + radius * direction
            frag.translate(target - frag.get_positions().mean(axis=0))
            pos = frag.get_positions()
            if np.any(pos < 0.7) or np.any(pos > box - 0.7):
                continue
            if not overlap_with_existing(frag, placed):
                placed += frag
                break
        else:
            raise RuntimeError("failed to place outer-sphere fragment without overlap")
    return placed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp2k-build-alkaline-box",
        description=(
            "Build randomized alkaline explicit-water CP2K boxes from a seed XYZ. "
            "Seed atoms remain first for restraint/TI bookkeeping."
        ),
    )
    parser.add_argument("seed_xyz", type=Path, help="Solute seed XYZ. Seed atoms remain first.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output box XYZ.")
    parser.add_argument("--project", default=None, help="CP2K project/file prefix.")
    parser.add_argument("--waters", type=int, default=None, help="Explicit water count.")
    parser.add_argument("--density-preset", choices=sorted(DENSITY_PRESETS), default="regular")
    parser.add_argument("--water-density", default=None, help="Water density in g/mL or preset.")
    parser.add_argument("--box", type=float, default=None, help="Cubic box length in Angstrom.")
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--min-box", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for packing.")
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--na", type=int, default=0, help="Number of outer-sphere Na+ ions.")
    parser.add_argument("--k", type=int, default=0, help="Number of outer-sphere K+ ions.")
    parser.add_argument("--ca", type=int, default=0, help="Number of outer-sphere Ca2+ ions.")
    parser.add_argument("--oh", type=int, default=0, help="Number of outer-sphere OH- fragments.")
    parser.add_argument("--cation-shell", type=float, default=None)
    parser.add_argument("--cation-jitter", type=float, default=0.8)
    parser.add_argument("--oh-shell", type=float, default=6.5)
    parser.add_argument("--oh-jitter", type=float, default=0.7)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--basis-file", default=str(Path(CP2K_DATA_DIR) / "BASIS_MOLOPT"))
    parser.add_argument(
        "--basis-file-extra",
        action="append",
        default=[],
        help="Additional CP2K BASIS_SET_FILE_NAME entries, e.g. BASIS_MOLOPT_UZH for Nb.",
    )
    parser.add_argument("--potential-file", default=str(Path(CP2K_DATA_DIR) / "GTH_POTENTIALS"))
    parser.add_argument("--d3-file", default=str(Path(CP2K_DATA_DIR) / "dftd3.dat"))
    parser.add_argument("--cutoff", type=int, default=300)
    parser.add_argument("--rel-cutoff", type=int, default=40)
    parser.add_argument("--geoopt-max-iter", type=int, default=200)
    parser.add_argument("--nvt-steps", type=int, default=3000)
    parser.add_argument("--timestep", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--print-each", type=int, default=10)
    parser.add_argument("--restart-each", type=int, default=100)
    parser.add_argument("--restraints", choices=("auto", "none"), default="auto")
    parser.add_argument("--metal-index", type=int, default=None)
    parser.add_argument("--ligand-cutoff", type=float, default=3.0)
    parser.add_argument("--max-restraints", type=int, default=None)
    parser.add_argument("--restraint-k", type=float, default=5.0)
    return parser


def infer_project(seed_xyz: Path, output: Path | None, project: str | None) -> str:
    if project:
        return project
    if output:
        stem = output.stem
        return stem[:-4] if stem.endswith("_box") else stem
    return seed_xyz.stem


def normalize_basis_files(primary: str, extras: list[str]) -> str:
    files = [primary, *extras]
    if not extras:
        return primary
    return primary + "".join(f"\n    BASIS_SET_FILE_NAME {item}" for item in extras)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.seed_xyz.is_file():
        parser.error(f"seed XYZ not found: {args.seed_xyz}")
    try:
        density = parse_density(args.water_density, args.density_preset)
    except ValueError as exc:
        parser.error(str(exc))

    _, molecule, read, write = require_ase()
    solute = read(str(args.seed_xyz))
    if len(solute) == 0:
        parser.error("seed XYZ is empty")

    project = infer_project(args.seed_xyz, args.output, args.project)
    args.project = project
    output = args.output or Path(f"{project}_box.xyz")
    if args.box is not None and args.waters is None:
        waters = density_to_water_count(args.box, density)
        box = float(args.box)
    elif args.box is None and args.waters is not None:
        waters = args.waters
        box = auto_box_length(solute, waters, density, padding=args.padding, min_box=args.min_box)
    elif args.box is None and args.waters is None:
        box = args.min_box
        waters = density_to_water_count(box, density)
        box = auto_box_length(solute, waters, density, padding=args.padding, min_box=args.min_box)
    else:
        waters = args.waters
        box = float(args.box)

    args.box_length = box
    args.basis_file = normalize_basis_files(args.basis_file, args.basis_file_extra)
    rng = np.random.default_rng(args.seed)
    center = np.array([box / 2.0, box / 2.0, box / 2.0])
    system = center_atoms(solute, center)
    system.set_cell([box, box, box])
    system.set_pbc([True, True, True])

    for symbol, count in (("Na", args.na), ("K", args.k), ("Ca", args.ca)):
        shell = args.cation_shell or DEFAULT_CATION_SHELLS[symbol]
        for _ in range(count):
            system = place_outer_sphere_single_atom(
                system,
                symbol,
                center,
                shell,
                args.cation_jitter,
                box,
                rng,
                args.max_attempts,
            )
    if args.oh:
        system = place_outer_sphere_fragments(
            system,
            build_oh_minus(),
            args.oh,
            box,
            center,
            args.oh_shell,
            args.oh_jitter,
            rng,
            args.max_attempts,
        )

    water = molecule("H2O")
    for _ in range(waters):
        system = place_fragment_random(system, water, box, rng, max_attempts=args.max_attempts)

    system.set_cell([box, box, box])
    system.set_pbc([True, True, True])
    comment = (
        f"seed={args.seed_xyz.name}; waters={waters}; density={density:.3f}g_ml; "
        f"Na={args.na}; K={args.k}; Ca={args.ca}; OH={args.oh}; "
        f"box={box:.3f}A; rng_seed={args.seed}; alkaline_box"
    )
    write(output, system, comment=comment)

    restraints: list[Restraint] = []
    if args.restraints == "auto":
        metal_index = find_metal_index(solute.get_chemical_symbols(), args.metal_index)
        restraints = detect_restraints(solute, metal_index, args.ligand_cutoff, args.max_restraints)

    colvars = render_colvars(restraints)
    constraints = render_constraints(restraints, args.restraint_k) if restraints else ""
    symbols = system.get_chemical_symbols()

    geoopt_path = Path(f"{project}_geoopt.inp")
    nvt_path = Path(f"{project}_nvt.inp")
    geoopt_path.write_text(
        render_geoopt_input(args, output.name, symbols, colvars, constraints),
        encoding="utf-8",
    )
    nvt_path.write_text(
        render_nvt_input(args, output.name, symbols, colvars, constraints),
        encoding="utf-8",
    )
    Path(f"{project}_restraints_colvar.inc").write_text(
        colvars + ("\n" if colvars else ""), encoding="utf-8"
    )
    Path(f"{project}_restraints_constraint.inc").write_text(constraints, encoding="utf-8")
    Path(f"{project}_restraints.tsv").write_text(
        "colvar\tmetal_index\tligand_index\ttarget_angstrom\n"
        + "".join(
            f"{r.index}\t{r.metal_index}\t{r.ligand_index}\t{r.target:.6f}\n" for r in restraints
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output}")
    print(f"Wrote {geoopt_path}")
    print(f"Wrote {nvt_path}")
    print(f"Total atoms: {len(system)}")
    print(f"Box length: {box:.3f} Angstrom")
    print(f"Waters: {waters} at density {density:.3f} g/mL")
    print(f"Na+: {args.na}; K+: {args.k}; Ca2+: {args.ca}; OH-: {args.oh}")
    print(f"Restraints: {len(restraints)}")
    print("Review CHARGE, KIND potentials, and CP2K data paths before production runs.")


if __name__ == "__main__":
    main(sys.argv[1:])
