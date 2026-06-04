"""Small cation/anion sublattice randomization wrapper for an existing POSCAR."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from atomi.vasp.poscar_project import main as project_poscar_main


def comma_items(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rand_cell",
        description=(
            "Randomize occupations on an existing POSCAR lattice while moving MAGMOM "
            "decorations with the shuffled atoms. This is a short wrapper around "
            "vasp-project-poscar for the case where the desired structure is already in place."
        ),
    )
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"), help="Input POSCAR/CONTCAR. Default: POSCAR.")
    parser.add_argument("--incar", type=Path, default=Path("INCAR"), help="Reference INCAR carrying MAGMOM/LDAU order. Default: INCAR.")
    parser.add_argument("--outdir", type=Path, default=Path("rand_cell"), help="Output folder. Default: rand_cell.")
    parser.add_argument(
        "--route",
        choices=("random", "atat", "both"),
        default="random",
        help="Write ranked random candidates, submit ATAT handoff, or both. Default: random.",
    )
    parser.add_argument(
        "--cation-elements",
        action="append",
        required=True,
        help="Cation sublattice elements to shuffle, comma-separated or repeatable, e.g. Gd,U.",
    )
    parser.add_argument(
        "--anion-elements",
        action="append",
        default=[],
        help="Anion sublattice elements. Default: O.",
    )
    parser.add_argument(
        "--species-order",
        action="append",
        default=[],
        help="Output POSCAR/INCAR species order, e.g. Gd,U,O. Default: cations then anions.",
    )
    parser.add_argument("--candidates", type=int, default=3, help="Ranked random candidates to write. Default: 3.")
    parser.add_argument("--pool-size", type=int, default=200, help="Random pool size before ranking. Default: 200.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed. Default: 12345.")
    parser.add_argument(
        "--sublattice",
        action="append",
        default=["cation"],
        help="Sublattice to randomize: cation, anion, or both. Default: cation.",
    )
    parser.add_argument("--oxidation-state", action="append", default=[], help="Forwarded to vasp-project-poscar, e.g. Gd=3,U=4,O=-2.")
    parser.add_argument("--magmom-oxidation", action="append", default=[], help="Forwarded to vasp-project-poscar, e.g. U:1=5,U:2=4,Gd:7=3.")
    parser.add_argument("--min-guest-distance", type=float, help="Hard filter for minority guest-cation distance in Angstrom.")
    parser.add_argument("--max-guest-vacancy-distance", type=float, help="Hard filter for guest-vacancy distance in Angstrom.")
    parser.add_argument("--atat-atoms", type=int, help="Target atom/site count passed to ATAT mcsqs -n.")
    parser.add_argument("--atat-job-name", default="atat-random", help="ATAT Slurm job name. Default: atat-random.")
    parser.add_argument("--atat-walltime", default="04:00:00", help="ATAT Slurm walltime. Default: 04:00:00.")
    parser.add_argument("--atat-pair-diameter", type=float, default=6.0, help="ATAT pair cluster diameter. Default: 6.0.")
    parser.add_argument("--atat-triplet-diameter", type=float, help="Optional ATAT triplet cluster diameter.")
    parser.add_argument("--atat-quadruplet-diameter", type=float, help="Optional ATAT quadruplet cluster diameter.")
    parser.add_argument("--atat-temperature", type=float, help="Optional ATAT mcsqs Monte Carlo temperature.")
    parser.add_argument("--atat-max-steps", type=int, help="Optional ATAT mcsqs max steps.")
    parser.add_argument(
        "--no-submit-atat",
        action="store_true",
        help="For --route atat/both, write atat_random/submit_mcsqs.sbatch but do not submit it.",
    )
    parser.add_argument("--allow-large-cation-distance", action="store_true", help="Forwarded safety override.")
    parser.add_argument("--allow-small-generated-distance", action="store_true", help="Forwarded diagnostic override.")
    return parser


def route_project_args(args: argparse.Namespace) -> list[str]:
    cations = comma_items(args.cation_elements)
    anions = comma_items(args.anion_elements) or ["O"]
    species_order = comma_items(args.species_order) or [*cations, *anions]
    project_args = [
        "--element-poscar",
        str(args.poscar),
        "--structure-poscar",
        str(args.poscar),
        "--incar-a",
        str(args.incar),
        "--outdir",
        str(args.outdir),
        "--cation-elements",
        ",".join(cations),
        "--anion-elements",
        ",".join(anions),
        "--species-order",
        ",".join(species_order),
        "--randomize-candidates",
        str(args.candidates),
        "--randomize-pool-size",
        str(args.pool_size),
        "--randomize-seed",
        str(args.seed),
        "--randomize-atat-job-name",
        args.atat_job_name,
        "--randomize-mcsqs-walltime",
        args.atat_walltime,
        "--randomize-mcsqs-pair-diameter",
        str(args.atat_pair_diameter),
    ]
    for sublattice in comma_items(args.sublattice):
        project_args.extend(["--randomize-sublattice", sublattice])
    for value in args.oxidation_state:
        project_args.extend(["--oxidation-state", value])
    for value in args.magmom_oxidation:
        project_args.extend(["--magmom-oxidation", value])
    if args.min_guest_distance is not None:
        project_args.extend(["--randomize-min-guest-distance", str(args.min_guest_distance)])
    if args.max_guest_vacancy_distance is not None:
        project_args.extend(["--randomize-max-guest-vacancy-distance", str(args.max_guest_vacancy_distance)])
    if args.atat_atoms is not None:
        project_args.extend(["--randomize-atat-atoms", str(args.atat_atoms)])
    if args.atat_triplet_diameter is not None:
        project_args.extend(["--randomize-mcsqs-triplet-diameter", str(args.atat_triplet_diameter)])
    if args.atat_quadruplet_diameter is not None:
        project_args.extend(["--randomize-mcsqs-quadruplet-diameter", str(args.atat_quadruplet_diameter)])
    if args.atat_temperature is not None:
        project_args.extend(["--randomize-mcsqs-temperature", str(args.atat_temperature)])
    if args.atat_max_steps is not None:
        project_args.extend(["--randomize-mcsqs-max-steps", str(args.atat_max_steps)])
    if args.allow_large_cation_distance:
        project_args.append("--allow-large-cation-distance")
    if args.allow_small_generated_distance:
        project_args.append("--allow-small-generated-distance")
    return project_args


def submit_atat(outdir: Path) -> None:
    atat_dir = outdir.expanduser().resolve() / "atat_random"
    submit_script = atat_dir / "submit_mcsqs.sbatch"
    if not submit_script.exists():
        raise FileNotFoundError(f"Missing ATAT submit script: {submit_script}")
    subprocess.run(["sbatch", str(submit_script)], cwd=atat_dir, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_poscar_main(route_project_args(args))
    print("")
    if args.route in {"random", "both"}:
        print(f"Random candidates : {args.outdir / 'candidates'}")
        print(f"Random runlist    : {args.outdir / 'randomized_runlist.txt'}")
    if args.route in {"atat", "both"}:
        print(f"ATAT folder       : {args.outdir / 'atat_random'}")
        if args.no_submit_atat:
            print(f"ATAT submit       : cd {args.outdir / 'atat_random'} && sbatch submit_mcsqs.sbatch")
        else:
            submit_atat(args.outdir)


if __name__ == "__main__":
    main()
