import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


AVOGADRO = 6.02214076e23
WATER_MOLAR_MASS_G_MOL = 18.01528
CP2K_DATA_DIR = os.environ.get("ATOMI_CP2K_DATA_DIR") or os.environ.get("CP2K_DATA_DIR") or "."

DENSITY_PRESETS = {
    "regular": 1.0,
    "normal": 1.0,
    "loose": 0.75,
    "very-loose": 0.60,
}

METALS = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", "Np", "Pu",
}

VDW_RADII = {
    "H": 1.20,
    "O": 1.52,
    "Na": 2.27,
    "K": 2.75,
    "Ca": 2.31,
    "Cl": 1.75,
    "Ga": 1.87,
    "Nb": 2.07,
}

PAIR_MIN_DIST = {
    frozenset(("H", "H")): 0.90,
    frozenset(("H", "O")): 1.00,
    frozenset(("H", "Cl")): 1.45,
    frozenset(("H", "Na")): 1.60,
    frozenset(("H", "Ga")): 1.60,
    frozenset(("O", "O")): 2.20,
    frozenset(("O", "Cl")): 2.35,
    frozenset(("O", "Na")): 2.10,
    frozenset(("O", "K")): 2.40,
    frozenset(("O", "Ca")): 2.30,
    frozenset(("O", "Ga")): 2.45,
    frozenset(("O", "Nb")): 2.00,
    frozenset(("Cl", "Cl")): 2.80,
    frozenset(("Cl", "Na")): 2.70,
    frozenset(("Cl", "K")): 3.00,
    frozenset(("Cl", "Ca")): 2.90,
    frozenset(("Cl", "Ga")): 2.10,
    frozenset(("Na", "Na")): 3.20,
    frozenset(("Na", "Ga")): 4.00,
    frozenset(("K", "K")): 3.50,
    frozenset(("Ca", "Ca")): 3.40,
    frozenset(("Ca", "Nb")): 5.00,
    frozenset(("Ga", "Ga")): 2.80,
    frozenset(("Nb", "Nb")): 3.20,
}

KIND_DEFAULTS = {
    "H": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q1"),
    "O": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q6"),
    "Cl": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q7"),
    "Na": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "K": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q9"),
    "Ca": ("DZVP-MOLOPT-PBE-GTH", "GTH-PBE-q10"),
    "Ga": ("DZVP-MOLOPT-SR-GTH", "GTH-PBE-q13"),
    "Nb": ("DZVP-MOLOPT-PBE-GTH-q13", "GTH-PBE-q13"),
}


@dataclass(frozen=True)
class Restraint:
    index: int
    metal_index: int
    ligand_index: int
    target: float


def density_to_water_count(box_length: float, density_g_ml: float) -> int:
    """Return the number of water molecules in a cubic Angstrom box at the requested density."""
    volume_cm3 = box_length**3 * 1.0e-24
    water_moles = density_g_ml * volume_cm3 / WATER_MOLAR_MASS_G_MOL
    return max(0, int(round(water_moles * AVOGADRO)))


def water_count_to_box_length(n_waters: int, density_g_ml: float) -> float:
    """Return the cubic Angstrom length needed for n water molecules at density_g_ml."""
    if n_waters <= 0:
        return 0.0
    volume_cm3 = n_waters * WATER_MOLAR_MASS_G_MOL / (density_g_ml * AVOGADRO)
    return (volume_cm3 / 1.0e-24) ** (1.0 / 3.0)


def parse_density(value: str | None, preset: str) -> float:
    if value is None:
        return DENSITY_PRESETS[preset]
    try:
        density = float(value)
    except ValueError:
        if value not in DENSITY_PRESETS:
            allowed = ", ".join(sorted(DENSITY_PRESETS))
            raise ValueError(f"unknown density preset {value!r}; use a number or one of: {allowed}")
        density = DENSITY_PRESETS[value]
    if density <= 0:
        raise ValueError("water density must be positive")
    return density


def pair_min_dist(sym1: str, sym2: str) -> float:
    key = frozenset((sym1, sym2))
    if key in PAIR_MIN_DIST:
        return PAIR_MIN_DIST[key]
    r1 = VDW_RADII.get(sym1, 1.7)
    r2 = VDW_RADII.get(sym2, 1.7)
    return 0.60 * (r1 + r2)


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3)],
        ]
    )


def require_ase() -> tuple[Any, Any, Any, Any]:
    try:
        from ase import Atoms
        from ase.build import molecule
        from ase.io import read, write
    except ImportError as exc:
        raise SystemExit(
            "cp2k-build-acid-box requires ASE. "
            "Install with: python -m pip install -e '.[materials]'"
        ) from exc
    return Atoms, molecule, read, write


def rotate_atoms(atoms: Any, rng: np.random.Generator) -> Any:
    rotated = atoms.copy()
    pos = rotated.get_positions()
    com = pos.mean(axis=0)
    pos_rot = (pos - com) @ random_rotation_matrix(rng).T + com
    rotated.set_positions(pos_rot)
    return rotated


def center_atoms(atoms: Any, center: np.ndarray) -> Any:
    centered = atoms.copy()
    centroid = centered.get_positions().mean(axis=0)
    centered.translate(center - centroid)
    return centered


def auto_box_length(
    solute: Any,
    n_waters: int,
    density_g_ml: float,
    padding: float = 5.0,
    min_box: float = 16.0,
) -> float:
    water_box = water_count_to_box_length(n_waters, density_g_ml) + 4.0
    span = np.max(solute.get_positions(), axis=0) - np.min(solute.get_positions(), axis=0)
    solute_box = float(np.max(span)) + 2 * padding
    return round(max(min_box, water_box, solute_box), 2)


def overlap_with_existing(trial_atoms: Any, existing_atoms: Any) -> bool:
    if len(existing_atoms) == 0:
        return False

    pos_new = trial_atoms.get_positions()
    sym_new = trial_atoms.get_chemical_symbols()
    pos_old = existing_atoms.get_positions()
    sym_old = existing_atoms.get_chemical_symbols()

    for i, p1 in enumerate(pos_new):
        for j, p2 in enumerate(pos_old):
            if np.linalg.norm(p1 - p2) < pair_min_dist(sym_new[i], sym_old[j]):
                return True
    return False


def place_fragment_random(
    system: Any,
    fragment: Any,
    box: float,
    rng: np.random.Generator,
    margin: float = 1.2,
    max_attempts: int = 5000,
) -> Any:
    placed = system.copy()
    for _ in range(max_attempts):
        frag = fragment.copy()
        if len(frag) > 1:
            frag = rotate_atoms(frag, rng)
        target = rng.uniform(low=margin, high=box - margin, size=3)
        frag.translate(target - frag.get_positions().mean(axis=0))
        pos = frag.get_positions()
        if np.any(pos < 0.0) or np.any(pos > box):
            continue
        if not overlap_with_existing(frag, placed):
            placed += frag
            return placed
    raise RuntimeError(
        "failed to place fragment without overlap; try a larger box or lower density"
    )


def build_h3o() -> Any:
    Atoms, _, _, _ = require_ase()
    oxygen = np.array([0.0, 0.0, 0.0])
    directions = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=float,
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    hydrogens = oxygen + 0.98 * directions
    return Atoms(symbols=["O", "H", "H", "H"], positions=np.vstack([oxygen, hydrogens]))


def place_outer_sphere_single_atom(
    system: Any,
    symbol: str,
    solute_center: np.ndarray,
    shell_radius: float,
    jitter: float,
    box: float,
    rng: np.random.Generator,
    max_attempts: int = 3000,
) -> Any:
    Atoms, _, _, _ = require_ase()
    placed = system.copy()
    for _ in range(max_attempts):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = max(3.5, shell_radius + rng.normal(scale=jitter))
        pos = solute_center + radius * direction
        atom = Atoms(symbol, positions=[pos])
        if np.any(pos < 1.0) or np.any(pos > box - 1.0):
            continue
        if not overlap_with_existing(atom, placed):
            placed += atom
            return placed
    raise RuntimeError(f"failed to place {symbol} in outer sphere")


def place_outer_sphere_h3o(
    system: Any,
    solute_center: np.ndarray,
    shell_radius: float,
    jitter: float,
    box: float,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> Any:
    placed = system.copy()
    for _ in range(max_attempts):
        frag = rotate_atoms(build_h3o(), rng)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = max(4.5, shell_radius + rng.normal(scale=jitter))
        target = solute_center + radius * direction
        frag.translate(target - frag.get_positions().mean(axis=0))
        pos = frag.get_positions()
        if np.any(pos < 0.5) or np.any(pos > box - 0.5):
            continue
        if not overlap_with_existing(frag, placed):
            placed += frag
            return placed
    raise RuntimeError("failed to place H3O+ in outer sphere")


def place_species_outer_then_random(
    system: Any,
    species: str,
    n: int,
    box: float,
    solute_center: np.ndarray,
    shell: float,
    jitter: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> Any:
    out = system.copy()
    for _ in range(n):
        if species == "Cl":
            out = place_outer_sphere_single_atom(
                out, "Cl", solute_center, shell, jitter, box, rng, max_attempts
            )
        elif species == "H3O":
            out = place_outer_sphere_h3o(out, solute_center, shell, jitter, box, rng, max_attempts)
        else:
            raise ValueError(f"unknown species: {species}")
    return out


def find_metal_index(symbols: list[str], requested_index: int | None = None) -> int:
    if requested_index is not None:
        if requested_index < 1 or requested_index > len(symbols):
            raise ValueError("--metal-index is 1-based and outside the seed atom range")
        return requested_index - 1
    for i, symbol in enumerate(symbols):
        if symbol in METALS:
            return i
    raise ValueError("could not auto-detect a metal atom; pass --metal-index")


def detect_restraints(
    solute: Any,
    metal_index: int,
    cutoff: float,
    max_restraints: int | None,
) -> list[Restraint]:
    symbols = solute.get_chemical_symbols()
    positions = solute.get_positions()
    metal_pos = positions[metal_index]
    candidates = []
    for i, symbol in enumerate(symbols):
        if i == metal_index or symbol == "H" or symbol in METALS:
            continue
        distance = float(np.linalg.norm(positions[i] - metal_pos))
        if distance <= cutoff:
            candidates.append((distance, i))
    candidates.sort()
    if max_restraints is not None:
        candidates = candidates[:max_restraints]
    return [
        Restraint(index=n, metal_index=metal_index + 1, ligand_index=i + 1, target=distance)
        for n, (distance, i) in enumerate(candidates, start=1)
    ]


def render_colvars(restraints: list[Restraint]) -> str:
    blocks = []
    for restraint in restraints:
        blocks.append(
            "\n".join(
                [
                    "    &COLVAR",
                    "      &DISTANCE",
                    f"        ATOMS {restraint.metal_index} {restraint.ligand_index}",
                    "      &END DISTANCE",
                    "    &END COLVAR",
                ]
            )
        )
    return "\n\n".join(blocks)


def render_constraints(restraints: list[Restraint], restraint_k: float) -> str:
    blocks = ["  &CONSTRAINT", "    CONSTRAINT_INIT T"]
    for restraint in restraints:
        blocks.extend(
            [
                "",
                "    &COLLECTIVE",
                f"      COLVAR {restraint.index}",
                "      INTERMOLECULAR T",
                f"      TARGET [angstrom] {restraint.target:.3f}",
                "      &RESTRAINT",
                f"        K [kcalmol] {restraint_k:.3f}",
                "      &END RESTRAINT",
                "    &END COLLECTIVE",
            ]
        )
    blocks.extend(["  &END CONSTRAINT", ""])
    return "\n".join(blocks)


def render_kinds(symbols: list[str]) -> str:
    lines = []
    for symbol in sorted(set(symbols), key=lambda s: (s != "Ga", s)):
        basis, potential = KIND_DEFAULTS.get(symbol, ("DZVP-MOLOPT-SR-GTH", "GTH-PBE"))
        lines.extend(
            [
                f"    &KIND {symbol}",
                f"      BASIS_SET {basis}",
                f"      POTENTIAL {potential}",
                "    &END KIND",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def render_force_eval(
    *,
    charge: int,
    multiplicity: int,
    box: float,
    xyz_name: str,
    symbols: list[str],
    basis_file: str,
    potential_file: str,
    d3_file: str,
    cutoff: int,
    rel_cutoff: int,
    scf_guess: str,
    md_extrapolation: bool,
    colvars: str,
) -> str:
    extrapolation = ""
    if md_extrapolation:
        extrapolation = "\n      EXTRAPOLATION ASPC\n      EXTRAPOLATION_ORDER 3"
    return f"""&FORCE_EVAL
  METHOD QS

  &DFT
    BASIS_SET_FILE_NAME {basis_file}
    POTENTIAL_FILE_NAME {potential_file}
    CHARGE {charge}
    MULTIPLICITY {multiplicity}

    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
      NGRIDS 4
    &END MGRID

    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12{extrapolation}
    &END QS

    &SCF
      MAX_SCF 150
      EPS_SCF 1.0E-4
      SCF_GUESS {scf_guess}

      &OT
        MINIMIZER CG
        PRECONDITIONER FULL_ALL
        ENERGY_GAP 0.20
      &END OT

      &OUTER_SCF
        MAX_SCF 15
        EPS_SCF 1.0E-4
      &END OUTER_SCF
    &END SCF

    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL

      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME {d3_file}
          REFERENCE_FUNCTIONAL PBE
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT

  &SUBSYS
    &CELL
      ABC {box:.3f} {box:.3f} {box:.3f}
      PERIODIC XYZ
    &END CELL

    &TOPOLOGY
      COORD_FILE_FORMAT XYZ
      COORD_FILE_NAME {xyz_name}
    &END TOPOLOGY

{colvars}

{render_kinds(symbols)}
  &END SUBSYS
&END FORCE_EVAL"""


def render_geoopt_input(
    args: argparse.Namespace,
    xyz_name: str,
    symbols: list[str],
    colvars: str,
    constraints: str,
) -> str:
    force_eval = render_force_eval(
        charge=args.charge,
        multiplicity=args.multiplicity,
        box=args.box_length,
        xyz_name=xyz_name,
        symbols=symbols,
        basis_file=args.basis_file,
        potential_file=args.potential_file,
        d3_file=args.d3_file,
        cutoff=args.cutoff,
        rel_cutoff=args.rel_cutoff,
        scf_guess="ATOMIC",
        md_extrapolation=False,
        colvars=colvars,
    )
    return f"""&GLOBAL
  PROJECT {args.project}_geoopt
  RUN_TYPE GEO_OPT
  PRINT_LEVEL MEDIUM
&END GLOBAL

{force_eval}

&MOTION
  &GEO_OPT
    TYPE MINIMIZATION
    OPTIMIZER BFGS
    MAX_ITER {args.geoopt_max_iter}
    MAX_FORCE 4.5E-4
    RMS_FORCE 3.0E-4
    MAX_DR    2.0E-3
    RMS_DR    1.5E-3
  &END GEO_OPT

{constraints}  &PRINT
    &RESTART
      FILENAME ={args.project}_geoopt.restart
      BACKUP_COPIES 2
    &END RESTART
  &END PRINT
&END MOTION
"""


def render_nvt_input(
    args: argparse.Namespace,
    xyz_name: str,
    symbols: list[str],
    colvars: str,
    constraints: str,
) -> str:
    force_eval = render_force_eval(
        charge=args.charge,
        multiplicity=args.multiplicity,
        box=args.box_length,
        xyz_name=xyz_name,
        symbols=symbols,
        basis_file=args.basis_file,
        potential_file=args.potential_file,
        d3_file=args.d3_file,
        cutoff=args.cutoff,
        rel_cutoff=args.rel_cutoff,
        scf_guess="RESTART",
        md_extrapolation=True,
        colvars=colvars,
    )
    return f"""&GLOBAL
  PROJECT {args.project}_equil
  RUN_TYPE MD
  PRINT_LEVEL MEDIUM
&END GLOBAL

&EXT_RESTART
  RESTART_FILE_NAME {args.project}_geoopt.restart
  RESTART_DEFAULT F
  RESTART_POS T
  RESTART_CELL T
  RESTART_CONSTRAINT T
&END EXT_RESTART

{force_eval}

&MOTION
  &MD
    ENSEMBLE NVT
    STEPS {args.nvt_steps}
    TIMESTEP {args.timestep}
    TEMPERATURE {args.temperature}

    &THERMOSTAT
      TYPE CSVR
      REGION GLOBAL
      &CSVR
        TIMECON 5.0
      &END CSVR
    &END THERMOSTAT
  &END MD

{constraints}  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      FILENAME ={args.project}_equil-pos.xyz
      &EACH
        MD {args.print_each}
      &END EACH
    &END TRAJECTORY

    &VELOCITIES
      FILENAME ={args.project}_equil-vel.xyz
      &EACH
        MD {args.print_each}
      &END EACH
    &END VELOCITIES

    &RESTART
      FILENAME ={args.project}_equil.restart
      BACKUP_COPIES 2
      &EACH
        MD {args.restart_each}
      &END EACH
    &END RESTART

    &RESTART_HISTORY
      FILENAME ={args.project}_equil.restart_hist
      &EACH
        MD {args.restart_each}
      &END EACH
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp2k-build-acid-box",
        description="Build an acidified explicit-water CP2K AIMD box from a metal-ligand XYZ.",
    )
    parser.add_argument(
        "seed_xyz",
        type=Path,
        help="Metal-ligand seed XYZ. Seed atoms remain first.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output box XYZ.")
    parser.add_argument("--project", default=None, help="CP2K project/file prefix.")
    parser.add_argument(
        "--waters",
        type=int,
        default=None,
        help="Explicit water count; overrides density.",
    )
    parser.add_argument("--density-preset", choices=sorted(DENSITY_PRESETS), default="regular")
    parser.add_argument(
        "--water-density",
        default=None,
        help="Water density in g/mL or preset name.",
    )
    parser.add_argument("--cl", type=int, default=0, help="Number of chloride ions to add.")
    parser.add_argument("--h3o", type=int, default=0, help="Number of hydronium ions to add.")
    parser.add_argument("--box", type=float, default=None, help="Cubic box length in Angstrom.")
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Solute padding for automatic box.",
    )
    parser.add_argument("--min-box", type=float, default=16.0, help="Minimum automatic box length.")
    parser.add_argument("--cl-shell", type=float, default=7.0)
    parser.add_argument("--cl-jitter", type=float, default=0.7)
    parser.add_argument("--h3o-shell", type=float, default=8.0)
    parser.add_argument("--h3o-jitter", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for packing.")
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument(
        "--basis-file",
        default=str(Path(CP2K_DATA_DIR) / "BASIS_MOLOPT"),
    )
    parser.add_argument(
        "--potential-file",
        default=str(Path(CP2K_DATA_DIR) / "GTH_POTENTIALS"),
    )
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
    rng = np.random.default_rng(args.seed)
    center = np.array([box / 2.0, box / 2.0, box / 2.0])
    system = center_atoms(solute, center)
    system.set_cell([box, box, box])
    system.set_pbc([True, True, True])

    if args.cl > 0:
        system = place_species_outer_then_random(
            system,
            "Cl",
            args.cl,
            box,
            center,
            args.cl_shell,
            args.cl_jitter,
            rng,
            args.max_attempts,
        )
    if args.h3o > 0:
        system = place_species_outer_then_random(
            system,
            "H3O",
            args.h3o,
            box,
            center,
            args.h3o_shell,
            args.h3o_jitter,
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
        f"cl={args.cl}; h3o={args.h3o}; box={box:.3f}A; rng_seed={args.seed}"
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

    colvar_path = Path(f"{project}_restraints_colvar.inc")
    constraint_path = Path(f"{project}_restraints_constraint.inc")
    summary_path = Path(f"{project}_restraints.tsv")
    colvar_path.write_text(colvars + ("\n" if colvars else ""), encoding="utf-8")
    constraint_path.write_text(constraints, encoding="utf-8")
    summary_path.write_text(
        "colvar\tmetal_index\tligand_index\ttarget_angstrom\n"
        + "".join(
            f"{r.index}\t{r.metal_index}\t{r.ligand_index}\t{r.target:.6f}\n" for r in restraints
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output}")
    print(f"Wrote {geoopt_path}")
    print(f"Wrote {nvt_path}")
    print(f"Wrote {colvar_path}")
    print(f"Wrote {constraint_path}")
    print(f"Wrote {summary_path}")
    print(f"Total atoms: {len(system)}")
    print(f"Box length: {box:.3f} Angstrom")
    print(f"Waters: {waters} at density {density:.3f} g/mL")
    print(f"Cl-: {args.cl}")
    print(f"H3O+: {args.h3o}")
    print(f"Restraints: {len(restraints)}")
    print("Review CHARGE, KIND potentials, and CP2K data paths before production runs.")


if __name__ == "__main__":
    main(sys.argv[1:])
