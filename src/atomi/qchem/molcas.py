"""OpenMolcas embedded-cluster helpers.

This module prepares P1/no-symmetry embedded clusters for actinide spectroscopy
workflows: QM center(s) plus first-shell ligands and an XFIELD point-charge
embedding generated from a periodic POSCAR/CONTCAR.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MolcasClusterOptions:
    center_indices: tuple[int, ...]
    ligand_element: str = "O"
    ligand_cutoff_A: float = 2.8
    point_charge_cutoff_A: float = 12.0
    oxygen_charge: float = -2.0
    basis: str = "ANO-RCC-VDZP"
    group: str = "NoSym"
    spin: int = 2
    charge_mode: str = "neutral-average"


def _is_ints(parts: list[str]) -> bool:
    try:
        [int(p) for p in parts]
        return True
    except ValueError:
        return False


def read_poscar(path: Path) -> dict[str, object]:
    lines = [line.rstrip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path} is too short to be a POSCAR/CONTCAR")
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float) * scale
    p5 = lines[5].split()
    if _is_ints(p5):
        symbols = [f"X{i+1}" for i in range(len(p5))]
        counts = [int(x) for x in p5]
        mode_idx = 6
    else:
        symbols = p5
        counts = [int(x) for x in lines[6].split()]
        mode_idx = 7
    if lines[mode_idx].lower().startswith("s"):
        mode_idx += 1
    mode = lines[mode_idx].lower()
    natoms = sum(counts)
    raw = np.array([[float(x) for x in line.split()[:3]] for line in lines[mode_idx + 1 : mode_idx + 1 + natoms]], dtype=float)
    if mode.startswith("d"):
        frac = raw
        cart = frac @ lattice
    else:
        cart = raw * scale
        frac = cart @ np.linalg.inv(lattice)
    species: list[str] = []
    for symbol, count in zip(symbols, counts):
        species.extend([symbol] * count)
    return {"symbols": symbols, "counts": counts, "species": species, "lattice": lattice, "frac": frac, "cart": cart}


def mic_delta(frac_a: np.ndarray, frac_b: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    diff = frac_b - frac_a
    diff -= np.round(diff)
    return diff @ lattice


def mic_distance(frac_a: np.ndarray, frac_b: np.ndarray, lattice: np.ndarray) -> float:
    return float(np.linalg.norm(mic_delta(frac_a, frac_b, lattice)))


def formal_u_charge(species: Sequence[str], oxygen_charge: float = -2.0) -> float:
    n_u = sum(1 for s in species if s == "U")
    n_o = sum(1 for s in species if s == "O")
    if n_u <= 0:
        raise ValueError("formal U charge requires at least one U atom")
    return -oxygen_charge * n_o / n_u


def cluster_indices(data: dict[str, object], options: MolcasClusterOptions) -> list[int]:
    species = data["species"]  # type: ignore[assignment]
    lattice = data["lattice"]  # type: ignore[assignment]
    frac = data["frac"]  # type: ignore[assignment]
    centers0 = [idx - 1 for idx in options.center_indices]
    qset = set(centers0)
    for idx, symbol in enumerate(species):
        if symbol != options.ligand_element:
            continue
        if min(mic_distance(frac[c], frac[idx], lattice) for c in centers0) <= options.ligand_cutoff_A:
            qset.add(idx)
    return sorted(qset)


def unwrap_cluster(data: dict[str, object], qm_indices: Sequence[int], centers0: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lattice = data["lattice"]  # type: ignore[assignment]
    frac = data["frac"]  # type: ignore[assignment]
    ref_frac = frac[centers0[0]]
    coords = []
    for idx in qm_indices:
        diff = frac[idx] - ref_frac
        diff -= np.round(diff)
        coords.append(diff @ lattice)
    arr = np.array(coords, dtype=float)
    center_positions = [arr[list(qm_indices).index(i)] for i in centers0]
    origin = np.mean(center_positions, axis=0)
    return arr - origin, origin, ref_frac @ lattice


def make_point_charges(
    data: dict[str, object],
    qm_indices: Sequence[int],
    qm_coords: np.ndarray,
    origin: np.ndarray,
    ref_cart: np.ndarray,
    options: MolcasClusterOptions,
) -> tuple[list[dict[str, object]], float]:
    species = data["species"]  # type: ignore[assignment]
    lattice = data["lattice"]  # type: ignore[assignment]
    frac = data["frac"]  # type: ignore[assignment]
    q_u = formal_u_charge(species, options.oxygen_charge)
    q_by_species = {"U": q_u, "O": options.oxygen_charge}
    qm_home = set(qm_indices)
    nrep = int(math.ceil(options.point_charge_cutoff_A / float(np.min(np.linalg.norm(lattice, axis=1))))) + 1
    charges: list[dict[str, object]] = []
    for a in range(-nrep, nrep + 1):
        for b in range(-nrep, nrep + 1):
            for c in range(-nrep, nrep + 1):
                shift = np.array([a, b, c], dtype=float)
                shift_cart = shift @ lattice
                home = a == b == c == 0
                for idx, symbol in enumerate(species):
                    if home and idx in qm_home:
                        continue
                    if symbol not in q_by_species:
                        continue
                    pos = frac[idx] @ lattice + shift_cart - ref_cart - origin
                    if float(np.min(np.linalg.norm(qm_coords - pos, axis=1))) <= options.point_charge_cutoff_A:
                        charges.append(
                            {
                                "element": symbol,
                                "source_index": idx + 1,
                                "x": float(pos[0]),
                                "y": float(pos[1]),
                                "z": float(pos[2]),
                                "charge": float(q_by_species[symbol]),
                            }
                        )
    return charges, q_u


def template_charge(data: dict[str, object], qm_indices: Sequence[int], options: MolcasClusterOptions, q_u: float) -> int:
    species = data["species"]  # type: ignore[assignment]
    n_u = sum(1 for i in qm_indices if species[i] == "U")
    n_o = sum(1 for i in qm_indices if species[i] == "O")
    if options.charge_mode == "u5-like":
        return int(round(5.0 * n_u + options.oxygen_charge * n_o))
    if options.charge_mode == "u4-like":
        return int(round(4.0 * n_u + options.oxygen_charge * n_o))
    return int(round(q_u * n_u + options.oxygen_charge * n_o))


def write_xyz(path: Path, species: Sequence[str], qm_indices: Sequence[int], coords: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(qm_indices)}\n")
        handle.write("Atomi OpenMolcas embedded cluster, coordinates in Angstrom\n")
        for idx, xyz in zip(qm_indices, coords):
            handle.write(f"{species[idx]:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f} # VASP_atom={idx + 1}\n")


def write_xfield(path: Path, charges: Sequence[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(charges)} Angstrom 0\n")
        for row in charges:
            handle.write(
                f"{row['x']: .10f} {row['y']: .10f} {row['z']: .10f} {row['charge']: .8f} "
                f"! {row['element']}{row['source_index']}\n"
            )


def render_ground_input(title: str, xyz: str, xfield: str, charge: int, options: MolcasClusterOptions) -> str:
    return f"""* Atomi OpenMolcas embedded-cluster ground-state template
&GATEWAY
Title = {title}
Coord = {xyz}
Basis = {options.basis}
Group = {options.group}
RX2C
AMFI
XField = {xfield}
ANGM
0.0 0.0 0.0

&SEWARD
Cholesky

&SCF
Charge = {charge}
Spin = {options.spin}
Iterations = 100

* Replace/extend SCF with calibrated RASSCF after orbital inspection.
"""


def render_medge_input(title: str, xyz: str, xfield: str, charge: int, options: MolcasClusterOptions) -> str:
    return f"""* Atomi U M4,5-edge HERFD-XANES conceptual OpenMolcas template
* Use all-electron U basis. Do not use an ECP that removes U 3d core states.
&GATEWAY
Title = {title}_Medge
Coord = {xyz}
Basis = {options.basis}
Group = {options.group}
RX2C
AMFI
XField = {xfield}
ANGM
0.0 0.0 0.0

&SEWARD
Cholesky

* Ground-state RASSCF sketch:
* - U(V)-like single center: U 5f active shell with about 1 active electron.
* - U(IV)-like single center: U 5f active shell with about 2 active electrons.
* - Covalent/mixed case: add selected U-O bonding/antibonding or O 2p ligand orbitals.
&RASSCF
Title = {title}_ground_placeholder
Symmetry = 1
Spin = {options.spin}
Charge = {charge}
Iterations = 200 100
CIRoots = 5 5 1
* Set nActEl / Inactive / Ras1/Ras2/Ras3 after inspecting SCF orbitals.
* For M edge, put U 3d in RAS1 and U 5f in RAS2 for core-excited states,
* then couple ground and core-excited JobIph files with RASSI-SO.
"""


def render_sbatch(project: str, tasks: int = 8, hours: int = 24) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={project[:28]}
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
#SBATCH --nodes=1
#SBATCH --tasks-per-node={tasks}
#SBATCH --mem-per-cpu=4000M
#SBATCH --time={hours}:00:00
#SBATCH --gres=scratch:200

unset LANG; export LC_ALL=C
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -s 200000
ml chem/openmolcas/24.02
export MOLCAS_NPROCS=${{SLURM_NTASKS:-{tasks}}}
export MOLCAS_MEM=${{SLURM_MEM_PER_CPU:-4000}}
export MOLCAS_WORKDIR=${{SCRATCH:-/tmp/$USER}}/${{SLURM_JOB_NAME}}.${{SLURM_JOB_ID}}.molcas_scratch
mkdir -p "$MOLCAS_WORKDIR"
pymolcas -np "$MOLCAS_NPROCS" {project}.ground.inp > {project}.ground.out.${{SLURM_JOB_ID}}
status=$?
tar -czf {project}.molcas_work.${{SLURM_JOB_ID}}.tgz -C "$MOLCAS_WORKDIR" . || true
rm -rf "$MOLCAS_WORKDIR"
exit $status
"""


def write_cluster_workspace(poscar: Path, outdir: Path, options: MolcasClusterOptions, *, label: str = "molcas_cluster", overwrite: bool = False) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("xyz", "xfield", "ground.inp", "Medge_template.inp", "sbatch", "metadata.json"):
        target = outdir / f"{label}.{suffix}"
        if target.exists() and not overwrite:
            raise FileExistsError(f"{target} exists; pass --overwrite to replace it")
    data = read_poscar(poscar)
    qm = cluster_indices(data, options)
    centers0 = [i - 1 for i in options.center_indices]
    coords, origin, ref_cart = unwrap_cluster(data, qm, centers0)
    charges, q_u = make_point_charges(data, qm, coords, origin, ref_cart, options)
    charge = template_charge(data, qm, options, q_u)
    species = data["species"]  # type: ignore[assignment]
    write_xyz(outdir / f"{label}.xyz", species, qm, coords)
    write_xfield(outdir / f"{label}.xfield", charges)
    (outdir / f"{label}.ground.inp").write_text(
        render_ground_input(label, f"{label}.xyz", f"{label}.xfield", charge, options), encoding="utf-8"
    )
    (outdir / f"{label}.Medge_template.inp").write_text(
        render_medge_input(label, f"{label}.xyz", f"{label}.xfield", charge, options), encoding="utf-8"
    )
    sbatch = outdir / f"run_{label}.sbatch"
    sbatch.write_text(render_sbatch(label), encoding="utf-8")
    sbatch.chmod(0o755)
    metadata = {
        "schema": "atomi.qchem.molcas_cluster.v1",
        "source_structure": str(poscar),
        "options": asdict(options),
        "qm_atom_count": len(qm),
        "qm_indices_1based": [i + 1 for i in qm],
        "point_charge_count": len(charges),
        "formal_u_charge_embedding": q_u,
        "template_charge": charge,
        "outputs": {
            "xyz": str(outdir / f"{label}.xyz"),
            "xfield": str(outdir / f"{label}.xfield"),
            "ground_input": str(outdir / f"{label}.ground.inp"),
            "medge_template": str(outdir / f"{label}.Medge_template.inp"),
            "sbatch": str(sbatch),
        },
    }
    (outdir / f"{label}.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _parse_indices(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in text.replace(",", " ").split():
        if "-" in chunk:
            a, b = [int(x) for x in chunk.split("-", 1)]
            values.extend(range(a, b + 1))
        else:
            values.append(int(chunk))
    if not values:
        raise argparse.ArgumentTypeError("at least one center index is required")
    return tuple(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare OpenMolcas embedded-cluster inputs from POSCAR/CONTCAR.")
    parser.add_argument("--poscar", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("molcas_cluster"))
    parser.add_argument("--label", default="molcas_cluster")
    parser.add_argument("--centers", type=_parse_indices, required=True, help="1-based center atoms, e.g. 3-8 or 3,4,5")
    parser.add_argument("--ligand-element", default="O")
    parser.add_argument("--ligand-cutoff", type=float, default=2.8)
    parser.add_argument("--point-charge-cutoff", type=float, default=12.0)
    parser.add_argument("--oxygen-charge", type=float, default=-2.0)
    parser.add_argument("--basis", default="ANO-RCC-VDZP")
    parser.add_argument("--group", default="NoSym")
    parser.add_argument("--spin", type=int, default=2)
    parser.add_argument("--charge-mode", choices=("neutral-average", "u5-like", "u4-like"), default="neutral-average")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = MolcasClusterOptions(
        center_indices=args.centers,
        ligand_element=args.ligand_element,
        ligand_cutoff_A=args.ligand_cutoff,
        point_charge_cutoff_A=args.point_charge_cutoff,
        oxygen_charge=args.oxygen_charge,
        basis=args.basis,
        group=args.group,
        spin=args.spin,
        charge_mode=args.charge_mode,
    )
    metadata = write_cluster_workspace(args.poscar, args.outdir, options, label=args.label, overwrite=args.overwrite)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
