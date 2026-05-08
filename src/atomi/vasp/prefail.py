from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write

from atomi.vasp.prep import (
    DEFAULT_TEMPLATE_FILES,
    species_order_from_atoms,
    summarize_atoms,
    template_poscar,
    validate_vasp_template,
)

DEFAULT_SPECIES_ORDER: tuple[str, ...] = ()


@dataclass
class CandidateRecord:
    run_dir: Path
    parent_poscar: Path
    variant: str
    seed: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_from_base(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def parse_box_bounds(box_header: str, box_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    parts = box_header.strip().split()
    triclinic = ("xy" in parts) or ("xz" in parts) or ("yz" in parts)

    if not triclinic:
        xlo, xhi = map(float, box_lines[0].split()[:2])
        ylo, yhi = map(float, box_lines[1].split()[:2])
        zlo, zhi = map(float, box_lines[2].split()[:2])
        cell = np.array(
            [
                [xhi - xlo, 0.0, 0.0],
                [0.0, yhi - ylo, 0.0],
                [0.0, 0.0, zhi - zlo],
            ],
            dtype=float,
        )
        return cell, np.array([xlo, ylo, zlo], dtype=float)

    xlo_b, xhi_b, xy = map(float, box_lines[0].split()[:3])
    ylo_b, yhi_b, xz = map(float, box_lines[1].split()[:3])
    zlo_b, zhi_b, yz = map(float, box_lines[2].split()[:3])

    xlo = xlo_b - min(0.0, xy, xz, xy + xz)
    xhi = xhi_b - max(0.0, xy, xz, xy + xz)
    ylo = ylo_b - min(0.0, yz)
    yhi = yhi_b - max(0.0, yz)
    zlo = zlo_b
    zhi = zhi_b

    cell = np.array(
        [
            [xhi - xlo, 0.0, 0.0],
            [xy, yhi - ylo, 0.0],
            [xz, yz, zhi - zlo],
        ],
        dtype=float,
    )
    return cell, np.array([xlo, ylo, zlo], dtype=float)


def iter_lammps_dump_frames(dump_path: Path) -> Iterable[tuple[int, str, list[str], str, list[str]]]:
    with dump_path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            timestep = int(handle.readline().strip())

            line = handle.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Expected NUMBER OF ATOMS after timestep {timestep}")
            natoms = int(handle.readline().strip())

            box_header = handle.readline()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected BOX BOUNDS after timestep {timestep}")
            box_lines = [handle.readline().strip() for _ in range(3)]

            atoms_header = handle.readline()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Expected ATOMS after timestep {timestep}")
            atom_lines = [handle.readline().strip() for _ in range(natoms)]
            yield timestep, box_header, box_lines, atoms_header, atom_lines


def parse_atoms_block(
    atom_lines: list[str],
    atoms_header: str,
    atom_type_map: dict[int, str],
    origin: np.ndarray,
    cell: np.ndarray,
) -> Atoms:
    cols = atoms_header.strip().split()[2:]
    idx = {column: i for i, column in enumerate(cols)}

    has_cart = all(key in idx for key in ("id", "type", "x", "y", "z"))
    has_scaled = all(key in idx for key in ("id", "type", "xs", "ys", "zs"))
    if not (has_cart or has_scaled):
        raise ValueError("Unsupported ATOMS header. Need id/type plus x y z or xs ys zs.")

    records = []
    for line in atom_lines:
        parts = line.split()
        atom_id = int(parts[idx["id"]])
        atom_type = int(parts[idx["type"]])
        if atom_type not in atom_type_map:
            raise KeyError(f"Atom type {atom_type} not found in atom_type_map.")

        if has_cart:
            coord = np.array(
                [float(parts[idx["x"]]), float(parts[idx["y"]]), float(parts[idx["z"]])],
                dtype=float,
            )
            coord = coord - origin
            is_scaled = False
        else:
            coord = np.array(
                [float(parts[idx["xs"]]), float(parts[idx["ys"]]), float(parts[idx["zs"]])],
                dtype=float,
            )
            is_scaled = True

        records.append((atom_id, atom_type_map[atom_type], coord, is_scaled))

    records.sort(key=lambda item: item[0])
    atoms = Atoms(symbols=[r[1] for r in records], cell=cell, pbc=True)
    coords = np.array([r[2] for r in records], dtype=float)
    if records[0][3]:
        atoms.set_scaled_positions(coords)
    else:
        atoms.set_positions(coords)
    return atoms


def infer_replicate_from_counts(reference: Atoms, md_atoms: Atoms) -> tuple[int, int, int]:
    ratio = len(md_atoms) / len(reference)
    n = round(ratio ** (1.0 / 3.0))
    if n**3 * len(reference) != len(md_atoms):
        raise ValueError(
            "Could not infer cubic replicate from atom counts: "
            f"reference={len(reference)}, md={len(md_atoms)}"
        )
    return (n, n, n)


def infer_replicate_from_cell(reference: Atoms, md_atoms: Atoms) -> tuple[int, int, int]:
    ratios = np.array(md_atoms.cell.lengths()) / np.array(reference.cell.lengths())
    reps = np.rint(ratios).astype(int)
    if np.any(reps < 1):
        raise ValueError("Invalid inferred replicate from cell lengths.")
    return tuple(int(value) for value in reps)


def choose_reference(reference: Atoms, md_atoms: Atoms, replicate: tuple[int, int, int] | None) -> Atoms:
    if len(reference) == len(md_atoms):
        return reference.copy()

    if replicate is None:
        try:
            replicate = infer_replicate_from_cell(reference, md_atoms)
        except Exception:
            replicate = infer_replicate_from_counts(reference, md_atoms)

    ref_rep = make_supercell(reference, np.diag(replicate))
    if len(ref_rep) != len(md_atoms):
        raise ValueError(
            f"Replicated reference atom count ({len(ref_rep)}) does not match "
            f"MD atom count ({len(md_atoms)})."
        )
    return ref_rep


def frac_diff_pbc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    diff -= np.round(diff)
    return diff


def greedy_match_species(ref_frac: np.ndarray, md_frac: np.ndarray) -> list[int]:
    if len(ref_frac) != len(md_frac):
        raise ValueError("Species counts do not match between reference and MD snapshot.")

    pairs = []
    for i_md in range(len(md_frac)):
        dist = np.linalg.norm(frac_diff_pbc(md_frac[i_md][None, :], ref_frac), axis=1)
        for j_ref, value in enumerate(dist):
            pairs.append((float(value), i_md, j_ref))
    pairs.sort(key=lambda item: item[0])

    md_used: set[int] = set()
    ref_used: set[int] = set()
    ref_to_md = [-1] * len(ref_frac)
    for _, i_md, j_ref in pairs:
        if i_md in md_used or j_ref in ref_used:
            continue
        ref_to_md[j_ref] = i_md
        md_used.add(i_md)
        ref_used.add(j_ref)
        if len(md_used) == len(ref_frac):
            break

    if any(index < 0 for index in ref_to_md):
        raise RuntimeError("Failed to construct complete species mapping.")
    return ref_to_md


def reorder_md_to_reference(
    reference: Atoms,
    md_atoms: Atoms,
    species_order: tuple[str, ...] = DEFAULT_SPECIES_ORDER,
) -> Atoms:
    if not species_order:
        species_order = species_order_from_atoms(reference)
    ref_symbols = np.array(reference.get_chemical_symbols())
    md_symbols = np.array(md_atoms.get_chemical_symbols())
    ref_frac = reference.get_scaled_positions(wrap=True)
    md_frac = md_atoms.get_scaled_positions(wrap=True)

    ordered_global_md = []
    expected_ref_indices = []
    for species in species_order:
        ref_idx = np.where(ref_symbols == species)[0]
        md_idx = np.where(md_symbols == species)[0]
        if len(ref_idx) != len(md_idx):
            raise ValueError(
                f"Species counts differ for {species}: ref={len(ref_idx)} md={len(md_idx)}"
            )
        local_map = greedy_match_species(ref_frac[ref_idx], md_frac[md_idx])
        ordered_global_md.extend(md_idx[local_i] for local_i in local_map)
        expected_ref_indices.extend(ref_idx)

    new_atoms = md_atoms[ordered_global_md].copy()
    expected_symbols = list(ref_symbols[expected_ref_indices])
    if new_atoms.get_chemical_symbols() != expected_symbols:
        raise RuntimeError("Final reordered MD symbols do not match expected reference order.")
    return new_atoms


def write_poscar_manual(
    atoms: Atoms,
    filename: Path,
    species_order: tuple[str, ...] = DEFAULT_SPECIES_ORDER,
    title: str | None = None,
) -> None:
    if not species_order:
        species_order = species_order_from_atoms(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    scaled = atoms.get_scaled_positions(wrap=True)
    counts = [int(np.sum(symbols == species)) for species in species_order]
    title = title or " ".join(f"{species}{count}" for species, count in zip(species_order, counts))

    with filename.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("   1.00000000000000\n")
        for vec in atoms.cell.array:
            handle.write(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}\n")
        handle.write("   " + "   ".join(species_order) + "\n")
        handle.write("   " + "   ".join(f"{count:4d}" for count in counts) + "\n")
        handle.write("Direct\n")
        for row in scaled:
            handle.write(f"  {row[0]:16.10f}  {row[1]:16.10f}  {row[2]:16.10f}\n")


def extract_base_poscars(
    config_path: Path,
    selected_root_override: Path | None = None,
    template_dir: Path | None = None,
) -> list[Path]:
    cfg = read_json(config_path)
    base = config_path.parent
    dump_file = resolve_from_base(cfg["dump_file"], base)
    output_root = selected_root_override or resolve_from_base(
        cfg.get("output_root", "MD_SELECT"),
        base,
    )
    atom_type_map = {int(key): value for key, value in cfg["atom_type_map"].items()}
    if "reference_poscar" in cfg:
        reference_poscar = resolve_from_base(cfg["reference_poscar"], base)
    else:
        reference_poscar = template_poscar(template_dir)
        if reference_poscar is None:
            raise ValueError("JSON needs reference_poscar unless --vasp-template/POSCAR exists.")
        reference_poscar = reference_poscar.resolve()
    reference = read(reference_poscar)
    species_order = tuple(cfg.get("species_order", list(species_order_from_atoms(reference))))
    validate_vasp_template(template_dir, atoms=reference, require_poscar=False)

    replicate = None
    if "reference_replicate" in cfg:
        rep = cfg["reference_replicate"]
        if len(rep) != 3:
            raise ValueError("'reference_replicate' must have three integers.")
        replicate = (int(rep[0]), int(rep[1]), int(rep[2]))

    timestep_to_labels: dict[int, list[str]] = {}
    for job in cfg["jobs"]:
        for timestep in job["timesteps"]:
            timestep_to_labels.setdefault(int(timestep), []).append(job["label"])

    requested = set(timestep_to_labels)
    found: set[int] = set()
    parent_poscars = []
    ensure_dir(output_root)

    print(f"Reading dump file: {dump_file}")
    print(f"Using reference POSCAR: {reference_poscar}")
    print(f"Reference structure: {summarize_atoms(reference)}")
    print(f"Species order: {', '.join(species_order)}")
    print(f"Atom type map: {atom_type_map}")
    print(f"Requested timesteps: {len(requested)}")
    print(f"Writing selected parent POSCARs under: {output_root}")

    for timestep, box_header, box_lines, atoms_header, atom_lines in iter_lammps_dump_frames(dump_file):
        if timestep not in requested:
            continue
        cell, origin = parse_box_bounds(box_header, box_lines)
        md_atoms = parse_atoms_block(atom_lines, atoms_header, atom_type_map, origin, cell)
        ref_for_match = choose_reference(reference, md_atoms, replicate)
        reordered = reorder_md_to_reference(ref_for_match, md_atoms, species_order=species_order)

        for label in timestep_to_labels[timestep]:
            out_dir = ensure_dir(output_root / label / f"md_{timestep:06d}" / "base")
            out_file = out_dir / "POSCAR"
            write_poscar_manual(reordered, out_file, species_order=species_order)
            parent_poscars.append(out_file)
            print(f"Wrote {out_file}")
        found.add(timestep)

    missing = sorted(requested - found)
    if missing:
        print("\nWARNING: requested timesteps not found in dump:")
        for timestep in missing:
            print(f"  {timestep}")
    return sorted(parent_poscars)


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


def write_candidate(
    atoms: Atoms,
    out_dir: Path,
    template_dir: Path | None,
    copy_template_all: bool,
) -> None:
    ensure_dir(out_dir)
    write(out_dir / "POSCAR", atoms, format="vasp", direct=True, vasp5=True, sort=False)
    copy_vasp_template(template_dir, out_dir, copy_all=copy_template_all)


def isotropic_strain(atoms: Atoms, eps: float) -> Atoms:
    result = atoms.copy()
    result.set_cell(atoms.cell * (1.0 + eps), scale_atoms=True)
    return result


def anisotropic_strain(atoms: Atoms, scale_xyz: tuple[float, float, float]) -> Atoms:
    result = atoms.copy()
    cell = result.cell.array.copy()
    cell[0, :] *= scale_xyz[0]
    cell[1, :] *= scale_xyz[1]
    cell[2, :] *= scale_xyz[2]
    result.set_cell(cell, scale_atoms=True)
    return result


def shear_distortion(atoms: Atoms, mode: str, gamma: float) -> Atoms:
    result = atoms.copy()
    cell = result.cell.array.copy()
    if mode == "xy":
        cell[1, 0] += gamma * np.linalg.norm(cell[0])
    elif mode == "xz":
        cell[2, 0] += gamma * np.linalg.norm(cell[0])
    elif mode == "yz":
        cell[2, 1] += gamma * np.linalg.norm(cell[1])
    else:
        raise ValueError(f"Unknown shear mode: {mode}")
    frac = result.get_scaled_positions(wrap=True)
    result.set_cell(cell, scale_atoms=False)
    result.set_scaled_positions(frac)
    return result


def normalized_random_vectors(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    vectors = rng.normal(size=shape)
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0.0] = 1.0
    return vectors / norms[:, None]


def random_displacement(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> Atoms:
    result = atoms.copy()
    disp = normalized_random_vectors(result.positions.shape, rng)
    amps = amplitude * rng.uniform(0.7, 1.3, size=(len(result), 1))
    result.set_positions(result.positions + disp * amps)
    return result


def structured_small_displacement(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> Atoms:
    result = atoms.copy()
    disp = normalized_random_vectors(result.positions.shape, rng)
    amps = amplitude * rng.uniform(0.9, 1.1, size=(len(result), 1))
    result.set_positions(result.positions + disp * amps)
    return result


def species_biased_displacement(
    atoms: Atoms,
    mode: str,
    amplitude: float,
    rng: np.random.Generator,
) -> Atoms:
    result = atoms.copy()
    symbols = np.array(result.get_chemical_symbols())
    mask = symbols == "O" if mode == "O" else np.ones(len(symbols), dtype=bool)
    if int(mask.sum()) == 0:
        return result
    disp = np.zeros_like(result.positions)
    vectors = normalized_random_vectors((int(mask.sum()), 3), rng)
    amps = amplitude * rng.uniform(0.7, 1.3, size=(int(mask.sum()), 1))
    disp[mask] = vectors * amps
    result.set_positions(result.positions + disp)
    return result


def strain_plus_displacement(
    atoms: Atoms,
    scale_xyz: tuple[float, float, float],
    disp_amp: float,
    rng: np.random.Generator,
) -> Atoms:
    return structured_small_displacement(anisotropic_strain(atoms, scale_xyz), disp_amp, rng)


def candidate_family(atoms: Atoms, args: argparse.Namespace, rng: np.random.Generator) -> list[tuple[str, Atoms]]:
    return [
        ("base", atoms.copy()),
        ("rd_small_001", random_displacement(atoms, args.rd_small_amp, rng)),
        ("rd_small_002", random_displacement(atoms, args.rd_small_amp, rng)),
        ("iso_p", isotropic_strain(atoms, +args.iso_strain)),
        ("iso_m", isotropic_strain(atoms, -args.iso_strain)),
        ("bias_o", species_biased_displacement(atoms, "O", args.bias_o, rng)),
        ("bias_mix", species_biased_displacement(atoms, "mixed", args.bias_mix, rng)),
        ("disp_small_001", structured_small_displacement(atoms, args.disp_small_amp1, rng)),
        ("disp_small_002", structured_small_displacement(atoms, args.disp_small_amp2, rng)),
        ("ani_strain_001", anisotropic_strain(atoms, (1.0 + args.ani_eps1, 1.0 - args.ani_eps1, 1.0))),
        ("ani_strain_002", anisotropic_strain(atoms, (1.0 + args.ani_eps2, 1.0, 1.0 - args.ani_eps2))),
        ("shear_xy", shear_distortion(atoms, "xy", args.shear_gamma)),
        ("shear_xz", shear_distortion(atoms, "xz", args.shear_gamma)),
        ("shear_yz", shear_distortion(atoms, "yz", args.shear_gamma)),
        (
            "strain_disp_001",
            strain_plus_displacement(
                atoms,
                (1.0 + args.ani_eps1, 1.0 - args.ani_eps1, 1.0),
                args.strain_disp_amp1,
                rng,
            ),
        ),
        (
            "strain_disp_002",
            strain_plus_displacement(
                atoms,
                (1.0 + args.ani_eps2, 1.0, 1.0 - args.ani_eps2),
                args.strain_disp_amp2,
                rng,
            ),
        ),
    ]


def find_parent_poscars(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("**/md_*/base/POSCAR"))


def generate_candidates(
    parent_poscars: list[Path],
    input_root: Path,
    output_root: Path,
    template_dir: Path | None,
    args: argparse.Namespace,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for index, poscar_path in enumerate(parent_poscars):
        parent_dir = poscar_path.parent.parent
        rel_parent = parent_dir.relative_to(input_root)
        out_parent = output_root / rel_parent
        atoms = read(poscar_path)
        seed = args.seed + index
        rng = np.random.default_rng(seed)

        variants = candidate_family(atoms, args, rng)
        for variant, candidate in variants:
            run_dir = out_parent / variant
            write_candidate(candidate, run_dir, template_dir, args.copy_template_all)
            records.append(
                CandidateRecord(
                    run_dir=run_dir,
                    parent_poscar=poscar_path,
                    variant=variant,
                    seed=seed,
                )
            )

        metadata = {
            "parent_poscar": str(poscar_path.resolve()),
            "output_parent": str(out_parent.resolve()),
            "seed": seed,
            "n_per_parent": len(variants),
        }
        ensure_dir(out_parent)
        (out_parent / "generation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Prepared {len(variants)} candidates for {rel_parent}")
    return records


def write_runlist(records: list[CandidateRecord], runlist: Path) -> None:
    ensure_dir(runlist.parent)
    lines = []
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(runlist.parent.resolve())))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_index(records: list[CandidateRecord], index_path: Path) -> None:
    ensure_dir(index_path.parent)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run_dir", "parent_poscar", "variant", "seed"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "parent_poscar": str(record.parent_poscar.resolve()),
                    "variant": record.variant,
                    "seed": record.seed,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-prefail-candidates",
        description="Extract prefail LAMMPS frames and prepare distorted VASP candidates.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Dump frame selection JSON.")
    parser.add_argument(
        "--selected-root",
        type=Path,
        default=None,
        help="Override where extracted parent POSCARs are written.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root for final VASP candidate run directories.",
    )
    parser.add_argument("--vasp-template", type=Path, default=None)
    parser.add_argument(
        "--copy-template-all",
        action="store_true",
        help="Copy every file/subdirectory from the template except POSCAR.",
    )
    parser.add_argument("--runlist", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Use existing parent POSCARs under --selected-root instead of reading the dump.",
    )

    parser.add_argument("--rd-small-amp", type=float, default=0.02)
    parser.add_argument("--iso-strain", type=float, default=0.01)
    parser.add_argument("--bias-o", type=float, default=0.05)
    parser.add_argument("--bias-mix", type=float, default=0.04)
    parser.add_argument("--disp-small-amp1", type=float, default=0.01)
    parser.add_argument("--disp-small-amp2", type=float, default=0.02)
    parser.add_argument("--ani-eps1", type=float, default=0.02)
    parser.add_argument("--ani-eps2", type=float, default=0.03)
    parser.add_argument("--shear-gamma", type=float, default=0.02)
    parser.add_argument("--strain-disp-amp1", type=float, default=0.01)
    parser.add_argument("--strain-disp-amp2", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    template_dir = args.vasp_template.resolve() if args.vasp_template else None

    if args.skip_extract:
        if args.selected_root is None:
            parser.error("--skip-extract requires --selected-root")
        selected_root = args.selected_root.resolve()
        parent_poscars = find_parent_poscars(selected_root)
        if parent_poscars:
            validate_vasp_template(template_dir, atoms=read(parent_poscars[0]), require_poscar=False)
    else:
        parent_poscars = extract_base_poscars(config_path, args.selected_root, template_dir=template_dir)
        if args.selected_root is not None:
            selected_root = args.selected_root.resolve()
        else:
            selected_root = resolve_from_base(read_json(config_path).get("output_root", "MD_SELECT"), config_path.parent)

    if not parent_poscars:
        raise RuntimeError(
            f"No parent POSCARs found. Expected paths like {selected_root}/.../md_000000/base/POSCAR"
        )

    records = generate_candidates(parent_poscars, selected_root, output_root, template_dir, args)
    runlist = args.runlist.resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.resolve() if args.index else output_root / "candidate_index.csv"
    write_runlist(records, runlist)
    write_index(records, index)

    print("")
    print(f"Parent POSCARs: {len(parent_poscars)}")
    print(f"Candidate runs: {len(records)}")
    print("Candidate variants per parent: 16")
    print(f"Runlist: {runlist}")
    print(f"Index: {index}")
