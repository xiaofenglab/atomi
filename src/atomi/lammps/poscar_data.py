"""Convert VASP POSCAR/CONTCAR structures into LAMMPS data files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomi.vasp.magmom import read_poscar_species


def _read_atoms(path: Path):
    try:
        from ase.io import read
    except Exception as exc:  # pragma: no cover - dependency declared, but keep message helpful.
        raise RuntimeError("ASE is required for POSCAR to LAMMPS data conversion.") from exc
    return read(str(path), format="vasp")


def _write_lammps_data(
    atoms,
    out: Path,
    specorder: list[str],
    atom_style: str,
    units: str,
    masses: bool,
    reduce_cell: bool,
    force_skew: bool,
    atom_type_labels: bool,
) -> None:
    try:
        from ase.io.lammpsdata import write_lammps_data
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ASE lammps-data writer is unavailable.") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        write_lammps_data(
            handle,
            atoms,
            specorder=specorder,
            atom_style=atom_style,
            units=units,
            masses=masses,
            reduce_cell=reduce_cell,
            force_skew=force_skew,
            atom_type_labels=atom_type_labels,
        )


def poscar_species_order(path: Path) -> list[str]:
    try:
        return list(read_poscar_species(path).symbols)
    except Exception:
        atoms = _read_atoms(path)
        order: list[str] = []
        for symbol in atoms.get_chemical_symbols():
            if symbol not in order:
                order.append(symbol)
        return order


def validate_species_order(atoms, specorder: list[str]) -> None:
    present = list(dict.fromkeys(atoms.get_chemical_symbols()))
    missing = [symbol for symbol in present if symbol not in specorder]
    if missing:
        raise ValueError(
            "Species order is missing elements present in the structure: "
            + ", ".join(missing)
            + ". Pass --species-order with all elements in desired LAMMPS type order."
        )


def relative_to_config(path: Path, config_path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config_path.parent.resolve()))
    except ValueError:
        return str(path.resolve())


def update_config_initial_structure(config_path: Path, data_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["initial_structure"] = relative_to_config(data_path, config_path)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def convert_poscar_to_lammps_data(
    poscar: Path,
    out: Path,
    species_order: list[str] | None = None,
    replicate: tuple[int, int, int] = (1, 1, 1),
    atom_style: str = "atomic",
    units: str = "metal",
    masses: bool = True,
    reduce_cell: bool = False,
    force_skew: bool = False,
    atom_type_labels: bool = False,
    metadata_out: Path | None = None,
    update_config: Path | None = None,
) -> dict[str, object]:
    poscar = poscar.expanduser().resolve()
    out = out.expanduser()
    atoms = _read_atoms(poscar)
    if replicate != (1, 1, 1):
        atoms = atoms.repeat(replicate)
    atoms.wrap()
    specorder = species_order or poscar_species_order(poscar)
    validate_species_order(atoms, specorder)
    _write_lammps_data(
        atoms,
        out=out,
        specorder=specorder,
        atom_style=atom_style,
        units=units,
        masses=masses,
        reduce_cell=reduce_cell,
        force_skew=force_skew,
        atom_type_labels=atom_type_labels,
    )
    type_map = {symbol: index for index, symbol in enumerate(specorder, start=1)}
    metadata = {
        "schema": "atomi.lammps.poscar_data.v1",
        "input_poscar": str(poscar),
        "output_data": str(out.resolve()),
        "atom_style": atom_style,
        "units": units,
        "masses_written": masses,
        "replicate": list(replicate),
        "natoms": len(atoms),
        "species_order": specorder,
        "lammps_type_map": type_map,
        "md_engine_initial_structure": str(out),
    }
    if metadata_out is not None:
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if update_config is not None:
        update_config_initial_structure(update_config, out)
        metadata["updated_config"] = str(update_config.resolve())
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poscar2lammps",
        description="Convert POSCAR/CONTCAR to a LAMMPS data file for md-engine initialization.",
    )
    parser.add_argument("poscar", nargs="?", type=Path, default=Path("POSCAR"), help="Input POSCAR/CONTCAR.")
    parser.add_argument("--out", type=Path, default=Path("structures/initial.data"), help="Output LAMMPS data file.")
    parser.add_argument(
        "--species-order",
        nargs="+",
        help="LAMMPS atom type order. Defaults to POSCAR element order, e.g. U O.",
    )
    parser.add_argument("--replicate", nargs=3, type=int, metavar=("NX", "NY", "NZ"), default=(1, 1, 1))
    parser.add_argument("--atom-style", default="atomic", help="LAMMPS atom style. Default: atomic.")
    parser.add_argument("--units", default="metal", help="LAMMPS units. Default: metal.")
    parser.add_argument("--no-masses", action="store_true", help="Do not write the Masses section.")
    parser.add_argument("--reduce-cell", action="store_true", help="Ask ASE to reduce the LAMMPS prism cell.")
    parser.add_argument("--force-skew", action="store_true", help="Force triclinic LAMMPS box output.")
    parser.add_argument("--atom-type-labels", action="store_true", help="Write LAMMPS atom type labels when supported.")
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional JSON summary with atom type mapping. Default: <out>.json.",
    )
    parser.add_argument(
        "--update-config",
        type=Path,
        help="Optional md-engine config.json to update with initial_structure pointing to --out.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata_out = args.metadata_out
    if metadata_out is None:
        metadata_out = args.out.with_suffix(args.out.suffix + ".json")
    metadata = convert_poscar_to_lammps_data(
        poscar=args.poscar,
        out=args.out,
        species_order=args.species_order,
        replicate=tuple(args.replicate),
        atom_style=args.atom_style,
        units=args.units,
        masses=not args.no_masses,
        reduce_cell=args.reduce_cell,
        force_skew=args.force_skew,
        atom_type_labels=args.atom_type_labels,
        metadata_out=metadata_out,
        update_config=args.update_config,
    )
    print(f"Wrote LAMMPS data : {metadata['output_data']}")
    print(f"Atoms             : {metadata['natoms']}")
    print(f"Species/type map  : {metadata['lammps_type_map']}")
    print(f"Metadata          : {metadata_out}")
    if args.update_config is not None:
        print(f"Updated config    : {args.update_config}")


if __name__ == "__main__":  # pragma: no cover
    main()
