from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from atomi.vasp.prefail import copy_vasp_template
from atomi.vasp.prep import summarize_atoms, validate_vasp_template


@dataclass
class CloudRecord:
    run_dir: Path
    motif_id: str
    motif_index: int
    seed_poscar: Path
    variant: str
    family: str
    seed: int
    info: dict


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "motif"


def fmt_float(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return f"{sign}{abs(value):.4f}".replace(".", "p")


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


def structured_displacement(atoms: Atoms, amplitude: float, rng: np.random.Generator) -> Atoms:
    result = atoms.copy()
    disp = normalized_random_vectors(result.positions.shape, rng)
    mags = amplitude * rng.uniform(0.9, 1.1, size=(len(result), 1))
    result.set_positions(result.positions + disp * mags)
    return result


def isotropic_strain(atoms: Atoms, eps: float) -> Atoms:
    result = atoms.copy()
    result.set_cell(result.cell * (1.0 + eps), scale_atoms=True)
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
    if int(mask.sum()) == 0:
        return result
    disp = np.zeros_like(result.positions)
    vectors = normalized_random_vectors((int(mask.sum()), 3), rng)
    mags = amplitude * rng.uniform(0.7, 1.3, size=(int(mask.sum()), 1))
    disp[mask] = vectors * mags
    result.set_positions(result.positions + disp)
    return result


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


def read_seed_list(path: Path) -> list[Path]:
    seeds = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        seed = Path(line)
        if not seed.is_absolute():
            seed = path.parent / seed
        seeds.append(seed.resolve())
    return seeds


def should_exclude(path: Path, root: Path, patterns: list[str], template_dir: Path | None) -> bool:
    resolved = path.resolve()
    if template_dir is not None:
        try:
            resolved.relative_to(template_dir.resolve())
            return True
        except ValueError:
            pass
    try:
        rel = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = resolved.as_posix()
    return any(fnmatch.fnmatch(rel, pattern) for pattern in patterns)


def discover_seed_poscars(args: argparse.Namespace, template_dir: Path | None) -> list[Path]:
    seeds: list[Path] = []
    if args.seed_poscar:
        seeds.extend(path.expanduser().resolve() for path in args.seed_poscar)
    if args.seed_list:
        seeds.extend(read_seed_list(args.seed_list.expanduser().resolve()))
    if args.seed_root:
        root = args.seed_root.expanduser().resolve()
        for pattern in args.seed_glob:
            seeds.extend(path.resolve() for path in root.glob(pattern) if path.is_file())

    excluded = list(args.exclude_glob)
    root = args.seed_root.expanduser().resolve() if args.seed_root else Path.cwd().resolve()
    unique = []
    seen: set[Path] = set()
    for seed in seeds:
        if should_exclude(seed, root, excluded, template_dir):
            continue
        if seed in seen:
            continue
        seen.add(seed)
        unique.append(seed)
    return sorted(unique)


def motif_id_for_seed(path: Path, used: dict[str, int]) -> str:
    base = path.parent.name if path.name in {"POSCAR", "CONTCAR"} else path.stem
    name = safe_name(base)
    count = used.get(name, 0) + 1
    used[name] = count
    return name if count == 1 else f"{name}_{count:03d}"


def build_variant_pool(atoms: Atoms, args: argparse.Namespace, rng: np.random.Generator) -> list[tuple[str, str, Atoms, dict]]:
    variants: list[tuple[str, str, Atoms, dict]] = [
        ("base", "base", atoms.copy(), {}),
    ]
    for index in range(args.n_random):
        variants.append(
            (
                f"rd_small_{index + 1:03d}",
                "random_displacement",
                random_displacement(atoms, args.random_amp, rng),
                {"amplitude_A": args.random_amp},
            )
        )
    for eps in args.iso_strains:
        variants.append(
            (
                f"iso_{fmt_float(eps)}",
                "isotropic_strain",
                isotropic_strain(atoms, eps),
                {"linear_strain": eps, "volume_scale": float((1.0 + eps) ** 3)},
            )
        )
    if args.bias_species in set(atoms.get_chemical_symbols()):
        for index in range(args.n_bias_species):
            variants.append(
                (
                    f"bias_{safe_name(args.bias_species)}_{index + 1:03d}",
                    "species_biased_displacement",
                    species_biased_displacement(atoms, args.bias_species, args.bias_amp, rng),
                    {"species": args.bias_species, "amplitude_A": args.bias_amp},
                )
            )
    for index in range(args.n_mixed):
        variants.append(
            (
                f"mixed_{index + 1:03d}",
                "mixed_displacement",
                random_displacement(atoms, args.mixed_amp, rng),
                {"amplitude_A": args.mixed_amp},
            )
        )
    for index in range(args.n_structured):
        variants.append(
            (
                f"disp_small_{index + 1:03d}",
                "structured_displacement",
                structured_displacement(atoms, args.structured_amp, rng),
                {"amplitude_A": args.structured_amp},
            )
        )
    return variants


def compact_variants(atoms: Atoms, args: argparse.Namespace, rng: np.random.Generator) -> list[tuple[str, str, Atoms, dict]]:
    variants = build_variant_pool(atoms, args, rng)
    target = args.per_motif
    extra_index = 0
    while target is not None and len(variants) < target:
        extra_index += 1
        variants.append(
            (
                f"disp_small_extra_{extra_index:03d}",
                "structured_displacement",
                structured_displacement(atoms, args.structured_amp, rng),
                {"amplitude_A": args.structured_amp, "filled_to_per_motif": True},
            )
        )
    if target is not None:
        variants = variants[:target]
    return variants


def write_runlist(records: list[CloudRecord], runlist: Path) -> None:
    ensure_dir(runlist.parent)
    base = runlist.parent.resolve()
    lines = []
    for record in records:
        try:
            lines.append(str(record.run_dir.resolve().relative_to(base)))
        except ValueError:
            lines.append(str(record.run_dir.resolve()))
    runlist.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_index(records: list[CloudRecord], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "run_dir",
        "motif_id",
        "motif_index",
        "seed_poscar",
        "variant",
        "family",
        "seed",
        "composition",
        "details_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "run_dir": str(record.run_dir.resolve()),
                    "motif_id": record.motif_id,
                    "motif_index": record.motif_index,
                    "seed_poscar": str(record.seed_poscar.resolve()),
                    "variant": record.variant,
                    "family": record.family,
                    "seed": record.seed,
                    "composition": record.info["composition"],
                    "details_json": json.dumps(record.info, sort_keys=True),
                }
            )


def generate_cloud(
    seeds: list[Path],
    output_root: Path,
    template_dir: Path | None,
    args: argparse.Namespace,
) -> list[CloudRecord]:
    records: list[CloudRecord] = []
    used_names: dict[str, int] = {}
    for motif_index, seed_poscar in enumerate(seeds, start=1):
        atoms = read(seed_poscar)
        motif_id = motif_id_for_seed(seed_poscar, used_names)
        motif_seed = args.seed + (motif_index - 1) * 1009
        rng = np.random.default_rng(motif_seed)
        variants = compact_variants(atoms, args, rng)
        for variant, family, candidate, extra in variants:
            run_dir = output_root / motif_id / variant
            info = {
                "schema": "atomi.vasp.defect_cloud.case.v1",
                "motif_id": motif_id,
                "motif_index": motif_index,
                "seed_poscar": str(seed_poscar.resolve()),
                "variant": variant,
                "family": family,
                "seed": motif_seed,
                "composition": summarize_atoms(candidate),
                "preserve_atom_order": True,
                **extra,
            }
            write_case(candidate, run_dir, info, template_dir, args.copy_template_all)
            records.append(
                CloudRecord(
                    run_dir=run_dir,
                    motif_id=motif_id,
                    motif_index=motif_index,
                    seed_poscar=seed_poscar,
                    variant=variant,
                    family=family,
                    seed=motif_seed,
                    info=info,
                )
            )
        print(f"Prepared {len(variants)} defect-cloud structures for {motif_id}: {seed_poscar}")
    return records


def write_summary(records: list[CloudRecord], seeds: list[Path], output_root: Path, args: argparse.Namespace) -> None:
    by_motif: dict[str, dict[str, int]] = {}
    for record in records:
        motif = by_motif.setdefault(record.motif_id, {})
        motif[record.family] = motif.get(record.family, 0) + 1
    payload = {
        "schema": "atomi.vasp.defect_cloud.summary.v1",
        "n_seed_motifs": len(seeds),
        "n_candidate_runs": len(records),
        "per_motif_requested": args.per_motif,
        "seed": args.seed,
        "seed_poscars": [str(path.resolve()) for path in seeds],
        "families_by_motif": by_motif,
        "defaults": {
            "random_amp_A": args.random_amp,
            "structured_amp_A": args.structured_amp,
            "bias_species": args.bias_species,
            "bias_amp_A": args.bias_amp,
            "mixed_amp_A": args.mixed_amp,
            "iso_strains": args.iso_strains,
        },
    }
    (output_root / "defect_cloud_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-defect-cloud",
        description="Prepare compact local perturbation clouds from relaxed defect motif POSCARs.",
    )
    parser.add_argument("--seed-root", type=Path, default=None, help="Root containing relaxed motif POSCARs.")
    parser.add_argument(
        "--seed-glob",
        action="append",
        default=None,
        help="Glob under --seed-root for seed structures. Default: **/POSCAR.",
    )
    parser.add_argument("--seed-poscar", action="append", type=Path, default=None, help="Explicit seed POSCAR path.")
    parser.add_argument("--seed-list", type=Path, default=None, help="Text file containing one seed POSCAR path per line.")
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=["**/VASP_TEMPLATE/**", "VASP_TEMPLATE/**"],
        help="Exclude seed paths matching this glob relative to --seed-root. Repeat as needed.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--vasp-template", type=Path, default=None)
    parser.add_argument("--copy-template-all", action="store_true")
    parser.add_argument("--runlist", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--per-motif", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260518)

    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--random-amp", type=float, default=0.02)
    parser.add_argument("--iso-strains", type=float, nargs="*", default=[-0.01, 0.01])
    parser.add_argument("--bias-species", default="O")
    parser.add_argument("--n-bias-species", type=int, default=1)
    parser.add_argument("--bias-amp", type=float, default=0.05)
    parser.add_argument("--n-mixed", type=int, default=1)
    parser.add_argument("--mixed-amp", type=float, default=0.04)
    parser.add_argument("--n-structured", type=int, default=0)
    parser.add_argument("--structured-amp", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seed_glob is None:
        args.seed_glob = ["**/POSCAR"]
    if args.per_motif is not None and args.per_motif <= 0:
        parser.error("--per-motif must be positive")

    output_root = ensure_dir(args.output_root.expanduser().resolve())
    template_dir = args.vasp_template.expanduser().resolve() if args.vasp_template else None
    seeds = discover_seed_poscars(args, template_dir)
    if not seeds:
        parser.error("No seed POSCARs found. Provide --seed-root, --seed-poscar, or --seed-list.")

    first_atoms = read(seeds[0])
    validate_vasp_template(template_dir, atoms=first_atoms, require_poscar=False)
    records = generate_cloud(seeds, output_root, template_dir, args)

    runlist = args.runlist.expanduser().resolve() if args.runlist else output_root / "runlist.txt"
    index = args.index.expanduser().resolve() if args.index else output_root / "defect_cloud_index.csv"
    write_runlist(records, runlist)
    write_index(records, index)
    write_summary(records, seeds, output_root, args)

    print("")
    print(f"Seed motifs       : {len(seeds)}")
    print(f"Candidate runs    : {len(records)}")
    print(f"Per motif target  : {args.per_motif}")
    print(f"Output root       : {output_root}")
    print(f"Runlist           : {runlist}")
    print(f"Index             : {index}")
    print(f"Summary           : {output_root / 'defect_cloud_summary.json'}")


if __name__ == "__main__":
    main()
