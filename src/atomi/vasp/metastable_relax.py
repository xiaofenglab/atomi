"""Conservative VASP staging helpers for metastable structures."""

from __future__ import annotations

import argparse
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from atomi.vasp.checks import DONE_MARKER, _latest_vasp_energy
from atomi.vasp.compare_runs import angle_deg, det3, norm
from atomi.vasp.magmom import (
    PoscarSpecies,
    PoscarStructure,
    expand_incar_value_tokens,
    read_outcar_lines,
    read_poscar_structure,
    strip_incar_comment,
)


CATION_DEFAULT_EXCLUDE = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "H"}
F_ELECTRON_LDAU_SPECIES = {
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
}
STAGE_ORDER = ("00_static_scf", "01_gentle_relax", "02_continue_relax", "03_final_static")
COPY_INPUT_NAMES = ("POTCAR", "KPOINTS")
SUBMIT_CANDIDATES = ("submit.sh", "run_vasp.sh", "job.sh", "vasp.sbatch", "*.sbatch")


@dataclass
class PoscarDocument:
    comment: str
    scale: str
    cell_lines: list[str]
    species: PoscarSpecies
    selective: bool
    mode: str
    coordinates: list[list[str]]
    tails: list[list[str]]


@dataclass
class StageSpec:
    name: str
    description: str
    overrides: dict[str, str | None]
    relaxation: bool


STAGES = {
    "00_static_scf": StageSpec(
        name="00_static_scf",
        description="Electronic stabilization without ionic motion.",
        relaxation=False,
        overrides={
            "ISTART": "0",
            "ICHARG": "2",
            "NSW": "0",
            "IBRION": "-1",
            "ISIF": "2",
            "ISYM": "0",
            "LREAL": ".FALSE.",
            "ALGO": "Normal",
            "NELM": "300",
            "EDIFF": "1E-6",
            "AMIX": "0.05",
            "BMIX": "0.0001",
            "AMIX_MAG": "0.20",
            "BMIX_MAG": "0.0001",
            "MAXMIX": "80",
            "LCHARG": ".TRUE.",
            "LWAVE": ".FALSE.",
        },
    ),
    "01_gentle_relax": StageSpec(
        name="01_gentle_relax",
        description="Very short fixed-cell relaxation; useful for oxygen-only relaxation.",
        relaxation=True,
        overrides={
            "ISTART": "0",
            "ICHARG": "1",
            "IBRION": "2",
            "ISIF": "2",
            "NSW": "10",
            "POTIM": "0.05",
            "EDIFFG": "-0.05",
            "EDIFF": "1E-6",
            "ISYM": "0",
            "LREAL": ".FALSE.",
            "ALGO": "Normal",
            "LCHARG": ".TRUE.",
            "LWAVE": ".FALSE.",
        },
    ),
    "02_continue_relax": StageSpec(
        name="02_continue_relax",
        description="Continue fixed-cell relaxation only after the metastable fingerprint survives.",
        relaxation=True,
        overrides={
            "ISTART": "0",
            "ICHARG": "1",
            "IBRION": "2",
            "ISIF": "2",
            "NSW": "30",
            "POTIM": "0.10",
            "EDIFFG": "-0.01",
            "EDIFF": "1E-6",
            "ISYM": "0",
            "LREAL": "Auto",
            "ALGO": "Normal",
            "LCHARG": ".TRUE.",
            "LWAVE": ".FALSE.",
        },
    ),
    "03_final_static": StageSpec(
        name="03_final_static",
        description="Final static energy and charge/magnetization collection after the preceding ionic relaxation reaches EDIFFG=-0.01.",
        relaxation=False,
        overrides={
            "ISTART": "0",
            "ICHARG": "1",
            "NSW": "0",
            "IBRION": "-1",
            "ISIF": "2",
            "ISYM": "0",
            "LREAL": ".FALSE.",
            "EDIFF": "1E-7",
            "EDIFFG": None,
            "ALGO": "Normal",
            "LCHARG": ".TRUE.",
            "LWAVE": ".FALSE.",
            "LORBIT": "11",
            "NEDOS": "4001",
        },
    ),
}


def parse_poscar_document(path: Path) -> PoscarDocument:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 8:
        raise ValueError(f"POSCAR is too short: {path}")
    symbols = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    coord_index = 7
    selective = lines[coord_index].strip().lower().startswith("s")
    if selective:
        coord_index += 1
    mode = lines[coord_index].strip()
    coord_index += 1
    total = sum(counts)
    coordinates: list[list[str]] = []
    tails: list[list[str]] = []
    for line in lines[coord_index : coord_index + total]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Could not parse coordinate line in {path}: {line}")
        coordinates.append(parts[:3])
        tails.append(parts[3:])
    return PoscarDocument(
        comment=lines[0],
        scale=lines[1],
        cell_lines=lines[2:5],
        species=PoscarSpecies(symbols=symbols, counts=counts),
        selective=selective,
        mode=mode,
        coordinates=coordinates,
        tails=tails,
    )


def labels_for_species(species: PoscarSpecies) -> list[str]:
    labels: list[str] = []
    for symbol, count in zip(species.symbols, species.counts):
        labels.extend([symbol] * count)
    return labels


def is_cation(symbol: str) -> bool:
    return symbol not in CATION_DEFAULT_EXCLUDE


def freeze_flags_for_labels(
    labels: list[str],
    *,
    freeze_mode: str,
    freeze_species: set[str],
    freeze_indices: set[int],
    anchor_flags: tuple[str, str, str],
) -> list[tuple[str, str, str]]:
    flags: list[tuple[str, str, str]] = []
    for index, symbol in enumerate(labels, start=1):
        freeze = False
        custom: tuple[str, str, str] | None = None
        if freeze_mode == "none":
            freeze = False
        elif freeze_mode == "all_cations":
            freeze = is_cation(symbol)
        elif freeze_mode == "all_oxygen":
            freeze = symbol == "O"
        elif freeze_mode == "by_species":
            freeze = symbol in freeze_species
        elif freeze_mode == "by_index":
            freeze = index in freeze_indices
        elif freeze_mode == "small_displacement_anchor":
            if index in freeze_indices:
                custom = anchor_flags
        else:
            raise ValueError(f"Unknown freeze mode: {freeze_mode}")
        if custom is not None:
            flags.append(custom)
        else:
            flags.append(("F", "F", "F") if freeze else ("T", "T", "T"))
    return flags


def write_poscar_with_flags(
    source: Path,
    output: Path,
    *,
    freeze_mode: str,
    freeze_species: set[str] | None = None,
    freeze_indices: set[int] | None = None,
    anchor_flags: tuple[str, str, str] = ("F", "F", "F"),
) -> None:
    doc = parse_poscar_document(source)
    labels = labels_for_species(doc.species)
    flags = freeze_flags_for_labels(
        labels,
        freeze_mode=freeze_mode,
        freeze_species=freeze_species or set(),
        freeze_indices=freeze_indices or set(),
        anchor_flags=anchor_flags,
    )
    lines = [doc.comment, doc.scale, *doc.cell_lines]
    lines.append("  " + "  ".join(doc.species.symbols))
    lines.append("  " + "  ".join(str(value) for value in doc.species.counts))
    if freeze_mode != "none":
        lines.append("Selective dynamics")
    lines.append(doc.mode)
    for coords, flag in zip(doc.coordinates, flags):
        if freeze_mode == "none":
            lines.append("  " + "  ".join(coords))
        else:
            lines.append("  " + "  ".join(coords) + "   " + " ".join(flag))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_poscar_preserving_target_flags(source: Path, target: Path) -> None:
    """Copy source coordinates while preserving target selective-dynamics flags."""
    if not target.is_file():
        shutil.copy2(source, target)
        return
    source_doc = parse_poscar_document(source)
    target_doc = parse_poscar_document(target)
    if not target_doc.selective:
        shutil.copy2(source, target)
        return
    if source_doc.species.symbols != target_doc.species.symbols or source_doc.species.counts != target_doc.species.counts:
        raise ValueError(
            "Cannot preserve selective-dynamics flags during stage advance because "
            f"species/counts differ between {source} and {target}."
        )
    lines = [source_doc.comment, source_doc.scale, *source_doc.cell_lines]
    lines.append("  " + "  ".join(source_doc.species.symbols))
    lines.append("  " + "  ".join(str(value) for value in source_doc.species.counts))
    lines.append("Selective dynamics")
    lines.append(source_doc.mode)
    for coords, tails in zip(source_doc.coordinates, target_doc.tails):
        flags = tails[:3] if len(tails) >= 3 else ["T", "T", "T"]
        lines.append("  " + "  ".join(coords) + "   " + " ".join(flags))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_species_csv(text: str | None) -> set[str]:
    if not text:
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def split_indices_csv(text: str | None) -> set[int]:
    if not text:
        return set()
    indices: set[int] = set()
    for part in text.replace(",", " ").split():
        value = int(part)
        if value <= 0:
            raise ValueError("VASP atom indices are 1-based positive integers.")
        indices.add(value)
    return indices


def parse_anchor_flags(text: str) -> tuple[str, str, str]:
    parts = text.replace(",", " ").split()
    if len(parts) != 3 or any(part.upper() not in {"T", "F"} for part in parts):
        raise ValueError("--anchor-flags must be three T/F values, for example 'T T F'.")
    return parts[0].upper(), parts[1].upper(), parts[2].upper()


def incar_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "!")) or "=" not in stripped:
        return None
    return stripped.split("=", 1)[0].strip().upper()


def set_incar_overrides(text: str, overrides: dict[str, str | None]) -> str:
    remaining = {key.upper(): value for key, value in overrides.items()}
    output: list[str] = []
    for line in text.splitlines():
        key = incar_key(line)
        if key in remaining:
            value = remaining.pop(key)
            if value is not None:
                output.append(f"{key} = {value}")
        else:
            output.append(line)
    appendable = {key: value for key, value in remaining.items() if value is not None}
    if appendable:
        output.append("")
        output.append("# Atomi metastable-relax stage overrides")
        for key, value in appendable.items():
            output.append(f"{key} = {value}")
    return "\n".join(output).rstrip() + "\n"


def copy_input(src: Path, dst: Path, *, overwrite: bool, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def find_submit_scripts(root: Path) -> list[Path]:
    scripts: list[Path] = []
    for pattern in SUBMIT_CANDIDATES:
        scripts.extend(sorted(root.glob(pattern)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for script in scripts:
        resolved = script.resolve()
        if resolved not in seen and script.is_file():
            seen.add(resolved)
            unique.append(script)
    return unique


def validate_root(root: Path, *, allow_cell_relax: bool) -> list[str]:
    warnings: list[str] = []
    poscar = root / "POSCAR"
    incar = root / "INCAR"
    potcar = root / "POTCAR"
    if not poscar.is_file():
        raise FileNotFoundError(f"Missing POSCAR: {poscar}")
    if not incar.is_file():
        raise FileNotFoundError(f"Missing INCAR: {incar}")
    structure = read_poscar_structure(poscar)
    if any(symbol == "Va" for symbol in structure.species.symbols):
        raise ValueError("POSCAR contains Va pseudo-species; VASP production POSCARs cannot contain Va.")
    mind = min_distance_A(structure)
    if mind is not None and mind < 1.0:
        warnings.append(f"minimum interatomic distance is very small: {mind:.3f} A")
    incar_text = incar.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^\s*ISIF\s*=\s*3\b", incar_text, flags=re.IGNORECASE | re.MULTILINE) and not allow_cell_relax:
        warnings.append("root INCAR contains ISIF=3; staged outputs will force ISIF=2")
    warnings.extend(validate_potcar_order(poscar, potcar))
    warnings.extend(validate_ldau_species_order(poscar, incar))
    return warnings


def validate_potcar_order(poscar: Path, potcar: Path) -> list[str]:
    if not potcar.is_file():
        return [f"missing POTCAR: {potcar}"]
    species = read_poscar_structure(poscar).species.symbols
    titles: list[str] = []
    for line in potcar.read_text(encoding="utf-8", errors="replace").splitlines():
        if "TITEL" not in line:
            continue
        body = line.split("=", 1)[-1].strip()
        parts = body.split()
        if len(parts) >= 2:
            titles.append(parts[1].split("_", 1)[0])
    if titles[: len(species)] != species:
        return [f"POTCAR title order {titles[:len(species)]} does not match POSCAR species {species}"]
    return []


def incar_tag_values(incar: Path, tag: str) -> list[str] | None:
    pattern = re.compile(rf"^\s*{re.escape(tag)}\s*=", re.IGNORECASE)
    for line in incar.read_text(encoding="utf-8", errors="replace").splitlines():
        if not pattern.match(line):
            continue
        body = strip_incar_comment(line).split("=", 1)[-1]
        return expand_incar_value_tokens(body.split())
    return None


def validate_ldau_species_order(poscar: Path, incar: Path) -> list[str]:
    species = read_poscar_structure(poscar).species.symbols
    ldaul_values = incar_tag_values(incar, "LDAUL")
    if ldaul_values is None:
        return []
    if len(ldaul_values) != len(species):
        return [f"LDAUL has {len(ldaul_values)} values but POSCAR has {len(species)} species"]
    warnings: list[str] = []
    for symbol, value in zip(species, ldaul_values):
        if symbol in CATION_DEFAULT_EXCLUDE and value != "-1":
            warnings.append(f"LDAUL species-order check: {symbol} usually expects -1, found {value}")
        elif symbol in F_ELECTRON_LDAU_SPECIES and value != "3":
            warnings.append(f"LDAUL species-order check: {symbol} usually expects 3, found {value}")
    return warnings


def prepare_metastable_relax(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    outdir = args.output.expanduser().resolve() if args.output else root
    warnings = validate_root(root, allow_cell_relax=args.allow_cell_relax)
    if outdir.exists() and any((outdir / stage).exists() for stage in STAGE_ORDER) and not args.overwrite:
        raise FileExistsError(f"Staged folders already exist under {outdir}; pass --overwrite to update.")
    outdir.mkdir(parents=True, exist_ok=True)
    incar_text = (root / "INCAR").read_text(encoding="utf-8", errors="replace")
    freeze_species = split_species_csv(args.freeze_species)
    freeze_indices = split_indices_csv(args.freeze_indices)
    anchor_flags = parse_anchor_flags(args.anchor_flags)
    copied_submit_scripts = find_submit_scripts(root) if args.copy_submit else []

    for stage_name in STAGE_ORDER:
        spec = STAGES[stage_name]
        stage_dir = outdir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "INCAR").write_text(set_incar_overrides(incar_text, spec.overrides), encoding="utf-8")
        poscar_freeze_mode = freeze_mode_for_stage(stage_name, args) if spec.relaxation else "none"
        write_poscar_with_flags(
            root / "POSCAR",
            stage_dir / "POSCAR",
            freeze_mode=poscar_freeze_mode,
            freeze_species=freeze_species,
            freeze_indices=freeze_indices,
            anchor_flags=anchor_flags,
        )
        for name in COPY_INPUT_NAMES:
            src = root / name
            if src.is_file():
                copy_input(src, stage_dir / name, overwrite=args.overwrite, symlink=args.symlink_inputs)
        for script in copied_submit_scripts:
            copy_input(script, stage_dir / script.name, overwrite=args.overwrite, symlink=False)
        write_stage_readme(stage_dir, spec, freeze_mode=poscar_freeze_mode)

    write_workflow_readme(outdir, root, warnings, args.relax_sequence)
    print(f"Prepared metastable VASP stages under: {outdir}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def write_stage_readme(stage_dir: Path, spec: StageSpec, *, freeze_mode: str) -> None:
    previous = {
        "01_gentle_relax": "../00_static_scf/CHGCAR; CONTCAR copied to POSCAR if present",
        "02_continue_relax": "../01_gentle_relax/CHGCAR and CONTCAR copied to POSCAR",
        "03_final_static": "../02_continue_relax/CHGCAR and CONTCAR copied to POSCAR",
    }.get(spec.name, "root POSCAR and fresh charge")
    text = "\n".join(
        [
            f"# {spec.name}",
            "",
            spec.description,
            "",
            f"Suggested continuation source: {previous}.",
            f"Selective dynamics mode in POSCAR: {freeze_mode}.",
            "",
            "This workflow uses ISTART=0 and does not require WAVECAR.",
            "Use `atomi vasp-metastable-advance` after a stage finishes to copy CHGCAR and POSCAR.",
            "",
            "Before submitting, confirm POSCAR is the intended structure for this stage.",
        ]
    )
    (stage_dir / "README.md").write_text(text + "\n", encoding="utf-8")


def freeze_mode_for_stage(stage_name: str, args: argparse.Namespace) -> str:
    if args.relax_sequence == "anion_then_cation":
        if stage_name == "01_gentle_relax":
            return "all_oxygen"
        if stage_name == "02_continue_relax":
            return "all_cations"
    if args.relax_sequence == "cation_then_anion":
        if stage_name == "01_gentle_relax":
            return "all_cations"
        if stage_name == "02_continue_relax":
            return "all_oxygen"
    return args.freeze_mode


def write_workflow_readme(outdir: Path, root: Path, warnings: list[str], relax_sequence: str) -> None:
    lines = [
        "# Atomi Metastable VASP Relaxation Workflow",
        "",
        f"Source root: `{root}`",
        f"Relaxation freeze sequence: `{relax_sequence}`",
        "",
        "Run stages in order and gate each continuation with `vasp-structure-fingerprint`.",
        "The workflow intentionally uses fixed-cell `ISIF=2` by default.",
        "It uses `ISTART=0`, `ICHARG=1` after the first stage, `LCHARG=.TRUE.`, and `LWAVE=.FALSE.`.",
        "Advance stages with `vasp-metastable-advance` so `CHGCAR` is copied forward without relying on `WAVECAR`.",
        "",
        "Default conservative sequence:",
        "",
        "- `01_gentle_relax`: freeze oxygen and gently relax cations.",
        "- `02_continue_relax`: freeze cations and gently relax oxygen.",
        "",
        "After each completed stage, inspect energy, spin, and structure before continuing:",
        "",
        "```bash",
        "atomi vasp-metastable-status . --reference ../atomi_metastable_relax_input/POSCAR",
        "atomi vasp-spin-report --outcar 00_static_scf/OUTCAR --species 00_static_scf/CONTCAR --incar 00_static_scf/INCAR --format both --no-plot",
        "atomi vasp-structure-fingerprint 00_static_scf/CONTCAR --reference ../atomi_metastable_relax_input/POSCAR",
        "```",
    ]
    if warnings:
        lines.extend(["", "## Preparation Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (outdir / "README.metastable_relax.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def min_distance_A(structure: PoscarStructure) -> float | None:
    labels = labels_for_species(structure.species)
    if len(labels) < 2:
        return None
    best = float("inf")
    for i, fi in enumerate(structure.scaled_positions):
        for fj in structure.scaled_positions[i + 1 :]:
            df = [fj[k] - fi[k] for k in range(3)]
            df = [value - round(value) for value in df]
            cart = mat_vec(df, structure.cell)
            best = min(best, norm(cart))
    return best if math.isfinite(best) else None


def mat_vec(frac: list[float], cell: list[list[float]]) -> list[float]:
    return [
        frac[0] * cell[0][j] + frac[1] * cell[1][j] + frac[2] * cell[2][j]
        for j in range(3)
    ]


def cell_metrics(structure: PoscarStructure) -> dict[str, float]:
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


def nearest_pair_distances(structure: PoscarStructure) -> dict[tuple[str, str], float]:
    labels = labels_for_species(structure.species)
    distances: dict[tuple[str, str], float] = {}
    for i, li in enumerate(labels):
        for j in range(i + 1, len(labels)):
            lj = labels[j]
            key = tuple(sorted((li, lj)))
            fi = structure.scaled_positions[i]
            fj = structure.scaled_positions[j]
            df = [fj[k] - fi[k] for k in range(3)]
            df = [value - round(value) for value in df]
            dist = norm(mat_vec(df, structure.cell))
            distances[key] = min(distances.get(key, float("inf")), dist)
    return distances


def displacement_stats(structure: PoscarStructure, reference: PoscarStructure) -> dict[str, float] | None:
    if structure.species != reference.species:
        return None
    rows = []
    for fa, fb in zip(reference.scaled_positions, structure.scaled_positions):
        df = [fb[k] - fa[k] for k in range(3)]
        df = [value - round(value) for value in df]
        rows.append(norm(mat_vec(df, structure.cell)))
    if not rows:
        return None
    return {
        "mean_disp_A": sum(rows) / len(rows),
        "rms_disp_A": math.sqrt(sum(value * value for value in rows) / len(rows)),
        "max_disp_A": max(rows),
    }


def print_fingerprint(path: Path, *, reference: Path | None = None, collapse_threshold: float = 0.35) -> dict[str, object]:
    structure = read_poscar_structure(path)
    metrics = cell_metrics(structure)
    labels = labels_for_species(structure.species)
    print("VASP Structure Fingerprint")
    print("=" * 80)
    print(f"structure       : {path}")
    print(f"species         : {' '.join(structure.species.symbols)}")
    print(f"counts          : {' '.join(str(value) for value in structure.species.counts)}")
    print(f"natoms          : {len(labels)}")
    print(f"volume_A3       : {metrics['volume_A3']:.6f}")
    print(
        "cell_A          : "
        f"{metrics['a_A']:.6f} {metrics['b_A']:.6f} {metrics['c_A']:.6f}"
    )
    print(
        "angles_deg      : "
        f"{metrics['alpha_deg']:.4f} {metrics['beta_deg']:.4f} {metrics['gamma_deg']:.4f}"
    )
    mind = min_distance_A(structure)
    print(f"min_distance_A  : {mind:.6f}" if mind is not None else "min_distance_A  : NA")
    print()
    print("Nearest Pair Distances")
    print("-" * 80)
    for (a, b), dist in sorted(nearest_pair_distances(structure).items()):
        print(f"{a}-{b:<8} {dist:.6f} A")
    stats = None
    if reference is not None:
        ref = read_poscar_structure(reference)
        stats = displacement_stats(structure, ref)
        print()
        print("Reference Displacement")
        print("-" * 80)
        print(f"reference       : {reference}")
        if stats is None:
            print("warning         : species/order mismatch; no displacement comparison")
        else:
            print(f"mean_disp_A     : {stats['mean_disp_A']:.6f}")
            print(f"rms_disp_A      : {stats['rms_disp_A']:.6f}")
            print(f"max_disp_A      : {stats['max_disp_A']:.6f}")
            if stats["max_disp_A"] > collapse_threshold:
                print(f"collapse_warning: max displacement exceeds {collapse_threshold:.3f} A")
    return {"metrics": metrics, "min_distance_A": mind, "displacement": stats}


def choose_structure_for_stage(stage_dir: Path) -> Path | None:
    contcar = stage_dir / "CONTCAR"
    if contcar.is_file() and contcar.stat().st_size > 0:
        return contcar
    poscar = stage_dir / "POSCAR"
    if poscar.is_file():
        return poscar
    return None


def choose_output_for_stage(stage_dir: Path) -> Path | None:
    for name in ("OUTCAR", "OUTCAR.gz"):
        path = stage_dir / name
        if path.is_file():
            return path
    candidates = sorted(stage_dir.glob("vasp.out*"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def stage_status(stage_dir: Path, *, reference: Path | None, collapse_threshold: float) -> dict[str, object]:
    output = choose_output_for_stage(stage_dir)
    structure = choose_structure_for_stage(stage_dir)
    done = False
    n_elm_warning = False
    energy = None
    energy_kind = ""
    if output is not None:
        text = "\n".join(read_outcar_lines(output)[-500:])
        done = DONE_MARKER in text or "reached required accuracy" in text or "aborting loop because EDIFF is reached" in text
        n_elm_warning = "self-consistency" in text.lower() and "not achieved" in text.lower()
        energy, energy_kind = _latest_vasp_energy(output, preferred_kind="toten")
    disp = None
    if structure is not None and reference is not None:
        try:
            disp = displacement_stats(read_poscar_structure(structure), read_poscar_structure(reference))
        except Exception:
            disp = None
    status = "DONE" if done else "RUNNING/INCOMPLETE" if output is not None else "NOTSTART"
    collapsed = bool(disp and disp["max_disp_A"] > collapse_threshold)
    return {
        "stage": stage_dir.name,
        "status": status,
        "output": output,
        "structure": structure,
        "energy": energy,
        "energy_kind": energy_kind,
        "nelm_warning": n_elm_warning,
        "collapsed": collapsed,
        "displacement": disp,
    }


def print_metastable_status(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    reference = args.reference.expanduser().resolve() if args.reference else None
    print("VASP Metastable Relaxation Status")
    print("=" * 100)
    print(f"root      : {root}")
    print(f"reference : {reference or 'none'}")
    print()
    print(
        f"{'stage':<22} {'status':<18} {'energy_eV':>15} {'kind':>8} "
        f"{'max_disp_A':>12} {'warnings':<20}"
    )
    for stage in STAGE_ORDER:
        stage_dir = root / stage
        if not stage_dir.is_dir():
            print(f"{stage:<22} {'MISSING':<18} {'NA':>15} {'NA':>8} {'NA':>12} missing stage")
            continue
        row = stage_status(stage_dir, reference=reference, collapse_threshold=args.collapse_threshold)
        disp = row["displacement"] or {}
        warnings = []
        if row["nelm_warning"]:
            warnings.append("NELM")
        if row["collapsed"]:
            warnings.append("collapse?")
        if row["structure"] is None:
            warnings.append("no-structure")
        print(
            f"{stage:<22} {str(row['status']):<18} "
            f"{format_float(row['energy'], 15, 8)} {str(row['energy_kind'] or 'NA'):>8} "
            f"{format_float(disp.get('max_disp_A'), 12, 5)} {','.join(warnings) or '-':<20}"
        )


def next_stage_name(stage: str) -> str:
    try:
        return STAGE_ORDER[STAGE_ORDER.index(stage) + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"No next metastable stage after {stage!r}.") from exc


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def advance_stage(args: argparse.Namespace) -> None:
    root = args.root.expanduser().resolve()
    from_stage = args.from_stage
    to_stage = args.to_stage or next_stage_name(from_stage)
    source = root / from_stage
    target = root / to_stage
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source stage: {source}")
    if not target.is_dir():
        raise FileNotFoundError(f"Missing target stage: {target}")

    actions: list[tuple[Path, Path, str, bool]] = []
    chgcar = source / "CHGCAR"
    if nonempty(chgcar):
        actions.append((chgcar, target / "CHGCAR", "charge density", False))
    elif not args.allow_missing_chgcar:
        raise FileNotFoundError(f"Missing non-empty CHGCAR in completed stage: {chgcar}")

    contcar = source / "CONTCAR"
    if nonempty(contcar):
        actions.append((contcar, target / "POSCAR", "next starting structure", True))
    elif from_stage != "00_static_scf" and not args.allow_missing_contcar:
        raise FileNotFoundError(f"Missing non-empty CONTCAR in completed stage: {contcar}")

    for src, dst, label, preserve_flags in actions:
        print(f"{'DRY-RUN ' if args.dry_run else ''}copy {label}: {src} -> {dst}")
        if not args.dry_run:
            if preserve_flags:
                copy_poscar_preserving_target_flags(src, dst)
            else:
                shutil.copy2(src, dst)

    if not actions:
        print("No files copied.")

    print()
    print("Recommended checks before submitting the next stage:")
    outcar = source / "OUTCAR"
    species = contcar if nonempty(contcar) else source / "POSCAR"
    if outcar.is_file():
        print(
            "atomi vasp-spin-report "
            f"--outcar {outcar} --species {species} --incar {source / 'INCAR'} --format both --no-plot"
        )
    else:
        print(f"# OUTCAR not found yet for spin report: {outcar}")
    if nonempty(contcar):
        reference = args.reference.expanduser().resolve() if args.reference else root.parent / "atomi_metastable_relax_input" / "POSCAR"
        print(f"atomi vasp-structure-fingerprint {contcar} --reference {reference}")
    print(f"atomi vasp-metastable-status {root} --reference {args.reference or '../atomi_metastable_relax_input/POSCAR'}")


def format_float(value: object, width: int, precision: int) -> str:
    if value is None:
        return "NA".rjust(width)
    try:
        return f"{float(value):{width}.{precision}f}"
    except (TypeError, ValueError):
        return "NA".rjust(width)


def build_prepare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-metastable-relax",
        description="Prepare conservative fixed-cell VASP stages for metastable structures.",
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("."), help="VASP root folder with POSCAR/INCAR.")
    parser.add_argument("--output", type=Path, help="Output folder for staged workflow. Default: root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing staged inputs.")
    parser.add_argument("--allow-cell-relax", action="store_true", help="Allow root ISIF=3 without warning.")
    parser.add_argument("--symlink-inputs", action="store_true", help="Symlink POTCAR/KPOINTS instead of copying.")
    parser.add_argument("--copy-submit", action="store_true", help="Copy submit/sbatch scripts into each stage.")
    parser.add_argument(
        "--relax-sequence",
        choices=("anion_then_cation", "cation_then_anion", "same"),
        default="anion_then_cation",
        help=(
            "Freeze sequence for relaxation stages. Default freezes anions in stage 1 "
            "then cations in stage 2."
        ),
    )
    parser.add_argument(
        "--freeze-mode",
        choices=("none", "all_cations", "all_oxygen", "by_species", "by_index", "small_displacement_anchor"),
        default="all_cations",
        help="Selective dynamics mode when --relax-sequence same is used.",
    )
    parser.add_argument("--freeze-species", help="Comma-separated species for --freeze-mode by_species.")
    parser.add_argument("--freeze-indices", help="1-based atom indices for by_index or small_displacement_anchor.")
    parser.add_argument("--anchor-flags", default="F F F", help="T/F flags for small_displacement_anchor.")
    return parser


def build_selective_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-selective-dynamics",
        description="Apply VASP Selective dynamics flags to a POSCAR/CONTCAR.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("POSCAR.selective"))
    parser.add_argument(
        "--freeze-mode",
        choices=("none", "all_cations", "all_oxygen", "by_species", "by_index", "small_displacement_anchor"),
        required=True,
    )
    parser.add_argument("--freeze-species")
    parser.add_argument("--freeze-indices")
    parser.add_argument("--anchor-flags", default="F F F")
    return parser


def build_fingerprint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-structure-fingerprint",
        description="Fingerprint a VASP structure and optionally compare to a reference.",
    )
    parser.add_argument("structure", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--collapse-threshold", type=float, default=0.35)
    return parser


def build_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-metastable-status",
        description="Summarize staged metastable VASP relaxation folders.",
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--collapse-threshold", type=float, default=0.35)
    return parser


def build_advance_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-metastable-advance",
        description="Copy CHGCAR and CONTCAR/POSCAR handoff files between metastable VASP stages.",
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("."), help="Metastable staged workflow root.")
    parser.add_argument(
        "--from-stage",
        choices=STAGE_ORDER[:-1],
        required=True,
        help="Completed stage to advance from.",
    )
    parser.add_argument("--to-stage", choices=STAGE_ORDER[1:], help="Target stage. Defaults to the next stage.")
    parser.add_argument("--reference", type=Path, help="Reference POSCAR for printed fingerprint command.")
    parser.add_argument("--dry-run", action="store_true", help="Print copies without changing files.")
    parser.add_argument("--allow-missing-chgcar", action="store_true", help="Do not fail if CHGCAR is missing.")
    parser.add_argument("--allow-missing-contcar", action="store_true", help="Do not fail if CONTCAR is missing.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_prepare_parser().parse_args(argv)
    prepare_metastable_relax(args)


def selective_main(argv: list[str] | None = None) -> None:
    args = build_selective_parser().parse_args(argv)
    write_poscar_with_flags(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        freeze_mode=args.freeze_mode,
        freeze_species=split_species_csv(args.freeze_species),
        freeze_indices=split_indices_csv(args.freeze_indices),
        anchor_flags=parse_anchor_flags(args.anchor_flags),
    )
    print(f"Wrote selective-dynamics POSCAR: {args.output.expanduser().resolve()}")


def fingerprint_main(argv: list[str] | None = None) -> None:
    args = build_fingerprint_parser().parse_args(argv)
    print_fingerprint(
        args.structure.expanduser().resolve(),
        reference=args.reference.expanduser().resolve() if args.reference else None,
        collapse_threshold=args.collapse_threshold,
    )


def status_main(argv: list[str] | None = None) -> None:
    args = build_status_parser().parse_args(argv)
    print_metastable_status(args)


def advance_main(argv: list[str] | None = None) -> None:
    args = build_advance_parser().parse_args(argv)
    advance_stage(args)


if __name__ == "__main__":
    main()
