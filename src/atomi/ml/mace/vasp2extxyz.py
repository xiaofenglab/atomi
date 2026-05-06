#!/usr/bin/env python3
"""
vasp2extxyz.py

Collect energies, forces, and optionally stresses from many VASP runs into one
extxyz dataset suitable for ASE/MACE workflows.

Main design choices:
- Parse from vasprun.xml first, then OUTCAR / OUTCAR.gz
- Skip known failed or unconverged runs unless --allow-unconverged is used
- Store energy/forces/stress in a SinglePointCalculator
- Remove duplicate calculator-related keys from atoms.info / atoms.arrays
- Keep provenance / metadata in atoms.info
- Write a clean extxyz that ASE and MACE can read

Typical use:
  python3 ../vasp2extxyz.py \
    --runlist runlist.txt \
    --out prefail_v4_r2.extxyz \
    --index index_prefail_v4_r2.csv \
    --failed failed_prefail_v4_r2.txt

Optional:
  python3 vasp2extxyz.py \
    --runlist runlist_VD.txt \
    --out train.extxyz \
    --require-stress
"""

import argparse
import csv
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import read as ase_read, write as ase_write
except ImportError:
    np = None
    Atoms = Any
    SinglePointCalculator = None
    ase_read = None
    ase_write = None


# ------------------------- helpers -------------------------

def infer_labels(run_dir: Path) -> Dict[str, Any]:
    """
    Extract useful labels from a path like:
      .../V1.00/rd_001
      .../V1.00/strain_003_uni_xx_p
    """
    out: Dict[str, Any] = {}
    for p in run_dir.parts:
        if p.startswith("V"):
            try:
                out["volume_tag"] = p
                out["volume_frac"] = float(p[1:])
            except Exception:
                pass
        elif p.startswith("rd_"):
            out["config_type"] = "rd"
            out["config_tag"] = p
            try:
                out["config_id"] = int(p.split("_")[1])
            except Exception:
                pass
        elif p.startswith("strain_"):
            out["config_type"] = "strain"
            out["config_tag"] = p
    return out


def is_bad_run(run_dir: Path) -> Tuple[bool, str]:
    """
    Decide if a run should be skipped before parsing.

    Returns:
      (True, reason) if bad
      (False, "") if okay
    """
    for marker in ("SCF_FAILED", "VASP_FAILED", "TIMED_OUT"):
        if (run_dir / marker).exists():
            return True, marker

    outcar = run_dir / "OUTCAR"
    outcar_gz = run_dir / "OUTCAR.gz"

    try:
        if outcar.is_file():
            txt = outcar.read_text(errors="ignore")
            if "self-consistency was not achieved" in txt:
                return True, "OUTCAR_SCF_NOT_CONVERGED"

        if outcar_gz.is_file():
            with gzip.open(outcar_gz, "rt", errors="ignore") as f:
                txt = f.read()
            if "self-consistency was not achieved" in txt:
                return True, "OUTCAR_GZ_SCF_NOT_CONVERGED"
    except Exception:
        pass

    return False, ""


def safe_read_vasprun(path: Path) -> Optional[Atoms]:
    """Read last ionic step from vasprun.xml."""
    try:
        return ase_read(str(path), index=-1)
    except Exception:
        return None


def safe_read_outcar(path: Path) -> Optional[Atoms]:
    """Read last ionic step from OUTCAR or OUTCAR.gz."""
    try:
        return ase_read(str(path), index=-1)
    except Exception:
        if path.suffix == ".gz":
            try:
                tmp = path.with_suffix("")  # OUTCAR.gz -> OUTCAR
                created_tmp = False
                if not tmp.exists():
                    with gzip.open(path, "rt", errors="ignore") as fin, open(tmp, "wt") as fout:
                        fout.write(fin.read())
                    created_tmp = True

                atoms = ase_read(str(tmp), index=-1)

                if created_tmp:
                    try:
                        tmp.unlink()
                    except Exception:
                        pass

                return atoms
            except Exception:
                return None
        return None


def extract_properties(
    atoms: Atoms,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract:
      energy (eV)
      forces (eV/Ang)
      stress (6-vector in ASE Voigt order, eV/Ang^3)
    """
    energy = None
    forces = None
    stress = None

    for key in ("energy", "free_energy"):
        if key in atoms.info:
            try:
                energy = float(atoms.info[key])
                break
            except Exception:
                pass

    if energy is None:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            energy = None

    if "forces" in atoms.arrays:
        try:
            forces = np.array(atoms.arrays["forces"], dtype=float)
        except Exception:
            forces = None

    if forces is None:
        try:
            forces = np.array(atoms.get_forces(), dtype=float)
        except Exception:
            forces = None

    if "stress" in atoms.info:
        try:
            stress = np.array(atoms.info["stress"], dtype=float).reshape(-1)
        except Exception:
            stress = None

    if stress is None:
        try:
            stress = np.array(atoms.get_stress(voigt=True), dtype=float).reshape(-1)
        except Exception:
            stress = None

    if stress is not None and stress.size >= 6:
        stress = stress[:6]

    return energy, forces, stress


def strip_calc_conflicts(atoms: Atoms) -> None:
    """
    Remove keys that would conflict with calculator results when writing extxyz.
    """
    for key in ("energy", "free_energy", "stress", "virial"):
        atoms.info.pop(key, None)

    for key in ("forces", "stresses", "charges", "magmoms"):
        if key in atoms.arrays:
            del atoms.arrays[key]


def attach_structure_metadata(atoms: Atoms, run_dir: Path, source_file: str) -> None:
    """
    Attach provenance and inferred labels to Atoms.info.
    Keep only plain serializable metadata.
    """
    atoms.info["run_dir"] = str(run_dir)
    atoms.info["source_file"] = source_file
    atoms.info.update(infer_labels(run_dir))


def build_singlepoint_atoms(
    atoms: Atoms,
    energy: float,
    forces: np.ndarray,
    stress: Optional[np.ndarray],
) -> Atoms:
    """
    Return a clean Atoms object with results attached via SinglePointCalculator.
    """
    clean = atoms.copy()
    clean.calc = None
    strip_calc_conflicts(clean)

    calc_kwargs = {
        "energy": float(energy),
        "forces": np.array(forces, dtype=float),
    }
    if stress is not None and len(stress) >= 6:
        calc_kwargs["stress"] = np.array(stress[:6], dtype=float)

    clean.calc = SinglePointCalculator(clean, **calc_kwargs)
    return clean


def validate_atoms_for_mace(
    atoms: Atoms,
    require_stress: bool = False,
) -> Tuple[bool, str]:
    """
    Basic sanity check that the structure can be read back by ASE/MACE-style workflows.
    """
    try:
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
    except Exception as exc:
        return False, f"CALC_RESULTS_UNREADABLE: {exc}"

    if not np.isfinite(e):
        return False, "NONFINITE_ENERGY"

    if f.shape != (len(atoms), 3):
        return False, f"BAD_FORCE_SHAPE:{f.shape}"

    if not np.isfinite(f).all():
        return False, "NONFINITE_FORCES"

    if require_stress:
        try:
            s = atoms.get_stress(voigt=True)
        except Exception as exc:
            return False, f"MISSING_STRESS: {exc}"
        if len(s) < 6 or not np.isfinite(np.array(s)).all():
            return False, "BAD_STRESS"

    return True, ""


def get_run_dirs_from_runlist(runlist: Path) -> List[Path]:
    dirs: List[Path] = []
    with open(runlist, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            d = Path(s)
            if d.exists() and d.is_dir():
                dirs.append(d.resolve())
    return sorted(set(dirs))


def get_run_dirs_by_search(root: Path, patterns: List[str]) -> List[Path]:
    dirs: List[Path] = []
    for pat in patterns:
        for d in root.rglob(pat):
            if not d.is_dir():
                continue
            if (d / "vasprun.xml").is_file() or (d / "OUTCAR").is_file() or (d / "OUTCAR.gz").is_file():
                dirs.append(d.resolve())
    return sorted(set(dirs))


def collect_csv_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Build a stable union of all keys across rows.
    Keep common keys first, then append any extras.
    """
    preferred = [
        "i",
        "run_dir",
        "source_file",
        "natoms",
        "energy_eV",
        "energy_eV_per_atom",
        "force_rms_eVA",
        "force_max_eVA",
        "force_mean_eVA",
        "has_stress",
        "volume_tag",
        "volume_frac",
        "config_type",
        "config_tag",
        "config_id",
    ]

    seen = set()
    fieldnames: List[str] = []

    for k in preferred:
        for row in rows:
            if k in row and k not in seen:
                fieldnames.append(k)
                seen.add(k)
                break

    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    return fieldnames


def write_index_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = collect_csv_fieldnames(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ------------------------- main -------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="mace-vasp2extxyz",
        description="Collect VASP run directories into an extxyz dataset for ASE/MACE workflows.",
    )
    ap.add_argument("--root", type=str, default=".", help="Root directory for fallback search.")
    ap.add_argument(
        "--runlist",
        type=str,
        default=None,
        help="Optional runlist.txt listing run directories. Recommended.",
    )
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=["rd_*", "strain_*"],
        help="Fallback directory patterns to search if runlist is not provided.",
    )
    ap.add_argument("--out", type=str, default="train.extxyz", help="Output extxyz filename.")
    ap.add_argument("--index", type=str, default="index.csv", help="Output CSV index filename.")
    ap.add_argument("--failed", type=str, default="failed.txt", help="Output failed list filename.")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of structures (0 = no limit).")
    ap.add_argument("--require-stress", action="store_true", help="Skip entries missing stress.")
    ap.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="If set, do not skip SCF_FAILED / TIMED_OUT / unconverged OUTCAR runs.",
    )
    args = ap.parse_args(argv)
    if np is None or ase_read is None or ase_write is None or SinglePointCalculator is None:
        raise SystemExit(
            "Missing dependencies numpy and/or ase. Install/load them before running mace-vasp2extxyz."
        )

    root = Path(args.root).resolve()

    if args.runlist is not None:
        runlist = Path(args.runlist).resolve()
        if not runlist.exists():
            raise FileNotFoundError(f"runlist not found: {runlist}")
        run_dirs = get_run_dirs_from_runlist(runlist)
        print(f"Using runlist: {runlist}")
    else:
        if not root.exists():
            raise FileNotFoundError(f"root not found: {root}")
        run_dirs = get_run_dirs_by_search(root, args.patterns)
        print(f"Using fallback search under root: {root}")

    if args.limit and args.limit > 0:
        run_dirs = run_dirs[:args.limit]

    print(f"Found {len(run_dirs)} run directories")

    collected: List[Atoms] = []
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []

    for i, rdir in enumerate(run_dirs, start=1):
        if not args.allow_unconverged:
            bad, reason = is_bad_run(rdir)
            if bad:
                failed.append(f"{rdir}\t{reason}")
                continue

        vasprun = rdir / "vasprun.xml"
        outcar = rdir / "OUTCAR"
        outcar_gz = rdir / "OUTCAR.gz"

        atoms = None
        source = None

        if vasprun.is_file():
            atoms = safe_read_vasprun(vasprun)
            source = "vasprun.xml"

        if atoms is None:
            if outcar.is_file():
                atoms = safe_read_outcar(outcar)
                source = "OUTCAR"
            elif outcar_gz.is_file():
                atoms = safe_read_outcar(outcar_gz)
                source = "OUTCAR.gz"

        if atoms is None:
            failed.append(f"{rdir}\tPARSE_FAILED")
            continue

        energy, forces, stress = extract_properties(atoms)

        if energy is None or forces is None:
            failed.append(f"{rdir}\tMISSING_ENERGY_OR_FORCES")
            continue

        if args.require_stress and (stress is None or len(stress) < 6):
            failed.append(f"{rdir}\tMISSING_STRESS")
            continue

        try:
            clean_atoms = build_singlepoint_atoms(atoms, energy, forces, stress)
            attach_structure_metadata(clean_atoms, rdir, source)
            clean_atoms.info["has_stress"] = bool(stress is not None and len(stress) >= 6)

            ok, reason = validate_atoms_for_mace(clean_atoms, require_stress=args.require_stress)
            if not ok:
                failed.append(f"{rdir}\t{reason}")
                continue

            collected.append(clean_atoms)

            row: Dict[str, Any] = {
                "i": i,
                "run_dir": str(rdir),
                "source_file": source,
                "natoms": len(clean_atoms),
                "energy_eV": float(clean_atoms.get_potential_energy()),
                "energy_eV_per_atom": float(clean_atoms.get_potential_energy() / len(clean_atoms)),
                "has_stress": bool(clean_atoms.info.get("has_stress", False)),
            }

            try:
                ff = clean_atoms.get_forces()
                fmag = np.linalg.norm(ff, axis=1)
                row["force_rms_eVA"] = float(np.sqrt((ff ** 2).mean()))
                row["force_max_eVA"] = float(fmag.max())
                row["force_mean_eVA"] = float(fmag.mean())
            except Exception:
                pass

            for k in ("volume_tag", "volume_frac", "config_type", "config_tag", "config_id"):
                if k in clean_atoms.info:
                    row[k] = clean_atoms.info[k]

            rows.append(row)

        except Exception as exc:
            failed.append(f"{rdir}\tBUILD_FAILED:{exc}")
            continue

        if i % 50 == 0:
            print(f"  parsed {i}/{len(run_dirs)} ... kept={len(collected)} failed={len(failed)}")

    out_path = Path(args.out).resolve()
    idx_path = Path(args.index).resolve()
    failed_path = Path(args.failed).resolve()

    if collected:
        ase_write(str(out_path), collected, format="extxyz")
        print(f"Wrote {len(collected)} structures to {out_path}")
    else:
        print("No structures collected; extxyz not written.")

    write_index_csv(idx_path, rows)
    print(f"Wrote index CSV to {idx_path}")

    failed_path.write_text("\n".join(failed) + ("\n" if failed else ""))
    print(f"Wrote failed list ({len(failed)}) to {failed_path}")


if __name__ == "__main__":
    main()
