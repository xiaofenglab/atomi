"""First-pass CP2K constrained-window free-energy helpers."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from atomi.cp2k.aimd_common import (
    distance,
    iter_xyz_frames,
    mean,
    parse_cp2k_input,
    parse_lagrange_file,
    tail_values,
)

R_KCAL_MOL_K = 0.00198720425864083


def _latest(paths: list[Path]) -> Path | None:
    paths = [path for path in paths if path.exists()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _find_input(window: Path) -> Path | None:
    return _latest([p for p in window.glob("*.inp") if not p.name.endswith("~") and ".used_" not in p.name])


def _find_traj(window: Path) -> Path | None:
    return _latest(list(window.glob("*-pos.xyz")))


def _find_lagrange(window: Path) -> Path | None:
    return _latest(list(window.glob("*.lagrange")))


def _cv_distance_series(
    trajectory: Path,
    atoms: list[int],
    cell_abc_A: list[float] | None,
) -> list[float]:
    values: list[float] = []
    if len(atoms) < 2:
        return values
    a_idx, b_idx = atoms[0], atoms[1]
    for _comment, _symbols, coords in iter_xyz_frames(trajectory):
        values.append(distance(coords[a_idx - 1], coords[b_idx - 1], cell_abc_A))
    return values


def summarize_window(window: Path, *, colvar: int, tail_fraction: float) -> dict[str, object]:
    input_path = _find_input(window)
    if input_path is None:
        raise FileNotFoundError(f"No CP2K input found in {window}")
    input_info = parse_cp2k_input(input_path)
    trajectory = _find_traj(window)
    if trajectory is None:
        coord = input_info.get("trajectory_file") or input_info.get("coordinate_file")
        if isinstance(coord, str) and (window / coord).exists():
            trajectory = window / coord
    if trajectory is None:
        raise FileNotFoundError(f"No CP2K trajectory found in {window}")
    restraints = [item for item in input_info.get("restraints", []) if isinstance(item, dict)]
    restraint = next((item for item in restraints if item.get("colvar") == colvar), None)
    if restraint is None:
        raise ValueError(f"{window}: no restraint found for COLVAR {colvar}")
    colvars = [item for item in input_info.get("colvars", []) if isinstance(item, dict)]
    cv = next((item for item in colvars if item.get("index") == colvar), None)
    if cv is None:
        raise ValueError(f"{window}: no COLVAR {colvar} definition found")
    atoms = cv.get("atoms")
    if not isinstance(atoms, list):
        raise ValueError(f"{window}: COLVAR {colvar} has no atom pair")
    cell = input_info.get("cell_abc_A")
    series = _cv_distance_series(trajectory, [int(item) for item in atoms], cell if isinstance(cell, list) else None)
    tail = tail_values(series, tail_fraction)
    target = restraint.get("target_A")
    k = restraint.get("k_kcalmol")
    if not isinstance(target, float) or not isinstance(k, float):
        raise ValueError(f"{window}: missing target/K for COLVAR {colvar}")
    mean_tail = mean(tail)
    mean_force = k * ((mean_tail or target) - target)
    lagrange = _find_lagrange(window)
    return {
        "window": str(window),
        "input_file": str(input_path),
        "trajectory": str(trajectory),
        "lagrange_file": str(lagrange) if lagrange else None,
        "colvar": colvar,
        "atoms": atoms,
        "target_A": target,
        "k_kcalmol_A2": k,
        "sample_count": len(series),
        "tail_count": len(tail),
        "mean_tail_A": mean_tail,
        "min_tail_A": min(tail) if tail else None,
        "max_tail_A": max(tail) if tail else None,
        "mean_force_kcal_mol_A": mean_force,
        "lagrange": parse_lagrange_file(lagrange) if lagrange else {},
    }


def integrate_windows(windows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(windows, key=lambda row: float(row["target_A"]))
    profile: list[dict[str, object]] = []
    cumulative = 0.0
    previous = None
    for row in ordered:
        if previous is not None:
            dx = float(row["target_A"]) - float(previous["target_A"])
            f0 = float(previous["mean_force_kcal_mol_A"])
            f1 = float(row["mean_force_kcal_mol_A"])
            cumulative += 0.5 * (f0 + f1) * dx
        item = dict(row)
        item["pmf_kcal_mol"] = cumulative
        profile.append(item)
        previous = row
    if profile:
        reference = min(float(row["pmf_kcal_mol"]) for row in profile)
        for row in profile:
            row["pmf_relative_kcal_mol"] = float(row["pmf_kcal_mol"]) - reference
    return profile


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "window",
        "colvar",
        "atoms",
        "target_A",
        "k_kcalmol_A2",
        "sample_count",
        "tail_count",
        "mean_tail_A",
        "min_tail_A",
        "max_tail_A",
        "mean_force_kcal_mol_A",
        "pmf_kcal_mol",
        "pmf_relative_kcal_mol",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cp2k-blue-moon-ti",
        description=(
            "Build a first-pass constrained-window mean-force/PMF table from CP2K "
            "COLVAR restraints and trajectories."
        ),
    )
    parser.add_argument("window_dirs", type=Path, nargs="*")
    parser.add_argument("--window-glob", type=str, help="Glob for window directories if positional dirs are omitted.")
    parser.add_argument("--colvar", type=int, default=1)
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--standard-state-correction-kcal", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=Path("cp2k_blue_moon_pmf.csv"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    window_dirs = args.window_dirs
    if not window_dirs and args.window_glob:
        window_dirs = [Path(item) for item in sorted(Path().glob(args.window_glob))]
    if not window_dirs:
        raise SystemExit("Provide window directories or --window-glob.")

    windows = [
        summarize_window(window, colvar=args.colvar, tail_fraction=args.tail_fraction)
        for window in window_dirs
    ]
    profile = integrate_windows(windows)
    if profile:
        delta_g = float(profile[-1]["pmf_kcal_mol"]) - float(profile[0]["pmf_kcal_mol"])
        delta_g_standard = delta_g + args.standard_state_correction_kcal
        log10_k = -delta_g_standard / (R_KCAL_MOL_K * args.temperature_K * math.log(10.0))
    else:
        delta_g_standard = None
        log10_k = None

    write_csv(args.out, profile)
    print("CP2K constrained-window free-energy summary")
    print("-------------------------------------------")
    print(f"windows      : {len(profile)}")
    print(f"colvar       : {args.colvar}")
    print(f"out          : {args.out}")
    if delta_g_standard is not None:
        print(f"DeltaG std   : {delta_g_standard:.6g} kcal/mol")
        print(f"log10 K      : {log10_k:.6g}")
    if args.json_out:
        payload = {
            "windows": profile,
            "delta_g_standard_kcal_mol": delta_g_standard,
            "log10_K": log10_k,
            "temperature_K": args.temperature_K,
            "standard_state_correction_kcal": args.standard_state_correction_kcal,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
