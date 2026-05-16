#!/usr/bin/env python3
"""Prepare and analyze finite-temperature elastic LAMMPS workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np

from atomi.lammps.thermo_series import (
    collect_config_paths,
    discover_npt_records_from_md_root,
    discover_production_records,
)


VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
DEFAULT_MODES = ("xx", "yy", "zz", "yz", "xz", "xy")
BAR_TO_GPA = 1.0e-4


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    def normalize(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(v) for v in value]
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(data), indent=2) + "\n", encoding="utf-8")


def relative_to_root(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def resolve_root_path(path: Path, root: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def temperature_label(value: float) -> str:
    value = float(value)
    return str(int(round(value))) if abs(value - round(value)) < 1.0e-9 else f"{value:g}".replace(".", "p")


def strain_label(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return sign + f"{abs(value) * 100:g}".replace(".", "p")


def voigt_strain(mode: str, strain: float) -> list[float]:
    values = [0.0] * 6
    if mode in VOIGT:
        values[VOIGT.index(mode)] = float(strain)
    return values


def parse_float_list(values: list[str] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    out: list[float] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(float(item))
    return out


def find_template_config(md_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit.resolve()
    for name in ("config_elastic_base.json", "config_production.json", "config.json"):
        candidate = md_root / name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find config_production.json/config.json under {md_root}; pass --template-config."
    )


def discover_npt_records(args: argparse.Namespace) -> tuple[list[dict], Path, dict]:
    if args.config:
        config_paths = collect_config_paths(args.config, args.config_dir, args.config_glob)
        records = discover_production_records(config_paths, duplicate_policy=args.duplicate_policy)
        template_path = config_paths[0].resolve()
    else:
        md_root = args.md_root.resolve()
        template_path = find_template_config(md_root, args.template_config)
        records = discover_npt_records_from_md_root(
            md_root,
            duplicate_policy=args.duplicate_policy,
            timestep_ps=args.timestep_ps,
        )
    cfg = load_json(template_path)
    root = template_path.parent.resolve()
    return records, root, cfg


def select_temperature_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    out = []
    for rec in records:
        t = float(rec["temperature"])
        if args.T_min is not None and t < args.T_min:
            continue
        if args.T_max is not None and t > args.T_max:
            continue
        out.append(rec)
    if args.include_all_temperatures:
        return out
    if not out:
        return []
    t_max = max(float(rec["temperature"]) for rec in out)
    targets = []
    t = float(args.temperature_start)
    while t <= t_max + args.temperature_tol:
        targets.append(t)
        t += float(args.temperature_step)
    selected = []
    used: set[int] = set()
    for target in targets:
        candidates = [
            (abs(float(rec["temperature"]) - target), index, rec)
            for index, rec in enumerate(out)
            if index not in used and abs(float(rec["temperature"]) - target) <= args.temperature_tol
        ]
        if not candidates:
            continue
        _, index, rec = min(candidates)
        selected.append(rec)
        used.add(index)
    return selected


def stage_dir_from_record(record: dict) -> Path:
    log_path = Path(record["log_path"]).resolve()
    parent = log_path.parent
    if parent.name.startswith("chunk"):
        return parent.parent
    return parent


def find_restart_or_data(record: dict) -> tuple[Path, Optional[Path]]:
    stage_name = record["stage_name"]
    stage_dir = stage_dir_from_record(record)
    restart = stage_dir / f"{stage_name}.restart"
    data = stage_dir / f"{stage_name}.data"
    if restart.exists():
        return restart.resolve(), data.resolve() if data.exists() else None
    if data.exists():
        return data.resolve(), None
    chunk_dir = Path(record["log_path"]).resolve().parent
    restart_candidates = sorted(chunk_dir.glob("*.restart"), key=lambda p: p.stat().st_mtime)
    data_candidates = sorted(chunk_dir.glob("*.data"), key=lambda p: p.stat().st_mtime)
    if restart_candidates:
        return restart_candidates[-1].resolve(), data_candidates[-1].resolve() if data_candidates else None
    if data_candidates:
        return data_candidates[-1].resolve(), None
    raise FileNotFoundError(f"No restart/data found for {stage_name} near {stage_dir}")


def copy_elastic_base_config(template: dict, root: Path, args: argparse.Namespace) -> dict:
    keys = [
        "wrapper_script",
        "model_file",
        "timestep",
        "mass_O",
        "mass_U",
        "velocity_seed",
        "poll_seconds",
        "thermostat",
        "barostat",
        "relax",
        "performance",
        "equilibrium_rules",
        "instability_rules",
    ]
    cfg = {key: template[key] for key in keys if key in template}
    cfg.setdefault("velocity_seed", 12345)
    cfg.setdefault("poll_seconds", 10)
    cfg.setdefault("thermostat", {"tdamp": 0.8})
    cfg.setdefault("equilibrium_rules", template.get("equilibrium_rules", {}))
    cfg["generated_by"] = "atomi elastic_lammps prepare"
    cfg["description"] = (
        "Generated finite-temperature elastic NVT strain workflow. Each stage starts from "
        "a completed NPT restart/data file and applies a small box deformation before NVT."
    )
    fixed_steps = int(round(float(args.run_time_ps) / float(cfg.get("timestep", args.timestep_ps or 0.0001))))
    cfg["adaptive_steps"] = {
        "initial_small": fixed_steps,
        "initial_large": fixed_steps,
        "growth_factor": 1.0,
        "max_chunk_steps": fixed_steps,
    }
    cfg["max_chunks_small"] = 1
    cfg["max_chunks_large"] = 1
    cfg["elastic_settings"] = {
        "method": "finite-temperature strain-stress NVT",
        "paper_basis": "U3Si2 RUS/MLACS paper strain-stress workflow: NPT equilibrium, strained NVT, stress fit, Voigt-Reuss-Hill moduli.",
        "run_time_ps": args.run_time_ps,
        "analysis_window_ps_default": args.analysis_window_ps,
        "strains": args.strains,
        "modes": list(args.modes),
        "temperature_selection": {
            "start_K": args.temperature_start,
            "step_K": args.temperature_step,
            "tolerance_K": args.temperature_tol,
            "include_all_temperatures": args.include_all_temperatures,
        },
        "symmetry": args.symmetry,
        "stress_sign_convention": "LAMMPS pressure tensor is converted to tensile stress by sigma_GPa = -p_bar * 1e-4.",
    }
    return cfg


def build_elastic_stages(records: list[dict], root: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    stages = []
    manifest = []
    for rec in records:
        t = float(rec["temperature"])
        temp_label = temperature_label(t)
        restart, data = find_restart_or_data(rec)
        cases = [("ref", 0.0)]
        for mode in args.modes:
            for strain in args.strains:
                cases.append((mode, float(strain)))
                cases.append((mode, -float(strain)))
        for mode, strain in cases:
            name = f"elastic_T{temp_label}K_{mode}" if mode == "ref" else f"elastic_T{temp_label}K_{mode}_{strain_label(strain)}"
            stage = {
                "name": name,
                "type": "nvt",
                "large_cell": bool(rec.get("stage", {}).get("large_cell", False)),
                "temperature": t,
                "input_structure": relative_to_root(restart, root),
                "chunk_name": "chunk_elastic",
                "fixed_steps": int(round(float(args.run_time_ps) / float(args.timestep_ps or 0.0001))),
                "max_chunks": 1,
                "production_run": True,
                "elastic_run": True,
                "thermo_stress": True,
                "dump_every": args.dump_every,
                "source_npt_stage": rec["stage_name"],
                "source_npt_log": str(Path(rec["log_path"]).resolve()),
                "deformation": {
                    "mode": mode,
                    "strain": strain,
                    "voigt_order": list(VOIGT),
                    "voigt_strain": voigt_strain(mode, strain),
                },
            }
            if data is not None:
                stage["input_data_fallback"] = relative_to_root(data, root)
            stages.append(stage)
            manifest.append(
                {
                    "stage_name": name,
                    "temperature_K": t,
                    "mode": mode,
                    "strain": strain,
                    "source_npt_stage": rec["stage_name"],
                    "input_structure": stage["input_structure"],
                }
            )
    return stages, manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stage_name", "temperature_K", "mode", "strain", "source_npt_stage", "input_structure"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_main(args: argparse.Namespace) -> dict:
    records, root, template = discover_npt_records(args)
    records = select_temperature_records(records, args)
    if not records:
        raise RuntimeError(
            "No NPT records matched the elastic temperature grid. Use --include-all-temperatures "
            "or adjust --temperature-start/--temperature-step/--temperature-tol."
        )
    if args.timestep_ps is None:
        args.timestep_ps = float(template.get("timestep", 0.0001))
    args.strains = parse_float_list(args.strain, [0.01, 0.02])
    args.modes = tuple(args.mode or DEFAULT_MODES)
    outdir = resolve_root_path(args.outdir, root)
    config_out = resolve_root_path(args.config_out, root)
    cfg = copy_elastic_base_config(template, root, args)
    stages, manifest = build_elastic_stages(records, root, args)
    cfg["stages"] = stages
    write_json(config_out, cfg)
    manifest_path = outdir / "elastic_manifest.csv"
    write_manifest(manifest_path, manifest)
    plan = {
        "root": str(root),
        "config": str(config_out),
        "manifest": str(manifest_path),
        "n_temperatures": len(records),
        "n_stages": len(stages),
        "temperatures_K": [float(rec["temperature"]) for rec in records],
        "run_command": f"md-engine-array --config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'array', root)} --job-name elastic-array",
        "analyze_command": f"elastic_lammps analyze --elastic-config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'fit', root)}",
    }
    plan_path = outdir / "elastic_plan.json"
    write_json(plan_path, plan)
    print(f"Wrote elastic config: {config_out}")
    print(f"Wrote elastic manifest: {manifest_path}")
    print(f"Wrote elastic plan: {plan_path}")
    print("Run with:")
    print(f"  {plan['run_command']}")
    print("Analyze after jobs finish with:")
    print(f"  {plan['analyze_command']}")
    return plan


def read_lammps_thermo_table(path: Path) -> dict[str, np.ndarray]:
    headers: list[str] | None = None
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "Step":
                headers = parts
                rows = []
                continue
            if headers is None or len(parts) < len(headers):
                continue
            try:
                rows.append([float(x) for x in parts[: len(headers)]])
            except ValueError:
                continue
    if headers is None or not rows:
        raise ValueError(f"No LAMMPS thermo table found in {path}")
    arr = np.asarray(rows, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(headers)}


def select_tail_mask(data: dict[str, np.ndarray], timestep_ps: float, window_ps: float) -> np.ndarray:
    step = np.asarray(data["Step"], dtype=float)
    t_ps = (step - step[0]) * float(timestep_ps)
    cutoff = float(np.max(t_ps)) - float(window_ps)
    mask = t_ps >= cutoff - 1.0e-12
    if np.count_nonzero(mask) < 3:
        return np.ones_like(step, dtype=bool)
    return mask


def thermo_column(data: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError("/".join(names))


def latest_elastic_log(root: Path, stage: dict) -> Path:
    stage_dir = root / "stages" / stage["name"] / stage.get("chunk_name", "chunk_elastic")
    candidates = sorted(stage_dir.glob("log.in.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        candidates = sorted(stage_dir.glob("log.*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No elastic LAMMPS log found in {stage_dir}")
    return candidates[-1]


def summarize_elastic_stage(root: Path, stage: dict, timestep_ps: float, window_ps: float) -> dict:
    log_path = latest_elastic_log(root, stage)
    data = read_lammps_thermo_table(log_path)
    required = ["Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy", "Lx", "Ly", "Lz", "Temp"]
    missing = [key for key in required if key not in data]
    if "Vol" not in data and "Volume" not in data:
        missing.append("Vol/Volume")
    if missing:
        raise ValueError(f"{log_path} is missing elastic thermo columns: {missing}")
    mask = select_tail_mask(data, timestep_ps, window_ps)
    pressure = np.array([data["Pxx"], data["Pyy"], data["Pzz"], data["Pyz"], data["Pxz"], data["Pxy"]], dtype=float)
    stress = -pressure[:, mask].mean(axis=1) * BAR_TO_GPA
    return {
        "stage_name": stage["name"],
        "temperature_K": float(stage["temperature"]),
        "mode": stage.get("deformation", {}).get("mode", "ref"),
        "strain": float(stage.get("deformation", {}).get("strain", 0.0) or 0.0),
        "voigt_strain": stage.get("deformation", {}).get("voigt_strain", [0.0] * 6),
        "log_path": str(log_path),
        "n_window_points": int(np.count_nonzero(mask)),
        "temp_mean_K": float(data["Temp"][mask].mean()),
        "vol_mean_A3": float(thermo_column(data, "Vol", "Volume")[mask].mean()),
        "lx_mean_A": float(data["Lx"][mask].mean()),
        "ly_mean_A": float(data["Ly"][mask].mean()),
        "lz_mean_A": float(data["Lz"][mask].mean()),
        "stress_GPa": stress.tolist(),
    }


def infer_symmetry_from_cell(summary: dict, tolerance: float) -> str:
    lx = float(summary["lx_mean_A"])
    ly = float(summary["ly_mean_A"])
    lz = float(summary["lz_mean_A"])
    scale = max(lx, ly, lz, 1.0)
    xy = abs(lx - ly) / scale
    xz = abs(lx - lz) / scale
    yz = abs(ly - lz) / scale
    if max(xy, xz, yz) <= tolerance:
        return "cubic"
    if xy <= tolerance and max(xz, yz) > tolerance:
        return "tetragonal"
    return "orthorhombic"


def fit_elastic_tensor(stage_summaries: list[dict]) -> tuple[np.ndarray, dict]:
    ref = next((item for item in stage_summaries if item["mode"] == "ref"), None)
    if ref is None:
        raise ValueError("Each temperature needs one reference stage with mode=ref")
    ref_stress = np.asarray(ref["stress_GPa"], dtype=float)
    strain_rows = []
    stress_rows = []
    for item in stage_summaries:
        if item["mode"] == "ref":
            continue
        strain_rows.append(np.asarray(item["voigt_strain"], dtype=float))
        stress_rows.append(np.asarray(item["stress_GPa"], dtype=float) - ref_stress)
    if len(strain_rows) < 6:
        raise ValueError("At least six non-reference strain states are needed for full tensor fitting")
    e = np.vstack(strain_rows)
    s = np.vstack(stress_rows)
    coeff, residuals, rank, singular = np.linalg.lstsq(e, s, rcond=None)
    c = coeff.T
    c_sym = 0.5 * (c + c.T)
    diagnostics = {
        "n_strained_states": len(strain_rows),
        "fit_rank": int(rank),
        "singular_values": singular.tolist(),
        "residual_sum_squares": residuals.tolist() if residuals.size else [],
    }
    return c_sym, diagnostics


def reduce_tensor_by_symmetry(c: np.ndarray, symmetry: str) -> np.ndarray:
    c = np.asarray(c, dtype=float).copy()
    if symmetry == "full":
        return 0.5 * (c + c.T)
    out = np.zeros((6, 6), dtype=float)
    if symmetry == "cubic":
        c11 = float(np.mean([c[0, 0], c[1, 1], c[2, 2]]))
        c12 = float(np.mean([c[0, 1], c[0, 2], c[1, 2], c[1, 0], c[2, 0], c[2, 1]]))
        c44 = float(np.mean([c[3, 3], c[4, 4], c[5, 5]]))
        out[:3, :3] = c12
        np.fill_diagonal(out[:3, :3], c11)
        out[3, 3] = out[4, 4] = out[5, 5] = c44
        return out
    if symmetry == "tetragonal":
        c11 = float(np.mean([c[0, 0], c[1, 1]]))
        c33 = float(c[2, 2])
        c12 = float(np.mean([c[0, 1], c[1, 0]]))
        c13 = float(np.mean([c[0, 2], c[2, 0], c[1, 2], c[2, 1]]))
        c44 = float(np.mean([c[3, 3], c[4, 4]]))
        c66 = float(c[5, 5])
        out[0, 0] = out[1, 1] = c11
        out[2, 2] = c33
        out[0, 1] = out[1, 0] = c12
        out[0, 2] = out[2, 0] = out[1, 2] = out[2, 1] = c13
        out[3, 3] = out[4, 4] = c44
        out[5, 5] = c66
        return out
    if symmetry == "orthorhombic":
        for i in range(6):
            out[i, i] = c[i, i]
        for i, j in ((0, 1), (0, 2), (1, 2)):
            out[i, j] = out[j, i] = 0.5 * (c[i, j] + c[j, i])
        return out
    raise ValueError(f"Unknown symmetry: {symmetry}")


def voigt_reuss_hill(c: np.ndarray) -> dict:
    c = np.asarray(c, dtype=float)
    s = np.linalg.inv(c)
    kv = (c[0, 0] + c[1, 1] + c[2, 2] + 2.0 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0
    gv = (
        c[0, 0]
        + c[1, 1]
        + c[2, 2]
        - c[0, 1]
        - c[0, 2]
        - c[1, 2]
        + 3.0 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0
    kr = 1.0 / (s[0, 0] + s[1, 1] + s[2, 2] + 2.0 * (s[0, 1] + s[0, 2] + s[1, 2]))
    gr = 15.0 / (
        4.0 * (s[0, 0] + s[1, 1] + s[2, 2])
        - 4.0 * (s[0, 1] + s[0, 2] + s[1, 2])
        + 3.0 * (s[3, 3] + s[4, 4] + s[5, 5])
    )
    kh = 0.5 * (kv + kr)
    gh = 0.5 * (gv + gr)
    e = 9.0 * kh * gh / (3.0 * kh + gh)
    nu = (3.0 * kh - 2.0 * gh) / (2.0 * (3.0 * kh + gh))
    eig = np.linalg.eigvalsh(0.5 * (c + c.T))
    return {
        "K_V_GPa": kv,
        "G_V_GPa": gv,
        "K_R_GPa": kr,
        "G_R_GPa": gr,
        "K_H_GPa": kh,
        "G_H_GPa": gh,
        "E_H_GPa": e,
        "nu_H": nu,
        "mechanically_stable_positive_definite": bool(np.all(eig > 0.0)),
        "elastic_eigenvalues_GPa": eig.tolist(),
    }


def tensor_components(c: np.ndarray) -> dict[str, float]:
    names = ["C11", "C22", "C33", "C44", "C55", "C66"]
    values = {name + "_GPa": float(c[i, i]) for i, name in enumerate(names)}
    values.update(
        {
            "C12_GPa": float(c[0, 1]),
            "C13_GPa": float(c[0, 2]),
            "C23_GPa": float(c[1, 2]),
        }
    )
    return values


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_main(args: argparse.Namespace) -> dict:
    config = args.elastic_config.resolve()
    cfg = load_json(config)
    root = config.parent.resolve()
    outdir = resolve_root_path(args.outdir, root)
    timestep_ps = float(args.timestep_ps if args.timestep_ps is not None else cfg.get("timestep", 0.0001))
    window_ps = float(args.window_ps)
    stage_summaries = [summarize_elastic_stage(root, stage, timestep_ps, window_ps) for stage in cfg.get("stages", [])]
    write_json(outdir / "elastic_stage_summaries.json", {"stages": stage_summaries})
    by_t: dict[float, list[dict]] = {}
    for item in stage_summaries:
        by_t.setdefault(float(item["temperature_K"]), []).append(item)
    rows = []
    tensor_payload = {}
    for t in sorted(by_t):
        items = by_t[t]
        raw_c, fit_diag = fit_elastic_tensor(items)
        ref = next(item for item in items if item["mode"] == "ref")
        inferred = infer_symmetry_from_cell(ref, args.symmetry_tolerance)
        symmetry = inferred if args.symmetry == "auto" else args.symmetry
        c = reduce_tensor_by_symmetry(raw_c, symmetry)
        moduli = voigt_reuss_hill(c)
        row = {
            "temperature_K": t,
            "symmetry": symmetry,
            "inferred_symmetry": inferred,
            **tensor_components(c),
            **moduli,
            "n_strained_states": fit_diag["n_strained_states"],
            "fit_rank": fit_diag["fit_rank"],
        }
        rows.append(row)
        tensor_payload[str(t)] = {
            "temperature_K": t,
            "symmetry": symmetry,
            "inferred_symmetry": inferred,
            "voigt_order": list(VOIGT),
            "C_raw_GPa": raw_c,
            "C_symmetry_reduced_GPa": c,
            "moduli": moduli,
            "fit_diagnostics": fit_diag,
        }
        print(f"T={t:g} K symmetry={symmetry} inferred={inferred} E={moduli['E_H_GPa']:.3f} GPa nu={moduli['nu_H']:.4f}")
    write_rows(outdir / "elastic_moduli_T.csv", rows)
    write_json(outdir / "elastic_tensors.json", tensor_payload)
    write_json(
        outdir / "elastic_analysis_metadata.json",
        {
            "config": str(config),
            "timestep_ps": timestep_ps,
            "window_ps": window_ps,
            "stress_sign_convention": "sigma_GPa = -pressure_tensor_bar * 1e-4",
            "symmetry": args.symmetry,
            "symmetry_tolerance": args.symmetry_tolerance,
            "outputs": {
                "elastic_moduli_T.csv": str(outdir / "elastic_moduli_T.csv"),
                "elastic_tensors.json": str(outdir / "elastic_tensors.json"),
                "elastic_stage_summaries.json": str(outdir / "elastic_stage_summaries.json"),
            },
        },
    )
    print(f"Wrote elastic moduli: {outdir / 'elastic_moduli_T.csv'}")
    print(f"Wrote elastic tensors: {outdir / 'elastic_tensors.json'}")
    return {"rows": rows, "tensors": tensor_payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elastic_lammps", description="Prepare and fit finite-temperature LAMMPS elastic tensors.")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write config_elastic.json from completed NPT stages.")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", nargs="+", help="One or more LAMMPS MD engine config JSON files.")
    source.add_argument("--md-root", type=Path, help="MD engine root; NPT folders are scanned and NVT folders ignored.")
    prep.add_argument("--template-config", type=Path, help="Template config to provide model/wrapper/mass settings for --md-root.")
    prep.add_argument("--config-dir")
    prep.add_argument("--config-glob", default="*.json")
    prep.add_argument("--duplicate-policy", choices=["highest_config_order", "first", "error"], default="highest_config_order")
    prep.add_argument("--outdir", type=Path, default=Path("analysis/elastic_lammps"))
    prep.add_argument("--config-out", type=Path, default=Path("config_elastic.json"))
    prep.add_argument("--T-min", dest="T_min", type=float)
    prep.add_argument("--T-max", dest="T_max", type=float)
    prep.add_argument("--temperature-start", type=float, default=100.0)
    prep.add_argument("--temperature-step", type=float, default=200.0)
    prep.add_argument("--temperature-tol", type=float, default=1.0)
    prep.add_argument("--include-all-temperatures", action="store_true")
    prep.add_argument("--strain", action="append", help="Positive strain magnitudes. Repeat or comma-separate. Default: 0.01,0.02")
    prep.add_argument("--mode", action="append", choices=VOIGT, help="Deformation mode. Default: all six Voigt modes.")
    prep.add_argument("--run-time-ps", type=float, default=20.0, help="NVT time per elastic strain state. Default: 20 ps.")
    prep.add_argument("--analysis-window-ps", type=float, default=10.0, help="Recommended tail window for stress fitting. Default: 10 ps.")
    prep.add_argument("--timestep-ps", type=float, help="Override timestep in ps. Defaults to template config timestep.")
    prep.add_argument("--dump-every", type=int, default=500)
    prep.add_argument("--symmetry", choices=["auto", "cubic", "tetragonal", "orthorhombic", "full"], default="auto")

    ana = sub.add_parser("analyze", help="Fit Cij and Voigt-Reuss-Hill moduli from completed elastic runs.")
    ana.add_argument("--elastic-config", type=Path, default=Path("config_elastic.json"))
    ana.add_argument("--outdir", type=Path, default=Path("analysis/elastic_lammps/fit"))
    ana.add_argument("--window-ps", type=float, default=10.0, help="Tail window for stress averaging. Default: 10 ps.")
    ana.add_argument("--timestep-ps", type=float, help="Override timestep in ps.")
    ana.add_argument("--symmetry", choices=["auto", "cubic", "tetragonal", "orthorhombic", "full"], default="auto")
    ana.add_argument("--symmetry-tolerance", type=float, default=0.01, help="Relative length tolerance for cubic/tetragonal inference.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_main(args)
    elif args.command == "analyze":
        analyze_main(args)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
