"""Prepare and analyze MD-based Green-Kubo LAMMPS workflows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from atomi.lammps.elastic import (
    discover_npt_records,
    find_restart_or_data,
    read_lammps_thermo_table,
    relative_to_root,
    resolve_root_path,
    select_tail_mask,
    select_temperature_records,
    temperature_label,
)
from atomi.lammps.thermal_conductivity import green_kubo_rows, write_csv, write_json


EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1.0e-10
PS_TO_S = 1.0e-12
KB_J_PER_K = 1.380649e-23


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metal_heat_flux_scale(temperature_K: float, volume_A3: float) -> float:
    """Return scale for trapz(HCACF, time_ps) from LAMMPS metal heat/flux to W/m/K."""
    convert = (EV_TO_J * EV_TO_J) / (PS_TO_S * ANGSTROM_TO_M)
    return convert / (KB_J_PER_K * float(temperature_K) * float(temperature_K) * float(volume_A3))


def seed_values(args: argparse.Namespace) -> list[int]:
    if args.seed:
        values: list[int] = []
        for item in args.seed:
            values.extend(int(part.strip()) for part in str(item).split(",") if part.strip())
        return values
    return [int(args.seed_start) + i * int(args.seed_step) for i in range(int(args.n_seeds))]


def copy_gk_base_config(template: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "wrapper_script",
        "model_file",
        "timestep",
        "timestep_ps",
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
    timestep_ps = float(args.timestep_ps if args.timestep_ps is not None else template.get("timestep", 0.0001))
    run_steps = int(round(float(args.nve_time_ps) / timestep_ps))
    cfg["generated_by"] = "atomi thermal-gk-lammps prepare"
    cfg["description"] = (
        "Generated Green-Kubo NVE heat-current workflow. Each stage starts from "
        "a completed NPT restart/data file, optionally rethermalizes in NVT, then "
        "runs NVE and writes LAMMPS heat-flux autocorrelation data."
    )
    cfg["adaptive_steps"] = {
        "initial_small": run_steps,
        "initial_large": run_steps,
        "growth_factor": 1.0,
        "max_chunk_steps": run_steps,
    }
    cfg["max_chunks_small"] = 1
    cfg["max_chunks_large"] = 1
    cfg["green_kubo_settings"] = {
        "method": "LAMMPS compute heat/flux + fix ave/correlate",
        "nve_time_ps": float(args.nve_time_ps),
        "nvt_preequilibration_ps": float(args.nvt_preequilibration_ps),
        "sample_interval_ps": float(args.sample_interval_ps),
        "correlation_time_ps": float(args.correlation_time_ps),
        "plateau_window_ps": float(args.plateau_window_ps),
        "seed_count": len(seed_values(args)),
        "notes": [
            "NVT_stress elasticity runs are not reused for GK because GK should be collected during NVE.",
            "LAMMPS heat/flux requires the pair style to provide per-atom energy and virial consistently.",
        ],
    }
    return cfg


def build_gk_stages(records: list[dict[str, Any]], root: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    stages: list[dict] = []
    manifest: list[dict] = []
    timestep_ps = float(args.timestep_ps)
    run_steps = int(round(float(args.nve_time_ps) / timestep_ps))
    for rec in records:
        temp = float(rec["temperature"])
        t_label = temperature_label(temp)
        restart, data = find_restart_or_data(rec)
        for seed_index, seed in enumerate(seed_values(args), start=1):
            name = f"gk_T{t_label}K_s{seed_index:02d}"
            stage = {
                "name": name,
                "type": "nve",
                "large_cell": bool(rec.get("stage", {}).get("large_cell", False)),
                "temperature": temp,
                "input_structure": relative_to_root(restart, root),
                "chunk_name": "chunk_gk",
                "fixed_steps": run_steps,
                "max_chunks": 1,
                "production_run": True,
                "green_kubo_run": True,
                "recreate_velocity": True,
                "velocity_seed": int(seed),
                "source_npt_stage": rec["stage_name"],
                "source_npt_log": str(Path(rec["log_path"]).resolve()),
                "green_kubo_settings": {
                    "sample_interval_ps": float(args.sample_interval_ps),
                    "correlation_time_ps": float(args.correlation_time_ps),
                    "nvt_preequilibration_ps": float(args.nvt_preequilibration_ps),
                    "hcacf_file": "heatflux_hcacf.dat",
                    "timeseries_file": "heatflux_timeseries.dat",
                },
            }
            if data is not None:
                stage["input_data_fallback"] = relative_to_root(data, root)
            if args.walltime_hours is not None:
                stage["walltime_hours"] = float(args.walltime_hours)
            stages.append(stage)
            manifest.append(
                {
                    "stage_name": name,
                    "temperature_K": temp,
                    "seed": int(seed),
                    "source_npt_stage": rec["stage_name"],
                    "input_structure": stage["input_structure"],
                    "nve_time_ps": float(args.nve_time_ps),
                    "sample_interval_ps": float(args.sample_interval_ps),
                    "correlation_time_ps": float(args.correlation_time_ps),
                }
            )
    return stages, manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage_name",
        "temperature_K",
        "seed",
        "source_npt_stage",
        "input_structure",
        "nve_time_ps",
        "sample_interval_ps",
        "correlation_time_ps",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    records, root, template = discover_npt_records(args)
    if args.timestep_ps is None:
        args.timestep_ps = float(template.get("timestep_ps", template.get("timestep", 0.0001)))
    records = select_temperature_records(records, args)
    if not records:
        raise RuntimeError("No NPT records matched the Green-Kubo temperature selection.")
    outdir = resolve_root_path(args.outdir, root)
    config_out = resolve_root_path(args.config_out, root)
    cfg = copy_gk_base_config(template, args)
    stages, manifest = build_gk_stages(records, root, args)
    cfg["stages"] = stages
    write_json(config_out, cfg)
    manifest_path = outdir / "gk_manifest.csv"
    write_manifest(manifest_path, manifest)
    plan = {
        "root": str(root),
        "config": str(config_out),
        "manifest": str(manifest_path),
        "n_temperatures": len(records),
        "n_seeds_per_temperature": len(seed_values(args)),
        "n_stages": len(stages),
        "temperatures_K": [float(rec["temperature"]) for rec in records],
        "run_command": f"md-engine-array --config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'array', root)} --job-name gk-array --array-limit {args.array_limit}",
        "analyze_command": f"thermal-gk-lammps analyze --gk-config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'fit', root)}",
    }
    plan_path = outdir / "gk_plan.json"
    write_json(plan_path, plan)
    print(f"Wrote Green-Kubo config: {config_out}")
    print(f"Wrote Green-Kubo manifest: {manifest_path}")
    print(f"Wrote Green-Kubo plan: {plan_path}")
    print("Run with:")
    print(f"  {plan['run_command']}")
    print("Analyze after jobs finish with:")
    print(f"  {plan['analyze_command']}")
    return plan


def latest_gk_log(root: Path, stage: dict[str, Any]) -> Path | None:
    chunk_dir = root / "stages" / stage["name"] / stage.get("chunk_name", "chunk_gk")
    candidates = sorted(chunk_dir.glob("log.in.*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("log.*"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def latest_hcacf_path(root: Path, stage: dict[str, Any]) -> Path:
    chunk_dir = root / "stages" / stage["name"] / stage.get("chunk_name", "chunk_gk")
    settings = stage.get("green_kubo_settings", {})
    return chunk_dir / settings.get("hcacf_file", "heatflux_hcacf.dat")


def parse_hcacf_dat(path: Path, timestep_ps: float) -> list[dict[str, float]]:
    blocks: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 2:
                if current:
                    blocks.append(current)
                    current = []
                continue
            if len(values) >= 5:
                time_ps = values[1] * float(timestep_ps)
                current.append(
                    {
                        "time_ps": time_ps,
                        "HCACF_x": values[2],
                        "HCACF_y": values[3],
                        "HCACF_z": values[4],
                    }
                )
    if current:
        blocks.append(current)
    if not blocks:
        raise ValueError(f"No HCACF blocks found in {path}")
    return blocks[-1]


def write_hcacf_csv(path: Path, rows: list[dict[str, float]]) -> None:
    write_csv(path, rows, ["time_ps", "HCACF_x", "HCACF_y", "HCACF_z"])


def summarize_gk_log(path: Path | None, timestep_ps: float, window_ps: float) -> dict[str, float | None]:
    if path is None:
        return {"temperature_mean_K": None, "volume_mean_A3": None}
    try:
        data = read_lammps_thermo_table(path)
        mask = select_tail_mask(data, timestep_ps, window_ps)
    except Exception:
        return {"temperature_mean_K": None, "volume_mean_A3": None}
    volume = data.get("Vol")
    if volume is None:
        volume = data.get("Volume")
    return {
        "temperature_mean_K": float(np.asarray(data["Temp"])[mask].mean()) if "Temp" in data else None,
        "volume_mean_A3": float(np.asarray(volume)[mask].mean()) if volume is not None else None,
    }


def average_hcacf(tables: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    if not tables:
        return []
    n = min(len(table) for table in tables)
    if n == 0:
        return []
    out: list[dict[str, float]] = []
    for idx in range(n):
        times = [table[idx]["time_ps"] for table in tables]
        out.append(
            {
                "time_ps": float(np.mean(times)),
                "HCACF_x": float(np.mean([table[idx]["HCACF_x"] for table in tables])),
                "HCACF_y": float(np.mean([table[idx]["HCACF_y"] for table in tables])),
                "HCACF_z": float(np.mean([table[idx]["HCACF_z"] for table in tables])),
            }
        )
    return out


def stage_temperature_key(stage: dict[str, Any]) -> float:
    return float(stage.get("temperature", stage.get("temperature_end")))


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    config = args.gk_config.resolve()
    cfg = load_json(config)
    root = config.parent.resolve()
    outdir = resolve_root_path(args.outdir, root)
    timestep_ps = float(args.timestep_ps if args.timestep_ps is not None else cfg.get("timestep_ps", cfg.get("timestep", 0.0001)))
    plateau_start = args.plateau_start_ps
    if plateau_start is None:
        correlation = float(cfg.get("green_kubo_settings", {}).get("correlation_time_ps", args.correlation_time_ps))
        plateau_start = max(0.0, correlation - float(args.plateau_window_ps))
    seed_rows: list[dict[str, Any]] = []
    grouped_tables: dict[float, list[list[dict[str, float]]]] = {}
    grouped_summaries: dict[float, list[dict[str, Any]]] = {}
    seed_csv_dir = outdir / "hcacf_seeds"
    for stage in cfg.get("stages", []):
        if not stage.get("green_kubo_run", False):
            continue
        temp = stage_temperature_key(stage)
        hcacf = latest_hcacf_path(root, stage)
        if not hcacf.exists():
            seed_rows.append(
                {
                    "stage_name": stage["name"],
                    "temperature_K": temp,
                    "seed": stage.get("velocity_seed"),
                    "status": "missing_hcacf",
                    "hcacf_path": str(hcacf),
                }
            )
            continue
        table = parse_hcacf_dat(hcacf, timestep_ps)
        per_seed_csv = seed_csv_dir / f"{stage['name']}_hcacf.csv"
        write_hcacf_csv(per_seed_csv, table)
        log_summary = summarize_gk_log(latest_gk_log(root, stage), timestep_ps, args.thermo_window_ps)
        volume = log_summary.get("volume_mean_A3")
        temp_mean = log_summary.get("temperature_mean_K") or temp
        scale = args.green_kubo_scale
        scale_mode = "user"
        if scale is None:
            if args.scale_mode == "metal" and volume is not None:
                scale = metal_heat_flux_scale(float(temp_mean), float(volume))
                scale_mode = "metal_auto"
            else:
                scale = 1.0
                scale_mode = "raw_unscaled"
        k_rows, k_meta = green_kubo_rows(
            per_seed_csv,
            temperature_K=temp,
            scale_to_W_mK=float(scale),
            plateau_start_ps=plateau_start,
            plateau_fraction=args.plateau_fraction,
            label=f"GK_seed_{stage['name']}",
            meta={},
        )
        k_row = k_rows[0] if k_rows else {}
        seed_rows.append(
            {
                "stage_name": stage["name"],
                "temperature_K": temp,
                "seed": stage.get("velocity_seed"),
                "status": "ok",
                "hcacf_path": str(hcacf),
                "hcacf_csv": str(per_seed_csv),
                "temperature_mean_K": temp_mean,
                "volume_mean_A3": volume,
                "scale_mode": scale_mode,
                "scale_to_W_mK": scale,
                "k_seed_W_mK": k_row.get("k_W_mK"),
                "k_seed_std_axes_W_mK": k_row.get("k_std_W_mK"),
                "plateau_start_ps": k_meta.get("plateau_start_ps", plateau_start),
            }
        )
        grouped_tables.setdefault(temp, []).append(table)
        grouped_summaries.setdefault(temp, []).append({"temperature_K": temp_mean, "volume_A3": volume})
    final_rows: list[dict[str, Any]] = []
    average_sources: list[dict[str, Any]] = []
    for temp, tables in sorted(grouped_tables.items()):
        averaged = average_hcacf(tables)
        label = temperature_label(temp)
        averaged_csv = outdir / f"gk_hcacf_T{label}K_average.csv"
        write_hcacf_csv(averaged_csv, averaged)
        summaries = grouped_summaries.get(temp, [])
        temps = [float(item["temperature_K"]) for item in summaries if item.get("temperature_K") is not None]
        vols = [float(item["volume_A3"]) for item in summaries if item.get("volume_A3") is not None]
        t_scale = float(np.mean(temps)) if temps else float(temp)
        v_scale = float(np.mean(vols)) if vols else None
        scale = args.green_kubo_scale
        scale_mode = "user"
        if scale is None:
            if args.scale_mode == "metal" and v_scale is not None:
                scale = metal_heat_flux_scale(t_scale, v_scale)
                scale_mode = "metal_auto"
            else:
                scale = 1.0
                scale_mode = "raw_unscaled"
        rows, meta = green_kubo_rows(
            averaged_csv,
            temperature_K=temp,
            scale_to_W_mK=float(scale),
            plateau_start_ps=plateau_start,
            plateau_fraction=args.plateau_fraction,
            label=f"GK_MD_T{label}K",
            meta={},
        )
        for row in rows:
            row["n_gk_seeds"] = len(tables)
            row["scale_mode"] = scale_mode
            row["scale_to_W_mK"] = scale
            row["volume_mean_A3"] = v_scale
            final_rows.append(row)
        average_sources.append(
            {
                "temperature_K": temp,
                "hcacf_csv": str(averaged_csv),
                "n_seeds": len(tables),
                "scale_mode": scale_mode,
                "scale_to_W_mK": scale,
                "plateau_start_ps": meta.get("plateau_start_ps", plateau_start),
            }
        )
    seed_fields = [
        "stage_name",
        "temperature_K",
        "seed",
        "status",
        "hcacf_path",
        "hcacf_csv",
        "temperature_mean_K",
        "volume_mean_A3",
        "scale_mode",
        "scale_to_W_mK",
        "k_seed_W_mK",
        "k_seed_std_axes_W_mK",
        "plateau_start_ps",
    ]
    write_csv(outdir / "gk_seed_summary.csv", seed_rows, seed_fields)
    final_fields = [
        "T_K",
        "k_W_mK",
        "k_std_W_mK",
        "k_x_W_mK",
        "k_y_W_mK",
        "k_z_W_mK",
        "n_gk_seeds",
        "scale_mode",
        "scale_to_W_mK",
        "volume_mean_A3",
        "source",
        "source_file",
    ]
    write_csv(outdir / "thermal_conductivity_T.csv", final_rows, final_fields)
    metadata = {
        "schema": "atomi.lammps.green_kubo.v1",
        "config": str(config),
        "timestep_ps": timestep_ps,
        "plateau_start_ps": plateau_start,
        "plateau_window_ps": args.plateau_window_ps,
        "scale_mode": args.scale_mode if args.green_kubo_scale is None else "user",
        "outputs": {
            "thermal_conductivity_T.csv": str(outdir / "thermal_conductivity_T.csv"),
            "gk_seed_summary.csv": str(outdir / "gk_seed_summary.csv"),
        },
        "averaged_sources": average_sources,
        "notes": [
            "This analyzer expects LAMMPS fix ave/correlate files produced by Atomi green_kubo_run stages.",
            "For scale_mode=metal, HCACF columns are assumed to be unnormalized c_atomi_flux components in LAMMPS metal units.",
            "NMA trajectory projection is not performed here; combine this output with NMA mode tables through thermal-k-lammps.",
        ],
    }
    write_json(outdir / "gk_analysis_metadata.json", metadata)
    print(f"Wrote GK seed summary: {outdir / 'gk_seed_summary.csv'}")
    print(f"Wrote GK k(T): {outdir / 'thermal_conductivity_T.csv'}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-gk-lammps",
        description="Prepare and analyze MD-based Green-Kubo NVE heat-current workflows.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write config_gk.json from completed NPT stages.")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", nargs="+", help="One or more LAMMPS MD engine config JSON files.")
    source.add_argument("--md-root", type=Path, help="MD engine root; NPT folders are scanned and NVT folders ignored.")
    prep.add_argument("--template-config", type=Path, help="Template config to provide model/wrapper/mass settings for --md-root.")
    prep.add_argument("--config-dir")
    prep.add_argument("--config-glob", default="*.json")
    prep.add_argument("--duplicate-policy", choices=["highest_config_order", "first", "error"], default="highest_config_order")
    prep.add_argument("--outdir", type=Path, default=Path("analysis/gk_lammps"))
    prep.add_argument("--config-out", type=Path, default=Path("config_gk.json"))
    prep.add_argument("--T-min", dest="T_min", type=float)
    prep.add_argument("--T-max", dest="T_max", type=float)
    prep.add_argument("--temperature-start", type=float, default=300.0)
    prep.add_argument("--temperature-step", type=float, default=200.0)
    prep.add_argument("--temperature-tol", type=float, default=1.0)
    prep.add_argument("--temperature-grid", dest="include_all_temperatures", action="store_false", help="Use --temperature-start/step grid instead of all discovered NPT temperatures.")
    prep.set_defaults(include_all_temperatures=True)
    prep.add_argument("--n-seeds", type=int, default=10)
    prep.add_argument("--seed-start", type=int, default=91001)
    prep.add_argument("--seed-step", type=int, default=17)
    prep.add_argument("--seed", action="append", help="Explicit velocity seed. Repeat or comma-separate.")
    prep.add_argument("--nve-time-ps", type=float, default=400.0)
    prep.add_argument("--nvt-preequilibration-ps", type=float, default=20.0)
    prep.add_argument("--sample-interval-ps", type=float, default=0.01)
    prep.add_argument("--correlation-time-ps", type=float, default=50.0)
    prep.add_argument("--plateau-window-ps", type=float, default=5.0)
    prep.add_argument("--timestep-ps", type=float, help="Override timestep in ps. Defaults to template config timestep.")
    prep.add_argument("--array-limit", type=int, default=10, help="Suggested md-engine-array concurrency. Default: 10.")
    prep.add_argument("--walltime-hours", type=float, help="Optional walltime override for every GK seed stage.")

    ana = sub.add_parser("analyze", help="Integrate completed GK HCACF files into k(T).")
    ana.add_argument("--gk-config", type=Path, default=Path("config_gk.json"))
    ana.add_argument("--outdir", type=Path, default=Path("analysis/gk_lammps/fit"))
    ana.add_argument("--timestep-ps", type=float, help="Override timestep in ps.")
    ana.add_argument("--thermo-window-ps", type=float, default=20.0, help="Tail window for mean T/V scaling. Default: 20 ps.")
    ana.add_argument("--correlation-time-ps", type=float, default=50.0)
    ana.add_argument("--plateau-window-ps", type=float, default=5.0)
    ana.add_argument("--plateau-start-ps", type=float, help="Start time for plateau averaging. Default: correlation - plateau window.")
    ana.add_argument("--plateau-fraction", type=float, default=0.2)
    ana.add_argument(
        "--scale-mode",
        choices=["metal", "raw"],
        default="metal",
        help="Use LAMMPS metal heat/flux scaling from mean T,V, or leave raw/unscaled. Default: metal.",
    )
    ana.add_argument("--green-kubo-scale", type=float, help="Manual scale overriding --scale-mode.")
    return parser


def main(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    if args.command == "analyze":
        return analyze_main(args)
    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
