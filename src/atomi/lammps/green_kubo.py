"""Prepare and analyze MD-based Green-Kubo LAMMPS workflows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
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
from atomi.lammps.workflow import (
    SBATCH_RESOURCE_ENV,
    _apply_sbatch_resource_overrides,
    lammps_pair_lines,
    lammps_wrapper_text,
)


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


def _positive_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_gk_steps_per_hour(args: argparse.Namespace) -> float | None:
    value = args.gk_steps_per_hour
    if value is None:
        value = os.environ.get("ATOMI_LAMMPS_GK_STEPS_PER_HOUR")
    return _positive_float_or_none(value)


def _resolve_gk_walltime_safety_factor(args: argparse.Namespace) -> float:
    env_value = _positive_float_or_none(os.environ.get("ATOMI_LAMMPS_GK_WALLTIME_SAFETY_FACTOR"))
    value = args.gk_walltime_safety_factor if args.gk_walltime_safety_factor is not None else env_value
    return float(value if value is not None else 1.25)


def _gk_nvt_steps(args: argparse.Namespace) -> int:
    return max(0, int(round(float(args.nvt_preequilibration_ps) / float(args.timestep_ps))))


def _gk_walltime_hours(args: argparse.Namespace) -> float | None:
    steps_per_hour = _resolve_gk_steps_per_hour(args)
    if steps_per_hour is None:
        return None
    nve_steps = int(round(float(args.nve_time_ps) / float(args.timestep_ps)))
    total_steps = nve_steps + _gk_nvt_steps(args)
    return max((total_steps / steps_per_hour) * _resolve_gk_walltime_safety_factor(args), 0.25)


def apply_gk_runtime_performance(cfg: dict[str, Any], template: dict[str, Any], args: argparse.Namespace) -> None:
    """Record an observed GK/ML-IAP throughput so array walltimes do not reuse old MD timings."""
    steps_per_hour = _resolve_gk_steps_per_hour(args)
    if steps_per_hour is None:
        return
    template_perf = template.get("performance", {}) if isinstance(template.get("performance", {}), dict) else {}
    reference_atoms = int(
        args.gk_reference_atoms
        or template_perf.get("atoms")
        or template_perf.get("atoms_small")
        or template_perf.get("reference_atoms")
        or 1
    )
    cfg["performance"] = {
        "model": "observed_gk_mliap_steps_per_hour",
        "reference_atoms": reference_atoms,
        "atoms_small": reference_atoms,
        "atoms_large": reference_atoms,
        "reference_steps": float(steps_per_hour),
        "reference_hours": 1.0,
        "safety_factor": _resolve_gk_walltime_safety_factor(args),
        "notes": [
            "This performance block is for generated GK/ML-IAP stages only.",
            "It is based on observed ML-IAP steps/hour, not the old pair_style mace/kk MD timing.",
        ],
    }


def gk_runtime_estimate(args: argparse.Namespace, stage_count: int, array_limit: int | None) -> dict[str, Any]:
    nve_steps = int(round(float(args.nve_time_ps) / float(args.timestep_ps)))
    nvt_steps = _gk_nvt_steps(args)
    total_steps = nve_steps + nvt_steps
    steps_per_hour = _resolve_gk_steps_per_hour(args)
    estimate: dict[str, Any] = {
        "timestep_ps": float(args.timestep_ps),
        "timestep_fs": float(args.timestep_ps) * 1000.0,
        "nve_steps_per_stage": nve_steps,
        "nvt_preequilibration_steps_per_stage": nvt_steps,
        "estimated_total_md_steps_per_stage": total_steps,
        "n_stages": int(stage_count),
        "array_limit": int(array_limit) if array_limit else None,
    }
    if steps_per_hour is not None:
        safety = _resolve_gk_walltime_safety_factor(args)
        walltime = max((total_steps / steps_per_hour) * safety, 0.25)
        concurrency = max(1, int(array_limit or stage_count or 1))
        batches = math.ceil(stage_count / concurrency) if stage_count else 0
        estimate.update(
            {
                "observed_steps_per_hour": float(steps_per_hour),
                "walltime_safety_factor": safety,
                "estimated_walltime_hours_per_stage": walltime,
                "estimated_gpu_hours_all_stages": walltime * stage_count,
                "estimated_array_batches": batches,
                "estimated_elapsed_hours_at_array_limit": walltime * batches,
            }
        )
    return estimate


def copy_gk_base_config(template: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "wrapper_script",
        "model_file",
        "pair_style_backend",
        "model_elements",
        "lammps_pair_style",
        "lammps_pair_coeff",
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
        "slurm_resources",
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
        "disable_accelerated_suffix_for_heat_flux": not args.keep_accelerated_suffix_for_heat_flux,
        "heat_flux_preflight": not args.no_heat_flux_preflight,
        "notes": [
            "NVT_stress elasticity runs are not reused for GK because GK should be collected during NVE.",
            "LAMMPS heat/flux requires the pair style to provide per-atom energy and virial consistently.",
            "Atomi disables accelerated LAMMPS suffixes for heat-flux jobs by default because some GPU/Kokkos pair styles do not expose per-atom energy/virial.",
            "A run 0 heat-flux preflight is written before NVT pre-equilibration so unsupported pair styles fail immediately.",
        ],
    }
    return cfg


def scheduler_resource_key(option: str) -> str:
    return option.replace("-", "_")


def inherit_scheduler_resources(cfg: dict[str, Any], template: dict[str, Any]) -> None:
    """Carry private scheduler resources into generated GK configs.

    md-engine wrappers already know how to read ``slurm_resources`` plus
    ATOMI_LAMMPS_* environment variables. The GK prepare path writes a new
    config, so we snapshot the active private environment into that config too.
    """
    resources: dict[str, Any] = {}
    template_resources = template.get("slurm_resources", {})
    if isinstance(template_resources, dict):
        resources.update(
            {
                scheduler_resource_key(str(k)): v
                for k, v in template_resources.items()
                if v not in (None, "")
            }
        )

    for option, env_key in SBATCH_RESOURCE_ENV.items():
        key = scheduler_resource_key(option)
        for source_key in (key, option):
            value = template.get(source_key)
            if value not in (None, ""):
                resources.setdefault(key, value)
        value = os.environ.get(env_key)
        if value not in (None, ""):
            resources[key] = value

    if resources:
        cfg["slurm_resources"] = resources


def build_gk_stages(records: list[dict[str, Any]], root: Path, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    stages: list[dict] = []
    manifest: list[dict] = []
    timestep_ps = float(args.timestep_ps)
    run_steps = int(round(float(args.nve_time_ps) / timestep_ps))
    for rec in records:
        temp = float(rec["temperature"])
        t_label = temperature_label(temp)
        restart, data = find_restart_or_data(rec)
        input_structure = data if data is not None and not args.prefer_restart else restart
        for seed_index, seed in enumerate(seed_values(args), start=1):
            name = f"gk_T{t_label}K_s{seed_index:02d}"
            stage = {
                "name": name,
                "type": "nve",
                "large_cell": bool(rec.get("stage", {}).get("large_cell", False)),
                "temperature": temp,
                "input_structure": relative_to_root(input_structure, root),
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
                    "disable_accelerated_suffix_for_heat_flux": not args.keep_accelerated_suffix_for_heat_flux,
                    "heat_flux_preflight": not args.no_heat_flux_preflight,
                },
            }
            if data is not None and input_structure != data:
                stage["input_data_fallback"] = relative_to_root(data, root)
            if input_structure != restart:
                stage["input_restart_fallback"] = relative_to_root(restart, root)
            if args.walltime_hours is not None:
                stage["walltime_hours"] = float(args.walltime_hours)
            elif (estimated_walltime := _gk_walltime_hours(args)) is not None:
                stage["walltime_hours"] = float(estimated_walltime)
            stages.append(stage)
            manifest.append(
                {
                    "stage_name": name,
                    "temperature_K": temp,
                    "seed": int(seed),
                    "source_npt_stage": rec["stage_name"],
                    "input_structure": stage["input_structure"],
                    "input_kind": "data" if input_structure == data else "restart",
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
        "input_kind",
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
    inherit_scheduler_resources(cfg, template)
    if args.model_file is not None:
        cfg["model_file"] = relative_to_root(resolve_root_path(args.model_file, root), root)
    if args.pair_style_backend is not None:
        cfg["pair_style_backend"] = args.pair_style_backend
    if args.model_elements:
        elements: list[str] = []
        for item in args.model_elements:
            elements.extend(part.strip() for part in str(item).replace(",", " ").split() if part.strip())
        cfg["model_elements"] = elements
    if cfg.get("pair_style_backend") == "mliap":
        cfg["runtime_profile"] = "lammps_gk_mliap"
        args.keep_accelerated_suffix_for_heat_flux = True
        apply_gk_runtime_performance(cfg, template, args)
        cfg["green_kubo_settings"]["notes"].append(
            "This config requests pair_style_backend=mliap; use the private lammps_gk_mliap profile/GK ML-IAP LAMMPS binary."
        )
        cfg["green_kubo_settings"]["heat_flux_suffix"] = os.environ.get("ATOMI_LAMMPS_GK_SUFFIX", "kk")
        cfg["green_kubo_settings"]["notes"].append(
            "For MACE ML-IAP, Atomi keeps the KOKKOS suffix enabled so LAMMPS uses mliap/kk and the KOKKOS forward_exchange path."
        )
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
        "runtime_estimate": gk_runtime_estimate(args, len(stages), args.array_limit),
        "run_command": f"md-engine-array --config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'array', root)} --job-name gk-array --array-limit {args.array_limit}",
        "analyze_command": f"thermal-gk-lammps analyze --gk-config {relative_to_root(config_out, root)} --outdir {relative_to_root(outdir / 'fit', root)}",
    }
    cfg["green_kubo_settings"]["runtime_estimate"] = plan["runtime_estimate"]
    write_json(config_out, cfg)
    plan_path = outdir / "gk_plan.json"
    write_json(plan_path, plan)
    print(f"Wrote Green-Kubo config: {config_out}")
    print(f"Wrote Green-Kubo manifest: {manifest_path}")
    print(f"Wrote Green-Kubo plan: {plan_path}")
    print("Run with:")
    print(f"  {plan['run_command']}")
    estimate = plan["runtime_estimate"]
    if "estimated_walltime_hours_per_stage" in estimate:
        print(
            "Estimated GK walltime per stage: "
            f"{estimate['estimated_walltime_hours_per_stage']:.2f} h "
            f"({estimate['estimated_total_md_steps_per_stage']} steps at "
            f"{estimate['observed_steps_per_hour']:.0f} steps/hour, safety "
            f"{estimate['walltime_safety_factor']:.2f})"
        )
    print("Analyze after jobs finish with:")
    print(f"  {plan['analyze_command']}")
    return plan


def first_green_kubo_stage(cfg: dict[str, Any], stage_name: str | None = None) -> dict[str, Any]:
    stages = list(cfg.get("stages", []))
    if stage_name:
        for stage in stages:
            if stage.get("name") == stage_name:
                return stage
        raise ValueError(f"No stage named {stage_name!r} was found in the Green-Kubo config.")
    for stage in stages:
        if stage.get("green_kubo_run", False):
            return stage
    if stages:
        return stages[0]
    raise ValueError("The config has no stages to probe.")


def probe_read_command(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".restart" or name.startswith("restart."):
        return f"read_restart    {path.resolve()}"
    return f"read_data       {path.resolve()}"


def probe_suffix_command(suffix: str) -> str:
    if suffix == "none":
        return ""
    return f"suffix          {suffix}\n\n"


def build_heat_flux_probe_input(
    cfg: dict[str, Any],
    *,
    root: Path,
    input_structure: Path,
    temperature: float,
    suffix: str,
) -> str:
    cfg_for_pair = dict(cfg)
    cfg_for_pair["model_file"] = str(resolve_root_path(Path(cfg["model_file"]), root))
    pair_text = lammps_pair_lines(cfg_for_pair)
    timestep = float(cfg.get("timestep_ps", cfg.get("timestep", 0.001)))
    return f"""units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map yes
newton          on

{probe_read_command(input_structure)}

mass            1 {cfg["mass_O"]}
mass            2 {cfg["mass_U"]}

{probe_suffix_command(suffix)}{pair_text}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
timestep        {timestep}

velocity        all create {temperature} 987654 mom yes rot yes dist gaussian

compute         atomi_ke all ke/atom
compute         atomi_pe all pe/atom
compute         atomi_stress all stress/atom NULL virial
compute         atomi_flux all heat/flux atomi_ke atomi_pe atomi_stress
variable        atomi_Jx equal c_atomi_flux[1]/vol
variable        atomi_Jy equal c_atomi_flux[2]/vol
variable        atomi_Jz equal c_atomi_flux[3]/vol

thermo          1
thermo_style    custom step temp pe etotal press vol v_atomi_Jx v_atomi_Jy v_atomi_Jz
thermo_modify   flush yes

print           "Atomi GK probe: testing compute heat/flux compatibility"
run             0
print           "Atomi GK probe: PASS heat/flux preflight completed"
"""


def write_probe_runner(path: Path, input_name: str, lammps_command: str) -> None:
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f": \"${{LMP_CMD:={lammps_command}}}\"\n"
        f"eval \"$LMP_CMD -in {shlex.quote(input_name)}\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def replace_sbatch_option(script: str, option: str, value: str) -> str:
    pattern = rf"(?m)^#+SBATCH\s+--{re.escape(option)}(?:[=\s].*)?$"
    replacement = f"#SBATCH --{option}={value}"
    updated, count = re.subn(pattern, replacement, script)
    if count:
        return updated
    lines = updated.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if re.match(r"^#+SBATCH\b", line):
            insert_at = index + 1
    lines.insert(insert_at, replacement)
    return "\n".join(lines) + ("\n" if updated.endswith("\n") else "")


def write_probe_sbatch_runner(cfg: dict[str, Any], root: Path, outdir: Path, input_name: str, walltime: str) -> Path | None:
    wrapper = cfg.get("wrapper_script")
    if not wrapper:
        return None
    wrapper_path = resolve_root_path(Path(wrapper), root)
    if not wrapper_path.exists():
        return None
    script = lammps_wrapper_text(cfg)
    script = _apply_sbatch_resource_overrides(script, cfg)
    script = replace_sbatch_option(script, "time", walltime)
    path = outdir / "run_probe_sbatch.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    submit_path = outdir / "submit_probe.sh"
    exports = ""
    if cfg.get("pair_style_backend") == "mliap" or cfg.get("runtime_profile") == "lammps_gk_mliap":
        exports = "export ATOMI_LAMMPS_USE_GK_EXE=1\n"
    submit_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{exports}"
        f"sbatch {shlex.quote(path.name)} {shlex.quote(input_name)}\n",
        encoding="utf-8",
    )
    submit_path.chmod(0o755)
    return path


def classify_probe_log(text: str) -> str:
    lowered = text.lower()
    if "atomi gk/ml-iap preflight: pass" in lowered and "atomi gk probe: pass heat/flux preflight completed" in lowered:
        return "PASS: GK/ML-IAP environment and compute heat/flux preflight completed."
    if "atomi lammps preflight failed" in lowered:
        if "does not expose the ml-iap mliap pair style" in lowered or "unrecognized pair style 'mliap'" in lowered:
            return "FAIL: selected LAMMPS binary does not provide ML-IAP/mliap; check ATOMI_LMP_GK_EXE."
        if "lammps executable could not start" in lowered or "libcuda.so.1" in lowered:
            return "FAIL: selected LAMMPS binary could not start; check GPU allocation, CUDA modules, and LD_LIBRARY_PATH."
        if "libpython" in lowered or "unable to locate python shared library" in lowered:
            return "FAIL: ML-IAP cannot find libpython; add the active Python lib directory to LD_LIBRARY_PATH."
        if "required ml-iap python modules could not be imported" in lowered or "no module named 'lammps'" in lowered:
            return "FAIL: ML-IAP Python coupling is not importable; check ATOMI_LAMMPS_ENV and ATOMI_LAMMPS_PYTHONPATH."
        if "no module named 'cupy'" in lowered or "python import cupy" in lowered:
            return "FAIL: KOKKOS ML-IAP needs CuPy to wrap GPU device arrays; install cupy-cuda12x in the GK env."
        if "ml-iap model file not found" in lowered:
            return "FAIL: converted ML-IAP model file was not found."
        if "pair_style mliap unified" in lowered:
            return "FAIL: GK requested ML-IAP, but the generated input does not use pair_style mliap unified."
        return "FAIL: wrapper preflight failed before LAMMPS run 0."
    if "lammps -h preflight failed" in lowered and "mpi_init" in lowered:
        return "WARNING: LAMMPS help-mode preflight hit MPI_Init; rerun with ATOMI_LAMMPS_SKIP_HELP_PREFLIGHT=1 or continue to input run."
    if "loading mliappy unified module failure" in lowered:
        return "FAIL: ML-IAP unified Python module failed to load; check lammps/mliap_unified_couple imports and Python path."
    if "running mliappy unified module failure" in lowered:
        if "forward_exchange" in lowered:
            return (
                "FAIL: MACE/ML-IAP API mismatch: installed MACE expects MLIAPDataPy.forward_exchange, "
                "but the configured LAMMPS ML-IAP build does not provide it."
            )
        if "partially initialized module 'torch'" in lowered or "torch' has no attribute 'fx'" in lowered:
            return (
                "FAIL: embedded Python torch import failed inside ML-IAP; align Python torch with the libtorch "
                "used to build LAMMPS, or rebuild LAMMPS against the active Python torch/libtorch."
            )
        if "gpu requested but tensor is on cpu" in lowered:
            return "FAIL: MACE ML-IAP received CPU tensors; set/export MACE_ALLOW_CPU=true for a CPU-fallback diagnostic probe."
        if "torch.compiler" in lowered and "is_compiling" in lowered:
            return (
                "FAIL: cuequivariance_torch expects torch.compiler.is_compiling, but this Torch build does not expose it; "
                "use a cuequivariance_torch version compatible with torch 2.2 or add a torch._dynamo.is_compiling shim."
            )
        if "torch.fx._symbolic_trace" in lowered and "is_fx_symbolic_tracing" in lowered:
            return (
                "FAIL: cuequivariance_torch expects torch.fx._symbolic_trace.is_fx_symbolic_tracing, "
                "but this Torch build does not expose it; use a compatible cuequivariance_torch version or add a shim."
            )
        return (
            "FAIL: ML-IAP unified module loaded but failed while running the model; "
            "inspect the Python traceback in the Slurm .err file for model/device/dtype details."
        )
    if "module not founderror" in lowered or "no module named 'lammps'" in lowered:
        return "FAIL: required Python module for ML-IAP is missing."
    if "no module named 'cupy'" in lowered or "name 'cupy' is not defined" in lowered:
        return "FAIL: KOKKOS ML-IAP needs CuPy to wrap GPU device arrays; install cupy-cuda12x in the GK env."
    if "unrecognized pair style 'mliap'" in lowered:
        return "FAIL: selected LAMMPS binary does not have the ML-IAP package enabled."
    if "libcuda.so.1" in lowered:
        return "FAIL: CUDA driver library is not visible; run the probe on a GPU allocation/node."
    if "libpython" in lowered or "unable to locate python shared library" in lowered:
        return "FAIL: ML-IAP cannot find libpython; check active Python module/env and LD_LIBRARY_PATH."
    if "undefined symbol" in lowered and ("torch" in lowered or "libshm" in lowered or "c10" in lowered):
        return "FAIL: Python torch is loading incompatible Torch/C10 shared libraries; prioritize the active torch/lib in LD_LIBRARY_PATH."
    if "partially initialized module 'torch'" in lowered or "torch' has no attribute 'fx'" in lowered:
        return (
            "FAIL: embedded Python torch import failed inside ML-IAP; align Python torch with the libtorch "
            "used to build LAMMPS, or rebuild LAMMPS against the active Python torch/libtorch."
        )
    if "forward_exchange" in lowered:
        return (
            "FAIL: MACE/ML-IAP API mismatch: installed MACE expects MLIAPDataPy.forward_exchange, "
            "but the configured LAMMPS ML-IAP build does not provide it."
        )
    if "gpu requested but tensor is on cpu" in lowered:
        return "FAIL: MACE ML-IAP received CPU tensors; set/export MACE_ALLOW_CPU=true for a CPU-fallback diagnostic probe."
    if "torch.compiler" in lowered and "is_compiling" in lowered:
        return (
            "FAIL: cuequivariance_torch expects torch.compiler.is_compiling, but this Torch build does not expose it; "
            "use a cuequivariance_torch version compatible with torch 2.2 or add a torch._dynamo.is_compiling shim."
        )
    if "torch.fx._symbolic_trace" in lowered and "is_fx_symbolic_tracing" in lowered:
        return (
            "FAIL: cuequivariance_torch expects torch.fx._symbolic_trace.is_fx_symbolic_tracing, "
            "but this Torch build does not expose it; use a compatible cuequivariance_torch version or add a shim."
        )
    if "model file not found" in lowered or "cannot open" in lowered and "mliap" in lowered:
        return "FAIL: ML-IAP model file could not be opened."
    if "eflag_atom" in lowered or "vflag_atom" in lowered or "heat/flux" in lowered and "error" in lowered:
        return "FAIL: pair style does not support the per-atom energy/virial needed by compute heat/flux."
    if "atomi gk probe: pass heat/flux preflight completed" in lowered:
        return "PASS: compute heat/flux preflight completed."
    if "error" in lowered:
        return "FAIL: LAMMPS reported an error; inspect gk_heatflux_probe.log."
    return "UNKNOWN: probe finished without a recognizable PASS/FAIL marker."


def probe_main(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = args.config.resolve()
    cfg = load_json(cfg_path)
    root = cfg_path.parent.resolve()
    stage = first_green_kubo_stage(cfg, args.stage)
    input_value = args.input_structure or stage.get("input_structure")
    if not input_value:
        raise ValueError("No input structure found. Pass --input-structure or use a config_gk.json stage with input_structure.")
    input_structure = resolve_root_path(Path(input_value), root)
    temperature = float(args.temperature if args.temperature is not None else stage.get("temperature", 300.0))
    suffix = args.suffix
    if suffix == "auto":
        env_suffix = os.environ.get("ATOMI_LAMMPS_GK_SUFFIX", "").strip()
        if env_suffix:
            suffix = env_suffix
        else:
            suffix = "kk" if cfg.get("pair_style_backend") == "mliap" or cfg.get("runtime_profile") == "lammps_gk_mliap" else "off"
    outdir = resolve_root_path(args.outdir, root)
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = outdir / "gk_heatflux_probe.in"
    runner_path = outdir / "run_probe.sh"
    sbatch_runner_path = outdir / "run_probe_sbatch.sh"
    log_path = outdir / "gk_heatflux_probe.log"
    report_path = outdir / "gk_heatflux_probe_report.json"
    text = build_heat_flux_probe_input(
        cfg,
        root=root,
        input_structure=input_structure,
        temperature=temperature,
        suffix=suffix,
    )
    input_path.write_text(text, encoding="utf-8")
    write_probe_runner(runner_path, input_path.name, args.lammps_command)
    wrapper_runner = write_probe_sbatch_runner(cfg, root, outdir, input_path.name, args.sbatch_walltime)
    status = "not_executed"
    returncode = None
    if args.execute:
        command = shlex.split(args.lammps_command) + ["-in", input_path.name]
        result = subprocess.run(command, cwd=outdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        returncode = result.returncode
        log_path.write_text(result.stdout, encoding="utf-8")
        status = classify_probe_log(result.stdout)
    report = {
        "config": str(cfg_path),
        "stage": stage.get("name"),
        "input_structure": str(input_structure),
        "temperature_K": temperature,
        "suffix": suffix,
        "input": str(input_path),
        "runner": str(runner_path),
        "sbatch_runner": str(wrapper_runner) if wrapper_runner else None,
        "sbatch_submit": str(outdir / "submit_probe.sh") if wrapper_runner else None,
        "log": str(log_path) if args.execute else None,
        "executed": bool(args.execute),
        "returncode": returncode,
        "status": status,
        "notes": [
            "For ML-IAP GK configs, the wrapper checks the selected GK binary, ML-IAP package exposure, model path, and Python coupling modules before LAMMPS run 0.",
            "The LAMMPS probe input then runs compute heat/flux at run 0 to catch per-atom energy/virial support errors.",
            "For ML-IAP configs, the default probe suffix is kk so LAMMPS uses the KOKKOS forward_exchange path.",
            "For non-ML-IAP configs, the default probe suffix is off because some accelerated pair styles do not expose heat-flux per-atom virials.",
            "A PASS here confirms launch-time GK compatibility, not statistical convergence of kappa.",
        ],
    }
    write_json(report_path, report)
    print(f"Wrote GK heat-flux probe input : {input_path}")
    print(f"Wrote GK heat-flux probe runner: {runner_path}")
    if wrapper_runner:
        print(f"Wrote wrapper-based Slurm probe: {sbatch_runner_path}")
        print(f"Wrote wrapper-based submitter  : {outdir / 'submit_probe.sh'}")
    print(f"Wrote GK heat-flux probe report: {report_path}")
    if args.execute:
        print(f"Probe status                  : {status}")
        print(f"Probe log                     : {log_path}")
    else:
        print("Run on HPC with:")
        print(f"  cd {outdir}")
        if wrapper_runner:
            print("  ./submit_probe.sh")
            print("or direct, if the LAMMPS command is already on PATH:")
        print("  ./run_probe.sh")
        print("or with an explicit command:")
        print("  LMP_CMD='srun lmp' ./run_probe.sh")
    return report


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
    prep.add_argument("--model-file", type=Path, help="Override model file for GK stages, e.g. a converted MACE ML-IAP model.")
    prep.add_argument("--pair-style-backend", choices=("mace", "mliap"), help="Pair-style backend written to GK inputs.")
    prep.add_argument("--model-elements", nargs="+", help="Element/type order for pair_coeff, e.g. O U. Commas are accepted.")
    prep.add_argument("--array-limit", type=int, default=10, help="Suggested md-engine-array concurrency. Default: 10.")
    prep.add_argument("--walltime-hours", type=float, help="Optional walltime override for every GK seed stage.")
    prep.add_argument(
        "--gk-steps-per-hour",
        type=float,
        help=(
            "Observed GK/ML-IAP throughput used to estimate per-stage walltime. "
            "Also read from ATOMI_LAMMPS_GK_STEPS_PER_HOUR when unset."
        ),
    )
    prep.add_argument(
        "--gk-walltime-safety-factor",
        type=float,
        default=None,
        help="Safety multiplier for --gk-steps-per-hour estimates. Default: 1.25 or ATOMI_LAMMPS_GK_WALLTIME_SAFETY_FACTOR.",
    )
    prep.add_argument(
        "--gk-reference-atoms",
        type=int,
        help="Atom count for documenting the observed GK throughput. Defaults to the template performance atom count when available.",
    )
    prep.add_argument(
        "--prefer-restart",
        action="store_true",
        help=(
            "Use NPT restart files as GK inputs even when matching data files exist. "
            "Default prefers data files because GK recreates velocities and data "
            "inputs avoid carrying accelerated pair-style restart metadata."
        ),
    )
    prep.add_argument(
        "--keep-accelerated-suffix-for-heat-flux",
        action="store_true",
        help=(
            "Do not write 'suffix off' for GK heat-flux stages. By default Atomi "
            "disables accelerated suffixes because some /kk pair styles do not "
            "provide per-atom energy/virial needed by compute heat/flux."
        ),
    )
    prep.add_argument(
        "--no-heat-flux-preflight",
        action="store_true",
        help="Skip the run 0 heat-flux compatibility preflight before NVT pre-equilibration.",
    )

    probe = sub.add_parser("probe", help="Write or run a tiny run-0 LAMMPS heat-flux compatibility probe.")
    probe.add_argument("--config", type=Path, default=Path("config_gk.json"))
    probe.add_argument("--stage", help="Green-Kubo stage name to probe. Default: first green_kubo_run stage.")
    probe.add_argument("--input-structure", type=Path, help="Override structure/data/restart used by the probe.")
    probe.add_argument("--temperature", type=float, help="Override probe temperature. Default: stage temperature.")
    probe.add_argument("--outdir", type=Path, default=Path("analysis/gk_lammps/probe"))
    probe.add_argument(
        "--suffix",
        choices=("auto", "off", "kk", "on", "none"),
        default="auto",
        help="LAMMPS suffix command for the probe. Default auto uses kk for ML-IAP and off otherwise.",
    )
    probe.add_argument("--lammps-command", default="lmp", help="Command used with --execute and written into run_probe.sh.")
    probe.add_argument(
        "--sbatch-walltime",
        default="00:05:00",
        help="Walltime written into the wrapper-based Slurm probe script. Default: 00:05:00.",
    )
    probe.add_argument("--execute", action="store_true", help="Run the probe immediately on this machine/session.")

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
        prepare_main(args)
        return None
    if args.command == "probe":
        probe_main(args)
        return None
    if args.command == "analyze":
        analyze_main(args)
        return None
    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
