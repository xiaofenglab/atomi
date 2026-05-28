"""Prepare reverse NEMD LAMMPS workflows from completed NPT stages."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from atomi.lammps.elastic import (
    discover_npt_records,
    find_restart_or_data,
    read_lammps_thermo_table,
    relative_to_root,
    resolve_root_path,
    select_temperature_records,
    temperature_label,
)
from atomi.lammps.thermal_conductivity import write_json
from atomi.lammps.workflow import (
    SBATCH_RESOURCE_ENV,
    _apply_sbatch_resource_overrides,
    create_stage_wrapper,
    hours_to_slurm,
    lammps_pair_lines,
    lammps_wrapper_text,
)
from atomi.viz.vasp_live import ensure_gnuplot


DEFAULT_RNEMD_TIMESTEP_PS = 0.0001
DEFAULT_RNEMD_REPEAT = "1x1x3"
DEFAULT_RNEMD_WALLTIME_SAFETY_FACTOR = 1.5
EV_TO_J = 1.602176634e-19
ANGSTROM_TO_M = 1.0e-10
PS_TO_S = 1.0e-12


@dataclass(frozen=True)
class RNEMDRunPlan:
    timestep_ps: float
    run_steps: int
    replicate: str = ""
    direction: str = "z"
    swap_every: int | None = None
    nbin: int | None = None
    profile_nevery: int | None = None
    profile_nrepeat: int | None = None
    profile_nfreq: int | None = None


@dataclass(frozen=True)
class RNEMDStatus:
    current_steps: int
    expected_steps: int
    timestep_ps: float
    latest_temperature_K: float | None = None
    transferred_energy_eV: float | None = None

    @property
    def current_ps(self) -> float:
        return self.current_steps * self.timestep_ps

    @property
    def expected_ps(self) -> float:
        return self.expected_steps * self.timestep_ps

    @property
    def percent(self) -> float:
        if self.expected_steps <= 0:
            return 0.0
        return 100.0 * min(max(self.current_steps / self.expected_steps, 0.0), 1.0)


def _positive_float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int_or_none(value: Any) -> int | None:
    parsed = _positive_float_or_none(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_repeat(value: str | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)):
        parts = [int(item) for item in value]
    else:
        parts = [int(part) for part in re.split(r"[xX,\s]+", str(value).strip()) if part]
    if len(parts) != 3 or any(item < 1 for item in parts):
        raise ValueError(f"Expected a repeat like {DEFAULT_RNEMD_REPEAT}, got {value!r}.")
    return parts[0], parts[1], parts[2]


def _repeat_label(repeat: tuple[int, int, int]) -> str:
    return f"{repeat[0]}x{repeat[1]}x{repeat[2]}"


def _repeat_factor(repeat: tuple[int, int, int]) -> int:
    return int(repeat[0] * repeat[1] * repeat[2])


def _split_elements(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    elements: list[str] = []
    for item in values:
        elements.extend(part.strip() for part in str(item).replace(",", " ").split() if part.strip())
    return elements or None


def seed_values(args: argparse.Namespace) -> list[int]:
    if args.seed:
        values: list[int] = []
        for item in args.seed:
            values.extend(int(part.strip()) for part in str(item).split(",") if part.strip())
        return values
    return [int(args.seed_start) + i * int(args.seed_step) for i in range(int(args.n_seeds))]


def _resolve_prepare_timestep_ps(template: dict[str, Any], args: argparse.Namespace) -> float:
    if args.timestep_ps is not None:
        return float(args.timestep_ps)
    env_timestep = _positive_float_or_none(os.environ.get("ATOMI_LAMMPS_RNEMD_TIMESTEP_PS"))
    return float(env_timestep or template.get("timestep_ps", template.get("timestep", DEFAULT_RNEMD_TIMESTEP_PS)))


def _resolve_rnemd_walltime_safety_factor(args: argparse.Namespace) -> float:
    env_value = _positive_float_or_none(os.environ.get("ATOMI_LAMMPS_RNEMD_WALLTIME_SAFETY_FACTOR"))
    value = args.rnemd_walltime_safety_factor if args.rnemd_walltime_safety_factor is not None else env_value
    return float(value if value is not None else DEFAULT_RNEMD_WALLTIME_SAFETY_FACTOR)


def _performance_steps_per_hour(performance: dict[str, Any]) -> float | None:
    for key in ("steps_per_hour", "rnemd_steps_per_hour", "observed_steps_per_hour", "reference_steps_per_hour"):
        value = _positive_float_or_none(performance.get(key))
        if value is not None:
            return value
    reference_steps = _positive_float_or_none(performance.get("reference_steps"))
    reference_hours = _positive_float_or_none(
        performance.get("reference_hours") or performance.get("reference_walltime_hours")
    )
    if reference_steps is not None and reference_hours is not None:
        return reference_steps / reference_hours
    return None


def _resolve_reference_atoms(template: dict[str, Any], args: argparse.Namespace) -> int | None:
    if args.rnemd_reference_atoms is not None:
        return int(args.rnemd_reference_atoms)
    env_atoms = _positive_int_or_none(os.environ.get("ATOMI_LAMMPS_RNEMD_REFERENCE_ATOMS"))
    if env_atoms is not None:
        return env_atoms
    performance = template.get("performance", {})
    if isinstance(performance, dict):
        return _positive_int_or_none(
            performance.get("atoms")
            or performance.get("atoms_small")
            or performance.get("reference_atoms")
            or performance.get("reference_base_atoms")
        )
    return None


def _resolve_rnemd_steps_per_hour(
    template: dict[str, Any],
    args: argparse.Namespace,
    replicate_factor: int,
) -> tuple[float | None, dict[str, Any]]:
    if args.rnemd_steps_per_hour is not None:
        return float(args.rnemd_steps_per_hour), {"source": "cli", "scaled_by_replicate_factor": False}
    env_value = _positive_float_or_none(os.environ.get("ATOMI_LAMMPS_RNEMD_STEPS_PER_HOUR"))
    if env_value is not None:
        return env_value, {"source": "environment", "scaled_by_replicate_factor": False}

    performance = template.get("performance", {})
    if not isinstance(performance, dict):
        return None, {"source": None, "scaled_by_replicate_factor": False}
    base_steps_per_hour = _performance_steps_per_hour(performance)
    if base_steps_per_hour is None:
        return None, {"source": None, "scaled_by_replicate_factor": False}

    base_atoms = _resolve_reference_atoms(template, args)
    target_atoms = args.rnemd_target_atoms
    if target_atoms is None and base_atoms is not None:
        target_atoms = int(base_atoms * replicate_factor)
    if target_atoms is not None and base_atoms:
        scaled = base_steps_per_hour * (float(base_atoms) / float(target_atoms))
        return scaled, {
            "source": "template_performance_scaled_by_atoms",
            "base_steps_per_hour": base_steps_per_hour,
            "base_atoms": int(base_atoms),
            "target_atoms": int(target_atoms),
            "scaled_by_replicate_factor": True,
        }
    return base_steps_per_hour / float(replicate_factor), {
        "source": "template_performance_scaled_by_replicate_factor",
        "base_steps_per_hour": base_steps_per_hour,
        "replicate_factor": replicate_factor,
        "scaled_by_replicate_factor": True,
    }


def _parse_prompted_steps_per_hour(value: str, total_steps: int) -> float | None:
    text = value.strip().lower()
    if not text:
        return None
    if text.endswith("h"):
        hours = _positive_float_or_none(text[:-1])
        if hours is None:
            return None
        return float(total_steps) / hours
    return _positive_float_or_none(text)


def maybe_prompt_rnemd_steps_per_hour(
    template: dict[str, Any],
    args: argparse.Namespace,
    replicate_factor: int,
) -> None:
    if args.walltime_hours is not None:
        return
    total_steps = int(round(float(args.run_time_ps) / float(args.timestep_ps)))
    steps_per_hour, _meta = _resolve_rnemd_steps_per_hour(template, args, replicate_factor)
    if steps_per_hour is not None:
        return
    message = (
        "Reverse-NEMD timing is not configured. "
        f"This prepare request expects about {total_steps} MD steps per run after "
        f"replicate {_repeat_label(args.replicate_tuple)}. Store a reusable value in "
        "profiles.lammps_rnemd.performance.steps_per_hour or "
        "ATOMI_LAMMPS_RNEMD_STEPS_PER_HOUR."
    )
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(f"WARNING: {message}")
        return
    print(message)
    raw = input(
        "Enter observed rNEMD steps/hour, or observed hours for one run as e.g. 116h "
        "[blank to continue without estimate]: "
    )
    steps_per_hour = _parse_prompted_steps_per_hour(raw, total_steps)
    if steps_per_hour is None:
        if raw.strip():
            raise ValueError("Could not parse rNEMD timing input. Use steps/hour or hours like 116h.")
        return
    args.rnemd_steps_per_hour = steps_per_hour


def copy_rnemd_base_config(template: dict[str, Any], args: argparse.Namespace, root: Path) -> dict[str, Any]:
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
    if "wrapper_script" in cfg:
        cfg["wrapper_script"] = str(resolve_root_path(Path(cfg["wrapper_script"]), root))
    timestep_ps = float(args.timestep_ps)
    run_steps = int(round(float(args.run_time_ps) / timestep_ps))
    cfg["timestep"] = timestep_ps
    cfg["timestep_ps"] = timestep_ps
    cfg["generated_by"] = "atomi thermal-rnemd-lammps prepare"
    cfg["runtime_profile"] = "lammps_rnemd"
    cfg["description"] = (
        "Generated reverse NEMD workflow. Each stage starts from a completed NPT "
        "data file, replicates the relaxed cell, equilibrates velocities, runs NVE, "
        "uses fix thermal/conductivity to impose a heat flux, and writes a slab "
        "temperature profile for thermal-conductivity fitting."
    )
    cfg["adaptive_steps"] = {
        "initial_small": run_steps,
        "initial_large": run_steps,
        "growth_factor": 1.0,
        "max_chunk_steps": run_steps,
    }
    cfg["max_chunks_small"] = 1
    cfg["max_chunks_large"] = 1
    cfg["rnemd_settings"] = {
        "method": "Muller-Plathe reverse NEMD via LAMMPS fix thermal/conductivity",
        "run_time_ps": float(args.run_time_ps),
        "timestep_ps": timestep_ps,
        "timestep_fs": timestep_ps * 1000.0,
        "replicate": _repeat_label(args.replicate_tuple),
        "replicate_factor": _repeat_factor(args.replicate_tuple),
        "direction": args.direction,
        "nbin": int(args.nbin),
        "swap_every_steps": int(args.swap_every),
        "swap_count": int(args.swap_count),
        "profile_nevery": int(args.profile_nevery),
        "profile_nrepeat": int(args.profile_nrepeat),
        "profile_nfreq": int(args.profile_nfreq),
        "thermo_every": int(args.thermo_every),
        "dump_every": int(args.dump_every),
        "seed_count": len(seed_values(args)),
        "notes": [
            "rNEMD uses the normal LAMMPS/MACE runtime profile in m_lammps_env, not the GK ML-IAP profile.",
            "The default replicate is 1x1x3 so only the heat-flow axis is lengthened from the NPT-ready cell.",
            "For periodic heat-flow direction, the imposed heat flux is split into two directions in analysis.",
            "LAMMPS fix thermal/conductivity requires an even Nbin value.",
        ],
    }
    return cfg


def scheduler_resource_key(option: str) -> str:
    return option.replace("-", "_")


def inherit_scheduler_resources(cfg: dict[str, Any], template: dict[str, Any]) -> None:
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


def _read_command(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".restart" or name.startswith("restart."):
        return f"read_restart    {path.resolve()}"
    return f"read_data       {path.resolve()}"


def _input_kind(path: Path) -> str:
    name = path.name.lower()
    if path.suffix.lower() == ".restart" or name.startswith("restart."):
        return "restart"
    return "data"


def _suffix_command(suffix: str) -> str:
    if suffix in ("none", ""):
        return ""
    return f"suffix          {suffix}\n\n"


def build_rnemd_input(
    cfg: dict[str, Any],
    *,
    root: Path,
    input_structure: Path,
    stage_name: str,
    temperature: float,
    seed: int,
    args: argparse.Namespace,
) -> str:
    pair_cfg = dict(cfg)
    pair_cfg["model_file"] = str(resolve_root_path(Path(cfg["model_file"]), root))
    pair_text = lammps_pair_lines(pair_cfg)
    run_steps = int(round(float(args.run_time_ps) / float(args.timestep_ps)))
    repeat = args.replicate_tuple
    bin_width = 1.0 / float(args.nbin)
    swap_suffix = "" if int(args.swap_count) == 1 else f" swap {int(args.swap_count)}"
    dump_line = ""
    if int(args.dump_every) > 0:
        dump_line = (
            f"dump            rnemd_dump all custom {int(args.dump_every)} "
            f"dump.{stage_name}.rnemd id type x y z vx vy vz\n"
            "dump_modify     rnemd_dump sort id\n\n"
        )
    return f"""units           metal
dimension       3
boundary        p p p
atom_style      atomic
atom_modify     map yes
newton          on

{_read_command(input_structure)}
replicate       {repeat[0]} {repeat[1]} {repeat[2]}

mass            1 {cfg["mass_O"]}
mass            2 {cfg["mass_U"]}

{_suffix_command(args.suffix)}{pair_text}

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes
timestep        {float(args.timestep_ps)}

velocity        all create {float(temperature)} {int(seed)} mom yes rot yes dist gaussian
fix             rnemd_mom all momentum 1000 linear 1 1 1
fix             rnemd_int all nve
fix             rnemd_flux all thermal/conductivity {int(args.swap_every)} {args.direction} {int(args.nbin)}{swap_suffix}

compute         rnemd_ke all ke/atom
variable        rnemd_temp atom c_rnemd_ke/1.5
compute         rnemd_layers all chunk/atom bin/1d {args.direction} lower {bin_width:.12g} units reduced
fix             rnemd_profile all ave/chunk {int(args.profile_nevery)} {int(args.profile_nrepeat)} {int(args.profile_nfreq)} rnemd_layers v_rnemd_temp file rnemd_temperature_profile.dat

thermo          {int(args.thermo_every)}
thermo_style    custom step temp pe etotal press vol lx ly lz f_rnemd_flux
thermo_modify   flush yes

{dump_line}print           "Atomi rNEMD phase: NVE with Muller-Plathe heat-flux swaps"
run             {run_steps}

write_data      {stage_name}.rnemd_final.data
print           "Atomi rNEMD phase: completed"
"""


def _estimated_walltime_hours(
    template: dict[str, Any],
    args: argparse.Namespace,
    replicate_factor: int,
) -> tuple[float | None, dict[str, Any]]:
    if args.walltime_hours is not None:
        return max(float(args.walltime_hours), 0.25), {"source": "cli_walltime_hours"}
    steps_per_hour, meta = _resolve_rnemd_steps_per_hour(template, args, replicate_factor)
    if steps_per_hour is None:
        return None, meta
    steps = int(round(float(args.run_time_ps) / float(args.timestep_ps)))
    safety = _resolve_rnemd_walltime_safety_factor(args)
    return max((steps / steps_per_hour) * safety, 0.25), {
        **meta,
        "steps_per_hour": float(steps_per_hour),
        "walltime_safety_factor": safety,
    }


def rnemd_runtime_estimate(
    template: dict[str, Any],
    args: argparse.Namespace,
    stage_count: int,
    array_limit: int | None,
) -> dict[str, Any]:
    replicate_factor = _repeat_factor(args.replicate_tuple)
    run_steps = int(round(float(args.run_time_ps) / float(args.timestep_ps)))
    steps_per_hour, meta = _resolve_rnemd_steps_per_hour(template, args, replicate_factor)
    estimate: dict[str, Any] = {
        "timestep_ps": float(args.timestep_ps),
        "timestep_fs": float(args.timestep_ps) * 1000.0,
        "run_steps_per_stage": run_steps,
        "run_time_ps_per_stage": float(args.run_time_ps),
        "replicate": _repeat_label(args.replicate_tuple),
        "replicate_factor": replicate_factor,
        "n_stages": int(stage_count),
        "array_limit": int(array_limit) if array_limit else None,
        "throughput_source": meta.get("source"),
    }
    estimate.update({k: v for k, v in meta.items() if k != "source"})
    if steps_per_hour is not None:
        safety = _resolve_rnemd_walltime_safety_factor(args)
        walltime = max((run_steps / steps_per_hour) * safety, 0.25)
        concurrency = max(1, int(array_limit or stage_count or 1))
        batches = math.ceil(stage_count / concurrency) if stage_count else 0
        estimate.update(
            {
                "estimated_steps_per_hour": float(steps_per_hour),
                "walltime_safety_factor": safety,
                "estimated_walltime_hours_per_stage": walltime,
                "estimated_gpu_hours_all_stages": walltime * stage_count,
                "estimated_array_batches": batches,
                "estimated_elapsed_hours_at_array_limit": walltime * batches,
            }
        )
    return estimate


def _active_sbatch_lines(script: str) -> list[str]:
    skip = ("--job-name", "--output", "--error", "--time", "--array")
    lines: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH"):
            continue
        if any(flag in stripped for flag in skip):
            continue
        if stripped not in lines:
            lines.append(stripped)
    return lines


def _max_walltime(values: list[str]) -> str:
    def seconds(text: str) -> int:
        parts = [int(part) for part in text.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return parts[0]

    if not values:
        return "01:00:00"
    return max(values, key=seconds)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "stage_name",
        "temperature_K",
        "seed",
        "source_npt_stage",
        "input_structure",
        "input_kind",
        "replicate",
        "direction",
        "run_time_ps",
        "run_steps",
        "walltime",
        "chunk_dir",
        "input_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_submit_all(path: Path, manifest: list[dict[str, Any]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for row in manifest:
        lines.append(f"(cd {shlex.quote(row['chunk_dir'])} && sbatch run_stage.sh {shlex.quote(row['input_name'])})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_array_script(
    cfg: dict[str, Any],
    path: Path,
    manifest_path: Path,
    manifest: list[dict[str, Any]],
    *,
    array_limit: int,
    job_name: str,
) -> None:
    wrapper_text = _apply_sbatch_resource_overrides(lammps_wrapper_text(cfg), cfg)
    sbatch_lines = _active_sbatch_lines(wrapper_text)
    walltime = _max_walltime([str(row["walltime"]) for row in manifest])
    n_tasks = len(manifest)
    concurrency = max(1, min(int(array_limit), n_tasks or 1))
    content = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=rnemd_array.%A_%a.out",
        "#SBATCH --error=rnemd_array.%A_%a.err",
        *sbatch_lines,
        f"#SBATCH --time={walltime}",
        f"#SBATCH --array=1-{n_tasks}%{concurrency}",
        "",
        "set -euo pipefail",
        'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"',
        f"MANIFEST={shlex.quote(str(manifest_path.resolve()))}",
        "",
        "line=$(awk -F '\\t' -v task=\"$TASK_ID\" 'NR > 1 && $1 == task {print; exit}' \"$MANIFEST\" | tr -d '\\r')",
        'if [ -z "$line" ]; then',
        '  echo "ERROR: no rNEMD manifest row for task ${TASK_ID}"',
        "  exit 2",
        "fi",
        "IFS=$'\\t' read -r task_id stage_name temperature seed source_npt_stage input_structure input_kind "
        "replicate direction run_time_ps run_steps walltime chunk_dir input_name <<< \"$line\"",
        "task_id=${task_id//$'\\r'/}",
        "stage_name=${stage_name//$'\\r'/}",
        "temperature=${temperature//$'\\r'/}",
        "seed=${seed//$'\\r'/}",
        "source_npt_stage=${source_npt_stage//$'\\r'/}",
        "input_structure=${input_structure//$'\\r'/}",
        "input_kind=${input_kind//$'\\r'/}",
        "replicate=${replicate//$'\\r'/}",
        "direction=${direction//$'\\r'/}",
        "run_time_ps=${run_time_ps//$'\\r'/}",
        "run_steps=${run_steps//$'\\r'/}",
        "walltime=${walltime//$'\\r'/}",
        "chunk_dir=${chunk_dir//$'\\r'/}",
        "input_name=${input_name//$'\\r'/}",
        'echo "Running rNEMD array task ${task_id}: ${stage_name} T=${temperature} K seed=${seed}"',
        'echo "chunk=${chunk_dir}"',
        'echo "input=${input_name}"',
        'echo "walltime=${walltime} run_steps=${run_steps} run_time_ps=${run_time_ps}"',
        'if [ ! -d "$chunk_dir" ]; then',
        '  echo "ERROR: rNEMD chunk directory does not exist: ${chunk_dir}"',
        "  exit 2",
        "fi",
        'cd "$chunk_dir"',
        'if [ ! -f "$input_name" ]; then',
        '  echo "ERROR: rNEMD input file is missing in $(pwd): ${input_name}"',
        '  echo "Available files:"',
        "  ls -la",
        "  exit 2",
        "fi",
        'export SLURM_SUBMIT_DIR="$chunk_dir"',
        './run_stage.sh "$input_name"',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")
    path.chmod(0o755)


def _guard_old_env_backend(cfg: dict[str, Any]) -> None:
    backend = str(cfg.get("pair_style_backend", "mace")).lower()
    if backend == "mliap" or cfg.get("runtime_profile") == "lammps_gk_mliap":
        raise ValueError(
            "thermal-rnemd-lammps is intentionally configured for the normal old MACE/Kokkos "
            "LAMMPS environment. Use pair_style_backend=mace and profiles.lammps_md_engine/"
            "profiles.lammps_rnemd, not the ML-IAP GK profile."
        )


def build_rnemd_runs(
    cfg: dict[str, Any],
    records: list[dict[str, Any]],
    root: Path,
    outdir: Path,
    template: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    task_id = 0
    run_steps = int(round(float(args.run_time_ps) / float(args.timestep_ps)))
    walltime_hours, walltime_meta = _estimated_walltime_hours(template, args, _repeat_factor(args.replicate_tuple))
    walltime = hours_to_slurm(walltime_hours) if walltime_hours is not None else hours_to_slurm(24.0)

    for rec in records:
        temp = float(rec["temperature"])
        t_label = temperature_label(temp)
        restart, data = find_restart_or_data(rec)
        input_structure = data if data is not None and not args.prefer_restart else restart
        input_kind = _input_kind(Path(input_structure))
        for seed_index, seed in enumerate(seed_values(args), start=1):
            task_id += 1
            name = f"rnemd_T{t_label}K_s{seed_index:02d}"
            chunk_dir = outdir / name / "chunk_rnemd"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            input_name = f"in.{name}_production"
            input_text = build_rnemd_input(
                cfg,
                root=root,
                input_structure=Path(input_structure),
                stage_name=name,
                temperature=temp,
                seed=int(seed),
                args=args,
            )
            (chunk_dir / input_name).write_text(input_text, encoding="utf-8")
            create_stage_wrapper(cfg, chunk_dir, walltime)
            stage = {
                "name": name,
                "type": "rnemd",
                "temperature": temp,
                "input_structure": relative_to_root(input_structure, root),
                "input_kind": input_kind,
                "chunk_name": "chunk_rnemd",
                "fixed_steps": run_steps,
                "max_chunks": 1,
                "production_run": True,
                "rnemd_run": True,
                "velocity_seed": int(seed),
                "source_npt_stage": rec["stage_name"],
                "source_npt_log": str(Path(rec["log_path"]).resolve()),
                "replicate": _repeat_label(args.replicate_tuple),
                "direction": args.direction,
                "walltime_hours": walltime_hours,
                "walltime_metadata": walltime_meta,
                "input_file": relative_to_root(chunk_dir / input_name, root),
                "chunk_dir": relative_to_root(chunk_dir, root),
            }
            if data is not None and input_structure != data:
                stage["input_data_fallback"] = relative_to_root(data, root)
            if input_structure != restart:
                stage["input_restart_fallback"] = relative_to_root(restart, root)
            stages.append(stage)
            manifest.append(
                {
                    "task_id": task_id,
                    "stage_name": name,
                    "temperature_K": temp,
                    "seed": int(seed),
                    "source_npt_stage": rec["stage_name"],
                    "input_structure": stage["input_structure"],
                    "input_kind": input_kind,
                    "replicate": _repeat_label(args.replicate_tuple),
                    "direction": args.direction,
                    "run_time_ps": float(args.run_time_ps),
                    "run_steps": run_steps,
                    "walltime": walltime,
                    "chunk_dir": str(chunk_dir.resolve()),
                    "input_name": input_name,
                }
            )
    return stages, manifest


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    records, root, template = discover_npt_records(args)
    args.replicate_tuple = _parse_repeat(args.replicate)
    if int(args.nbin) % 2:
        raise ValueError("LAMMPS fix thermal/conductivity requires --nbin to be even.")
    args.timestep_ps = _resolve_prepare_timestep_ps(template, args)
    if args.profile_nfreq is None:
        args.profile_nfreq = int(args.profile_nevery) * int(args.profile_nrepeat)
    maybe_prompt_rnemd_steps_per_hour(template, args, _repeat_factor(args.replicate_tuple))
    records = select_temperature_records(records, args)
    if not records:
        raise RuntimeError("No NPT records matched the reverse-NEMD temperature selection.")

    outdir = resolve_root_path(args.outdir, root)
    config_out = resolve_root_path(args.config_out, root)
    cfg = copy_rnemd_base_config(template, args, root)
    inherit_scheduler_resources(cfg, template)
    if args.model_file is not None:
        cfg["model_file"] = relative_to_root(resolve_root_path(args.model_file, root), root)
    if args.pair_style_backend is not None:
        cfg["pair_style_backend"] = args.pair_style_backend
    _guard_old_env_backend(cfg)
    elements = _split_elements(args.model_elements)
    if elements is not None:
        cfg["model_elements"] = elements

    stages, manifest = build_rnemd_runs(cfg, records, root, outdir, template, args)
    cfg["stages"] = stages
    plan = {
        "root": str(root),
        "config": str(config_out),
        "n_temperatures": len(records),
        "n_seeds_per_temperature": len(seed_values(args)),
        "n_stages": len(stages),
        "temperatures_K": [float(rec["temperature"]) for rec in records],
        "runtime_estimate": rnemd_runtime_estimate(template, args, len(stages), args.array_limit),
    }
    cfg["rnemd_settings"]["runtime_estimate"] = plan["runtime_estimate"]
    write_json(config_out, cfg)
    manifest_path = outdir / "rnemd_manifest.tsv"
    write_manifest(manifest_path, manifest)
    write_submit_all(outdir / "submit_rnemd_all.sh", manifest)
    array_path = outdir / "array" / "run_rnemd_array.sh"
    write_array_script(
        cfg,
        array_path,
        manifest_path,
        manifest,
        array_limit=int(args.array_limit),
        job_name=args.job_name,
    )
    plan["manifest"] = str(manifest_path)
    plan["array_script"] = str(array_path)
    plan["submit_all_script"] = str(outdir / "submit_rnemd_all.sh")
    plan["run_command"] = f"sbatch {relative_to_root(array_path, root)}"
    plan_path = outdir / "rnemd_plan.json"
    write_json(plan_path, plan)

    print(f"Wrote reverse-NEMD config: {config_out}")
    print(f"Wrote reverse-NEMD manifest: {manifest_path}")
    print(f"Wrote reverse-NEMD plan: {plan_path}")
    print(f"Wrote reverse-NEMD array script: {array_path}")
    estimate = plan["runtime_estimate"]
    if "estimated_walltime_hours_per_stage" in estimate:
        print(
            "Estimated rNEMD walltime per stage: "
            f"{estimate['estimated_walltime_hours_per_stage']:.2f} h "
            f"({estimate['run_steps_per_stage']} steps at "
            f"{estimate['estimated_steps_per_hour']:.0f} steps/hour, safety "
            f"{estimate['walltime_safety_factor']:.2f})"
        )
        print(
            "Estimated array elapsed time: "
            f"{estimate['estimated_elapsed_hours_at_array_limit']:.2f} h "
            f"for {estimate['n_stages']} run(s) at array limit {estimate['array_limit']}"
        )
    else:
        print(
            "WARNING: rNEMD walltime was not estimated; configure "
            "profiles.lammps_rnemd.performance.steps_per_hour in the local kit JSON."
        )
    print("Submit with:")
    print(f"  {plan['run_command']}")
    return plan


def _load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_log(chunk_dir: Path, input_name: str | None = None) -> Path:
    candidates: list[Path] = []
    if input_name:
        candidates.extend(chunk_dir.glob(f"log.{input_name}"))
        candidates.extend(chunk_dir.glob(f"log.{input_name}.*"))
    candidates.extend(chunk_dir.glob("log.in.rnemd*_production"))
    candidates.extend(chunk_dir.glob("log.*"))
    candidates = sorted({path.resolve() for path in candidates if path.exists()}, key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No LAMMPS log found in {chunk_dir}")
    return candidates[-1]


def latest_rnemd_input(chunk_dir: Path) -> Path | None:
    candidates = sorted(chunk_dir.glob("in.rnemd*_production"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("in.*"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def latest_rnemd_log(chunk_dir: Path, input_name: str | None = None) -> Path | None:
    try:
        return _latest_log(chunk_dir, input_name=input_name)
    except FileNotFoundError:
        return None


def read_rnemd_run_plan(input_file: Path) -> RNEMDRunPlan:
    text = input_file.read_text(encoding="utf-8", errors="replace")
    timestep = _first_float(r"(?m)^\s*timestep\s+([0-9.eE+-]+)", text) or DEFAULT_RNEMD_TIMESTEP_PS
    run_matches = [int(match.group(1)) for match in re.finditer(r"(?m)^\s*run\s+(\d+)", text)]
    replicate_match = re.search(r"(?m)^\s*replicate\s+(\d+)\s+(\d+)\s+(\d+)", text)
    replicate = "x".join(replicate_match.groups()) if replicate_match else ""
    flux_match = re.search(
        r"(?m)^\s*fix\s+rnemd_flux\s+all\s+thermal/conductivity\s+(\d+)\s+([xyz])\s+(\d+)",
        text,
    )
    profile_match = re.search(
        r"(?m)^\s*fix\s+rnemd_profile\s+all\s+ave/chunk\s+(\d+)\s+(\d+)\s+(\d+)",
        text,
    )
    return RNEMDRunPlan(
        timestep_ps=float(timestep),
        run_steps=run_matches[-1] if run_matches else 0,
        replicate=replicate,
        direction=flux_match.group(2) if flux_match else "z",
        swap_every=int(flux_match.group(1)) if flux_match else None,
        nbin=int(flux_match.group(3)) if flux_match else None,
        profile_nevery=int(profile_match.group(1)) if profile_match else None,
        profile_nrepeat=int(profile_match.group(2)) if profile_match else None,
        profile_nfreq=int(profile_match.group(3)) if profile_match else None,
    )


def summarize_rnemd_status(log_file: Path | None, plan: RNEMDRunPlan) -> RNEMDStatus:
    if log_file is None or not log_file.exists():
        return RNEMDStatus(0, plan.run_steps, plan.timestep_ps)
    try:
        thermo = read_lammps_thermo_table(log_file)
    except ValueError:
        return RNEMDStatus(0, plan.run_steps, plan.timestep_ps)
    step = np.asarray(thermo.get("Step", []), dtype=float)
    if step.size == 0:
        return RNEMDStatus(0, plan.run_steps, plan.timestep_ps)
    current_steps = max(0, int(round(float(step[-1] - step[0]))))
    latest_temperature = float(thermo["Temp"][-1]) if "Temp" in thermo and len(thermo["Temp"]) else None
    try:
        transferred_energy = float(_thermo_flux_column(thermo)[-1])
    except (KeyError, IndexError):
        transferred_energy = None
    return RNEMDStatus(
        current_steps=current_steps,
        expected_steps=plan.run_steps,
        timestep_ps=plan.timestep_ps,
        latest_temperature_K=latest_temperature,
        transferred_energy_eV=transferred_energy,
    )


def print_rnemd_summary(
    chunk_dir: Path,
    input_file: Path | None = None,
    log_file: Path | None = None,
) -> tuple[RNEMDRunPlan, RNEMDStatus]:
    input_file = input_file or latest_rnemd_input(chunk_dir)
    if input_file is None:
        raise FileNotFoundError(f"No rNEMD input file found in {chunk_dir}")
    log_file = log_file or latest_rnemd_log(chunk_dir, input_name=input_file.name)
    plan = read_rnemd_run_plan(input_file)
    status = summarize_rnemd_status(log_file, plan)
    profile_path = chunk_dir / "rnemd_temperature_profile.dat"
    blocks = _read_profile_blocks(profile_path) if profile_path.exists() else []

    print("rNEMD run status")
    print("----------------")
    print(f"chunk        : {chunk_dir}")
    print(f"input        : {input_file}")
    print(f"log          : {log_file if log_file else 'missing'}")
    print(f"timestep     : {plan.timestep_ps:g} ps ({plan.timestep_ps * 1000:g} fs)")
    print(f"replicate    : {plan.replicate or 'unknown'}")
    print(f"direction    : {plan.direction}")
    print(f"run target   : {plan.run_steps} steps = {plan.run_steps * plan.timestep_ps:g} ps")
    if plan.swap_every and plan.nbin:
        print(f"rNEMD swaps  : every {plan.swap_every} steps over {plan.nbin} bins")
    if plan.profile_nevery and plan.profile_nrepeat and plan.profile_nfreq:
        print(
            "profile ave  : "
            f"nevery={plan.profile_nevery}, nrepeat={plan.profile_nrepeat}, "
            f"nfreq={plan.profile_nfreq} ({plan.profile_nfreq * plan.timestep_ps:g} ps/window)"
        )
    print(
        f"current      : {status.current_steps}/{status.expected_steps} steps "
        f"= {status.current_ps:g}/{status.expected_ps:g} ps ({status.percent:.1f}%)"
    )
    if status.latest_temperature_K is not None:
        print(f"latest Temp  : {status.latest_temperature_K:.3g} K")
    if status.transferred_energy_eV is not None:
        print(f"flux energy  : {status.transferred_energy_eV:.6g} eV")
    if blocks:
        latest = blocks[-1]
        print(
            f"profile rows : {len(blocks)} block(s), latest step {latest['timestep']} "
            f"= {latest['timestep'] * plan.timestep_ps:g} ps"
        )
    else:
        print(f"profile rows : none yet ({profile_path.name})")
    return plan, status


def _read_profile_blocks(path: Path) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                timestep = int(float(parts[0]))
                nchunk = int(float(parts[1]))
                total_count = float(parts[2])
            except ValueError:
                continue
            current = {"timestep": timestep, "nchunk": nchunk, "total_count": total_count, "rows": []}
            blocks.append(current)
            continue
        if current is None or len(parts) < 4:
            continue
        try:
            current["rows"].append(
                {
                    "chunk": int(float(parts[0])),
                    "coord": float(parts[1]),
                    "count": float(parts[2]),
                    "temp": float(parts[3]),
                }
            )
        except ValueError:
            continue
    return [block for block in blocks if len(block["rows"]) == int(block["nchunk"])]


def _mean_profile(blocks: list[dict[str, Any]], tail_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if not blocks:
        raise ValueError("No complete temperature-profile blocks were found.")
    start = max(0, int(math.floor(len(blocks) * (1.0 - float(tail_fraction)))))
    selected = blocks[start:] or blocks
    nchunk = int(selected[-1]["nchunk"])
    coords = np.zeros(nchunk, dtype=float)
    temps = np.zeros(nchunk, dtype=float)
    counts = np.zeros(nchunk, dtype=float)
    for block in selected:
        rows = sorted(block["rows"], key=lambda item: item["chunk"])
        coords += np.asarray([row["coord"] for row in rows], dtype=float)
        temps += np.asarray([row["temp"] for row in rows], dtype=float)
        counts += np.asarray([row["count"] for row in rows], dtype=float)
    denom = float(len(selected))
    return coords / denom, temps / denom, counts / denom, len(selected)


def _fit_profile_slopes(coords: np.ndarray, temps: np.ndarray) -> dict[str, Any]:
    nbin = len(coords)
    if nbin < 8 or nbin % 2:
        raise ValueError(f"Expected an even profile with at least 8 bins, got {nbin}.")
    mid = nbin // 2
    first = np.arange(1, mid)
    second = np.arange(mid + 1, nbin)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Too few bins remain after excluding swap layers.")
    slope1, intercept1 = np.polyfit(coords[first], temps[first], 1)
    slope2, intercept2 = np.polyfit(coords[second], temps[second], 1)
    slope_abs = float(np.mean([abs(slope1), abs(slope2)]))
    spread = abs(abs(slope1) - abs(slope2)) / slope_abs if slope_abs > 0 else math.inf
    return {
        "slope1_K_per_reduced": float(slope1),
        "slope2_K_per_reduced": float(slope2),
        "intercept1_K": float(intercept1),
        "intercept2_K": float(intercept2),
        "mean_abs_slope_K_per_reduced": slope_abs,
        "slope_disagreement_fraction": float(spread),
    }


def _tail_mask(values: np.ndarray, tail_fraction: float) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    start = max(0, int(math.floor(values.size * (1.0 - float(tail_fraction)))))
    mask = np.zeros(values.size, dtype=bool)
    mask[start:] = True
    if np.count_nonzero(mask) < min(3, values.size):
        mask[:] = True
    return mask


def _direction_box_and_area(thermo: dict[str, np.ndarray], direction: str, mask: np.ndarray) -> tuple[float, float]:
    lengths = {
        "x": ("Lx", "Ly", "Lz"),
        "y": ("Ly", "Lx", "Lz"),
        "z": ("Lz", "Lx", "Ly"),
    }
    along_key, a_key, b_key = lengths[direction]
    along = float(np.mean(thermo[along_key][mask]))
    area = float(np.mean(thermo[a_key][mask]) * np.mean(thermo[b_key][mask]))
    return along, area


def _thermo_flux_column(thermo: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("f_rnemd_flux", "f_mp", "f_1"):
        if key in thermo:
            return thermo[key]
    for key in thermo:
        if key.startswith("f_"):
            return thermo[key]
    raise KeyError("No fix thermal/conductivity energy column found in thermo output.")


def plot_rnemd_once(
    chunk_dir: Path,
    *,
    window: int = 220,
    profile_tail_fraction: float = 0.5,
) -> None:
    chunk_dir = chunk_dir.resolve()
    plan, _status = print_rnemd_summary(chunk_dir)
    log_file = latest_rnemd_log(chunk_dir, input_name=(latest_rnemd_input(chunk_dir) or Path("")).name)
    profile_path = chunk_dir / "rnemd_temperature_profile.dat"
    thermo_rows = _rnemd_thermo_rows(log_file, plan, window=window) if log_file else []
    profile_rows = _rnemd_profile_rows(profile_path, profile_tail_fraction) if profile_path.exists() else []
    if not thermo_rows and not profile_rows:
        print("No plottable rNEMD thermo/profile rows are available yet.")
        return
    ensure_gnuplot()
    if thermo_rows:
        _plot_rnemd_thermo(thermo_rows)
    else:
        print(f"No thermo rows found in {log_file}")
    if profile_rows:
        _plot_rnemd_profile(profile_rows)
    else:
        print(f"No complete profile blocks found in {profile_path}")


def _rnemd_thermo_rows(log_file: Path | None, plan: RNEMDRunPlan, *, window: int) -> list[dict[str, float]]:
    if log_file is None or not log_file.exists():
        return []
    try:
        thermo = read_lammps_thermo_table(log_file)
        step = np.asarray(thermo["Step"], dtype=float)
        temp = np.asarray(thermo["Temp"], dtype=float)
        flux = np.asarray(_thermo_flux_column(thermo), dtype=float)
    except (KeyError, ValueError):
        return []
    if step.size == 0:
        return []
    start_step = float(step[0])
    rows = [
        {
            "time_ps": float((step[i] - start_step) * plan.timestep_ps),
            "temperature_K": float(temp[i]),
            "flux_energy_eV": float(flux[i]),
        }
        for i in range(step.size)
    ]
    return rows[-max(1, int(window)) :]


def _rnemd_profile_rows(profile_path: Path, tail_fraction: float) -> list[dict[str, float]]:
    try:
        blocks = _read_profile_blocks(profile_path)
        coords, temps, counts, _nblocks = _mean_profile(blocks, tail_fraction)
    except (OSError, ValueError):
        return []
    return [
        {
            "coord": float(coords[i]),
            "temperature_K": float(temps[i]),
            "count": float(counts[i]),
        }
        for i in range(len(coords))
    ]


def _plot_rnemd_thermo(rows: list[dict[str, float]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as handle:
        path = Path(handle.name)
        for row in rows:
            handle.write(f"{row['time_ps']} {row['temperature_K']} {row['flux_energy_eV']}\n")
    try:
        script = f"""
set term dumb ansi 120 24
set grid
set key outside
set y2tics
set title "rNEMD live thermo"
set xlabel "time (ps)"
set ylabel "Temp (K)"
set y2label "transferred energy (eV)"
plot "{_gnuplot_quote(path)}" using 1:2 with lines title "Temp", \
     "{_gnuplot_quote(path)}" using 1:3 axes x1y2 with lines title "f_rnemd_flux"
unset y2tics
"""
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
    finally:
        path.unlink(missing_ok=True)


def _plot_rnemd_profile(rows: list[dict[str, float]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as handle:
        path = Path(handle.name)
        for row in rows:
            handle.write(f"{row['coord']} {row['temperature_K']} {row['count']}\n")
    try:
        script = f"""
set term dumb ansi 120 24
set grid
set key outside
set title "rNEMD slab temperature profile"
set xlabel "reduced coordinate"
set ylabel "Temp (K)"
plot "{_gnuplot_quote(path)}" using 1:2 with linespoints title "T profile"
"""
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
    finally:
        path.unlink(missing_ok=True)


def analyze_stage(cfg: dict[str, Any], stage: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    chunk_dir = resolve_root_path(Path(stage["chunk_dir"]), root)
    input_name = Path(stage.get("input_file", "")).name or None
    profile_path = chunk_dir / "rnemd_temperature_profile.dat"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing rNEMD temperature profile: {profile_path}")
    log_path = _latest_log(chunk_dir, input_name=input_name)
    blocks = _read_profile_blocks(profile_path)
    coords, temps, counts, n_profile_blocks_used = _mean_profile(blocks, args.profile_tail_fraction)
    slope_info = _fit_profile_slopes(coords, temps)
    thermo = read_lammps_thermo_table(log_path)
    step = np.asarray(thermo["Step"], dtype=float)
    flux_energy = np.asarray(_thermo_flux_column(thermo), dtype=float)
    thermo_mask = _tail_mask(step, args.thermo_tail_fraction)
    timestep_ps = float(cfg.get("timestep_ps", cfg.get("timestep", DEFAULT_RNEMD_TIMESTEP_PS)))
    time_ps = (step - step[0]) * timestep_ps
    if np.count_nonzero(thermo_mask) < 2:
        raise ValueError(f"Not enough thermo rows in {log_path} to fit energy transfer rate.")
    energy_rate_eV_per_ps, energy_intercept = np.polyfit(time_ps[thermo_mask], flux_energy[thermo_mask], 1)
    direction = str(stage.get("direction") or cfg.get("rnemd_settings", {}).get("direction", "z"))
    length_A, area_A2 = _direction_box_and_area(thermo, direction, thermo_mask)
    slope_K_per_A = slope_info["mean_abs_slope_K_per_reduced"] / length_A
    heat_flux_W_m2 = abs(float(energy_rate_eV_per_ps)) * EV_TO_J / (
        2.0 * area_A2 * (ANGSTROM_TO_M**2) * PS_TO_S
    )
    k_W_mK = heat_flux_W_m2 / (slope_K_per_A / ANGSTROM_TO_M) if slope_K_per_A > 0 else math.nan
    row = {
        "stage_name": stage["name"],
        "temperature_K": float(stage["temperature"]),
        "seed": int(stage.get("velocity_seed", 0)),
        "chunk_dir": str(chunk_dir),
        "log_path": str(log_path),
        "profile_path": str(profile_path),
        "direction": direction,
        "replicate": str(stage.get("replicate", cfg.get("rnemd_settings", {}).get("replicate", ""))),
        "timestep_ps": timestep_ps,
        "run_steps": int(stage.get("fixed_steps", 0)),
        "profile_blocks": len(blocks),
        "profile_blocks_used": n_profile_blocks_used,
        "mean_temperature_K": float(np.average(temps, weights=np.maximum(counts, 1.0))),
        "length_A": length_A,
        "area_A2": area_A2,
        "slope1_K_per_reduced": slope_info["slope1_K_per_reduced"],
        "slope2_K_per_reduced": slope_info["slope2_K_per_reduced"],
        "slope_disagreement_fraction": slope_info["slope_disagreement_fraction"],
        "mean_abs_slope_K_per_A": slope_K_per_A,
        "energy_rate_eV_per_ps": float(energy_rate_eV_per_ps),
        "energy_intercept_eV": float(energy_intercept),
        "heat_flux_W_m2": heat_flux_W_m2,
        "k_W_mK": float(k_W_mK),
    }
    profile_rows = [
        {
            "stage_name": stage["name"],
            "chunk": int(index + 1),
            "coord_reduced": float(coords[index]),
            "temperature_K": float(temps[index]),
            "count": float(counts[index]),
        }
        for index in range(len(coords))
    ]
    row["profile_rows"] = profile_rows
    return row


def analyze_main(args: argparse.Namespace) -> dict[str, Any]:
    config = args.config.resolve()
    cfg = _load_json(config)
    root = config.parent.resolve()
    outdir = resolve_root_path(args.outdir, root)
    stage_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for stage in cfg.get("stages", []):
        if not stage.get("rnemd_run", False):
            continue
        row = analyze_stage(cfg, stage, root, args)
        profile_rows.extend(row.pop("profile_rows"))
        stage_rows.append(row)
    if not stage_rows:
        raise RuntimeError(f"No rNEMD stages were found in {config}")

    seed_fields = [
        "stage_name",
        "temperature_K",
        "seed",
        "k_W_mK",
        "heat_flux_W_m2",
        "mean_abs_slope_K_per_A",
        "slope_disagreement_fraction",
        "energy_rate_eV_per_ps",
        "mean_temperature_K",
        "profile_blocks",
        "profile_blocks_used",
        "direction",
        "replicate",
        "length_A",
        "area_A2",
        "chunk_dir",
    ]
    _write_csv(outdir / "rnemd_seed_summary.csv", stage_rows, seed_fields)
    _write_csv(
        outdir / "rnemd_temperature_profiles.csv",
        profile_rows,
        ["stage_name", "chunk", "coord_reduced", "temperature_K", "count"],
    )
    temperatures = sorted({float(row["temperature_K"]) for row in stage_rows})
    summary_rows: list[dict[str, Any]] = []
    for temp in temperatures:
        rows = [row for row in stage_rows if float(row["temperature_K"]) == temp]
        k_values = np.asarray([row["k_W_mK"] for row in rows], dtype=float)
        valid = np.isfinite(k_values) & (k_values > 0)
        ok = k_values[valid]
        summary_rows.append(
            {
                "temperature_K": temp,
                "k_mean_W_mK": float(np.mean(ok)) if ok.size else math.nan,
                "k_std_W_mK": float(np.std(ok, ddof=1)) if ok.size > 1 else 0.0,
                "seed_count": len(rows),
                "ok_seed_count": int(ok.size),
                "seed_cv_fraction": float(np.std(ok, ddof=1) / np.mean(ok)) if ok.size > 1 and np.mean(ok) else 0.0,
                "slope_disagreement_mean_fraction": float(
                    np.mean([row["slope_disagreement_fraction"] for row in rows])
                ),
            }
        )
    _write_csv(
        outdir / "thermal_conductivity_rnemd_T.csv",
        summary_rows,
        [
            "temperature_K",
            "k_mean_W_mK",
            "k_std_W_mK",
            "seed_count",
            "ok_seed_count",
            "seed_cv_fraction",
            "slope_disagreement_mean_fraction",
        ],
    )
    result = {
        "config": str(config),
        "outdir": str(outdir),
        "seed_summary": str(outdir / "rnemd_seed_summary.csv"),
        "temperature_summary": str(outdir / "thermal_conductivity_rnemd_T.csv"),
        "profile_summary": str(outdir / "rnemd_temperature_profiles.csv"),
        "n_stages": len(stage_rows),
        "n_temperatures": len(summary_rows),
    }
    write_json(outdir / "rnemd_analysis_summary.json", result)
    print(f"Wrote rNEMD seed summary: {result['seed_summary']}")
    print(f"Wrote rNEMD k(T): {result['temperature_summary']}")
    for row in summary_rows:
        print(
            f"T={row['temperature_K']:g} K  k={row['k_mean_W_mK']:.4g} W/m/K  "
            f"ok_seeds={row['ok_seed_count']}/{row['seed_count']}  "
            f"slope_mismatch={row['slope_disagreement_mean_fraction']:.1%}"
        )
    return result


def validate_main(args: argparse.Namespace) -> dict[str, Any]:
    fit_dir = args.fit_dir.resolve()
    summary_path = fit_dir / "thermal_conductivity_rnemd_T.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run thermal-rnemd-lammps analyze first; missing {summary_path}")
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    reports: list[dict[str, Any]] = []
    for row in rows:
        warnings: list[str] = []
        status = "pass"
        ok_seed_count = int(float(row["ok_seed_count"]))
        seed_count = int(float(row["seed_count"]))
        k_value = float(row["k_mean_W_mK"])
        slope_mismatch = float(row["slope_disagreement_mean_fraction"])
        seed_cv = float(row["seed_cv_fraction"])
        if ok_seed_count < args.min_seeds:
            warnings.append(f"Only {ok_seed_count}/{seed_count} valid seed(s); target at least {args.min_seeds}.")
            status = "warn"
        if not math.isfinite(k_value) or k_value <= 0:
            warnings.append("Mean thermal conductivity is not positive/finite.")
            status = "fail"
        if slope_mismatch >= args.slope_disagreement_fail_fraction:
            warnings.append("The two mirrored temperature-gradient slopes disagree strongly.")
            status = "fail"
        elif slope_mismatch >= args.slope_disagreement_warn_fraction and status != "fail":
            warnings.append("The two mirrored temperature-gradient slopes disagree; inspect profile linearity.")
            status = "warn"
        if seed_cv >= args.seed_cv_fail_fraction:
            warnings.append("Seed-to-seed spread is very large.")
            status = "fail"
        elif seed_cv >= args.seed_cv_warn_fraction and status != "fail":
            warnings.append("Seed-to-seed spread is large.")
            status = "warn"
        reports.append(
            {
                "temperature_K": float(row["temperature_K"]),
                "status": status,
                "k_W_mK": k_value,
                "ok_seed_count": ok_seed_count,
                "seed_count": seed_count,
                "seed_cv_fraction": seed_cv,
                "slope_disagreement_fraction": slope_mismatch,
                "warnings": warnings,
            }
        )
    output = {"fit_dir": str(fit_dir), "reports": reports}
    json_out = args.json_out or (fit_dir / "rnemd_validation_summary.json")
    write_json(json_out, output)
    print(f"Wrote rNEMD validation JSON: {json_out}")
    print("rNEMD Validation Summary")
    print("------------------------")
    for report in reports:
        print(
            f"T={report['temperature_K']:g} K  status={report['status']}  "
            f"k={report['k_W_mK']:.4g} W/m/K  ok_seeds={report['ok_seed_count']}/{report['seed_count']}  "
            f"slope_mismatch={report['slope_disagreement_fraction']:.1%}  "
            f"seed_cv={report['seed_cv_fraction']:.1%}"
        )
        for warning in report["warnings"]:
            print(f"  - {warning}")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermal-rnemd-lammps",
        description="Prepare reverse NEMD LAMMPS thermal-conductivity workflows from completed NPT data.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="Write reverse-NEMD inputs from completed NPT stages.")
    source = prep.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", nargs="+", help="One or more LAMMPS MD engine config JSON files.")
    source.add_argument("--md-root", type=Path, help="MD engine root; NPT folders are scanned and NVT folders ignored.")
    prep.add_argument("--template-config", type=Path, help="Template config to provide model/wrapper/mass settings for --md-root.")
    prep.add_argument("--config-dir")
    prep.add_argument("--config-glob", default="*.json")
    prep.add_argument("--duplicate-policy", choices=["highest_config_order", "first", "error"], default="highest_config_order")
    prep.add_argument("--outdir", type=Path, default=Path("analysis/rnemd_lammps"))
    prep.add_argument("--config-out", type=Path, default=Path("config_rnemd.json"))
    prep.add_argument("--T-min", dest="T_min", type=float)
    prep.add_argument("--T-max", dest="T_max", type=float)
    prep.add_argument("--temperature-start", type=float, default=300.0)
    prep.add_argument("--temperature-step", type=float, default=200.0)
    prep.add_argument("--temperature-tol", type=float, default=1.0)
    prep.add_argument(
        "--temperature-grid",
        dest="include_all_temperatures",
        action="store_false",
        help="Use --temperature-start/step grid instead of all discovered NPT temperatures.",
    )
    prep.set_defaults(include_all_temperatures=True)
    prep.add_argument("--n-seeds", type=int, default=1)
    prep.add_argument("--seed-start", type=int, default=71001)
    prep.add_argument("--seed-step", type=int, default=17)
    prep.add_argument("--seed", action="append", help="Explicit velocity seed. Repeat or comma-separate.")
    prep.add_argument("--run-time-ps", type=float, default=20.0, help="NVE/rNEMD production time after replication.")
    prep.add_argument("--timestep-ps", type=float, help="Override timestep in ps. Default reads template config/env.")
    prep.add_argument(
        "--replicate",
        default=DEFAULT_RNEMD_REPEAT,
        help=f"LAMMPS replicate factors for the NPT-ready data. Default: {DEFAULT_RNEMD_REPEAT}.",
    )
    prep.add_argument("--direction", choices=("x", "y", "z"), default="z", help="Heat-flow direction for swaps. Default: z.")
    prep.add_argument("--nbin", type=int, default=20, help="Even number of layers along --direction. Default: 20.")
    prep.add_argument("--swap-every", type=int, default=100, help="Perform kinetic-energy swaps every N steps. Default: 100.")
    prep.add_argument("--swap-count", type=int, default=1, help="LAMMPS thermal/conductivity swap count. Default: 1.")
    prep.add_argument("--profile-nevery", type=int, default=100, help="fix ave/chunk Nevery for slab temperatures.")
    prep.add_argument("--profile-nrepeat", type=int, default=100, help="fix ave/chunk Nrepeat for slab temperatures.")
    prep.add_argument("--profile-nfreq", type=int, help="fix ave/chunk Nfreq. Default: profile-nevery * profile-nrepeat.")
    prep.add_argument("--thermo-every", type=int, default=1000)
    prep.add_argument("--dump-every", type=int, default=0, help="Dump atom snapshots every N steps; 0 disables dumps. Default: 0.")
    prep.add_argument("--suffix", choices=("kk", "off", "none"), default="kk")
    prep.add_argument("--model-file", type=Path, help="Override model file for rNEMD stages.")
    prep.add_argument(
        "--pair-style-backend",
        choices=("mace",),
        help="Pair-style backend written to rNEMD inputs. rNEMD is kept on the old fast MACE/Kokkos route.",
    )
    prep.add_argument("--model-elements", nargs="+", help="Element/type order for pair_coeff, e.g. O U. Commas are accepted.")
    prep.add_argument("--array-limit", type=int, default=3, help="Slurm array concurrency. Default: 3.")
    prep.add_argument("--job-name", default="rnemd-array")
    prep.add_argument("--walltime-hours", type=float, help="Optional walltime override for every rNEMD stage.")
    prep.add_argument(
        "--rnemd-steps-per-hour",
        type=float,
        help=(
            "Observed throughput for the replicated rNEMD cell. Also read from "
            "ATOMI_LAMMPS_RNEMD_STEPS_PER_HOUR, which confighpc can export from "
            "profiles.lammps_rnemd.performance.steps_per_hour."
        ),
    )
    prep.add_argument(
        "--rnemd-walltime-safety-factor",
        type=float,
        default=None,
        help=(
            "Safety multiplier for throughput estimates. Default: 1.5 or "
            "ATOMI_LAMMPS_RNEMD_WALLTIME_SAFETY_FACTOR."
        ),
    )
    prep.add_argument(
        "--rnemd-reference-atoms",
        type=int,
        help="Base atom count for scaling template performance by replicated atom count.",
    )
    prep.add_argument(
        "--rnemd-target-atoms",
        type=int,
        help="Replicated atom count for scaling template performance. Default: reference_atoms * replicate factor.",
    )
    prep.add_argument(
        "--prefer-restart",
        action="store_true",
        help="Use NPT restart files even when matching data files exist. Default prefers data files.",
    )

    ana = sub.add_parser("analyze", help="Fit rNEMD temperature profiles and imposed heat flux into k(T).")
    ana.add_argument("--config", type=Path, default=Path("config_rnemd.json"))
    ana.add_argument("--outdir", type=Path, default=Path("analysis/rnemd_lammps/fit"))
    ana.add_argument("--profile-tail-fraction", type=float, default=0.5)
    ana.add_argument("--thermo-tail-fraction", type=float, default=0.5)

    val = sub.add_parser("validate", help="Summarize rNEMD fit quality and decision-rule warnings.")
    val.add_argument("--fit-dir", type=Path, default=Path("analysis/rnemd_lammps/fit"))
    val.add_argument("--json-out", type=Path, help="Optional validation report JSON.")
    val.add_argument("--min-seeds", type=int, default=3)
    val.add_argument("--slope-disagreement-warn-fraction", type=float, default=0.25)
    val.add_argument("--slope-disagreement-fail-fraction", type=float, default=0.50)
    val.add_argument("--seed-cv-warn-fraction", type=float, default=0.25)
    val.add_argument("--seed-cv-fail-fraction", type=float, default=0.50)

    status = sub.add_parser("status", help="Print live progress for one rNEMD chunk directory.")
    status.add_argument("chunk_dir", type=Path, nargs="?", default=Path("."))
    status.add_argument("--input", type=Path, help="Specific rNEMD input file.")
    status.add_argument("--log", type=Path, help="Specific rNEMD LAMMPS log file.")

    plot = sub.add_parser("plot", help="Plot live rNEMD thermo and slab-profile diagnostics in the terminal.")
    plot.add_argument("chunk_dir", type=Path, nargs="?", default=Path("."))
    plot.add_argument("--window", type=int, default=220, help="Thermo rows to show. Default: 220.")
    plot.add_argument(
        "--profile-tail-fraction",
        type=float,
        default=0.5,
        help="Fraction of complete profile blocks to average for the profile plot. Default: 0.5.",
    )
    return parser


def main(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    if args.command == "analyze":
        return analyze_main(args)
    if args.command == "validate":
        return validate_main(args)
    if args.command == "status":
        return print_rnemd_summary(args.chunk_dir.resolve(), input_file=args.input, log_file=args.log)
    if args.command == "plot":
        return plot_rnemd_once(
            args.chunk_dir,
            window=args.window,
            profile_tail_fraction=args.profile_tail_fraction,
        )
    parser.error(f"unknown command {args.command}")


def plot_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plotrnemd")
    parser.add_argument("chunk_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--window", type=int, default=220, help="Thermo rows to show. Default: 220.")
    parser.add_argument(
        "--profile-tail-fraction",
        type=float,
        default=0.5,
        help="Fraction of complete profile blocks to average for the profile plot. Default: 0.5.",
    )
    args = parser.parse_args(argv)
    plot_rnemd_once(
        args.chunk_dir,
        window=args.window,
        profile_tail_fraction=args.profile_tail_fraction,
    )


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
