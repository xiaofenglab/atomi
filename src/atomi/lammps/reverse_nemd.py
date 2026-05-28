"""Prepare reverse NEMD LAMMPS workflows from completed NPT stages."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from atomi.lammps.elastic import (
    discover_npt_records,
    find_restart_or_data,
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


DEFAULT_RNEMD_TIMESTEP_PS = 0.0001
DEFAULT_RNEMD_WALLTIME_SAFETY_FACTOR = 1.5


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
        raise ValueError(f"Expected a repeat like 1x3x3, got {value!r}.")
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
            "rNEMD uses the normal LAMMPS/MACE runtime profile, not the GK ML-IAP profile.",
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
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
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
        "line=$(awk -F '\\t' -v task=\"$TASK_ID\" 'NR > 1 && $1 == task {print; exit}' \"$MANIFEST\")",
        'if [ -z "$line" ]; then',
        '  echo "ERROR: no rNEMD manifest row for task ${TASK_ID}"',
        "  exit 2",
        "fi",
        "IFS=$'\\t' read -r task_id stage_name temperature seed source_npt_stage input_structure input_kind "
        "replicate direction run_time_ps run_steps walltime chunk_dir input_name <<< \"$line\"",
        'echo "Running rNEMD array task ${task_id}: ${stage_name} T=${temperature} K seed=${seed}"',
        'echo "chunk=${chunk_dir}"',
        'echo "input=${input_name}"',
        'echo "walltime=${walltime} run_steps=${run_steps} run_time_ps=${run_time_ps}"',
        'cd "$chunk_dir"',
        'export SLURM_SUBMIT_DIR="$chunk_dir"',
        './run_stage.sh "$input_name"',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")
    path.chmod(0o755)


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
    prep.add_argument("--replicate", default="1x3x3", help="LAMMPS replicate factors for the NPT-ready data. Default: 1x3x3.")
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
    prep.add_argument("--pair-style-backend", choices=("mace", "mliap"), help="Pair-style backend written to rNEMD inputs.")
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
    return parser


def main(argv: list[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_main(args)
    parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
