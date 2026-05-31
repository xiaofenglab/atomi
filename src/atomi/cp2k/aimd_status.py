"""Summarize live or completed CP2K AIMD run folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomi.cp2k.aimd_common import (
    parse_cp2k_input,
    parse_energy_file,
    scan_cp2k_log,
    xyz_frame_summary,
)


def _latest(paths: list[Path]) -> Path | None:
    paths = [path for path in paths if path.exists()]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _find_input(run_dir: Path) -> Path | None:
    candidates = [
        path
        for path in run_dir.glob("*.inp")
        if not path.name.endswith("~") and ".used_" not in path.name
    ]
    used = [path for path in run_dir.glob("*.used_*.inp") if not path.name.endswith("~")]
    return _latest(candidates) or _latest(used)


def _find_log(run_dir: Path) -> Path | None:
    return _latest(list(run_dir.glob("*.log")) + list(run_dir.glob("*.log.gz")))


def _find_energy(run_dir: Path) -> Path | None:
    return _latest(list(run_dir.glob("*-1.ener")) + list(run_dir.glob("*.ener")))


def _find_trajectory(run_dir: Path) -> Path | None:
    pos = list(run_dir.glob("*-pos.xyz"))
    return _latest(pos or list(run_dir.glob("*.xyz")))


def _find_slurm(run_dir: Path) -> Path | None:
    return _latest(list(run_dir.glob("slurm.*.out")))


def summarize_run(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    input_path = _find_input(run_dir)
    log_path = _find_log(run_dir)
    energy_path = _find_energy(run_dir)
    trajectory_path = _find_trajectory(run_dir)
    slurm_path = _find_slurm(run_dir)

    input_info = parse_cp2k_input(input_path) if input_path else {}
    energy_info = parse_energy_file(energy_path) if energy_path else {}
    log_info = scan_cp2k_log(log_path) if log_path else {}
    trajectory_info = xyz_frame_summary(trajectory_path) if trajectory_path else {}

    target_steps = input_info.get("steps")
    latest_step = energy_info.get("latest_step") or log_info.get("last_step")
    percent = None
    if isinstance(target_steps, int) and target_steps > 0 and isinstance(latest_step, int):
        percent = 100.0 * latest_step / target_steps

    eta_seconds = None
    mean_step_seconds = energy_info.get("mean_step_seconds")
    if (
        isinstance(target_steps, int)
        and isinstance(latest_step, int)
        and isinstance(mean_step_seconds, float)
    ):
        eta_seconds = max(0.0, (target_steps - latest_step) * mean_step_seconds)

    return {
        "run_dir": str(run_dir),
        "input_file": str(input_path) if input_path else None,
        "log_file": str(log_path) if log_path else None,
        "energy_file": str(energy_path) if energy_path else None,
        "trajectory_file": str(trajectory_path) if trajectory_path else None,
        "slurm_file": str(slurm_path) if slurm_path else None,
        "input": input_info,
        "energy": energy_info,
        "log": log_info,
        "trajectory": trajectory_info,
        "latest_step": latest_step,
        "target_steps": target_steps,
        "percent_complete": percent,
        "eta_seconds": eta_seconds,
        "status": "finished"
        if log_info.get("finished")
        else "running_or_incomplete"
        if latest_step
        else "unknown",
    }


def _format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    seconds = int(round(float(value)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def print_summary(summary: dict[str, object]) -> None:
    print("CP2K AIMD status")
    print("----------------")
    print(f"run dir      : {summary['run_dir']}")
    print(f"input        : {summary['input_file'] or 'missing'}")
    print(f"log          : {summary['log_file'] or 'missing'}")
    print(f"energy       : {summary['energy_file'] or 'missing'}")
    print(f"trajectory   : {summary['trajectory_file'] or 'missing'}")
    print(f"status       : {summary['status']}")
    input_info = summary.get("input", {})
    if isinstance(input_info, dict):
        project = input_info.get("project")
        if project:
            print(f"project      : {project}")
        if input_info.get("ensemble") or input_info.get("temperature_K") or input_info.get("timestep_fs"):
            print(
                "md settings  : "
                f"ensemble={input_info.get('ensemble') or 'unknown'}, "
                f"T={input_info.get('temperature_K') or 'unknown'} K, "
                f"dt={input_info.get('timestep_fs') or 'unknown'} fs"
            )
        if input_info.get("colvars"):
            print(f"colvars      : {input_info['colvars']}")
        if input_info.get("restraints"):
            print(f"restraints   : {input_info['restraints']}")
    if summary.get("latest_step") is not None:
        percent = summary.get("percent_complete")
        percent_text = f" ({percent:.1f}%)" if isinstance(percent, float) else ""
        print(f"progress     : {summary['latest_step']}/{summary.get('target_steps') or '?'} steps{percent_text}")
    energy = summary.get("energy", {})
    if isinstance(energy, dict) and energy.get("latest_temperature_K") is not None:
        print(f"latest T     : {float(energy['latest_temperature_K']):.3g} K")
    trajectory = summary.get("trajectory", {})
    if isinstance(trajectory, dict) and trajectory.get("frame_count") is not None:
        print(f"frames       : {trajectory.get('frame_count')} saved")
    log = summary.get("log", {})
    if isinstance(log, dict):
        print(
            "log health   : "
            f"failures={log.get('failure_count', 0)}, warnings={log.get('warning_count', 0)}, "
            f"finished={bool(log.get('finished'))}"
        )
    if summary.get("eta_seconds") is not None:
        print(f"eta          : {_format_seconds(summary['eta_seconds'])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cp2k-aimd-status",
        description="Summarize CP2K AIMD input, log, energy, trajectory, and restart health.",
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--json-out", type=Path, help="Write machine-readable status JSON.")
    args = parser.parse_args(argv)

    summary = summarize_run(args.run_dir)
    print_summary(summary)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
