"""Config-driven helpers for MOOSE application workflows."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from atomi.core.doctor import load_hpc_config


DEFAULT_PROFILE = "moose_gpu_kokkos"


def load_moose_profile(
    hpc_config: Path | None = None,
    profile_name: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Load one MOOSE profile from the local atomi HPC config."""
    config = load_hpc_config(hpc_config)
    profile = config.get("profiles", {}).get(profile_name)
    if not isinstance(profile, dict):
        available = sorted(config.get("profiles", {}).keys())
        available_text = ", ".join(available) if available else "none"
        raise SystemExit(
            f"MOOSE profile {profile_name!r} was not found. Available profiles: {available_text}"
        )
    return profile


def _shell_join(command: list[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(part) for part in command)


def _profile_command(profile: dict[str, Any], app: str | None = None) -> str:
    if app:
        return shlex.quote(app) + " --help"
    executable = profile.get("test_executable") or profile.get("executable")
    if executable:
        return shlex.quote(str(executable)) + " --help"
    return "moose-opt --help"


def activation_lines(profile: dict[str, Any]) -> list[str]:
    """Return shell lines that activate the configured MOOSE runtime."""
    lines = ["set -euo pipefail"]
    activation = profile.get("activation_script")
    if activation:
        lines.append(f"source {shlex.quote(str(activation))}")
        return lines

    module_commands = profile.get("module_commands") or []
    if module_commands:
        lines.extend(str(command) for command in module_commands)
    elif profile.get("modules"):
        lines.append("module purge")
        lines.extend(f"module load {shlex.quote(str(module))}" for module in profile["modules"])

    python_env = profile.get("python_env") or profile.get("env_path")
    if python_env:
        lines.append(f"source {shlex.quote(str(python_env))}/bin/activate")

    exports = profile.get("build_environment_exports") or {}
    for key, value in sorted(exports.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    if profile.get("mpi", {}).get("psm2_cuda"):
        lines.append(f"export PSM2_CUDA={shlex.quote(str(profile['mpi']['psm2_cuda']))}")
    return lines


def render_slurm_submit(
    profile: dict[str, Any],
    command: str | list[str] | None = None,
    *,
    job_name: str = "moose",
    output: str = "moose_%j.out",
    error: str = "moose_%j.err",
) -> str:
    """Render a portable Slurm submission script for one configured MOOSE profile."""
    run_command = _shell_join(command) if command else _profile_command(profile)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
    ]
    if profile.get("partition"):
        lines.append(f"#SBATCH --partition={profile['partition']}")
    if profile.get("gres"):
        lines.append(f"#SBATCH --gres={profile['gres']}")
    if profile.get("nodes"):
        lines.append(f"#SBATCH --nodes={profile['nodes']}")
    if profile.get("ntasks"):
        lines.append(f"#SBATCH --ntasks={profile['ntasks']}")
    if profile.get("cpus_per_task"):
        lines.append(f"#SBATCH --cpus-per-task={profile['cpus_per_task']}")
    if profile.get("time"):
        lines.append(f"#SBATCH --time={profile['time']}")
    if profile.get("mem"):
        lines.append(f"#SBATCH --mem={profile['mem']}")
    if profile.get("mem_per_cpu"):
        lines.append(f"#SBATCH --mem-per-cpu={profile['mem_per_cpu']}")

    lines.append("")
    lines.extend(activation_lines(profile))
    lines.append("")
    lines.append(run_command)
    return "\n".join(lines) + "\n"


def build_info(profile: dict[str, Any], profile_name: str) -> dict[str, Any]:
    """Return a compact profile summary suitable for JSON or text output."""
    return {
        "profile": profile_name,
        "status": profile.get("status"),
        "scheduler": profile.get("scheduler"),
        "partition": profile.get("partition"),
        "gres": profile.get("gres"),
        "modules": profile.get("modules", []),
        "activation_script": profile.get("activation_script"),
        "moose_root": profile.get("moose_root"),
        "test_executable": profile.get("test_executable"),
        "python_env": profile.get("python_env") or profile.get("env_path"),
        "calphad_work": profile.get("calphad_work"),
        "cuda": profile.get("cuda", {}),
        "next_tests": profile.get("next_tests", []),
    }


def run_smoke(profile: dict[str, Any], app: str | None = None, timeout: int = 30) -> dict[str, Any]:
    """Run a local MOOSE --help smoke command in the current environment."""
    command = shlex.split(_profile_command(profile, app=app))
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except OSError as exc:
        return {"command": command, "returncode": None, "output": str(exc)}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {"command": command, "returncode": None, "timed_out": True, "output": output}
    return {"command": command, "returncode": result.returncode, "output": result.stdout}


def info_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-info",
        description="Print MOOSE profile information from the local atomi HPC config.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    profile = load_moose_profile(args.hpc_config, args.profile)
    info = build_info(profile, args.profile)
    if args.json:
        print(json.dumps(info, indent=2))
        return
    print("Atomi MOOSE profile")
    for key in ("profile", "status", "scheduler", "partition", "gres"):
        if info.get(key):
            print(f"{key}: {info[key]}")
    if info.get("activation_script"):
        print(f"activation_script: {info['activation_script']}")
    if info.get("test_executable"):
        print(f"test_executable: {info['test_executable']}")
    if info.get("python_env"):
        print(f"python_env: {info['python_env']}")


def smoke_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-smoke",
        description="Run a local MOOSE executable --help smoke check.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--app", help="Override the configured executable.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = run_smoke(load_moose_profile(args.hpc_config, args.profile), args.app, args.timeout)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("$ " + " ".join(report["command"]))
    print(report.get("output", "").rstrip())
    if report.get("returncode") not in (0, None):
        raise SystemExit(report["returncode"])


def write_submit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="moose-write-submit",
        description="Write a Slurm MOOSE submission script from a local profile.",
    )
    parser.add_argument("--hpc-config", type=Path, help="Local atomi HPC JSON config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="MOOSE profile name.")
    parser.add_argument("--output", type=Path, default=Path("submit_moose.sh"))
    parser.add_argument("--job-name", default="moose")
    parser.add_argument("--command", help="Command to run after environment activation.")
    args = parser.parse_args(argv)

    profile = load_moose_profile(args.hpc_config, args.profile)
    script = render_slurm_submit(profile, command=args.command, job_name=args.job_name)
    args.output.write_text(script, encoding="utf-8")
    args.output.chmod(0o755)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    info_main()
