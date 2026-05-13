import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata, util
from pathlib import Path
from typing import Any


CONFIG_ENV_VAR = "ATOMI_HPC_CONFIG"
LOCAL_CONFIG = Path("atomi_hpc_config.json")
USER_CONFIG = Path("~/.config/atomi/hpc.json").expanduser()

EXECUTABLES = {
    "core": ["python", "python3"],
    "scheduler": ["sbatch", "squeue", "srun", "qsub"],
    "visualization": ["gnuplot"],
    "engines": ["vasp_std", "vasp_gam", "vasp_ncl", "cp2k", "lmp", "lammps", "nvidia-smi"],
    "environment": ["module", "conda", "git", "cmake", "mpiexec", "mpirun", "mpicc", "mpicxx", "mpif90"],
}

HPC_PROBE_WHICH = [
    "sbatch",
    "qsub",
    "srun",
    "mpiexec",
    "mpirun",
    "mpicc",
    "mpicxx",
    "mpif90",
    "python3",
    "conda",
    "git",
    "cmake",
]

HPC_PROBE_COMMANDS = [
    {
        "key": "module_version_head",
        "command": "module --version 2>&1 | head",
    },
    {
        "key": "module_avail_gcc_head60",
        "command": "module avail gcc 2>&1 | head -60",
    },
    {
        "key": "module_avail_mpi_head80",
        "command": "module avail mpi 2>&1 | head -80",
    },
    {
        "key": "module_avail_cmake_head60",
        "command": "module avail cmake 2>&1 | head -60",
    },
    {
        "key": "python3_version",
        "command": "python3 --version",
    },
    {
        "key": "home_scratch_df",
        "command": 'df -h "$HOME" "${SCRATCH:-$HOME}" 2>/dev/null',
    },
]

PYTHON_PACKAGES = [
    "numpy",
    "ase",
    "matplotlib",
    "torch",
    "mace",
]

HPC_ASSUMPTIONS = [
    {
        "key": "gnuplot_on_path",
        "applies_to": ["plotvasp", "plotvasp4", "plotlammps", "plotcp2k", "plotmace"],
        "note": "Live terminal plotting requires gnuplot to be available on PATH, often by module load.",
    },
    {
        "key": "slurm_available",
        "applies_to": ["convertmace", "mace-convert-lammps"],
        "note": "Default MACE-to-LAMMPS conversion submits with sbatch on Slurm systems.",
    },
    {
        "key": "mace_lammps_environment",
        "applies_to": ["convertmace", "mace-convert-lammps"],
        "note": "The conversion needs a Python environment containing mace and its dependencies.",
    },
    {
        "key": "gpu_resource_names",
        "applies_to": ["convertmace", "mace-energy-outliers", "md-engine"],
        "note": "GPU partitions and gres strings vary by cluster.",
    },
    {
        "key": "lammps_md_engine_runtime",
        "applies_to": ["md-engine-init", "md-engine"],
        "note": "The MD engine needs Slurm sbatch/squeue, a LAMMPS executable, GPU modules, and MACE/LAMMPS runtime libraries configured for each HPC.",
    },
]


def find_config_path(explicit: Path | None = None) -> Path | None:
    """Return the first existing atomi HPC config path."""
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend([LOCAL_CONFIG, USER_CONFIG])
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_hpc_config(explicit: Path | None = None) -> dict[str, Any]:
    """Load an atomi HPC JSON config, returning an empty dict if none exists."""
    path = find_config_path(explicit)
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mace_lammps_defaults(config: dict[str, Any]) -> dict[str, str]:
    """Return MACE LAMMPS conversion defaults from config with portable fallbacks."""
    profile = config.get("profiles", {}).get("mace_lammps", {})
    return {
        "env_path": str(profile.get("env_path") or "~/m_lammps_env"),
        "partition": str(profile.get("partition") or "gpu"),
        "gres": str(profile.get("gres") or "gpu:1"),
        "time": str(profile.get("time") or "00:15:00"),
    }


def executable_report() -> dict[str, dict[str, dict[str, str | bool | None]]]:
    report = {}
    for group, names in EXECUTABLES.items():
        report[group] = {}
        for name in names:
            path = shutil.which(name)
            version = _executable_version(name) if path else None
            report[group][name] = {"available": path is not None, "path": path, "version": version}
    return report


def _executable_version(name: str) -> str | None:
    version_args = {
        "gnuplot": ["gnuplot", "--version"],
        "sbatch": ["sbatch", "--version"],
        "srun": ["srun", "--version"],
        "qsub": ["qsub", "--version"],
        "squeue": ["squeue", "--version"],
        "cp2k": ["cp2k", "--version"],
        "nvidia-smi": ["nvidia-smi"],
        "lmp": ["lmp", "-help"],
        "lammps": ["lammps", "-help"],
    }
    command = version_args.get(name)
    if command is None:
        return None
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        clean = line.strip()
        if clean:
            return clean[:200]
    return None


def _run_shell_probe(command: str, timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {"command": command, "returncode": None, "output": "bash not found"}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "command": command,
            "returncode": None,
            "timed_out": True,
            "output": output.strip(),
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "output": result.stdout.strip(),
    }


def hpc_probe_report() -> dict[str, Any]:
    """Return a shell-level HPC portability probe requested by other projects."""
    which_report = {}
    for name in HPC_PROBE_WHICH:
        which_report[name] = shutil.which(name)

    command_report = {}
    for item in HPC_PROBE_COMMANDS:
        command_report[item["key"]] = _run_shell_probe(item["command"])

    return {
        "hostname": platform.node(),
        "shell_argv0": _run_shell_probe('echo "$0"'),
        "shell_env": os.environ.get("SHELL", ""),
        "pwd": str(Path.cwd()),
        "which": which_report,
        "commands": command_report,
    }


def python_package_report() -> dict[str, dict[str, str | bool | None]]:
    report = {}
    for package in PYTHON_PACKAGES:
        available = util.find_spec(package) is not None
        version = None
        if available:
            try:
                version = metadata.version(package)
            except metadata.PackageNotFoundError:
                version = "unknown"
        report[package] = {"available": available, "version": version}
    return report


def build_report(include_hpc_probe: bool = False) -> dict[str, Any]:
    executables = executable_report()
    packages = python_package_report()
    report = {
        "schema_version": 1,
        "generated_by": "atomi doctor",
        "platform": {
            "hostname": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
        "environment": {
            "path": os.environ.get("PATH", ""),
            "pythonpath": os.environ.get("PYTHONPATH", ""),
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "modulepath": os.environ.get("MODULEPATH", ""),
        },
        "executables": executables,
        "python_packages": packages,
        "profiles": {
            "mace_lammps": {
                "env_path": "~/m_lammps_env",
                "partition": "gpu",
                "gres": "gpu:1",
                "time": "00:15:00",
            },
            "lammps_md_engine": {
                "env_path": "~/m_lammps_env",
                "partition": "gpu",
                "gres": "gpu:1",
                "modules": ["compiler/gnu", "mpi/openmpi", "numlib/mkl/2020.2", "devel/cuda/12.3"],
                "lammps_prefix": "~/projects/lammps/gup_run",
                "lammps_executable": "~/projects/lammps/gup_run/install/bin/lmp",
                "libtorch_lib": "~/projects/lammps/gup_run/src/libtorch-gpu/lib",
            }
        },
        "hpc_assumptions": HPC_ASSUMPTIONS,
    }
    if include_hpc_probe:
        report["hpc_probe"] = hpc_probe_report()
    return report


def print_summary(report: dict[str, Any]) -> None:
    print("atomi HPC doctor")
    print(f"Host: {report['platform']['hostname']}")
    print(f"Python: {report['platform']['python']} ({report['platform']['python_version']})")
    print("")
    print("Executables:")
    for group, entries in report["executables"].items():
        found = [name for name, item in entries.items() if item["available"]]
        missing = [name for name, item in entries.items() if not item["available"]]
        print(f"  {group}: found {', '.join(found) if found else 'none'}")
        for name in found:
            version = entries[name].get("version")
            if version:
                print(f"    {name}: {version}")
        if missing:
            print(f"  {group}: missing {', '.join(missing)}")
    print("")
    print("Python packages:")
    for name, item in report["python_packages"].items():
        status = item["version"] if item["available"] else "missing"
        print(f"  {name}: {status}")
    print("")
    print("Cluster-specific assumptions to review:")
    for item in report["hpc_assumptions"]:
        print(f"  {item['key']}: {item['note']}")
    if "hpc_probe" in report:
        probe = report["hpc_probe"]
        print("")
        print("HPC shell probe:")
        print(f"  hostname: {probe['hostname']}")
        print(f"  shell: {probe['shell_env'] or probe['shell_argv0'].get('output', 'unknown')}")
        print(f"  pwd: {probe['pwd']}")
        found = [name for name, path in probe["which"].items() if path]
        missing = [name for name, path in probe["which"].items() if not path]
        print(f"  which found: {', '.join(found) if found else 'none'}")
        if missing:
            print(f"  which missing: {', '.join(missing)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atomi doctor",
        description="Inspect an HPC environment and optionally write an atomi JSON config.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the full report to a JSON file.")
    parser.add_argument(
        "--hpc-probe",
        action="store_true",
        help="Run shell-level HPC probes: scheduler/MPI/compiler paths, module avail, python3, and df.",
    )
    args = parser.parse_args(argv)

    report = build_report(include_hpc_probe=args.hpc_probe)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.write}")
    elif args.json:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)


if __name__ == "__main__":
    main()
