import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import textwrap
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
    "engines": ["vasp_std", "vasp_gam", "vasp_ncl", "cp2k", "lmp", "lammps"],
    "gpu": ["nvidia-smi", "nvcc"],
    "moose": ["moose-opt", "moose-dbg", "moose-devel", "moose_test-opt"],
    "environment": [
        "module",
        "conda",
        "git",
        "cmake",
        "mpiexec",
        "mpirun",
        "mpicc",
        "mpicxx",
        "mpif90",
    ],
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
    "nvidia-smi",
    "nvcc",
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
        "key": "module_avail_cuda_head80",
        "command": "module avail cuda 2>&1 | head -80",
    },
    {
        "key": "module_avail_nvidia_head80",
        "command": "module avail nvidia 2>&1 | head -80",
    },
    {
        "key": "module_avail_gpu_head80",
        "command": "module avail gpu 2>&1 | head -80",
    },
    {
        "key": "module_list_head80",
        "command": "module list 2>&1 | head -80",
    },
    {
        "key": "nvidia_smi_list",
        "command": "nvidia-smi -L 2>&1",
    },
    {
        "key": "nvidia_smi_query",
        "command": "nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1",
    },
    {
        "key": "nvcc_version_head",
        "command": "nvcc --version 2>&1 | head",
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
    "pycalphad",
]

SENSITIVE_CONFIG_KEYS = {
    "basis_file",
    "data_dir",
    "d3_file",
    "env_path",
    "gres",
    "home_candidates",
    "lammps_executable",
    "lammps_prefix",
    "libtorch_lib",
    "micromamba_env",
    "micromamba_root",
    "module",
    "module_commands",
    "modules",
    "partition",
    "potential_file",
    "time",
}

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
    {
        "key": "moose_application_executable",
        "applies_to": ["moose-doctor"],
        "note": "MOOSE workflows usually run project-specific app executables; record the app-opt path and required compiler/MPI/PETSc modules per cluster.",
    },
    {
        "key": "pycalphad_database_paths",
        "applies_to": ["calphad-doctor"],
        "note": "CALPHAD workflows need pycalphad installed and explicit local paths to thermodynamic database files.",
    },
]

DISCOVERY_MODULE_KEYWORDS = [
    "gcc",
    "gnu",
    "intel",
    "openmpi",
    "mpi",
    "mkl",
    "cuda",
    "cmake",
    "python",
    "vasp",
    "lammps",
    "cp2k",
    "phonopy",
    "gnuplot",
]

DISCOVERY_STACK_ENV_VARS = {
    "vasp_cpu": "ATOMI_PROBE_VASP_MODULES",
    "lammps_gpu": "ATOMI_PROBE_LAMMPS_GPU_MODULES",
    "cp2k": "ATOMI_PROBE_CP2K_MODULES",
    "phonopy": "ATOMI_PROBE_PHONOPY_MODULES",
}


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


def hpc_config_report(explicit: Path | None = None, include_private_values: bool = False) -> dict[str, Any]:
    """Return a redacted summary of private HPC config, or exact values by request."""
    path = find_config_path(explicit)
    if path is None:
        return {"found": False, "path": str(explicit.expanduser()) if explicit else None}
    config = load_hpc_config(path)
    if include_private_values:
        return {"found": True, "path": str(path), "config": config}

    profiles = config.get("profiles", {})
    profile_summary = {}
    for name, profile in profiles.items():
        if not isinstance(profile, dict):
            profile_summary[name] = {"type": type(profile).__name__}
            continue
        sensitive = sorted(key for key in profile if key in SENSITIVE_CONFIG_KEYS)
        public_keys = sorted(key for key in profile if key not in SENSITIVE_CONFIG_KEYS)
        profile_summary[name] = {
            "keys": sorted(profile),
            "public_keys": public_keys,
            "private_keys_redacted": sensitive,
        }

    return {
        "found": True,
        "path": str(path),
        "site": config.get("site"),
        "schema_version": config.get("schema_version"),
        "profile_names": sorted(profiles),
        "profiles": profile_summary,
        "private_values": "redacted; pass --show-private-config only for local/private reports",
    }


def build_hpc_config_template(site: str = "") -> dict[str, Any]:
    """Return a private local HPC config template with no site-specific values."""
    return {
        "schema_version": 1,
        "site": site or "new_hpc",
        "privacy": "local-only; do not commit or push",
        "notes": [
            "Fill this file on the HPC after running atomi_hpc_discover.sh.",
            "Keep module names, executable paths, partitions, accounts, and user paths private.",
            "Set ATOMI_HPC_CONFIG to this file when running Atomi commands.",
        ],
        "profiles": {
            "vasp_cpu": {
                "scheduler": "slurm",
                "partition": "",
                "account": "",
                "modules": [],
                "module_commands": [],
                "executables": {
                    "vasp_std": "",
                    "vasp_gam": "",
                    "vasp_ncl": "",
                },
            },
            "lammps_md_engine": {
                "scheduler": "slurm",
                "partition": "",
                "gres": "",
                "account": "",
                "env_path": "",
                "modules": [],
                "module_commands": [],
                "lammps_executable": "",
                "lammps_prefix": "",
                "libtorch_lib": "",
                "environment": {
                    "ATOMI_LAMMPS_ENV": "",
                    "ATOMI_LAMMPS_MODULES": "",
                    "ATOMI_LMP_EXE": "",
                    "ATOMI_LAMMPS_PREFIX": "",
                    "PSM2_CUDA": "",
                },
            },
            "mace_lammps": {
                "env_path": "",
                "partition": "",
                "gres": "",
                "time": "00:15:00",
            },
            "cp2k": {
                "modules": [],
                "cp2k_executable": "",
                "data_dir": "",
                "basis_file": "",
                "potential_file": "",
                "d3_file": "",
                "environment": {
                    "ATOMI_CP2K_DATA_DIR": "",
                },
            },
            "phonopy": {
                "modules": [],
                "phonopy": "",
                "phonopy_load": "",
                "environment": {
                    "ATOMI_PHONOPY_MODULE": "",
                },
            },
            "pymol": {
                "modules": [],
                "pymol_executable": "",
                "ffmpeg_executable": "",
                "micromamba_root": "",
                "micromamba_env": "",
            },
            "moose": {
                "modules": [],
                "app_executable": "",
                "environment": {},
            },
            "calphad": {
                "python_env": "",
                "database_paths": [],
            },
        },
        "discovery": {
            "module_keyword_searches": DISCOVERY_MODULE_KEYWORDS,
            "stack_env_variables": DISCOVERY_STACK_ENV_VARS,
            "required_commands_by_profile": {
                "vasp_cpu": ["vasp_std", "vasp_gam", "vasp_ncl"],
                "lammps_gpu": ["lmp", "lammps", "mpicc", "mpicxx", "nvcc", "nvidia-smi"],
                "cp2k": ["cp2k"],
                "phonopy": ["phonopy", "phonopy-load"],
                "pymol": ["pymol", "ffmpeg"],
            },
        },
    }


def build_discovery_script() -> str:
    """Return a shell script that discovers local HPC module/runtime details."""
    keywords = " ".join(DISCOVERY_MODULE_KEYWORDS)
    stack_exports = "\n".join(
        f"#   export {env_var}=\"module_a module_b module_c\""
        for env_var in DISCOVERY_STACK_ENV_VARS.values()
    )
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        # Local-only Atomi HPC discovery helper.
        # Run this on a new HPC login node. For GPU checks, run again inside a GPU allocation.
        # The output may include private paths, module names, accounts, and usernames. Do not commit it.
        set -u

        OUT="${{1:-atomi_hpc_discovery.$(hostname).$(date +%Y%m%d_%H%M%S).log}}"
        exec > >(tee "$OUT") 2>&1
        INTERACTIVE="${{ATOMI_DISCOVERY_INTERACTIVE:-0}}"

        section() {{
            printf '\\n===== %s =====\\n' "$1"
        }}

        run() {{
            printf '\\n$ %s\\n' "$*"
            bash -lc "$*" || true
        }}

        show_which() {{
            printf '\\n$ which %s\\n' "$*"
            for exe in "$@"; do
                if command -v "$exe" >/dev/null 2>&1; then
                    printf '  %s -> %s\\n' "$exe" "$(command -v "$exe")"
                else
                    printf '  %s -> MISSING\\n' "$exe"
                fi
            done
        }}

        ask_stack() {{
            local label="$1"
            local current="$2"
            if [ "$INTERACTIVE" != "1" ] || [ ! -t 0 ]; then
                printf '%s' "$current"
                return
            fi
            printf '\\nInteractive selection for %s\\n' "$label" > /dev/tty
            if [ -n "$current" ]; then
                printf 'Current stack: %s\\n' "$current" > /dev/tty
                printf 'Press Enter to keep it, type a replacement stack, or type skip: ' > /dev/tty
            else
                printf 'Enter module stack to test, or press Enter/skip to skip: ' > /dev/tty
            fi
            local answer
            IFS= read -r answer < /dev/tty || answer=""
            if [ -z "$answer" ]; then
                printf '%s' "$current"
            elif [ "$answer" = "skip" ]; then
                printf ''
            else
                printf '%s' "$answer"
            fi
        }}

        probe_stack() {{
            local label="$1"
            local modules="$2"
            shift 2
            section "Module stack: $label"
            if ! command -v module >/dev/null 2>&1; then
                echo "module command is not available in this shell"
                return
            fi
            if [ -z "$modules" ]; then
                echo "No module stack provided; set the matching ATOMI_PROBE_*_MODULES variable to test exact loads."
                return
            fi
            module purge || true
            for mod in $modules; do
                echo "+ module load $mod"
                module load "$mod" || true
            done
            module list 2>&1 || true
            show_which "$@"
            run 'python3 --version'
            run 'gcc --version | head -3'
            run 'mpicc --version | head -3'
            run 'nvcc --version | head -6'
            run 'nvidia-smi -L'
        }}

        section "Host and scheduler"
        run 'hostname'
        run 'echo "$0"'
        run 'pwd'
        run 'python3 --version'
        show_which sbatch qsub srun squeue mpiexec mpirun mpicc mpicxx mpif90 python3 conda git cmake gnuplot vasp_std vasp_gam vasp_ncl cp2k lmp lammps nvcc nvidia-smi phonopy phonopy-load pymol ffmpeg
        run 'df -h "$HOME" "${{SCRATCH:-$HOME}}" 2>/dev/null'

        section "Loaded modules"
        run 'module --version 2>&1 | head'
        run 'module list 2>&1 | head -80'

        section "Module keyword discovery"
        for key in {keywords}; do
            run "module avail $key 2>&1 | head -80"
            run "module spider $key 2>&1 | head -80"
        done

        section "Python package discovery"
        run 'python3 - <<'"'"'PY'"'"'
        import importlib.util
        import sys
        from importlib import metadata
        print("python:", sys.executable)
        for name in ["numpy", "ase", "matplotlib", "torch", "mace", "pycalphad"]:
            ok = importlib.util.find_spec(name) is not None
            version = None
            if ok:
                try:
                    version = metadata.version(name)
                except metadata.PackageNotFoundError:
                    version = "unknown"
            print(f"{{name}}: {{version if ok else 'MISSING'}}")
        PY'

        section "Optional exact module-stack tests"
        echo "Set any of these private variables before running this script to test exact stacks:"
        {stack_exports}
        echo "For interactive choice after reviewing candidates, run with: ATOMI_DISCOVERY_INTERACTIVE=1 bash $0"
        VASP_STACK="$(ask_stack "VASP CPU" "${{ATOMI_PROBE_VASP_MODULES:-}}")"
        LAMMPS_GPU_STACK="$(ask_stack "LAMMPS GPU" "${{ATOMI_PROBE_LAMMPS_GPU_MODULES:-}}")"
        CP2K_STACK="$(ask_stack "CP2K" "${{ATOMI_PROBE_CP2K_MODULES:-}}")"
        PHONOPY_STACK="$(ask_stack "phonopy" "${{ATOMI_PROBE_PHONOPY_MODULES:-}}")"
        probe_stack "VASP CPU" "$VASP_STACK" vasp_std vasp_gam vasp_ncl
        probe_stack "LAMMPS GPU" "$LAMMPS_GPU_STACK" lmp lammps mpicc mpicxx nvcc nvidia-smi
        probe_stack "CP2K" "$CP2K_STACK" cp2k
        probe_stack "phonopy" "$PHONOPY_STACK" phonopy phonopy-load

        section "Atomi private config reminders"
        cat <<'EOF'
        Put confirmed local values into a private ignored JSON file, for example:
          ~/.config/atomi/hpc.json
          ./atomi_hpc_config.json
          ./atomi_hpc_config.<site>.local.json

        Useful private environment variables for generated scripts:
          ATOMI_HPC_CONFIG
          ATOMI_LAMMPS_ENV
          ATOMI_LAMMPS_MODULES
          ATOMI_LMP_EXE
          ATOMI_LAMMPS_PREFIX
          ATOMI_CP2K_DATA_DIR
          ATOMI_PHONOPY_MODULE
        EOF

        echo
        echo "Wrote $OUT"
        """
    ).lstrip()
    return "\n".join(line[8:] if line.startswith("        ") else line for line in script.splitlines()) + "\n"


def write_private_template(path: Path, site: str = "", overwrite: bool = False) -> None:
    """Write a local-only HPC config template."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_hpc_config_template(site=site), indent=2) + "\n", encoding="utf-8")


def write_discovery_script(path: Path, overwrite: bool = False) -> None:
    """Write the local-only HPC discovery shell script."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_discovery_script(), encoding="utf-8")
    path.chmod(0o755)


def mace_lammps_defaults(config: dict[str, Any]) -> dict[str, str]:
    """Return MACE LAMMPS conversion defaults from config with portable fallbacks."""
    profile = config.get("profiles", {}).get("mace_lammps", {})
    return {
        "env_path": str(profile.get("env_path") or ""),
        "partition": str(profile.get("partition") or ""),
        "gres": str(profile.get("gres") or ""),
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
        "nvcc": ["nvcc", "--version"],
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


def build_report(
    include_hpc_probe: bool = False,
    hpc_config_path: Path | None = None,
    include_private_config: bool = False,
) -> dict[str, Any]:
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
                "env_path": "",
                "partition": "",
                "gres": "",
                "time": "00:15:00",
                "note": "Set env_path, partition, and gres in private local config before submitting.",
            },
            "lammps_md_engine": {
                "env_path": "",
                "partition": "",
                "gres": "",
                "modules": [],
                "module_commands": ["module purge"],
                "lammps_prefix": "",
                "lammps_executable": "",
                "libtorch_lib": "",
                "gpu_checks": ["nvidia-smi -L", "nvcc --version", "mpicc --version"],
                "note": "Keep site-specific module names in a private atomi_hpc_config.json or ATOMI_LAMMPS_MODULES, not in the public package.",
            },
            "gpu_lammps": {
                "description": "GPU LAMMPS build/runtime module stack to verify on each HPC before installing or running.",
                "scheduler": "slurm",
                "partition": "",
                "gres": "",
                "modules": [],
                "module_commands": ["module purge"],
                "checks": ["which nvidia-smi nvcc mpicc", "nvidia-smi -L", "nvcc --version"],
                "note": "Populate modules privately per cluster; doctor only reports public generic fields.",
            }
        },
        "hpc_assumptions": HPC_ASSUMPTIONS,
    }
    if include_hpc_probe:
        report["hpc_probe"] = hpc_probe_report()
    config_path = find_config_path(hpc_config_path)
    if hpc_config_path is not None or config_path is not None:
        report["hpc_config"] = hpc_config_report(
            hpc_config_path,
            include_private_values=include_private_config,
        )
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
        gpu_list = probe["commands"].get("nvidia_smi_list", {}).get("output", "")
        if gpu_list:
            first_gpu_line = gpu_list.splitlines()[0]
            print(f"  gpu: {first_gpu_line[:160]}")
    if "hpc_config" in report:
        config = report["hpc_config"]
        print("")
        print("Private HPC config:")
        if not config["found"]:
            print(f"  not found: {config.get('path') or '<auto>'}")
        else:
            print(f"  path: {config['path']}")
            print(f"  site: {config.get('site') or '<unset>'}")
            names = config.get("profile_names", [])
            print(f"  profiles: {', '.join(names) if names else 'none'}")
            if not config.get("config"):
                print("  values: redacted")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="atomi doctor",
        description="Inspect an HPC environment and optionally write an atomi JSON config.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--write", type=Path, help="Write the full report to a JSON file.")
    parser.add_argument("--hpc-config", type=Path, help="Read a private local atomi HPC config.")
    parser.add_argument(
        "--write-config-template",
        type=Path,
        help="Write a private local HPC config template. Use a .local.json path and do not commit it.",
    )
    parser.add_argument(
        "--write-discovery-script",
        type=Path,
        help="Write a local shell script that probes module stacks and executable paths on a new HPC.",
    )
    parser.add_argument("--site", default="", help="Site label for --write-config-template.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated template/script outputs.")
    parser.add_argument(
        "--show-private-config",
        action="store_true",
        help="Include exact private HPC config values in JSON/write output. Do not share this report.",
    )
    parser.add_argument(
        "--hpc-probe",
        action="store_true",
        help="Run shell-level HPC probes: scheduler/MPI/compiler paths, module avail, python3, and df.",
    )
    args = parser.parse_args(argv)

    wrote_helper = False
    if args.write_config_template:
        write_private_template(args.write_config_template, site=args.site, overwrite=args.overwrite)
        print(f"Wrote private HPC config template {args.write_config_template}")
        wrote_helper = True
    if args.write_discovery_script:
        write_discovery_script(args.write_discovery_script, overwrite=args.overwrite)
        print(f"Wrote HPC discovery script {args.write_discovery_script}")
        wrote_helper = True
    if wrote_helper and not (args.json or args.write or args.hpc_probe or args.hpc_config):
        return

    report = build_report(
        include_hpc_probe=args.hpc_probe,
        hpc_config_path=args.hpc_config,
        include_private_config=args.show_private_config,
    )
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
