import argparse
import shutil
import subprocess
from pathlib import Path

from atomi.core.doctor import load_hpc_config, mace_lammps_defaults


def find_default_model() -> Path | None:
    """Return the first .model file in the current directory, if any."""
    models = sorted(Path(".").glob("*.model"))
    return models[0] if models else None


def build_slurm_script(
    model: Path,
    env_path: Path,
    partition: str,
    gres: str,
    time_limit: str,
    model_format: str | None = None,
) -> str:
    """Build a Slurm script that converts a MACE model for LAMMPS."""
    format_arg = f" --format={model_format}" if model_format else ""
    return f"""#!/bin/bash
#SBATCH --job-name=mace_convert
#SBATCH --output=mace_convert.%j.out
#SBATCH --error=mace_convert.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --gres={gres}

set -euo pipefail

unset PYTHONPATH
source "{env_path}/bin/activate"

echo "Running in directory:"
pwd

echo "Model file: {model}"
echo "Format    : {model_format or 'default'}"

python -m mace.cli.create_lammps_model "{model}"{format_arg}

echo "Conversion finished"
"""


def convert_mace_model_local(model: Path, model_format: str | None = None) -> None:
    """Run MACE's LAMMPS model converter in the active environment."""
    command = ["python", "-m", "mace.cli.create_lammps_model", str(model)]
    if model_format:
        command.append(f"--format={model_format}")
    subprocess.run(command, check=True)


def submit_mace_conversion(
    model: Path,
    env_path: Path,
    partition: str,
    gres: str,
    time_limit: str = "00:15:00",
    model_format: str | None = None,
    dry_run: bool = False,
) -> None:
    """Submit a Slurm job for MACE to LAMMPS model conversion."""
    if not model.is_file():
        raise FileNotFoundError(f"model file not found: {model}")

    script = build_slurm_script(
        model=model,
        env_path=env_path,
        partition=partition,
        gres=gres,
        time_limit=time_limit,
        model_format=model_format,
    )
    if dry_run:
        print(script)
        return
    activate = env_path / "bin" / "activate"
    if not activate.is_file():
        raise FileNotFoundError(f"environment activate script not found: {activate}")
    if shutil.which("sbatch") is None:
        raise RuntimeError("sbatch was not found on PATH; use --local or run on a Slurm login node.")
    subprocess.run(["sbatch"], input=script, text=True, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="convertmace",
        description="Convert a trained MACE .model file to a LAMMPS .pt model.",
    )
    parser.add_argument("model", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--hpc-config",
        type=Path,
        help="Read cluster defaults from this atomi doctor JSON config.",
    )
    parser.add_argument("--env", type=Path)
    parser.add_argument("--partition")
    parser.add_argument("--gres")
    parser.add_argument("--time")
    parser.add_argument(
        "--format",
        dest="model_format",
        choices=("mliap",),
        help="Pass a MACE converter output format, e.g. mliap for pair_style mliap unified.",
    )
    parser.add_argument("--local", action="store_true", help="Run conversion in the active environment.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Slurm script without submitting.")
    args = parser.parse_args(argv)

    defaults = mace_lammps_defaults(load_hpc_config(args.hpc_config))
    env_default = defaults["env_path"]
    env_path = args.env or (Path(env_default).expanduser() if env_default else None)
    partition = args.partition or defaults["partition"]
    gres = args.gres or defaults["gres"]
    time_limit = args.time or defaults["time"]

    model = args.model or find_default_model()
    if model is None:
        raise SystemExit("No .model file found in this directory. Usage: convertmace modelname.model")

    if args.local:
        convert_mace_model_local(model, model_format=args.model_format)
        return

    missing = []
    if env_path is None:
        missing.append("--env or profiles.mace_lammps.env_path")
    if not partition:
        missing.append("--partition or profiles.mace_lammps.partition")
    if not gres:
        missing.append("--gres or profiles.mace_lammps.gres")
    if missing:
        raise SystemExit(
            "Missing cluster-specific conversion settings: "
            + ", ".join(missing)
            + ". Keep these in a private atomi_hpc_config.json or pass them explicitly."
        )

    submit_mace_conversion(
        model=model,
        env_path=env_path,
        partition=partition,
        gres=gres,
        time_limit=time_limit,
        model_format=args.model_format,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
