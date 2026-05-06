import argparse
import shutil
import subprocess
from pathlib import Path


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
) -> str:
    """Build a Slurm script that converts a MACE model for LAMMPS."""
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

python -m mace.cli.create_lammps_model "{model}"

echo "Conversion finished"
"""


def convert_mace_model_local(model: Path) -> None:
    """Run MACE's LAMMPS model converter in the active environment."""
    subprocess.run(["python", "-m", "mace.cli.create_lammps_model", str(model)], check=True)


def submit_mace_conversion(
    model: Path,
    env_path: Path,
    partition: str = "gpu",
    gres: str = "gpu:1",
    time_limit: str = "00:15:00",
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
    parser.add_argument("--env", type=Path, default=Path("~/m_lammps_env").expanduser())
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--time", default="00:15:00")
    parser.add_argument("--local", action="store_true", help="Run conversion in the active environment.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Slurm script without submitting.")
    args = parser.parse_args(argv)

    model = args.model or find_default_model()
    if model is None:
        raise SystemExit("No .model file found in this directory. Usage: convertmace modelname.model")

    if args.local:
        convert_mace_model_local(model)
        return

    submit_mace_conversion(
        model=model,
        env_path=args.env,
        partition=args.partition,
        gres=args.gres,
        time_limit=args.time,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
