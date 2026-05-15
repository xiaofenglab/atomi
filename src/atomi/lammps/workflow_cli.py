import argparse
import shutil
from importlib.resources import files
from pathlib import Path

from atomi.lammps.workflow import main as workflow_main


TEMPLATE_FILES = {
    "run_workflow.sh": "run_workflow.sh",
    "run_lammps_gpu.sh": "run_lammps_gpu.sh",
    "config_equilibration_example.json": "config.json",
    "config_production_example.json": "config_production.json",
}


def init_workflow(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="md-engine-init",
        description="Copy LAMMPS MD engine templates into a project directory.",
    )
    parser.add_argument("path", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--force", action="store_true", help="Overwrite existing template files.")
    args = parser.parse_args(argv)

    root = args.path.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "stages").mkdir(exist_ok=True)
    (root / "analysis").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "structures").mkdir(exist_ok=True)

    template_root = files("atomi").joinpath("templates", "lammps_workflow")
    for source_name, target_name in TEMPLATE_FILES.items():
        target = root / target_name
        if target.exists() and not args.force:
            print(f"Keeping existing {target}")
            continue
        source = Path(str(template_root.joinpath(source_name)))
        shutil.copy2(source, target)
        if target.suffix == ".sh":
            target.chmod(0o755)
        print(f"Wrote {target}")


def run_workflow(argv: list[str] | None = None) -> None:
    workflow_main(argv)


def production_array(argv: list[str] | None = None) -> None:
    from atomi.lammps.production_array import main

    main(argv)
