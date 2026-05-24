from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path

from atomi.lammps.workflow import (
    _apply_sbatch_resource_overrides,
    _resolve_stage_input,
    estimate_walltime,
    hours_to_slurm,
    load_config,
    lammps_wrapper_text,
    production_stage_selected,
    resolve_run_steps,
    run_production_stage,
    set_project_root,
    stage_temperature,
)


MANIFEST_FIELDS = [
    "task_id",
    "stage_name",
    "temperature_K",
    "run_steps",
    "walltime",
    "chunk_dir",
    "input_structure",
]


def _split_repeated_values(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def _parse_temperature_values(values: list[str]) -> set[float]:
    temperatures: set[float] = set()
    for item in _split_repeated_values(values):
        raw = item.rstrip("Kk")
        temperatures.add(float(raw))
    return temperatures


def _parse_stage_values(values: list[str]) -> set[str]:
    return set(_split_repeated_values(values))


def _parse_t_range(value: str | None) -> tuple[float | None, float | None]:
    if not value:
        return None, None
    parts = value.replace(",", ":").split(":")
    if len(parts) != 2:
        raise ValueError("--T-range must look like START:STOP, for example 300:1500")
    return float(parts[0].rstrip("Kk")), float(parts[1].rstrip("Kk"))


def _stage_completed(root: Path, stage: dict) -> bool:
    stage_name = stage["name"]
    stage_dir = root / "stages" / stage_name
    return (stage_dir / "PASS").exists() and (stage_dir / f"{stage_name}.restart").exists()


def select_production_stages(cfg: dict, args: argparse.Namespace, root: Path) -> list[dict]:
    exact_temperatures = _parse_temperature_values(args.temperature)
    stage_names = _parse_stage_values(args.stage)
    range_min, range_max = _parse_t_range(args.T_range)
    t_min = args.T_min if args.T_min is not None else range_min
    t_max = args.T_max if args.T_max is not None else range_max

    selected = []
    for stage in cfg.get("stages", []):
        if not production_stage_selected(stage):
            continue
        name = stage["name"]
        temperature = float(stage_temperature(stage))
        if stage_names and name not in stage_names:
            continue
        if exact_temperatures and not any(abs(temperature - target) < 1.0e-6 for target in exact_temperatures):
            continue
        if t_min is not None and temperature < float(t_min):
            continue
        if t_max is not None and temperature > float(t_max):
            continue
        if args.skip_completed and _stage_completed(root, stage):
            continue
        selected.append(stage)

    selected.sort(key=lambda item: (float(stage_temperature(item)), item["name"]))
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]
    return selected


def build_manifest_rows(root: Path, cfg: dict, stages: list[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for task_id, stage in enumerate(stages, start=1):
        stage_name = stage["name"]
        structure_path = _resolve_stage_input(stage)
        run_steps = resolve_run_steps(cfg, stage)
        walltime = hours_to_slurm(estimate_walltime(cfg, stage, run_steps))
        rows.append(
            {
                "task_id": str(task_id),
                "stage_name": stage_name,
                "temperature_K": f"{float(stage_temperature(stage)):g}",
                "run_steps": str(run_steps),
                "walltime": walltime,
                "chunk_dir": str((root / "stages" / stage_name / stage.get("chunk_name", "chunk_production")).resolve()),
                "input_structure": str(structure_path),
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _resolve_user_path(path: Path, root: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _active_sbatch_lines(wrapper_text: str) -> list[str]:
    lines = []
    excluded = {"job-name", "output", "error", "time", "array"}
    for line in wrapper_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#SBATCH "):
            continue
        option = stripped.removeprefix("#SBATCH ").split("=", 1)[0].split()[0].lstrip("-")
        if option not in excluded:
            lines.append(stripped)
    return lines


def write_array_script(
    path: Path,
    *,
    root: Path,
    config: Path,
    manifest: Path,
    cfg: dict,
    rows: list[dict[str, str]],
    array_limit: int | None,
    python_exe: str,
    job_name: str,
    walltime: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_text = lammps_wrapper_text(cfg)
    wrapper_text = _apply_sbatch_resource_overrides(wrapper_text, cfg)
    active_lines = _active_sbatch_lines(wrapper_text)
    task_count = len(rows)
    max_walltime = walltime or max((row["walltime"] for row in rows), default="01:00:00")
    array_spec = f"1-{task_count}"
    if array_limit and array_limit > 0:
        array_spec += f"%{array_limit}"

    gk_mliap = cfg.get("runtime_profile") == "lammps_gk_mliap" or cfg.get("pair_style_backend") == "mliap"
    gk_environment_lines: list[str] = []
    if gk_mliap:
        gk_environment_lines = [
            'if command -v confighpc >/dev/null 2>&1; then',
            '  if [ -n "${ATOMI_HPC_CONFIG:-}" ]; then',
            '    eval "$(confighpc --config "$ATOMI_HPC_CONFIG" --shell)"',
            '  elif [ -f "$HOME/atomi_hpc/atomi_hpc_config.kit.local.json" ]; then',
            '    eval "$(confighpc --config "$HOME/atomi_hpc/atomi_hpc_config.kit.local.json" --shell)"',
            '  fi',
            'fi',
            'export TORCH_DISABLE_ADDR2LINE="${TORCH_DISABLE_ADDR2LINE:-1}"',
            'if [ -n "${ATOMI_LAMMPS_GK_EXTRA_LD_LIBRARY_PATH:-}" ]; then',
            '  export LD_LIBRARY_PATH="${ATOMI_LAMMPS_GK_EXTRA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"',
            'fi',
            'echo "ATOMI_LMP_GK_EXE=${ATOMI_LMP_GK_EXE:-}"',
            'echo "ATOMI_LAMMPS_GK_ENV=${ATOMI_LAMMPS_GK_ENV:-}"',
            'echo "ATOMI_LAMMPS_GK_EXTRA_LD_LIBRARY_PATH=${ATOMI_LAMMPS_GK_EXTRA_LD_LIBRARY_PATH:-}"',
            '',
        ]

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=md_prod_array.%A_%a.out",
        "#SBATCH --error=md_prod_array.%A_%a.err",
        *active_lines,
        f"#SBATCH --time={max_walltime}",
        f"#SBATCH --array={array_spec}",
        "",
        "set -euo pipefail",
        'TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"',
        f"PYTHON_EXE={shlex.quote(python_exe)}",
        f"ROOT_DIR={shlex.quote(str(root))}",
        f"CONFIG={shlex.quote(str(config))}",
        f"MANIFEST={shlex.quote(str(manifest))}",
        "",
        *gk_environment_lines,
        'echo "Running md-engine production array task ${TASK_ID}"',
        '"${PYTHON_EXE}" -m atomi.lammps.production_array \\',
        '  --run-task \\',
        '  --task-id "${TASK_ID}" \\',
        '  --root "${ROOT_DIR}" \\',
        '  --config "${CONFIG}" \\',
        '  --manifest "${MANIFEST}" \\',
        "  --resume",
        "",
    ]
    path.write_text("\n".join(script_lines), encoding="utf-8")
    path.chmod(0o755)


def run_task(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    set_project_root(root)
    cfg = load_config(args.config)
    rows = read_manifest(args.manifest)
    matches = [row for row in rows if int(row["task_id"]) == int(args.task_id)]
    if not matches:
        raise SystemExit(f"No manifest task_id={args.task_id} in {args.manifest}")
    stage_name = matches[0]["stage_name"]
    stages = [stage for stage in cfg.get("stages", []) if stage["name"] == stage_name]
    if not stages:
        raise SystemExit(f"Manifest stage {stage_name} is not present in {cfg['_config_path']}")
    print(f"Running production array task {args.task_id}: {stage_name}", flush=True)
    run_production_stage(cfg, stages[0], resume_mode=args.resume, submit_mode=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="md-engine-array",
        description="Generate or run a Slurm array for independent md-engine production stages.",
    )
    parser.add_argument("--config", type=Path, default=Path("config_production.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--outdir", type=Path, default=Path("analysis/md_engine_array"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument("--stage", action="append", default=[], help="Production stage name. Repeat or comma-separate.")
    parser.add_argument("--T", "--temperature", dest="temperature", action="append", default=[], help="Temperature block in K. Repeat or comma-separate.")
    parser.add_argument("--T-min", dest="T_min", type=float, default=None)
    parser.add_argument("--T-max", dest="T_max", type=float, default=None)
    parser.add_argument("--T-range", default=None, help="Inclusive temperature range START:STOP, e.g. 300:1500.")
    parser.add_argument("--max-tasks", type=int, default=None, help="After filtering, keep only this many tasks.")
    parser.add_argument(
        "--array-limit",
        "--max-running",
        dest="array_limit",
        type=int,
        default=1,
        help="Maximum simultaneous Slurm array tasks. Default: 1.",
    )
    parser.add_argument("--include-completed", dest="skip_completed", action="store_false", help="Include stages already marked PASS.")
    parser.set_defaults(skip_completed=True)
    parser.add_argument("--walltime", default=None, help="Override one walltime for every array task.")
    parser.add_argument("--job-name", default="md-prod-array")
    parser.add_argument("--python", default=sys.executable, help="Python executable used inside the array job.")
    parser.add_argument("--submit", action="store_true", help="Submit the generated array script with sbatch.")
    parser.add_argument("--run-task", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--task-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.run_task:
        if args.task_id is None or args.manifest is None:
            parser.error("--run-task needs --task-id and --manifest")
        run_task(args)
        return

    root = args.root.resolve()
    set_project_root(root)
    cfg = load_config(args.config)
    config_path = Path(cfg["_config_path"]).resolve()
    outdir = _resolve_user_path(args.outdir, root)
    manifest = _resolve_user_path(args.manifest, root) if args.manifest else (outdir / "md_engine_array_manifest.tsv")
    script = _resolve_user_path(args.script, root) if args.script else (outdir / "run_md_production_array.sh")

    stages = select_production_stages(cfg, args, root)
    if not stages:
        parser.error("No production stages matched the requested selection.")

    rows = build_manifest_rows(root, cfg, stages)
    write_manifest(manifest, rows)
    write_array_script(
        script,
        root=root,
        config=config_path,
        manifest=manifest,
        cfg=cfg,
        rows=rows,
        array_limit=args.array_limit,
        python_exe=args.python,
        job_name=args.job_name,
        walltime=args.walltime,
    )

    print(f"Selected {len(rows)} production stage(s):")
    for row in rows:
        print(f"  task {row['task_id']}: {row['stage_name']} ({row['temperature_K']} K)")
    print(f"Wrote manifest: {manifest}")
    print(f"Wrote Slurm array script: {script}")
    print(f"Array concurrency limit: {args.array_limit}")
    if args.submit:
        out = subprocess.check_output(["sbatch", str(script)], cwd=root).decode().strip()
        print(out)
    else:
        print(f"Submit with: sbatch {script}")


if __name__ == "__main__":
    main()
