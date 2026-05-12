import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

from atomi.vasp.qha_summary import is_finite, selected_volume_dirs, summarize_volume


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_summary_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scan_volume_rows(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> list[dict]:
    dirs = selected_volume_dirs(args, parser)
    summary_args = argparse.Namespace(
        parent_pattern=args.parent_pattern,
        disp_pattern=args.disp_pattern,
        atoms_per_fu=args.atoms_per_fu,
    )
    return [summarize_volume(directory, summary_args) for directory in dirs]


def source_rows(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict]:
    if args.summary_csv:
        return load_summary_csv(args.summary_csv.resolve())
    return scan_volume_rows(args, parser)


def sort_key(row: dict, sort_by: str) -> tuple:
    if sort_by == "volume":
        return (as_float(row.get("volume_A3")), row.get("volume_folder", ""))
    if sort_by == "folder":
        return (row.get("volume_folder", ""),)
    scale = as_float(row.get("scale_factor"))
    if not is_finite(scale):
        scale = math.inf
    return (scale, row.get("volume_folder", ""))


def qha_rows(rows: list[dict], sort_by: str) -> list[dict]:
    valid = [
        row
        for row in rows
        if is_finite(row.get("volume_A3")) and is_finite(row.get("energy_eV"))
    ]
    valid.sort(key=lambda row: sort_key(row, sort_by))
    return valid


def resolve_thermal_path(row: dict, thermal_file: str) -> Path:
    root = row.get("root")
    if not root:
        raise ValueError(f"row for {row.get('volume_folder', '<unknown>')} has no root path")
    return (Path(root) / thermal_file).resolve()


def thermal_paths(
    args: argparse.Namespace,
    rows: list[dict],
    parser: argparse.ArgumentParser,
) -> list[Path]:
    if args.thermal_yaml:
        paths = [path.resolve() for path in args.thermal_yaml]
        if len(paths) != len(rows):
            parser.error(
                f"--thermal-yaml count ({len(paths)}) does not match valid E-V points "
                f"({len(rows)}). phonopy-qha needs one thermal_properties.yaml per row."
            )
        missing = [path for path in paths if not path.exists()]
        if missing:
            parser.error(
                "missing explicit thermal YAML: " + ", ".join(str(path) for path in missing)
            )
        return paths

    paths = [resolve_thermal_path(row, args.thermal_file) for row in rows]
    missing = [path for path in paths if not path.exists()]
    if missing:
        parser.error("missing thermal YAML: " + ", ".join(str(path) for path in missing))
    return paths


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path, base)
    except ValueError:
        return str(path)


def write_ev_dat(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            volume = as_float(row["volume_A3"])
            energy = as_float(row["energy_eV"])
            handle.write(f"{volume:.10f}  {energy:.10f}\n")


def write_manifest(path: Path, rows: list[dict], thermals: list[Path], outdir: Path) -> None:
    fields = ["volume_folder", "scale_factor", "volume_A3", "energy_eV", "thermal_yaml"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, thermal in zip(rows, thermals):
            writer.writerow(
                {
                    "volume_folder": row.get("volume_folder", ""),
                    "scale_factor": row.get("scale_factor", ""),
                    "volume_A3": row.get("volume_A3", ""),
                    "energy_eV": row.get("energy_eV", ""),
                    "thermal_yaml": rel_or_abs(thermal, outdir),
                }
            )


def write_run_script(
    path: Path,
    ev_path: Path,
    thermals: list[Path],
    outdir: Path,
    phonopy_module: str | None,
    phonopy_qha: str,
    plot_script: Path,
    plot_output_dir: str,
    plot_t_min: float | None,
    plot_t_max: float | None,
    plot_after_qha: bool,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        if phonopy_module:
            handle.write(f"module load {shlex.quote(phonopy_module)}\n\n")
        handle.write(f"{shlex.quote(phonopy_qha)} {shlex.quote(rel_or_abs(ev_path, outdir))} \\\n")
        for index, thermal in enumerate(thermals):
            suffix = " \\" if index < len(thermals) - 1 else ""
            handle.write(f"  {shlex.quote(rel_or_abs(thermal, outdir))}{suffix}\n")
        if plot_after_qha:
            command = [
                "python",
                rel_or_abs(plot_script, outdir),
                "--outdir",
                plot_output_dir,
            ]
            if plot_t_min is not None:
                command.extend(["--t-min", str(plot_t_min)])
            if plot_t_max is not None:
                command.extend(["--t-max", str(plot_t_max)])
            handle.write("\n")
            handle.write(" ".join(shlex.quote(part) for part in command) + "\n")
    path.chmod(0o755)


def write_sbatch_script(
    path: Path,
    run_script: Path,
    outdir: Path,
    job_name: str,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}

set -euo pipefail
mkdir -p logs
bash {shlex.quote(rel_or_abs(run_script, outdir))}
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def write_plot_script(path: Path) -> None:
    script = r'''
#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


PLOT_FILES = [
    ("volume-temperature.dat", "Volume vs temperature", "Temperature (K)", "Volume"),
    ("thermal_expansion.dat", "Thermal expansion", "Temperature (K)", "Thermal expansion"),
    ("gibbs-temperature.dat", "Gibbs free energy", "Temperature (K)", "Gibbs free energy"),
    (
        "helmholtz-temperature.dat",
        "Helmholtz free energy",
        "Temperature (K)",
        "Helmholtz free energy",
    ),
    ("entropy-temperature.dat", "Entropy", "Temperature (K)", "Entropy"),
    ("Cv-temperature.dat", "Cv", "Temperature (K)", "Cv"),
    ("Cp-temperature.dat", "Cp", "Temperature (K)", "Cp"),
    ("dsdv-temperature.dat", "dS/dV", "Temperature (K)", "dS/dV"),
    ("bulk_modulus-temperature.dat", "Bulk modulus", "Temperature (K)", "Bulk modulus"),
    ("gruneisen-temperature.dat", "Gruneisen parameter", "Temperature (K)", "Gruneisen parameter"),
    ("helmholtz-volume.dat", "Helmholtz free energy vs volume", "Volume", "Helmholtz free energy"),
    ("gibbs-volume.dat", "Gibbs free energy vs volume", "Volume", "Gibbs free energy"),
    ("Cv-volume.dat", "Cv vs volume", "Volume", "Cv"),
    ("Cp-temperature_polyfit.dat", "Cp polynomial fit", "Temperature (K)", "Cp"),
]


def read_numeric_table(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue
        if len(values) >= 2:
            rows.append(values)
    return rows


def filtered_rows(rows, t_min, t_max):
    if t_min is None and t_max is None:
        return rows
    kept = []
    for row in rows:
        x = row[0]
        if t_min is not None and x < t_min:
            continue
        if t_max is not None and x > t_max:
            continue
        kept.append(row)
    return kept


def plot_table(path, outdir, meta, t_min, t_max):
    import matplotlib.pyplot as plt

    rows = filtered_rows(read_numeric_table(path), t_min, t_max)
    if not rows:
        return None
    ncols = max(len(row) for row in rows)
    xs = [row[0] for row in rows if len(row) == ncols]
    series = []
    for col in range(1, ncols):
        ys = [row[col] for row in rows if len(row) == ncols]
        if len(ys) == len(xs):
            series.append((col, ys))
    if not series:
        return None

    plt.figure(figsize=(7.2, 4.8))
    for col, ys in series:
        label = "value" if len(series) == 1 else f"col{col + 1}"
        plt.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.4, label=label)
    if len(series) > 1:
        plt.legend(fontsize=8)
    plt.xlabel(meta[2])
    plt.ylabel(meta[3])
    plt.title(meta[1])
    plt.tight_layout()
    outpath = outdir / f"{path.stem}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    return outpath


def write_index(path, outputs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_dat", "plot_png"])
        for source, plot in outputs:
            writer.writerow([source.name, plot.name])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot phonopy-qha .dat outputs.")
    parser.add_argument("--qha-dir", type=Path, default=Path("."))
    parser.add_argument("--outdir", type=Path, default=Path("qha_plots"))
    parser.add_argument("--t-min", type=float, default=None)
    parser.add_argument("--t-max", type=float, default=None)
    args = parser.parse_args(argv)

    qha_dir = args.qha_dir.resolve()
    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = qha_dir / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for file_name, title, xlabel, ylabel in PLOT_FILES:
        path = qha_dir / file_name
        if not path.exists():
            continue
        plot = plot_table(path, outdir, (file_name, title, xlabel, ylabel), args.t_min, args.t_max)
        if plot is not None:
            outputs.append((path, plot))
            print(plot)
    write_index(outdir / "plot_index.csv", outputs)
    if not outputs:
        print("No supported phonopy-qha .dat files found with plottable numeric data.")


if __name__ == "__main__":
    main()
'''
    path.write_text(textwrap.dedent(script).lstrip(), encoding="utf-8")
    path.chmod(0o755)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-qha-run",
        description="Prepare e-v.dat and a phonopy-qha run script from QHA volume folders.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="Folder containing QHA volume subfolders.")
    source.add_argument(
        "--volume-folder",
        type=Path,
        action="append",
        help="Explicit QHA volume folder. Repeat once per volume.",
    )
    source.add_argument("--summary-csv", type=Path, help="CSV from vasp-qha-summary.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--volume-pattern", default="V*")
    parser.add_argument("--disp-pattern", default="disp-*")
    parser.add_argument("--parent-pattern", default="parent*")
    parser.add_argument("--atoms-per-fu", type=float, default=3.0)
    parser.add_argument("--thermal-file", default="thermal_properties.yaml")
    parser.add_argument(
        "--thermal-yaml",
        type=Path,
        action="append",
        help="Explicit thermal YAML in E-V order. Repeat once per E-V row.",
    )
    parser.add_argument("--phonopy-module", default=os.environ.get("ATOMI_PHONOPY_MODULE"))
    parser.add_argument("--phonopy-qha", default="phonopy-qha")
    parser.add_argument("--ev-file", default="e-v.dat")
    parser.add_argument("--script-name", default="run_phonopy_qha.sh")
    parser.add_argument("--sbatch-script-name", default="submit_phonopy_qha.sbatch")
    parser.add_argument("--plot-script-name", default="plot_qha_results.py")
    parser.add_argument("--plot-output-dir", default="qha_plots")
    parser.add_argument("--plot-t-min", type=float, default=None)
    parser.add_argument("--plot-t-max", type=float, default=None)
    parser.add_argument(
        "--plot-after-qha",
        action="store_true",
        help="Append a plotting step to the generated phonopy-qha shell script.",
    )
    parser.add_argument("--job-name", default="phonopy_qha")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--sort-by", choices=("scale", "volume", "folder"), default="scale")
    parser.add_argument("--execute", action="store_true", help="Run the generated shell script.")
    parser.add_argument("--submit", action="store_true", help="Run sbatch on the generated Slurm script.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.atoms_per_fu <= 0:
        parser.error("--atoms-per-fu must be positive")

    try:
        rows = qha_rows(source_rows(args, parser), args.sort_by)
    except ValueError as exc:
        parser.error(str(exc))
    if not rows:
        parser.error("no valid QHA E-V rows found")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    thermals = thermal_paths(args, rows, parser)
    ev_path = outdir / args.ev_file
    script_path = outdir / args.script_name
    sbatch_path = outdir / args.sbatch_script_name
    plot_script_path = outdir / args.plot_script_name
    manifest_path = outdir / "qha_inputs.csv"

    write_ev_dat(ev_path, rows)
    write_manifest(manifest_path, rows, thermals, outdir)
    write_plot_script(plot_script_path)
    write_run_script(
        script_path,
        ev_path,
        thermals,
        outdir,
        args.phonopy_module,
        args.phonopy_qha,
        plot_script_path,
        args.plot_output_dir,
        args.plot_t_min,
        args.plot_t_max,
        args.plot_after_qha,
    )
    write_sbatch_script(
        sbatch_path,
        script_path,
        outdir,
        args.job_name,
        args.time,
        args.cpus,
        args.mem,
    )

    print(ev_path)
    print(manifest_path)
    print(script_path)
    print(sbatch_path)
    print(plot_script_path)
    if args.execute:
        subprocess.run(["bash", str(script_path)], cwd=outdir, check=True)
    if args.submit:
        subprocess.run(["sbatch", str(sbatch_path)], cwd=outdir, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
