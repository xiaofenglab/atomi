import argparse
import sys
from pathlib import Path

from atomi.cli.vasp import extv
from atomi.core.project import create_project
from atomi.core.scheduler import render_submit_script
from atomi.viz.cp2k import plot_cp2k, plot_cp2k_all
from atomi.viz.lammps import format_summary, plot_lammps_live, read_thermo_rows, summarize_thermo
from atomi.viz.mace import plot_mace_live
from atomi.viz.vasp_live import plot_vasp_live, plot_vasp_live4
from atomi.ml.mace.datasets import main as mace_build_dataset_main


SUPPORTED_CODES = ("vasp", "cp2k", "lammps", "turbomole", "molcas")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atomi",
        description="HPC automation helpers for atomistic modeling.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    init_project = subparsers.add_parser(
        "init-project",
        help="Create a standardized calculation folder.",
    )
    init_project.add_argument("path", type=Path)
    init_project.add_argument("--code", choices=SUPPORTED_CODES, required=True)

    write_submit = subparsers.add_parser(
        "write-submit",
        help="Write a scheduler submission script from a reusable template.",
    )
    write_submit.add_argument("--scheduler", choices=("slurm", "pbs"), default="slurm")
    write_submit.add_argument("--profile", default="generic_cpu")
    write_submit.add_argument("--output", type=Path, default=Path("submit.sh"))
    write_submit.add_argument("--job-name", default=None)
    write_submit.add_argument(
        "--command",
        dest="run_command",
        default=None,
        help="Run command, for example: vasp_std > vasp.out",
    )

    inspect = subparsers.add_parser(
        "inspect",
        help="Print a quick summary of a calculation folder.",
    )
    inspect.add_argument("path", type=Path, nargs="?", default=Path("."))

    vasp_live = subparsers.add_parser(
        "vasp-live",
        help="Live terminal plot for one VASP output file.",
    )
    vasp_live.add_argument("output_file", type=Path)
    vasp_live.add_argument("--window", type=int, default=100)
    vasp_live.add_argument("--interval", type=float, default=2.0)
    vasp_live.add_argument("--once", action="store_true", help="Draw once and exit.")

    vasp_live4 = subparsers.add_parser(
        "vasp-live4",
        help="Live terminal plot for one to four VASP output files.",
    )
    vasp_live4.add_argument("output_files", type=Path, nargs="+")
    vasp_live4.add_argument("--window", type=int, default=100)
    vasp_live4.add_argument("--interval", type=float, default=5.0)
    vasp_live4.add_argument("--once", action="store_true", help="Draw once and exit.")

    lammps_live = subparsers.add_parser(
        "lammps-live",
        help="Live terminal plot for a LAMMPS log file.",
    )
    lammps_live.add_argument("logfile", type=Path)
    lammps_live.add_argument("--window", type=int, default=40)
    lammps_live.add_argument("--interval", type=float, default=10.0)
    lammps_live.add_argument("--once", action="store_true", help="Draw once and exit.")
    lammps_live.add_argument(
        "--keep-going",
        action="store_true",
        help="Keep monitoring after a Loop time line appears.",
    )

    lammps_summary = subparsers.add_parser(
        "lammps-summary",
        help="Summarize LAMMPS thermo data from a log file.",
    )
    lammps_summary.add_argument("logfile", type=Path)
    lammps_summary.add_argument("--last-fraction", type=float, default=0.5)

    cp2k_live = subparsers.add_parser(
        "cp2k-live",
        help="Auto-detect CP2K MD/GEO logs and launch the terminal monitor.",
    )
    cp2k_live.add_argument("logfile", type=Path)
    cp2k_live.add_argument("xyzfile", type=Path, nargs="?")
    cp2k_live.add_argument("--mode", choices=("auto", "md", "geo"), default="auto")
    cp2k_live.add_argument("--window", type=int, default=300)
    cp2k_live.add_argument("--refresh", type=int, default=15)

    cp2k_all = subparsers.add_parser(
        "cp2k-all",
        help="Full CP2K GEO convergence dashboard.",
    )
    cp2k_all.add_argument("logfile", type=Path)

    mace_live = subparsers.add_parser(
        "mace-live",
        help="Live terminal plot for MACE training logs.",
    )
    mace_live.add_argument("logfile", type=Path)
    mace_live.add_argument("--window", type=int, default=100)
    mace_live.add_argument("--refresh", type=int, default=5)

    vasp_outcar = subparsers.add_parser(
        "vasp-outcar",
        help="Quick VASP OUTCAR summary.",
    )
    vasp_outcar.add_argument("outcar", type=Path, nargs="?", default=Path("OUTCAR"))

    mace_build_dataset = subparsers.add_parser(
        "mace-build-dataset",
        help="Build adaptive MACE train/validation extxyz datasets.",
    )
    mace_build_dataset.add_argument("dataset_args", nargs=argparse.REMAINDER)

    mace_energy_outliers = subparsers.add_parser(
        "mace-energy-outliers",
        help="Find high energy-error outliers for a MACE model.",
    )
    mace_energy_outliers.add_argument("outlier_args", nargs=argparse.REMAINDER)

    mace_update_outliers = subparsers.add_parser(
        "mace-update-outliers",
        help="Remove outlier frames and optionally append rerun extxyz frames.",
    )
    mace_update_outliers.add_argument("update_args", nargs=argparse.REMAINDER)

    mace_check_extxyz = subparsers.add_parser(
        "mace-check-extxyz",
        help="Check extxyz labels, composition, histograms, and optional REF-key rewriting.",
    )
    mace_check_extxyz.add_argument("check_args", nargs=argparse.REMAINDER)

    mace_vasp2extxyz = subparsers.add_parser(
        "mace-vasp2extxyz",
        help="Collect VASP run directories into an extxyz dataset.",
    )
    mace_vasp2extxyz.add_argument("convert_args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> None:
    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] == "mace-build-dataset":
        mace_build_dataset_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-energy-outliers":
        from atomi.ml.mace.outliers import main as mace_energy_outliers_main

        mace_energy_outliers_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-update-outliers":
        from atomi.ml.mace.update_outliers import main as mace_update_outliers_main

        mace_update_outliers_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-check-extxyz":
        from atomi.ml.mace.check_extxyz import main as mace_check_extxyz_main

        mace_check_extxyz_main(raw_args[1:])
        return
    if raw_args and raw_args[0] == "mace-vasp2extxyz":
        from atomi.ml.mace.vasp2extxyz import main as mace_vasp2extxyz_main

        mace_vasp2extxyz_main(raw_args[1:])
        return

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.subcommand == "init-project":
        created = create_project(path=args.path, code=args.code)
        print(f"Created {created}")
        return

    if args.subcommand == "write-submit":
        script = render_submit_script(
            scheduler=args.scheduler,
            profile_name=args.profile,
            job_name=args.job_name or args.output.parent.resolve().name,
            command=args.run_command or "echo 'Replace this command with your executable'",
        )
        args.output.write_text(script, encoding="utf-8")
        args.output.chmod(0o755)
        print(f"Wrote {args.output}")
        return

    if args.subcommand == "inspect":
        files = sorted(item.name for item in args.path.iterdir())
        print(f"Path: {args.path.resolve()}")
        print("Files:")
        for name in files:
            print(f"  {name}")
        return

    if args.subcommand == "vasp-live":
        plot_vasp_live(
            output_file=args.output_file,
            window=args.window,
            interval=args.interval,
            once=args.once,
        )
        return

    if args.subcommand == "vasp-live4":
        plot_vasp_live4(
            output_files=args.output_files,
            window=args.window,
            interval=args.interval,
            once=args.once,
        )
        return

    if args.subcommand == "lammps-live":
        plot_lammps_live(
            logfile=args.logfile,
            window=args.window,
            interval=args.interval,
            once=args.once,
            stop_on_finish=not args.keep_going,
        )
        return

    if args.subcommand == "lammps-summary":
        rows = read_thermo_rows(args.logfile)
        summary = summarize_thermo(rows, last_fraction=args.last_fraction)
        print(format_summary(summary))
        return

    if args.subcommand == "cp2k-live":
        plot_cp2k(
            logfile=args.logfile,
            xyzfile=args.xyzfile,
            mode=args.mode,
            window=args.window,
            refresh=args.refresh,
        )
        return

    if args.subcommand == "cp2k-all":
        plot_cp2k_all(args.logfile)
        return

    if args.subcommand == "mace-live":
        plot_mace_live(logfile=args.logfile, window=args.window, refresh=args.refresh)
        return

    if args.subcommand == "vasp-outcar":
        extv([str(args.outcar)])
        return

    if args.subcommand == "mace-build-dataset":
        mace_build_dataset_main(args.dataset_args)
        return

    if args.subcommand == "mace-energy-outliers":
        from atomi.ml.mace.outliers import main as mace_energy_outliers_main

        mace_energy_outliers_main(args.outlier_args)
        return

    if args.subcommand == "mace-update-outliers":
        from atomi.ml.mace.update_outliers import main as mace_update_outliers_main

        mace_update_outliers_main(args.update_args)
        return

    if args.subcommand == "mace-check-extxyz":
        from atomi.ml.mace.check_extxyz import main as mace_check_extxyz_main

        mace_check_extxyz_main(args.check_args)
        return

    if args.subcommand == "mace-vasp2extxyz":
        from atomi.ml.mace.vasp2extxyz import main as mace_vasp2extxyz_main

        mace_vasp2extxyz_main(args.convert_args)
        return


if __name__ == "__main__":
    main()
