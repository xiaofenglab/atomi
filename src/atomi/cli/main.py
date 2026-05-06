import argparse
from pathlib import Path

from atomi.core.project import create_project
from atomi.core.scheduler import render_submit_script
from atomi.viz.lammps import format_summary, plot_lammps_live, read_thermo_rows, summarize_thermo
from atomi.viz.vasp_live import plot_vasp_live, plot_vasp_live4


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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

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


if __name__ == "__main__":
    main()
