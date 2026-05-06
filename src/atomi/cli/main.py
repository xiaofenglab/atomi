import argparse
from pathlib import Path

from atomi.core.project import create_project
from atomi.core.scheduler import render_submit_script
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


if __name__ == "__main__":
    main()
