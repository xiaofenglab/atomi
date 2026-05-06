import argparse
from pathlib import Path

from atomi.viz.cp2k import plot_cp2k, plot_cp2k_all
from atomi.viz.lammps import plot_lammps_live
from atomi.viz.vasp_live import plot_vasp_live, plot_vasp_live4


def plotvasp(argv: list[str] | None = None) -> None:
    """Compatibility command: plotvasp vasp.out [window_steps]."""
    parser = argparse.ArgumentParser(prog="plotvasp")
    parser.add_argument("output_file", type=Path)
    parser.add_argument("window", type=int, nargs="?", default=100)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    plot_vasp_live(
        output_file=args.output_file,
        window=args.window,
        interval=args.interval,
        once=args.once,
    )


def plotvasp4(argv: list[str] | None = None) -> None:
    """Compatibility command: plotvasp4 file1 [file2 file3 file4] [window_steps]."""
    parser = argparse.ArgumentParser(prog="plotvasp4")
    parser.add_argument("items", nargs="+")
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)

    files, window = _split_files_and_optional_window(args.items, default_window=100)
    if not 1 <= len(files) <= 4:
        parser.error("expected one to four VASP output files, plus optional window_steps")

    plot_vasp_live4(
        output_files=[Path(item) for item in files],
        window=window,
        interval=args.interval,
        once=args.once,
    )


def plotlammps(argv: list[str] | None = None) -> None:
    """Compatibility command: plotlammps log.lammps [window_steps]."""
    parser = argparse.ArgumentParser(prog="plotlammps")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("window", type=int, nargs="?", default=40)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    plot_lammps_live(
        logfile=args.logfile,
        window=args.window,
        interval=args.interval,
        once=args.once,
        stop_on_finish=not args.keep_going,
    )


def plotcp2k(argv: list[str] | None = None) -> None:
    """Compatibility command: plotcp2k cp2k.log [trajectory.xyz]."""
    parser = argparse.ArgumentParser(prog="plotcp2k")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("xyzfile", type=Path, nargs="?")
    parser.add_argument("--mode", choices=("auto", "md", "geo"), default="auto")
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--refresh", type=int, default=15)
    args = parser.parse_args(argv)

    plot_cp2k(
        logfile=args.logfile,
        xyzfile=args.xyzfile,
        mode=args.mode,
        window=args.window,
        refresh=args.refresh,
    )


def plotcp2kall(argv: list[str] | None = None) -> None:
    """Compatibility command: plotcp2kall cp2k_geoopt.log."""
    parser = argparse.ArgumentParser(prog="plotcp2kall")
    parser.add_argument("logfile", type=Path)
    args = parser.parse_args(argv)

    plot_cp2k_all(args.logfile)


def _split_files_and_optional_window(items: list[str], default_window: int) -> tuple[list[str], int]:
    if len(items) >= 2 and items[-1].isdigit():
        return items[:-1], int(items[-1])
    return items, default_window
