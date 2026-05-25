import argparse
from pathlib import Path

from atomi.viz.cp2k import plot_cp2k, plot_cp2k_all
from atomi.viz.gk import main as plot_gk_main
from atomi.viz.lammps import plot_lammps_live
from atomi.viz.mace import plot_mace_live
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
    parser.add_argument(
        "--timestep-ps",
        type=float,
        default=None,
        help="Override the LAMMPS timestep in ps, for example 0.0001 for 0.1 fs or 0.00025 for 0.25 fs.",
    )
    args = parser.parse_args(argv)

    plot_lammps_live(
        logfile=args.logfile,
        window=args.window,
        interval=args.interval,
        once=args.once,
        stop_on_finish=not args.keep_going,
        timestep_ps=args.timestep_ps,
    )


def plotgk(argv: list[str] | None = None) -> None:
    """Compatibility command: plotgk [chunk_dir|heatflux_hcacf.dat]."""
    plot_gk_main(argv)


def plotcp2k(argv: list[str] | None = None) -> None:
    """Compatibility command: plotcp2k cp2k.log [trajectory.xyz]."""
    parser = argparse.ArgumentParser(prog="plotcp2k")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("xyzfile", type=Path, nargs="?")
    parser.add_argument("--mode", choices=("auto", "md", "geo"), default="auto")
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--refresh", type=int, default=15)
    parser.add_argument(
        "--track-atom",
        type=int,
        help="Track one 1-based trajectory atom as a metal-distance trace.",
    )
    args = parser.parse_args(argv)

    plot_cp2k(
        logfile=args.logfile,
        xyzfile=args.xyzfile,
        mode=args.mode,
        window=args.window,
        refresh=args.refresh,
        track_atom=args.track_atom,
    )


def plotcp2kall(argv: list[str] | None = None) -> None:
    """Compatibility command: plotcp2kall cp2k_geoopt.log."""
    parser = argparse.ArgumentParser(prog="plotcp2kall")
    parser.add_argument("logfile", type=Path)
    args = parser.parse_args(argv)

    plot_cp2k_all(args.logfile)


def plotmace(argv: list[str] | None = None) -> None:
    """Compatibility command: plotmace mace_train.log [window_epochs] [refresh_seconds]."""
    parser = argparse.ArgumentParser(prog="plotmace")
    parser.add_argument("logfile", type=Path)
    parser.add_argument("window", type=int, nargs="?", default=100)
    parser.add_argument("refresh", type=int, nargs="?", default=5)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("always", "onchange"),
        default="always",
        help="Accepted for compatibility; packaged gnuplot handles live refresh.",
    )
    args = parser.parse_args(argv)

    plot_mace_live(logfile=args.logfile, window=args.window, refresh=args.refresh)


def _split_files_and_optional_window(items: list[str], default_window: int) -> tuple[list[str], int]:
    if len(items) >= 2 and items[-1].isdigit():
        return items[:-1], int(items[-1])
    return items, default_window
