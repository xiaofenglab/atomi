import shutil
import subprocess
import time
from importlib.resources import files
from pathlib import Path


def count_dav_steps(path: Path) -> int:
    """Count VASP electronic steps written as DAV lines."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.startswith("DAV:"))
    except FileNotFoundError:
        return 0


def ensure_gnuplot() -> None:
    """Raise a clear error if gnuplot is not available."""
    if shutil.which("gnuplot") is None:
        raise RuntimeError("gnuplot was not found on PATH; load/install gnuplot first.")


def plot_vasp_live(
    output_file: Path,
    window: int = 100,
    interval: float = 2.0,
    once: bool = False,
) -> None:
    """Monitor one VASP output file and redraw SCF convergence when it changes."""
    _validate_window(window)
    _validate_files([output_file])
    ensure_gnuplot()

    script = files("atomi").joinpath("viz", "gnuplot", "vasp_live.gp")
    previous_count = -1

    try:
        while True:
            current_count = count_dav_steps(output_file)
            if current_count != previous_count or once:
                _clear_terminal()
                _run_gnuplot(
                    [
                        f"file='{_gnuplot_quote(output_file)}'",
                        f"win={window}",
                    ],
                    script,
                )
                previous_count = current_count
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped VASP live plot.")
        return


def plot_vasp_live4(
    output_files: list[Path],
    window: int = 100,
    interval: float = 5.0,
    once: bool = False,
) -> None:
    """Monitor one to four VASP output files in a terminal multiplot."""
    _validate_window(window)
    if not 1 <= len(output_files) <= 4:
        raise ValueError("plot_vasp_live4 expects one to four output files.")
    _validate_files(output_files)
    ensure_gnuplot()

    script = files("atomi").joinpath("viz", "gnuplot", "vasp_live4.gp")
    previous_count = -1

    try:
        while True:
            current_count = sum(count_dav_steps(path) for path in output_files)
            if current_count != previous_count or once:
                _clear_terminal()
                args = [f"nfiles={len(output_files)}", f"win={window}"]
                args.extend(
                    f"file{index}='{_gnuplot_quote(path)}'"
                    for index, path in enumerate(output_files, start=1)
                )
                _run_gnuplot(args, script)
                previous_count = current_count
            if once:
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped VASP live plot.")
        return


def _run_gnuplot(assignments: list[str], script: Path) -> None:
    expression = "; ".join(assignments)
    subprocess.run(["gnuplot", "-e", expression, str(script)], check=True)


def _validate_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")


def _validate_window(window: int) -> None:
    if window < 1:
        raise ValueError("window must be a positive integer.")


def _clear_terminal() -> None:
    subprocess.run(["clear"], check=False)


def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")
