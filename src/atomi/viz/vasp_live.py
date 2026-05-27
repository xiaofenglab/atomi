import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class DavTimingSummary:
    timing_file: Path
    dav_count: int
    timed_steps: int
    latest_seconds: float | None
    mean_seconds: float | None
    recent_mean_seconds: float | None


def count_dav_steps(path: Path) -> int:
    """Count VASP electronic steps written as DAV lines."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return sum(1 for line in handle if line.lstrip().startswith("DAV:"))
    except FileNotFoundError:
        return 0


def dav_timing_path(path: Path) -> Path:
    """Return the sidecar file used for live DAV timing observations."""
    return path.with_name(f"{path.name}.dav_timing.dat")


def update_dav_timing(
    path: Path,
    *,
    timing_file: Path | None = None,
    now: float | None = None,
    recent: int = 10,
) -> DavTimingSummary:
    """Update live DAV timing sidecar from observed DAV-count changes.

    VASP DAV lines do not carry timestamps. The monitor therefore measures only
    intervals observed while it is running: the first observed DAV count is the
    post-initialization baseline, and later count increases become per-DAV
    timing estimates.
    """
    timing_file = timing_file or dav_timing_path(path)
    now = time.time() if now is None else now
    dav_count = count_dav_steps(path)
    state_count, state_time, rows = _read_dav_timing(timing_file)
    if dav_count <= 0:
        _write_dav_timing(timing_file, dav_count, now, [])
        return _summarize_dav_timing(timing_file, dav_count, [], recent)
    if state_count is None or state_time is None or dav_count < state_count:
        _write_dav_timing(timing_file, dav_count, now, [])
        return _summarize_dav_timing(timing_file, dav_count, [], recent)
    if dav_count > state_count:
        increment = dav_count - state_count
        seconds_per_dav = max(0.0, now - state_time) / increment
        rows.extend((step, now, seconds_per_dav, increment) for step in range(state_count + 1, dav_count + 1))
        _write_dav_timing(timing_file, dav_count, now, rows)
    return _summarize_dav_timing(timing_file, dav_count, rows, recent)


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
                timing = update_dav_timing(output_file)
                _clear_terminal()
                _run_gnuplot(
                    [
                        f"file='{_gnuplot_quote(output_file)}'",
                        f'fileshell="{_gnuplot_double_quote_shell(output_file)}"',
                        f"timefile='{_gnuplot_quote(timing.timing_file)}'",
                        f'timefileshell="{_gnuplot_double_quote_shell(timing.timing_file)}"',
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
                timings = [update_dav_timing(path) for path in output_files]
                _clear_terminal()
                args = [f"nfiles={len(output_files)}", f"win={window}"]
                args.extend(
                    f"file{index}='{_gnuplot_quote(path)}'"
                    for index, path in enumerate(output_files, start=1)
                )
                args.extend(
                    f'fileshell{index}="{_gnuplot_double_quote_shell(path)}"'
                    for index, path in enumerate(output_files, start=1)
                )
                args.extend(
                    f"timefile{index}='{_gnuplot_quote(timing.timing_file)}'"
                    for index, timing in enumerate(timings, start=1)
                )
                args.extend(
                    f'timefileshell{index}="{_gnuplot_double_quote_shell(timing.timing_file)}"'
                    for index, timing in enumerate(timings, start=1)
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
    command = ["gnuplot", "-e", expression, str(script)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        message = [
            f"gnuplot failed with exit code {result.returncode}.",
            f"script: {script}",
            f"assignments: {expression}",
        ]
        if result.stderr:
            message.extend(["gnuplot stderr:", result.stderr.rstrip()])
        raise RuntimeError("\n".join(message))


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


def _gnuplot_double_quote_shell(path: Path) -> str:
    return shlex.quote(str(path)).replace("\\", "\\\\").replace('"', '\\"')


def _read_dav_timing(path: Path) -> tuple[int | None, float | None, list[tuple[int, float, float, int]]]:
    state_count: int | None = None
    state_time: float | None = None
    rows: list[tuple[int, float, float, int]] = []
    if not path.is_file():
        return state_count, state_time, rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 4 and parts[:2] == ["#", "state"]:
                try:
                    state_count = int(parts[2])
                    state_time = float(parts[3])
                except ValueError:
                    state_count = None
                    state_time = None
            elif len(parts) >= 4 and not line.startswith("#"):
                try:
                    rows.append((int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
                except ValueError:
                    continue
    return state_count, state_time, rows


def _write_dav_timing(path: Path, state_count: int, state_time: float, rows: list[tuple[int, float, float, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# atomi DAV timing from live plotvasp observation\n")
        handle.write("# first observed DAV count is a baseline; initialization time is excluded\n")
        handle.write(f"# state {state_count} {state_time:.6f}\n")
        handle.write("# columns: dav_step epoch seconds_per_dav observed_increment\n")
        for step, epoch, seconds, increment in rows:
            handle.write(f"{step:d} {epoch:.6f} {seconds:.6f} {increment:d}\n")


def _summarize_dav_timing(
    timing_file: Path,
    dav_count: int,
    rows: list[tuple[int, float, float, int]],
    recent: int,
) -> DavTimingSummary:
    seconds = [row[2] for row in rows]
    recent_seconds = seconds[-recent:] if recent > 0 else seconds
    return DavTimingSummary(
        timing_file=timing_file,
        dav_count=dav_count,
        timed_steps=len(seconds),
        latest_seconds=seconds[-1] if seconds else None,
        mean_seconds=sum(seconds) / len(seconds) if seconds else None,
        recent_mean_seconds=sum(recent_seconds) / len(recent_seconds) if recent_seconds else None,
    )
