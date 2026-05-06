import math
import subprocess
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from atomi.viz.vasp_live import ensure_gnuplot


@dataclass(frozen=True)
class LammpsThermoRow:
    step: float
    temp: float | None = None
    potential_energy: float | None = None
    total_energy: float | None = None
    pressure: float | None = None
    volume: float | None = None


@dataclass(frozen=True)
class LammpsThermoSummary:
    npoints: int
    last_step: float
    temp_avg: float | None
    pressure_avg: float | None
    volume_avg: float | None
    potential_energy_avg: float | None
    temp_std: float | None
    pressure_std: float | None
    volume_std: float | None
    relative_volume_std_percent: float | None
    volume_drift_percent: float | None


def read_thermo_rows(logfile: Path) -> list[LammpsThermoRow]:
    """Read numeric rows from LAMMPS thermo blocks that start with Step."""
    rows: list[LammpsThermoRow] = []
    in_block = False
    with logfile.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Step"):
                in_block = True
                continue
            if stripped.startswith("Loop time"):
                in_block = False
                continue
            if not in_block or not stripped:
                continue

            parts = stripped.split()
            if not parts or not _is_number(parts[0]):
                continue
            values = [_to_float(part) for part in parts]
            rows.append(
                LammpsThermoRow(
                    step=values[0],
                    temp=_get(values, 1),
                    potential_energy=_get(values, 2),
                    total_energy=_get(values, 3),
                    pressure=_get(values, 4),
                    volume=_get(values, 5),
                )
            )
    return rows


def summarize_thermo(rows: list[LammpsThermoRow], last_fraction: float = 0.5) -> LammpsThermoSummary:
    """Summarize LAMMPS thermo rows using the last fraction of samples."""
    if not rows:
        raise ValueError("No LAMMPS thermo rows were found.")
    if not 0 < last_fraction <= 1:
        raise ValueError("last_fraction must be between 0 and 1.")

    start = max(0, int(len(rows) * (1.0 - last_fraction)))
    window = rows[start:]
    temps = _series(window, "temp")
    pressures = _series(window, "pressure")
    volumes = _series(window, "volume")
    pes = _series(window, "potential_energy")
    volume_avg = _mean(volumes)
    volume_std = _std(volumes)

    return LammpsThermoSummary(
        npoints=len(rows),
        last_step=rows[-1].step,
        temp_avg=_mean(temps),
        pressure_avg=_mean(pressures),
        volume_avg=volume_avg,
        potential_energy_avg=_mean(pes),
        temp_std=_std(temps),
        pressure_std=_std(pressures),
        volume_std=volume_std,
        relative_volume_std_percent=_relative_percent(volume_std, volume_avg),
        volume_drift_percent=_volume_drift_percent(volumes),
    )


def format_summary(summary: LammpsThermoSummary) -> str:
    """Format a compact LAMMPS thermo summary for terminal output."""
    lines = [
        "LAMMPS thermo summary",
        "---------------------",
        f"Thermo points: {summary.npoints}",
        f"Latest step  : {_fmt(summary.last_step)}",
        "",
        "Averages over selected window",
        f"Temp         : {_fmt(summary.temp_avg)} K",
        f"Pressure     : {_fmt(summary.pressure_avg)}",
        f"Volume       : {_fmt(summary.volume_avg)} A^3",
        f"PotEng       : {_fmt(summary.potential_energy_avg)}",
        "",
        "Fluctuation/drift",
        f"Temp std     : {_fmt(summary.temp_std)} K",
        f"Pressure std : {_fmt(summary.pressure_std)}",
        f"Volume std   : {_fmt(summary.volume_std)} A^3",
        f"rel std(V)   : {_fmt(summary.relative_volume_std_percent)} %",
        f"V drift      : {_fmt(summary.volume_drift_percent)} %",
    ]
    return "\n".join(lines)


def plot_lammps_live(
    logfile: Path,
    window: int = 40,
    interval: float = 10.0,
    once: bool = False,
    stop_on_finish: bool = True,
) -> None:
    """Monitor a LAMMPS log file and redraw terminal thermo plots when rows change."""
    if window < 1:
        raise ValueError("window must be a positive integer.")
    if not logfile.is_file():
        raise FileNotFoundError(f"file not found: {logfile}")
    ensure_gnuplot()

    script = files("atomi").joinpath("viz", "gnuplot", "lammps_thermo.gp")
    previous_count = -1

    while True:
        rows = read_thermo_rows(logfile)
        current_count = len(rows)
        if current_count != previous_count or once:
            subprocess.run(["clear"], check=False)
            _print_live_header(logfile, rows, interval)
            _run_gnuplot(logfile, script, window)
            previous_count = current_count
        if once:
            return
        if stop_on_finish and _contains_loop_time(logfile):
            print("\nRun appears finished, found 'Loop time'. Exiting monitor.")
            return
        time.sleep(interval)


def _print_live_header(logfile: Path, rows: list[LammpsThermoRow], interval: float) -> None:
    print("============================================================")
    print(" Live LAMMPS thermo monitor")
    print(f" File    : {logfile}")
    print(f" Refresh : {interval:g}s")
    print(" Ctrl+C to stop")
    print("============================================================")
    if not rows:
        print("\nNo thermo data found yet.")
        return

    latest = rows[-1]
    print(f" Thermo points : {len(rows)}")
    print(f" Latest step   : {_fmt(latest.step)}")
    print(f" Latest T      : {_fmt(latest.temp)} K")
    print(f" Latest P      : {_fmt(latest.pressure)}")
    print(f" Latest V      : {_fmt(latest.volume)}")
    print(f" Latest PE     : {_fmt(latest.potential_energy)}")
    print("============================================================")
    print()


def _run_gnuplot(logfile: Path, script: Path, window: int) -> None:
    expression = f"file='{_gnuplot_quote(logfile)}'; win={window}"
    subprocess.run(["gnuplot", "-e", expression, str(script)], check=True)


def _contains_loop_time(logfile: Path) -> bool:
    with logfile.open(encoding="utf-8", errors="replace") as handle:
        return any(line.startswith("Loop time") for line in handle)


def _series(rows: list[LammpsThermoRow], field: str) -> list[float]:
    values = [getattr(row, field) for row in rows]
    return [value for value in values if value is not None]


def _get(values: list[float], index: int) -> float | None:
    return values[index] if index < len(values) else None


def _is_number(text: str) -> bool:
    return _to_float_or_none(text) is not None


def _to_float(text: str) -> float:
    return float(text)


def _to_float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = sum(values) / len(values)
    return math.sqrt(max(0.0, sum((value - avg) ** 2 for value in values) / len(values)))


def _relative_percent(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return 100.0 * value / reference


def _volume_drift_percent(volumes: list[float]) -> float | None:
    if len(volumes) < 2:
        return None
    avg = _mean(volumes)
    if avg in (None, 0):
        return None
    return 100.0 * abs(volumes[-1] - volumes[0]) / avg


def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace("'", "\\'")


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.6g}"
