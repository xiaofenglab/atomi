import math
import re
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


@dataclass(frozen=True)
class LammpsRunProgress:
    timestep_ps: float | None
    current_steps: float
    expected_steps: float | None
    current_ps: float | None
    expected_ps: float | None
    percent: float | None


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


def summarize_lammps_run_progress(
    logfile: Path,
    rows: list[LammpsThermoRow],
    timestep_ps: float | None = None,
) -> LammpsRunProgress | None:
    """Summarize active LAMMPS run progress from echoed timestep/run commands."""
    if not rows:
        return None
    text = logfile.read_text(encoding="utf-8", errors="replace")
    timestep = float(timestep_ps) if timestep_ps is not None else _infer_timestep_ps(text)
    first_step, latest_step = _latest_thermo_block_step_span(logfile)
    if first_step is None or latest_step is None:
        first_step = rows[0].step
        latest_step = rows[-1].step
    current_steps = max(0.0, latest_step - first_step)
    expected_steps = _infer_latest_run_steps(text)
    current_ps = current_steps * timestep if timestep is not None else None
    expected_ps = expected_steps * timestep if timestep is not None and expected_steps is not None else None
    percent = None
    if expected_steps not in (None, 0):
        percent = 100.0 * min(max(current_steps / expected_steps, 0.0), 1.0)
    return LammpsRunProgress(
        timestep_ps=timestep,
        current_steps=current_steps,
        expected_steps=expected_steps,
        current_ps=current_ps,
        expected_ps=expected_ps,
        percent=percent,
    )


def plot_lammps_live(
    logfile: Path,
    window: int = 40,
    interval: float = 10.0,
    once: bool = False,
    stop_on_finish: bool = True,
    timestep_ps: float | None = None,
) -> None:
    """Monitor a LAMMPS log file and redraw terminal thermo plots when rows change."""
    if window < 1:
        raise ValueError("window must be a positive integer.")
    if not logfile.is_file():
        raise FileNotFoundError(f"file not found: {logfile}")
    ensure_gnuplot()

    script = files("atomi").joinpath("viz", "gnuplot", "lammps_thermo.gp")
    previous_count = -1

    try:
        while True:
            rows = read_thermo_rows(logfile)
            current_count = len(rows)
            if current_count != previous_count or once:
                subprocess.run(["clear"], check=False)
                _print_live_header(logfile, rows, interval, timestep_ps=timestep_ps)
                _run_gnuplot(logfile, script, window)
                previous_count = current_count
            if once:
                return
            if stop_on_finish and _contains_loop_time(logfile):
                print("\nRun appears finished, found 'Loop time'. Exiting monitor.")
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped LAMMPS live plot.")
        return


def _print_live_header(
    logfile: Path,
    rows: list[LammpsThermoRow],
    interval: float,
    *,
    timestep_ps: float | None = None,
) -> None:
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
    progress = summarize_lammps_run_progress(logfile, rows, timestep_ps=timestep_ps)
    if progress:
        print(f" Timestep      : {_fmt(progress.timestep_ps)} ps ({_fmt(_ps_to_fs(progress.timestep_ps))} fs)")
        if progress.expected_steps is not None:
            print(
                " MD progress   : "
                f"{_fmt(progress.current_steps)}/{_fmt(progress.expected_steps)} steps, "
                f"{_fmt(progress.current_ps)}/{_fmt(progress.expected_ps)} ps "
                f"({_fmt(progress.percent)}%)"
            )
        else:
            print(f" MD progress   : {_fmt(progress.current_steps)} steps, {_fmt(progress.current_ps)} ps")
    recent = summarize_recent_runtime_fraction(rows, fraction=0.2)
    if recent:
        print(" Last 20% avg")
        print(f"   T  : {_fmt_avg_error(recent['temp'], recent['temp_std'], recent['temp_std_percent'])} K")
        print(f"   P  : {_fmt_avg_error(recent['pressure'], recent['pressure_std'], recent['pressure_std_percent'])}")
        print(f"   V  : {_fmt_avg_error(recent['volume'], recent['volume_std'], recent['volume_std_percent'])}")
        print(
            "   PE : "
            f"{_fmt_avg_error(recent['potential_energy'], recent['potential_energy_std'], recent['potential_energy_std_percent'])}"
        )
        print(
            f" Avg window    : {int(recent['npoints'])} pts, "
            f"steps {_fmt(recent['step_min'])} to {_fmt(recent['step_max'])}"
        )
    print("============================================================")
    print()


def summarize_recent_runtime_fraction(rows: list[LammpsThermoRow], fraction: float = 0.2) -> dict[str, float | None]:
    """Average thermo values over the last fraction of elapsed step span."""
    window = _recent_runtime_rows(rows, fraction=fraction)
    if not window:
        return {}
    temps = _series(window, "temp")
    pressures = _series(window, "pressure")
    volumes = _series(window, "volume")
    potential_energies = _series(window, "potential_energy")
    temp_avg = _mean(temps)
    pressure_avg = _mean(pressures)
    volume_avg = _mean(volumes)
    potential_energy_avg = _mean(potential_energies)
    temp_std = _std(temps)
    pressure_std = _std(pressures)
    volume_std = _std(volumes)
    potential_energy_std = _std(potential_energies)
    return {
        "npoints": float(len(window)),
        "step_min": window[0].step,
        "step_max": window[-1].step,
        "temp": temp_avg,
        "temp_std": temp_std,
        "temp_std_percent": _relative_abs_percent(temp_std, temp_avg),
        "pressure": pressure_avg,
        "pressure_std": pressure_std,
        "pressure_std_percent": _relative_abs_percent(pressure_std, pressure_avg),
        "volume": volume_avg,
        "volume_std": volume_std,
        "volume_std_percent": _relative_abs_percent(volume_std, volume_avg),
        "potential_energy": potential_energy_avg,
        "potential_energy_std": potential_energy_std,
        "potential_energy_std_percent": _relative_abs_percent(potential_energy_std, potential_energy_avg),
    }


def _recent_runtime_rows(rows: list[LammpsThermoRow], fraction: float = 0.2) -> list[LammpsThermoRow]:
    if not rows:
        return []
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    first_step = rows[0].step
    last_step = rows[-1].step
    span = last_step - first_step
    if span > 0:
        cutoff = last_step - fraction * span
        window = [row for row in rows if row.step >= cutoff]
        return window or [rows[-1]]
    start = max(0, math.floor(len(rows) * (1.0 - fraction)))
    return rows[start:]


def _run_gnuplot(logfile: Path, script: Path, window: int) -> None:
    expression = f"file='{_gnuplot_quote(logfile)}'; win={window}"
    subprocess.run(["gnuplot", "-e", expression, str(script)], check=True)


def _contains_loop_time(logfile: Path) -> bool:
    with logfile.open(encoding="utf-8", errors="replace") as handle:
        return any(line.startswith("Loop time") for line in handle)


def _infer_timestep_ps(text: str) -> float | None:
    matches = re.findall(r"(?m)^\s*timestep\s+([0-9.eE+-]+)\b", text)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"(?m)^\s*Time step\s*:\s*([0-9.eE+-]+)\b", text)
    if matches:
        return float(matches[-1])
    return None


def _infer_latest_run_steps(text: str) -> float | None:
    matches = re.findall(r"(?m)^\s*run\s+([0-9]+)\b", text)
    return float(matches[-1]) if matches else None


def _latest_thermo_block_step_span(logfile: Path) -> tuple[float | None, float | None]:
    first_step: float | None = None
    latest_step: float | None = None
    in_block = False
    with logfile.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("Step"):
                in_block = True
                first_step = None
                latest_step = None
                continue
            if stripped.startswith("Loop time"):
                in_block = False
                continue
            if not in_block:
                continue
            parts = stripped.split()
            if not parts or not _is_number(parts[0]):
                continue
            step = float(parts[0])
            first_step = step if first_step is None else first_step
            latest_step = step
    return first_step, latest_step


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


def _relative_abs_percent(value: float | None, reference: float | None) -> float | None:
    if value is None or reference in (None, 0):
        return None
    return 100.0 * value / abs(reference)


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


def _ps_to_fs(value: float | None) -> float | None:
    return None if value is None else value * 1000.0


def _fmt_avg_error(avg: float | None, err: float | None, err_percent: float | None) -> str:
    return f"{_fmt(avg)} +/- {_fmt(err)} ({_fmt(err_percent)}%)"
