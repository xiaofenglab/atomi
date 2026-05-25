from __future__ import annotations

import argparse
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from atomi.viz.vasp_live import ensure_gnuplot


@dataclass(frozen=True)
class GKRunPlan:
    timestep_ps: float
    nvt_steps: int
    nve_steps: int
    nevery: int | None = None
    nrepeat: int | None = None
    nfreq: int | None = None


@dataclass(frozen=True)
class GKStatus:
    phase: str
    current_steps: int
    expected_steps: int
    timestep_ps: float

    @property
    def current_ps(self) -> float:
        return self.current_steps * self.timestep_ps

    @property
    def expected_ps(self) -> float:
        return self.expected_steps * self.timestep_ps

    @property
    def percent(self) -> float:
        if self.expected_steps <= 0:
            return 0.0
        return 100.0 * min(max(self.current_steps / self.expected_steps, 0.0), 1.0)


def read_gk_run_plan(input_file: Path) -> GKRunPlan:
    text = input_file.read_text(encoding="utf-8", errors="replace")
    timestep = _first_float(r"(?m)^\s*timestep\s+([0-9.eE+-]+)", text) or 0.00025
    nvt_match = re.search(
        r"Atomi GK phase: short NVT pre-equilibration.*?^\s*run\s+(\d+)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    nve_match = re.search(
        r"Atomi GK phase: NVE heat-current production.*?^\s*run\s+(\d+)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    jj_match = re.search(r"(?m)^\s*fix\s+JJ\s+all\s+ave/correlate\s+(\d+)\s+(\d+)\s+(\d+)", text)
    return GKRunPlan(
        timestep_ps=float(timestep),
        nvt_steps=int(nvt_match.group(1)) if nvt_match else 0,
        nve_steps=int(nve_match.group(1)) if nve_match else 0,
        nevery=int(jj_match.group(1)) if jj_match else None,
        nrepeat=int(jj_match.group(2)) if jj_match else None,
        nfreq=int(jj_match.group(3)) if jj_match else None,
    )


def latest_gk_input(chunk_dir: Path) -> Path | None:
    candidates = sorted(chunk_dir.glob("in.gk*_production"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("in.*"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def latest_gk_log(chunk_dir: Path) -> Path | None:
    candidates = sorted(chunk_dir.glob("log.in.gk*_production"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("log.in.*"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        candidates = sorted(chunk_dir.glob("log.*"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def summarize_gk_status(log_file: Path | None, plan: GKRunPlan) -> GKStatus:
    phase = "not_started"
    latest_nvt: int | None = None
    first_nvt: int | None = None
    latest_nve: int | None = None
    first_nve: int | None = None
    if log_file and log_file.exists():
        active_phase = "preflight"
        in_block = False
        with log_file.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "Atomi GK phase: short NVT" in line:
                    active_phase = "nvt"
                    phase = "nvt"
                elif "Atomi GK phase: NVE heat-current production" in line:
                    active_phase = "nve"
                    phase = "nve"
                stripped = line.strip()
                if stripped.startswith("Step"):
                    in_block = True
                    continue
                if stripped.startswith("Loop time"):
                    in_block = False
                    continue
                if not in_block:
                    continue
                parts = stripped.split()
                if not parts or _to_float_or_none(parts[0]) is None:
                    continue
                step = int(round(float(parts[0])))
                if active_phase == "nvt":
                    first_nvt = step if first_nvt is None else first_nvt
                    latest_nvt = step
                elif active_phase == "nve":
                    first_nve = step if first_nve is None else first_nve
                    latest_nve = step
    if phase == "nve":
        progress = _phase_progress(first_nve, latest_nve)
        return GKStatus("nve", progress, plan.nve_steps, plan.timestep_ps)
    if phase == "nvt":
        progress = _phase_progress(first_nvt, latest_nvt)
        return GKStatus("nvt", progress, plan.nvt_steps, plan.timestep_ps)
    return GKStatus(phase, 0, plan.nvt_steps or plan.nve_steps, plan.timestep_ps)


def read_hcacf_rows(path: Path, timestep_ps: float) -> list[dict[str, float]]:
    blocks: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                values = [float(part) for part in parts]
            except ValueError:
                continue
            if len(values) == 2:
                if current:
                    blocks.append(current)
                    current = []
                continue
            if len(values) >= 6:
                index, lag_steps, count = values[:3]
                hcx, hcy, hcz = values[3:6]
            elif len(values) >= 5:
                index, lag_steps = values[:2]
                count = math.nan
                hcx, hcy, hcz = values[2:5]
            else:
                continue
            avg = (hcx + hcy + hcz) / 3.0
            current.append(
                {
                    "index": index,
                    "lag_steps": lag_steps,
                    "time_ps": lag_steps * timestep_ps,
                    "count": count,
                    "HCACF_x": hcx,
                    "HCACF_y": hcy,
                    "HCACF_z": hcz,
                    "HCACF_avg": avg,
                }
            )
    if current:
        blocks.append(current)
    return blocks[-1] if blocks else []


def read_heatflux_timeseries(path: Path, timestep_ps: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.lower().startswith("step"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                step, jx, jy, jz = [float(part) for part in parts[:4]]
            except ValueError:
                continue
            rows.append(
                {
                    "step": step,
                    "time_ps": step * timestep_ps,
                    "Jx": jx,
                    "Jy": jy,
                    "Jz": jz,
                }
            )
    return rows


def print_gk_summary(chunk_dir: Path, input_file: Path | None = None, log_file: Path | None = None) -> tuple[GKRunPlan, GKStatus]:
    input_file = input_file or latest_gk_input(chunk_dir)
    if input_file is None:
        raise FileNotFoundError(f"No GK input file found in {chunk_dir}")
    log_file = log_file or latest_gk_log(chunk_dir)
    plan = read_gk_run_plan(input_file)
    status = summarize_gk_status(log_file, plan)
    print("GK run status")
    print("-------------")
    print(f"chunk        : {chunk_dir}")
    print(f"input        : {input_file}")
    print(f"log          : {log_file if log_file else 'missing'}")
    print(f"timestep     : {plan.timestep_ps:g} ps ({plan.timestep_ps * 1000:g} fs)")
    print(f"NVT target   : {plan.nvt_steps} steps = {plan.nvt_steps * plan.timestep_ps:g} ps")
    print(f"NVE target   : {plan.nve_steps} steps = {plan.nve_steps * plan.timestep_ps:g} ps")
    if plan.nevery and plan.nrepeat and plan.nfreq:
        print(
            "HCACF sampling: "
            f"nevery={plan.nevery}, nrepeat={plan.nrepeat}, nfreq={plan.nfreq} "
            f"({plan.nfreq * plan.timestep_ps:g} ps/window)"
        )
    print(
        f"current phase: {status.phase} "
        f"{status.current_steps}/{status.expected_steps} steps "
        f"= {status.current_ps:g}/{status.expected_ps:g} ps "
        f"({status.percent:.1f}%)"
    )
    return plan, status


def plot_gk_once(
    chunk_or_file: Path,
    *,
    timestep_ps: float | None = None,
    window: int = 220,
    timeseries: bool = False,
) -> None:
    chunk_or_file = chunk_or_file.resolve()
    chunk_dir = chunk_or_file if chunk_or_file.is_dir() else chunk_or_file.parent
    input_file = latest_gk_input(chunk_dir)
    plan = read_gk_run_plan(input_file) if input_file else GKRunPlan(timestep_ps or 0.00025, 0, 0)
    if timestep_ps is not None:
        plan = GKRunPlan(timestep_ps, plan.nvt_steps, plan.nve_steps, plan.nevery, plan.nrepeat, plan.nfreq)
    print_gk_summary(chunk_dir, input_file=input_file)
    hcacf = chunk_or_file if chunk_or_file.is_file() else chunk_dir / "heatflux_hcacf.dat"
    rows = read_hcacf_rows(hcacf, plan.timestep_ps) if hcacf.exists() else []
    if not rows:
        print(f"No HCACF rows found in {hcacf}")
        return
    print_hcacf_column_notes()
    print(f"HCACF rows    : {len(rows)} latest lag {rows[-1]['time_ps']:.6g} ps")
    ensure_gnuplot()
    if timeseries:
        ts_rows = read_heatflux_timeseries(chunk_dir / "heatflux_timeseries.dat", plan.timestep_ps)
        if ts_rows:
            _plot_timeseries(ts_rows[-window:])
    _plot_hcacf(rows[-window:])


def print_hcacf_column_notes() -> None:
    print("HCACF columns")
    print("-------------")
    print("1 index        : lag row within the current fix ave/correlate block")
    print("2 lag_steps    : correlation lag in MD timesteps")
    print("3 count        : number of time origins contributing to that lag")
    print("4 JxJx         : <Jx(0) Jx(t)>")
    print("5 JyJy         : <Jy(0) Jy(t)>")
    print("6 JzJz         : <Jz(0) Jz(t)>")
    print("Look for decay toward noisy zero and, in the running integral, a stable plateau.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="plotgk")
    parser.add_argument("chunk_or_hcacf", type=Path, nargs="?", default=Path("."))
    parser.add_argument("--timestep-ps", type=float, help="Override timestep in ps. Default reads the GK input file.")
    parser.add_argument("--window", type=int, default=220, help="Rows to show in the terminal plot. Default: 220.")
    parser.add_argument("--timeseries", action="store_true", help="Also plot heatflux_timeseries.dat when present.")
    args = parser.parse_args(argv)
    plot_gk_once(
        args.chunk_or_hcacf,
        timestep_ps=args.timestep_ps,
        window=args.window,
        timeseries=args.timeseries,
    )


def _plot_hcacf(rows: list[dict[str, float]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as handle:
        path = Path(handle.name)
        integral = 0.0
        previous = None
        for row in rows:
            if previous is not None:
                integral += 0.5 * (previous["HCACF_avg"] + row["HCACF_avg"]) * (row["time_ps"] - previous["time_ps"])
            previous = row
            handle.write(
                f"{row['time_ps']} {row['HCACF_x']} {row['HCACF_y']} {row['HCACF_z']} {row['HCACF_avg']} {integral}\n"
            )
    try:
        script = f"""
set term dumb ansi 120 36
set grid
set key outside
set multiplot layout 2,1 title "GK heat-current autocorrelation"
set title "HCACF components and average"
set xlabel "correlation time (ps)"
set ylabel "HCACF"
plot "{_gnuplot_quote(path)}" using 1:2 with lines title "JxJx", \
     "{_gnuplot_quote(path)}" using 1:3 with lines title "JyJy", \
     "{_gnuplot_quote(path)}" using 1:4 with lines title "JzJz", \
     "{_gnuplot_quote(path)}" using 1:5 with lines title "avg"
set title "Raw running integral of average HCACF"
set xlabel "correlation time (ps)"
set ylabel "raw integral"
plot "{_gnuplot_quote(path)}" using 1:6 with lines title "integral"
unset multiplot
"""
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
    finally:
        path.unlink(missing_ok=True)


def _plot_timeseries(rows: list[dict[str, float]]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as handle:
        path = Path(handle.name)
        for row in rows:
            handle.write(f"{row['time_ps']} {row['Jx']} {row['Jy']} {row['Jz']}\n")
    try:
        script = f"""
set term dumb ansi 120 18
set grid
set key outside
set title "Instantaneous heat current per volume"
set xlabel "NVE time (ps)"
set ylabel "J/vol"
plot "{_gnuplot_quote(path)}" using 1:2 with lines title "Jx", \
     "{_gnuplot_quote(path)}" using 1:3 with lines title "Jy", \
     "{_gnuplot_quote(path)}" using 1:4 with lines title "Jz"
"""
        subprocess.run(["gnuplot"], input=script, text=True, check=True)
    finally:
        path.unlink(missing_ok=True)


def _phase_progress(first_step: int | None, latest_step: int | None) -> int:
    if first_step is None or latest_step is None:
        return 0
    return max(0, latest_step - first_step)


def _first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _to_float_or_none(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _gnuplot_quote(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
