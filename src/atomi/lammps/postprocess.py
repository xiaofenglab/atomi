#!/usr/bin/env python3
"""
postprocess.py

Post-process LAMMPS NPT production thermo logs for UO2 thermodynamic analysis.

Designed for LAMMPS `units metal` thermo output like:
  step temp pe etotal press vol lx ly lz

New in v2
---------
Adds finer window analysis:
  - one selected global analysis window, as before
  - optional fixed-size subwindows inside that selected window
  - optional sliding-window analysis
  - writes per-window CSV/JSON summaries
  - helps diagnose stationarity before trusting Cp, CTE, KT, and H

Why this matters
----------------
For small-cell NPT, especially 96-atom UO2, volume and enthalpy can show slow
oscillations. A single 25–50 ps average can hide drift or breathing modes.
Window analysis lets you ask:
  - Is V stable block by block?
  - Is H stable block by block?
  - Does Cp depend strongly on which window I choose?
  - Which part of the trajectory is most stationary?

Units assumed
-------------
LAMMPS metal:
  T        K
  PE/E     eV per simulation cell
  Pressure bar
  Volume   Angstrom^3

Computed quantities
-------------------
For each selected window:
  - T_mean, T_std
  - P_mean_bar, P_mean_GPa
  - V_mean, V_std
  - lattice proxy a = (4*V/n_formula_units)^(1/3)
  - PE_mean, Etot_mean
  - enthalpy H = Etot + P*V, with P*V converted from bar*Angstrom^3 to eV
  - Cp from NPT enthalpy fluctuation:
        Cp_cell = Var(H) / (k_B T^2)
        Cp_molar = Cp_cell * eV_to_J * N_A / n_formula_units
  - isothermal bulk modulus from volume fluctuation:
        K_T = k_B T <V> / Var(V), reported in GPa
  - drift slopes for V, PE, H in the selected window

Examples
--------
Analyze last 25 ps as one window:
  python3 postprocess_npt_thermo_v2.py \
    --log log.in.npt_eqm_1400K_m04 \
    --temperature 1400 \
    --timestep-ps 0.0001 \
    --natoms 96 \
    --discard-ps 25 \
    --outdir analysis/npt_1400K

Analyze last 30 ps, split into non-overlapping 5 ps windows:
  python3 postprocess_npt_thermo_v2.py \
    --log log.in.npt_eqm_1400K_m04 \
    --temperature 1400 \
    --timestep-ps 0.0001 \
    --natoms 96 \
    --discard-ps 20 \
    --window-ps 5 \
    --outdir analysis/npt_1400K_win5ps

Analyze last 30 ps with sliding 5 ps windows every 1 ps.
Raw points are plotted as transparent gray dots; binned means as thick curves.
Use --plot-bin-ps to control the visual smoothing bin width:
  python3 postprocess_npt_thermo_v2.py \
    --log log.in.npt_eqm_1400K_m04 \
    --temperature 1400 \
    --timestep-ps 0.0001 \
    --natoms 96 \
    --discard-ps 20 \
    --window-ps 5 \
    --window-stride-ps 1 \
    --plot-bin-ps 0.5 \
    --outdir analysis/npt_1400K_slide5ps

Analyze specific timestep interval:
  python3 postprocess_npt_thermo_v2.py \
    --log log.in.npt_eqm_1400K_m04 \
    --temperature 1400 \
    --timestep-ps 0.0001 \
    --natoms 96 \
    --start-step 3900000 \
    --end-step 4080000 \
    --window-ps 5 \
    --outdir analysis/npt_1400K_specific
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

_mpl_cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KB_EV_PER_K = 8.617333262145e-5
EV_TO_J = 1.602176634e-19
NA = 6.02214076e23
BAR_A3_TO_EV = 6.241509074e-7
EV_A3_TO_GPA = 160.21766208


THERMO_RE = re.compile(
    r"^\s*(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)(?:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+))?"
)


def infer_temperature_from_name(path: Path) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)K", str(path))
    if m:
        return float(m.group(1))
    return None


def parse_lammps_thermo(log_path: Path) -> dict[str, np.ndarray]:
    step, temp, pe, etot, press, vol, lx, ly, lz = [], [], [], [], [], [], [], [], []
    with log_path.open(errors="ignore") as f:
        for line in f:
            m = THERMO_RE.match(line)
            if not m:
                continue
            g = m.groups()
            step.append(int(g[0]))
            temp.append(float(g[1]))
            pe.append(float(g[2]))
            etot.append(float(g[3]))
            press.append(float(g[4]))
            vol.append(float(g[5]))
            lx.append(float(g[6]) if g[6] is not None else np.nan)
            ly.append(float(g[7]) if g[7] is not None else np.nan)
            lz.append(float(g[8]) if g[8] is not None else np.nan)

    if not step:
        raise RuntimeError(f"No thermo rows found in {log_path}")

    return {
        "step": np.array(step, dtype=float),
        "temp": np.array(temp, dtype=float),
        "pe": np.array(pe, dtype=float),
        "etot": np.array(etot, dtype=float),
        "press_bar": np.array(press, dtype=float),
        "vol_A3": np.array(vol, dtype=float),
        "lx_A": np.array(lx, dtype=float),
        "ly_A": np.array(ly, dtype=float),
        "lz_A": np.array(lz, dtype=float),
    }


def initial_selection_mask(data: dict[str, np.ndarray],
                           timestep_ps: float,
                           discard_fraction: Optional[float],
                           discard_ps: Optional[float],
                           start_step: Optional[float],
                           end_step: Optional[float]) -> np.ndarray:
    step = data["step"]
    mask = np.ones(len(step), dtype=bool)

    if discard_ps is not None:
        cutoff = step[0] + discard_ps / timestep_ps
        mask &= step >= cutoff
    elif discard_fraction is not None:
        n_discard = int(round(len(step) * discard_fraction))
        tmp = np.zeros(len(step), dtype=bool)
        tmp[n_discard:] = True
        mask &= tmp

    if start_step is not None:
        mask &= step >= start_step
    if end_step is not None:
        mask &= step <= end_step

    if mask.sum() < 5:
        raise RuntimeError(f"Selected analysis window has too few points: {mask.sum()}")
    return mask


def make_subwindow_masks(step: np.ndarray,
                         base_mask: np.ndarray,
                         timestep_ps: float,
                         window_ps: Optional[float],
                         window_stride_ps: Optional[float],
                         min_points: int = 5) -> list[tuple[str, np.ndarray]]:
    """
    Return list of named masks.
    Always includes ("selected_full", base_mask).
    If window_ps is set, also includes fixed windows inside base_mask.
    If window_stride_ps is omitted, non-overlapping windows are used.
    If window_stride_ps is provided, sliding windows are used.
    """
    windows = [("selected_full", base_mask.copy())]

    if window_ps is None:
        return windows

    selected_steps = step[base_mask]
    if len(selected_steps) < min_points:
        return windows

    start = selected_steps[0]
    end = selected_steps[-1]
    window_steps = window_ps / timestep_ps
    stride_steps = window_steps if window_stride_ps is None else window_stride_ps / timestep_ps

    if window_steps <= 0 or stride_steps <= 0:
        raise ValueError("window_ps and window_stride_ps must be positive.")

    i = 0
    left = start
    while left + window_steps <= end + 1e-9:
        right = left + window_steps
        mask = base_mask & (step >= left) & (step < right)
        if mask.sum() >= min_points:
            label = f"window_{i:03d}_step_{int(left)}_{int(right)}"
            windows.append((label, mask))
            i += 1
        left += stride_steps

    return windows


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xm = np.mean(x)
    ym = np.mean(y)
    den = np.sum((x - xm) ** 2)
    if den == 0:
        return 0.0
    return float(np.sum((x - xm) * (y - ym)) / den)


def block_average_sem(y: np.ndarray, nblocks: int = 5) -> tuple[float, float]:
    if len(y) < nblocks:
        return float(np.mean(y)), float("nan")
    blocks = np.array_split(y, nblocks)
    means = np.array([np.mean(b) for b in blocks if len(b) > 0])
    sem = np.std(means, ddof=1) / math.sqrt(len(means)) if len(means) > 1 else float("nan")
    return float(np.mean(y)), float(sem)


def summarize_window(data: dict[str, np.ndarray],
                     mask: np.ndarray,
                     label: str,
                     target_temperature: Optional[float],
                     timestep_ps: float,
                     natoms: int,
                     atoms_per_formula_unit: int,
                     nblocks: int) -> tuple[dict, dict[str, np.ndarray]]:
    sel = {k: v[mask] for k, v in data.items()}
    nfu = natoms / atoms_per_formula_unit

    P_bar = sel["press_bar"]
    P_GPa = P_bar * 1.0e-4
    V = sel["vol_A3"]
    T = sel["temp"]
    PE = sel["pe"]
    Etot = sel["etot"]
    H_eV = Etot + P_bar * V * BAR_A3_TO_EV

    Tmean = float(np.mean(T))
    T_for_fluct = target_temperature if target_temperature is not None else Tmean

    H_var = float(np.var(H_eV, ddof=1)) if len(H_eV) > 1 else float("nan")
    Cp_cell_eV_per_K = H_var / (KB_EV_PER_K * T_for_fluct**2)
    Cp_J_per_mol_fu_K = Cp_cell_eV_per_K * EV_TO_J * NA / nfu

    V_var = float(np.var(V, ddof=1)) if len(V) > 1 else float("nan")
    if V_var and V_var > 0:
        KT_eV_A3 = KB_EV_PER_K * T_for_fluct * float(np.mean(V)) / V_var
        KT_GPa = KT_eV_A3 * EV_A3_TO_GPA
    else:
        KT_GPa = float("nan")

    V_mean, V_sem = block_average_sem(V, nblocks=nblocks)
    H_mean, H_sem = block_average_sem(H_eV, nblocks=nblocks)
    PE_mean, PE_sem = block_average_sem(PE, nblocks=nblocks)

    a_proxy = (4.0 * V / nfu) ** (1.0 / 3.0)
    a_mean, a_sem = block_average_sem(a_proxy, nblocks=nblocks)

    time_ps = (sel["step"] - sel["step"][0]) * timestep_ps

    summary = {
        "window_label": label,
        "target_temperature_K": target_temperature,
        "n_used_points": int(mask.sum()),
        "used_step_min": int(sel["step"][0]),
        "used_step_max": int(sel["step"][-1]),
        "used_time_ps": float((sel["step"][-1] - sel["step"][0]) * timestep_ps),
        "T_mean_K": float(np.mean(T)),
        "T_std_K": float(np.std(T, ddof=1)) if len(T) > 1 else float("nan"),
        "T_min_K": float(np.min(T)),
        "T_max_K": float(np.max(T)),
        "P_mean_bar": float(np.mean(P_bar)),
        "P_std_bar": float(np.std(P_bar, ddof=1)) if len(P_bar) > 1 else float("nan"),
        "P_mean_GPa": float(np.mean(P_GPa)),
        "P_std_GPa": float(np.std(P_GPa, ddof=1)) if len(P_GPa) > 1 else float("nan"),
        "V_mean_A3": V_mean,
        "V_sem_A3": V_sem,
        "V_std_A3": float(np.std(V, ddof=1)) if len(V) > 1 else float("nan"),
        "a_proxy_mean_A": a_mean,
        "a_proxy_sem_A": a_sem,
        "PE_mean_eV_cell": PE_mean,
        "PE_sem_eV_cell": PE_sem,
        "Etot_mean_eV_cell": float(np.mean(Etot)),
        "H_mean_eV_cell": H_mean,
        "H_sem_eV_cell": H_sem,
        "Cp_cell_eV_per_K": float(Cp_cell_eV_per_K),
        "Cp_J_per_mol_UO2_K": float(Cp_J_per_mol_fu_K),
        "KT_GPa_from_V_fluct": float(KT_GPa),
        "V_slope_A3_per_ps": linear_slope(time_ps, V),
        "PE_slope_eV_per_ps": linear_slope(time_ps, PE),
        "H_slope_eV_per_ps": linear_slope(time_ps, H_eV),
    }

    series = {
        "step": sel["step"],
        "time_ps": time_ps,
        "T": T,
        "P_bar": P_bar,
        "P_GPa": P_GPa,
        "V": V,
        "PE": PE,
        "Etot": Etot,
        "H": H_eV,
        "a_proxy": a_proxy,
    }
    return summary, series


def write_timeseries_csv(path: Path, series: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time_ps", "T_K", "P_bar", "P_GPa", "V_A3", "PE_eV", "Etot_eV", "H_eV", "a_proxy_A"])
        for i in range(len(series["step"])):
            w.writerow([
                int(series["step"][i]), float(series["time_ps"][i]),
                float(series["T"][i]), float(series["P_bar"][i]), float(series["P_GPa"][i]),
                float(series["V"][i]), float(series["PE"][i]), float(series["Etot"][i]),
                float(series["H"][i]), float(series["a_proxy"][i]),
            ])


def binned_mean_curve(x: np.ndarray, y: np.ndarray, bin_ps: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return binned mean x/y values.

    If bin_ps is None or <=0, choose an automatic bin width that gives ~80 bins.
    x is assumed to be time in ps.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return x, y

    xmin = float(np.min(x))
    xmax = float(np.max(x))
    span = xmax - xmin

    if span <= 0:
        return x, y

    if bin_ps is None or bin_ps <= 0:
        bin_ps = max(span / 80.0, 1.0e-12)

    edges = np.arange(xmin, xmax + bin_ps, bin_ps)
    if len(edges) < 2:
        return x, y

    xb = []
    yb = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if np.any(mask):
            xb.append(float(np.mean(x[mask])))
            yb.append(float(np.mean(y[mask])))

    return np.array(xb), np.array(yb)


def plot_raw_and_binned(ax, x: np.ndarray, y: np.ndarray, ylabel: str, bin_ps: float | None,
                        raw_alpha: float = 0.22, raw_size: float = 7.0) -> None:
    ax.scatter(x, y, s=raw_size, alpha=raw_alpha, color="0.45", linewidths=0)
    xb, yb = binned_mean_curve(x, y, bin_ps)
    ax.plot(xb, yb, linewidth=2.2)
    ax.set_ylabel(ylabel)


def plot_full_window(outpath: Path, series: dict[str, np.ndarray], bin_ps: float | None = None,
                     raw_alpha: float = 0.22, raw_size: float = 7.0) -> None:
    fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)

    x = series["time_ps"]
    plot_raw_and_binned(ax[0], x, series["T"], "T (K)", bin_ps, raw_alpha, raw_size)
    plot_raw_and_binned(ax[1], x, series["P_GPa"], "P (GPa)", bin_ps, raw_alpha, raw_size)
    plot_raw_and_binned(ax[2], x, series["V"], "V (Å$^3$)", bin_ps, raw_alpha, raw_size)
    plot_raw_and_binned(ax[3], x, series["PE"], "PE (eV)", bin_ps, raw_alpha, raw_size)
    plot_raw_and_binned(ax[4], x, series["H"], "H (eV)", bin_ps, raw_alpha, raw_size)

    ax[4].set_xlabel("Selected-window time (ps)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_window_summaries(outpath: Path, window_summaries: list[dict]) -> None:
    if len(window_summaries) <= 1:
        return

    ws = [w for w in window_summaries if w["window_label"] != "selected_full"]
    if not ws:
        return

    x = np.array([0.5 * (w["used_step_min"] + w["used_step_max"]) for w in ws], dtype=float)
    x = x - x[0]

    fig, ax = plt.subplots(5, 1, figsize=(8, 11), sharex=True)
    ax[0].plot(x, [w["T_mean_K"] for w in ws], marker="o")
    ax[0].set_ylabel("<T> K")
    ax[1].plot(x, [w["P_mean_GPa"] for w in ws], marker="o")
    ax[1].set_ylabel("<P> GPa")
    ax[2].plot(x, [w["V_mean_A3"] for w in ws], marker="o")
    ax[2].set_ylabel("<V> Å$^3$")
    ax[3].plot(x, [w["H_mean_eV_cell"] for w in ws], marker="o")
    ax[3].set_ylabel("<H> eV")
    ax[4].plot(x, [w["Cp_J_per_mol_UO2_K"] for w in ws], marker="o")
    ax[4].set_ylabel("Cp J/mol/K")
    ax[4].set_xlabel("Window midpoint relative step")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def analyze_one(log_path: Path,
                outdir: Path,
                target_temperature: Optional[float],
                timestep_ps: float,
                natoms: int,
                atoms_per_formula_unit: int,
                discard_fraction: Optional[float],
                discard_ps: Optional[float],
                start_step: Optional[float],
                end_step: Optional[float],
                window_ps: Optional[float],
                window_stride_ps: Optional[float],
                nblocks: int = 5,
                plot_bin_ps: Optional[float] = None,
                raw_alpha: float = 0.22,
                raw_size: float = 7.0) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    data = parse_lammps_thermo(log_path)

    base_mask = initial_selection_mask(data, timestep_ps, discard_fraction, discard_ps, start_step, end_step)
    windows = make_subwindow_masks(
        step=data["step"],
        base_mask=base_mask,
        timestep_ps=timestep_ps,
        window_ps=window_ps,
        window_stride_ps=window_stride_ps,
    )

    all_summaries = []
    full_summary = None

    for label, mask in windows:
        summary, series = summarize_window(
            data=data,
            mask=mask,
            label=label,
            target_temperature=target_temperature,
            timestep_ps=timestep_ps,
            natoms=natoms,
            atoms_per_formula_unit=atoms_per_formula_unit,
            nblocks=nblocks,
        )
        summary["log_file"] = str(log_path.resolve())
        summary["timestep_ps"] = timestep_ps
        summary["natoms"] = natoms
        summary["atoms_per_formula_unit"] = atoms_per_formula_unit
        summary["n_formula_units"] = natoms / atoms_per_formula_unit

        all_summaries.append(summary)

        if label == "selected_full":
            full_summary = summary
            write_timeseries_csv(outdir / "selected_timeseries.csv", series)
            plot_full_window(outdir / "thermo_selected_window.png", series, bin_ps=plot_bin_ps, raw_alpha=raw_alpha, raw_size=raw_size)

    with (outdir / "thermo_summary.json").open("w") as f:
        json.dump(full_summary, f, indent=2)

    with (outdir / "window_summaries.json").open("w") as f:
        json.dump(all_summaries, f, indent=2)

    keys = list(all_summaries[0].keys())
    with (outdir / "window_summaries.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in all_summaries:
            w.writerow(row)

    plot_window_summaries(outdir / "window_summary_trends.png", all_summaries)

    return full_summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lammps-postprocess")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--log", help="Single LAMMPS log file")
    group.add_argument("--glob", help="Glob pattern for multiple LAMMPS log files")

    ap.add_argument("--temperature", type=float, default=None, help="Target temperature. If omitted, infer from filename or use mean T.")
    ap.add_argument("--timestep-ps", type=float, default=0.0001)
    ap.add_argument("--natoms", type=int, default=96)
    ap.add_argument("--atoms-per-formula-unit", type=int, default=3)

    ap.add_argument("--discard-fraction", type=float, default=0.5, help="Discard this fraction from the beginning unless --discard-ps is set")
    ap.add_argument("--discard-ps", type=float, default=None, help="Discard this many ps from beginning")
    ap.add_argument("--start-step", type=float, default=None)
    ap.add_argument("--end-step", type=float, default=None)

    ap.add_argument("--window-ps", type=float, default=None, help="Analyze fixed subwindows of this length inside selected window")
    ap.add_argument("--window-stride-ps", type=float, default=None, help="Sliding window stride. If omitted, use non-overlapping windows")

    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--plot-bin-ps", type=float, default=None, help="Bin width in ps for thick mean curve in plots. Default: automatic ~80 bins")
    ap.add_argument("--raw-alpha", type=float, default=0.22, help="Transparency for raw scatter points")
    ap.add_argument("--raw-size", type=float, default=7.0, help="Marker size for raw scatter points")
    ap.add_argument("--outdir", default="analysis/npt_thermo")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)

    if args.log:
        log_files = [Path(args.log)]
    else:
        log_files = sorted(Path(".").glob(args.glob))

    if not log_files:
        raise RuntimeError("No log files found.")

    summaries = []
    for log_path in log_files:
        T = args.temperature
        if T is None:
            T = infer_temperature_from_name(log_path)

        od = outdir if len(log_files) == 1 else outdir / log_path.parent.name
        print(f"Analyzing {log_path} -> {od}")
        summary = analyze_one(
            log_path=log_path,
            outdir=od,
            target_temperature=T,
            timestep_ps=args.timestep_ps,
            natoms=args.natoms,
            atoms_per_formula_unit=args.atoms_per_formula_unit,
            discard_fraction=args.discard_fraction,
            discard_ps=args.discard_ps,
            start_step=args.start_step,
            end_step=args.end_step,
            window_ps=args.window_ps,
            window_stride_ps=args.window_stride_ps,
            nblocks=args.nblocks,
            plot_bin_ps=args.plot_bin_ps,
            raw_alpha=args.raw_alpha,
            raw_size=args.raw_size,
        )
        summaries.append(summary)

    if len(summaries) > 1:
        outdir.mkdir(parents=True, exist_ok=True)
        keys = list(summaries[0].keys())
        with (outdir / "all_temperatures_summary.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summaries:
                w.writerow(s)
        with (outdir / "all_temperatures_summary.json").open("w") as f:
            json.dump(summaries, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
