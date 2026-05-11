#!/usr/bin/env python3
"""
thermo_series_uq_v4.py

Locate each T-dependent NPT production run from either a production config JSON
or an MD run root folder, auto-select a stable analysis window from each run,
compute per-temperature MD thermodynamic summaries, then build MD-only
T-dependent thermodynamic curves.

This script integrates the core logic of postprocess_npt_thermo_v3.py, but
automates the full temperature series.

Main goals
----------
1) Read an MD root folder or config_production.json and find NPT stages such as:
     npt_prod_50K, npt_prod_100K, ..., npt_prod_1400K
   NVT ramp folders are ignored.

2) For each stage, locate its production LAMMPS log, usually:
     stages/<stage_name>/chunk_production/log.in.<stage_name>_production

3) Auto-select a stable window of at least 20 ps from each trajectory.
   The selected window is chosen by scanning fixed-size windows and minimizing
   a stationarity score based on:
     - temperature closeness to target
     - volume drift
     - enthalpy drift
     - energy drift
     - mean pressure magnitude

4) Compute per-T quantities:
     <T>, <P>, <V>, lattice proxy a, density
     <E>, <H>
     Cp from NPT enthalpy fluctuation
     KT from volume fluctuation
     slopes/drift diagnostics

5) Plot diagnostics for each T:
     raw gray points + thick binned mean curve
     selected window highlighted

6) Also plot one combined diagnostic figure containing all T stages:
     T, P, V, PE, H panels
     raw gray points + thick binned curves
     selected step/time windows highlighted by transparent shaded regions

6) Combine all temperatures:
     all_T_summary.csv/json
     per_T_analysis/all_T_MD_diagnostics_with_selected_windows.png
     V(T), a(T), density(T), H(T), Cp(T), KT(T)
     CTE alpha_V(T), alpha_L(T)
     S(T)-S(T0), G(T)-G(T0), relative functions by numerical integration

Usage
-----
A) Auto post-process from production config:
    python3 connect_npt_temperature_series.py \
      --config config_production.json \
      --outdir analysis/production_thermo_series \
      --window-ps 20 \
      --window-stride-ps 2 \
      --natoms 96 \
      --plot-bin-ps 0.5

B) Use existing manual analysis directories instead of re-parsing logs:
    python3 connect_npt_temperature_series.py \
      --manual-analysis-root analysis/manual_windows \
      --outdir analysis/production_thermo_series_from_manual

Manual analysis mode expects files like:
    analysis/manual_windows/npt_prod_600K/thermo_summary.json
or:
    analysis/manual_windows/*/thermo_summary.json

C) Force a larger stable window, e.g. 30 ps:
    python3 connect_npt_temperature_series.py \
      --config config_production.json \
      --outdir analysis/production_thermo_series_win30ps \
      --window-ps 30 \
      --window-stride-ps 2


python3 ./analysis/thermo_series_uq_v4_anchor.py \
  --config config_production.json \
  --outdir analysis/thermo_anchor_300K \
  --min-window-ps 20 \
  --window-stride-ps 2 \
  --plot-bin-ps 0.5 \
  --natoms 96 \
  --plot-T-min 300 \
  --plot-T-max 1500 \
  --plot-T-step 10 \
  --cp-source dH \
  --thermo-anchor-T 300 \
  --thermo-anchor-S 78.0 \
  --thermo-anchor-Cp 64.0 \
  --use-anchor-for-integration \
  --use-anchor-Cp-in-fit \
  --n-bootstrap 100


Important units
---------------
Per-mole convention:
  For UO2 fluorite, Z=4 per conventional cell. A 2x2x2 conventional supercell has
  96 atoms = 32 UO2 formula units. The script uses n_formula_units = natoms/3,
  so with --natoms 96 the molar outputs are per mole of UO2 formula units.
  Absolute H from MD is also reported in kJ/mol-UO2, but its zero depends on the MLIP energy reference.


LAMMPS metal:
  pressure = bar
  volume = Angstrom^3
  energy = eV per simulation cell

Conversions:
  1 bar = 1e-4 GPa
  P*V in bar*Angstrom^3 = 6.241509074e-7 eV

Notes
-----
- Cp from enthalpy fluctuations can be noisy for a 96-atom cell.
- KT from volume fluctuations can be very sensitive to slow NPT breathing.
- For CTE, the most robust output is usually from fitted V(T) or a(T).
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

from atomi.thermo_db import jaea_anchor


# -----------------------------
# constants
# -----------------------------

KB_EV_PER_K = 8.617333262145e-5
EV_TO_J = 1.602176634e-19
NA = 6.02214076e23
BAR_A3_TO_EV = 6.241509074e-7
EV_A3_TO_GPA = 160.21766208
EV_CELL_TO_KJ_PER_MOL_UO2 = EV_TO_J * NA / 1000.0  # multiply by eV_cell/n_formula_units

# approximate molar mass UO2 in g/mol
MOLAR_MASS_UO2_G_MOL = 238.0289 + 2.0 * 15.999

THERMO_RE = re.compile(
    r"^\s*(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)(?:\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+))?"
)

LATTICE_PARAMETER_SPECS = [
    {
        "key": "a",
        "ylabel": "Lattice a (Å)",
        "qha_names": ["a-temperature.dat", "lattice_a-temperature.dat", "lattice-temperature.dat"],
        "summary_keys": ["a_mean_A", "a_proxy_mean_A"],
    },
    {
        "key": "b",
        "ylabel": "Lattice b (Å)",
        "qha_names": ["b-temperature.dat", "lattice_b-temperature.dat"],
        "summary_keys": ["b_mean_A", "ly_mean_A"],
    },
    {
        "key": "c",
        "ylabel": "Lattice c (Å)",
        "qha_names": ["c-temperature.dat", "lattice_c-temperature.dat"],
        "summary_keys": ["c_mean_A", "lz_mean_A"],
    },
    {
        "key": "alpha",
        "ylabel": "Lattice alpha (deg)",
        "qha_names": ["alpha-temperature.dat", "lattice_alpha-temperature.dat"],
        "summary_keys": ["alpha_mean_deg"],
    },
    {
        "key": "beta",
        "ylabel": "Lattice beta (deg)",
        "qha_names": ["beta-temperature.dat", "lattice_beta-temperature.dat"],
        "summary_keys": ["beta_mean_deg"],
    },
    {
        "key": "gamma",
        "ylabel": "Lattice gamma (deg)",
        "qha_names": ["gamma-temperature.dat", "lattice_gamma-temperature.dat"],
        "summary_keys": ["gamma_mean_deg"],
    },
]


# -----------------------------
# general helpers
# -----------------------------

def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def infer_temperature_from_name(path_or_name) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)K", str(path_or_name))
    if m:
        return float(m.group(1))
    return None


def resolve_relative_to(base: Path, p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    den = float(np.sum((x - xm) ** 2))
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


def trapz_cumulative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i in range(1, len(x)):
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return out


def split_indices_into_blocks(n: int, nblocks: int) -> list[np.ndarray]:
    nblocks = max(1, min(int(nblocks), int(n)))
    return [b for b in np.array_split(np.arange(n), nblocks) if len(b) > 0]


def block_mean_sem_full(y: np.ndarray, nblocks: int = 5) -> tuple[float, float, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return float("nan"), float("nan"), np.array([])
    blocks = split_indices_into_blocks(len(y), nblocks)
    means = np.array([float(np.mean(y[b])) for b in blocks], dtype=float)
    sem = float(np.std(means, ddof=1) / math.sqrt(len(means))) if len(means) > 1 else float("nan")
    return float(np.mean(y)), sem, means


def percentile_band(samples: np.ndarray, low: float = 16.0, high: float = 84.0) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(samples, low, axis=0), np.nanpercentile(samples, high, axis=0)


def fit_eval_functions(T: np.ndarray, V: np.ndarray, a: np.ndarray, H_eV: np.ndarray, Cp: np.ndarray,
                       T_grid: np.ndarray, fit_degree: int, cp_source: str,
                       nfu: float, anchor_zero: bool) -> dict[str, np.ndarray]:
    deg_fit = min(fit_degree, max(len(T) - 1, 1))

    pV = np.poly1d(np.polyfit(T, V, deg_fit))
    dpV = np.polyder(pV)
    pa = np.poly1d(np.polyfit(T, a, deg_fit))
    dpa = np.polyder(pa)
    pH = np.poly1d(np.polyfit(T, H_eV, deg_fit))
    dpH = np.polyder(pH)

    V_grid = pV(T_grid)
    a_grid = pa(T_grid)
    H_grid_eV = pH(T_grid)
    H_abs_kJ_per_mol_UO2_grid = H_grid_eV * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu

    alpha_V_grid = dpV(T_grid) / V_grid
    alpha_L_grid = dpa(T_grid) / a_grid
    Cp_from_H_grid = dpH(T_grid) * EV_TO_J * NA / nfu

    if cp_source == "dH":
        Cp_grid = Cp_from_H_grid.copy()
    else:
        finite = np.isfinite(Cp)
        if finite.sum() >= 3:
            cp_deg = min(3, finite.sum() - 1)
            pCp = np.poly1d(np.polyfit(T[finite], Cp[finite], cp_deg))
            Cp_grid = pCp(T_grid)
        elif finite.sum() >= 2:
            Cp_grid = np.interp(T_grid, T[finite], Cp[finite])
        else:
            Cp_grid = np.zeros_like(T_grid)

    if anchor_zero and len(T_grid) and abs(T_grid[0]) < 1.0e-12:
        Cp_grid[0] = 0.0

    Cp_over_T = np.zeros_like(Cp_grid)
    nz = T_grid > 1.0e-12
    Cp_over_T[nz] = Cp_grid[nz] / T_grid[nz]

    H_rel = trapz_cumulative(T_grid, Cp_grid)
    S_rel = trapz_cumulative(T_grid, Cp_over_T)
    G_rel = H_rel - T_grid * S_rel

    mass_g = nfu * MOLAR_MASS_UO2_G_MOL / NA
    rho_grid = mass_g / (V_grid * 1.0e-24)

    return {
        "V_grid": V_grid,
        "a_grid": a_grid,
        "density_grid": rho_grid,
        "H_grid_eV": H_grid_eV,
        "H_abs_kJ_per_mol_UO2_grid": H_grid_eV * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu,
        "Cp_grid": Cp_grid,
        "Cp_from_H_grid": Cp_from_H_grid,
        "alpha_V_grid": alpha_V_grid,
        "alpha_L_grid": alpha_L_grid,
        "H_rel_J_mol_grid": H_rel,
        "S_rel_J_mol_K_grid": S_rel,
        "G_rel_J_mol_grid": G_rel,
    }


def bootstrap_temperature_functions(T: np.ndarray, V: np.ndarray, V_sem: np.ndarray,
                                    a: np.ndarray, a_sem: np.ndarray,
                                    H: np.ndarray, H_sem: np.ndarray,
                                    Cp: np.ndarray, Cp_sem: np.ndarray,
                                    T_grid: np.ndarray, fit_degree: int,
                                    cp_source: str, nfu: float, anchor_zero: bool,
                                    n_boot: int, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    samples: dict[str, list[np.ndarray]] = {}

    def safe_sigma(y, s):
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        fallback = np.nanmedian(np.abs(y - np.nanmedian(y))) * 0.05
        if not np.isfinite(fallback) or fallback == 0:
            fallback = max(abs(float(np.nanmean(y))) * 1e-6, 1e-12)
        return np.where(np.isfinite(s) & (s > 0), s, fallback)

    sigV = safe_sigma(V, V_sem)
    siga = safe_sigma(a, a_sem)
    sigH = safe_sigma(H, H_sem)
    sigCp = safe_sigma(Cp, Cp_sem)

    for _ in range(int(n_boot)):
        Vb = rng.normal(V, sigV)
        ab = rng.normal(a, siga)
        Hb = rng.normal(H, sigH)
        Cpb = rng.normal(Cp, sigCp)
        try:
            funcs = fit_eval_functions(T, Vb, ab, Hb, Cpb, T_grid, fit_degree, cp_source, nfu, anchor_zero)
        except Exception:
            continue
        for k, v in funcs.items():
            samples.setdefault(k, []).append(v)

    bands = {}
    for k, vals in samples.items():
        arr = np.array(vals)
        if len(arr) >= 5:
            bands[k] = percentile_band(arr, 16.0, 84.0)
    return bands


def plot_function_with_band(outpath: Path, T_data: np.ndarray, y_data: np.ndarray,
                            y_sem: Optional[np.ndarray], T_grid: np.ndarray, y_grid: np.ndarray,
                            band: Optional[tuple[np.ndarray, np.ndarray]],
                            xlabel: str, ylabel: str, title: str) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    if y_sem is not None:
        ax.errorbar(T_data, y_data, yerr=y_sem, fmt="o", color="0.35", alpha=0.75, capsize=3,
                    label="MD window mean ± block SEM")
    else:
        ax.scatter(T_data, y_data, color="0.35", alpha=0.75, label="MD window mean")
    if band is not None:
        lo, hi = band
        ax.fill_between(T_grid, lo, hi, alpha=0.22, label="MD UQ band, 16–84%")
    ax.plot(T_grid, y_grid, linewidth=2.3, label="fit/function")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_hybrid_grid(outpath: Path,
                     T_grid: np.ndarray,
                     hybrid_y: np.ndarray,
                     ylabel: str,
                     title: str,
                     qha_T: Optional[np.ndarray] = None,
                     qha_y: Optional[np.ndarray] = None,
                     md_T: Optional[np.ndarray] = None,
                     md_y: Optional[np.ndarray] = None,
                     band: Optional[tuple[np.ndarray, np.ndarray]] = None,
                     blend_start: Optional[float] = None,
                     blend_end: Optional[float] = None,
                     db_points: Optional[list[dict]] = None,
                     neel_region: Optional[tuple[float, float]] = None) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    if qha_T is not None and qha_y is not None and len(qha_T) and len(qha_y):
        ax.plot(qha_T, qha_y, "-.", color="0.45", alpha=0.45, linewidth=1.4, label="QHA reference")
    if md_T is not None and md_y is not None and len(md_T) and len(md_y):
        ax.plot(md_T, md_y, "--", color="0.35", alpha=0.45, linewidth=1.3, label="MD reference")
    if band is not None:
        lo, hi = band
        ax.fill_between(T_grid, lo, hi, color="#111111", alpha=0.12, label="hybrid MD UQ")
    ax.plot(T_grid, hybrid_y, color="#111111", linewidth=2.3, label="hybrid")
    if db_points:
        ax.plot(
            [point["T_K"] for point in db_points],
            [point["value"] for point in db_points],
            "o",
            color="#111111",
            markeredgecolor="#111111",
            markerfacecolor="none",
            markeredgewidth=1.8,
            markersize=7.0,
            label=db_points[0].get("label", "database"),
        )
    if blend_start is not None and blend_end is not None:
        if abs(blend_end - blend_start) <= 1.0e-12:
            ax.axvline(blend_start, color="0.25", linestyle=":", linewidth=1.1)
        else:
            ax.axvspan(blend_start, blend_end, color="#111111", alpha=0.08, label="blend interval")
    if neel_region is not None:
        ax.axvspan(
            neel_region[0],
            neel_region[1],
            color="#6f6f6f",
            alpha=0.12,
            label="Neel transition region",
        )
    finite = hybrid_y[np.isfinite(hybrid_y)]
    if len(finite):
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        if abs(ymax - ymin) <= 1.0e-12:
            pad = max(1.0, 0.08 * abs(ymax - ymin))
        else:
            pad = 0.08 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_qha_md_overlap(outpath: Path,
                        qha_T: np.ndarray,
                        qha_y: np.ndarray,
                        md_T: np.ndarray,
                        md_y: np.ndarray,
                        ylabel: str,
                        title: str,
                        blend_start: Optional[float] = None,
                        blend_end: Optional[float] = None) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qha_T, qha_y, "-", color="#1f77b4", linewidth=2.0, label="QHA")
    ax.plot(md_T, md_y, "o--", color="#d62728", markersize=3.5, linewidth=1.5, label="MD")
    if blend_start is not None and blend_end is not None:
        if abs(blend_end - blend_start) <= 1.0e-12:
            ax.axvline(blend_start, color="0.25", linestyle=":", linewidth=1.1)
        else:
            ax.axvspan(blend_start, blend_end, color="#111111", alpha=0.08, label="blend")
    ax.set_xlabel("T (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_structural_hybrid_detail(outpath: Path,
                                  *,
                                  T_grid: np.ndarray,
                                  hybrid_y: np.ndarray,
                                  ylabel: str,
                                  title: str,
                                  qha_T_raw: Optional[np.ndarray] = None,
                                  qha_y_raw: Optional[np.ndarray] = None,
                                  md_T_raw: Optional[np.ndarray] = None,
                                  md_y_raw: Optional[np.ndarray] = None,
                                  qha_T_corrected: Optional[np.ndarray] = None,
                                  qha_y_corrected: Optional[np.ndarray] = None,
                                  md_T_corrected: Optional[np.ndarray] = None,
                                  md_y_corrected: Optional[np.ndarray] = None,
                                  blend_start: Optional[float] = None,
                                  blend_end: Optional[float] = None) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    if qha_T_raw is not None and qha_y_raw is not None:
        ax.plot(qha_T_raw, qha_y_raw, "--", color="#1f77b4", alpha=0.65, label="raw QHA")
    if md_T_raw is not None and md_y_raw is not None:
        ax.plot(md_T_raw, md_y_raw, ":", color="#d62728", alpha=0.55, label="raw MD")
    if qha_T_corrected is not None and qha_y_corrected is not None:
        ax.plot(
            qha_T_corrected,
            qha_y_corrected,
            "-.",
            color="#1f77b4",
            alpha=0.75,
            label="corrected QHA",
        )
    if md_T_corrected is not None and md_y_corrected is not None:
        ax.plot(
            md_T_corrected,
            md_y_corrected,
            "--",
            color="#d62728",
            alpha=0.75,
            label="corrected MD",
        )
    ax.plot(T_grid, hybrid_y, "-", color="#111111", linewidth=2.3, label="hybrid")
    if blend_start is not None and blend_end is not None:
        if abs(blend_end - blend_start) <= 1.0e-12:
            ax.axvline(blend_start, color="0.25", linestyle=":", linewidth=1.1)
        else:
            ax.axvspan(blend_start, blend_end, color="#111111", alpha=0.08, label="blend")
    finite = hybrid_y[np.isfinite(hybrid_y)]
    if len(finite):
        ymin = float(np.min(finite))
        ymax = float(np.max(finite))
        pad = max(abs(ymax - ymin) * 0.08, 1.0e-12)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# -----------------------------
# parsing LAMMPS thermo
# -----------------------------

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


def add_derived_series(data: dict[str, np.ndarray], natoms: int, atoms_per_formula_unit: int) -> dict[str, np.ndarray]:
    nfu = natoms / atoms_per_formula_unit
    out = dict(data)
    out["press_GPa"] = out["press_bar"] * 1.0e-4
    out["enthalpy_eV"] = out["etot"] + out["press_bar"] * out["vol_A3"] * BAR_A3_TO_EV
    out["a_proxy_A"] = (4.0 * out["vol_A3"] / nfu) ** (1.0 / 3.0)
    # density g/cm3: mass = nfu*M g/mol / NA; volume A3 = 1e-24 cm3
    mass_g = nfu * MOLAR_MASS_UO2_G_MOL / NA
    out["density_g_cm3"] = mass_g / (out["vol_A3"] * 1.0e-24)
    return out


# -----------------------------
# window selection
# -----------------------------

def make_candidate_windows(step: np.ndarray,
                           timestep_ps: float,
                           window_ps: float,
                           stride_ps: float) -> list[tuple[int, int, np.ndarray]]:
    windows = []
    start = step[0]
    end = step[-1]
    window_steps = window_ps / timestep_ps
    stride_steps = stride_ps / timestep_ps

    left = start
    while left + window_steps <= end + 1e-9:
        right = left + window_steps
        mask = (step >= left) & (step < right)
        if mask.sum() >= 5:
            windows.append((int(left), int(right), mask))
        left += stride_steps

    return windows


def score_window(data: dict[str, np.ndarray],
                 mask: np.ndarray,
                 target_T: float,
                 timestep_ps: float) -> dict:
    step = data["step"][mask]
    tps = (step - step[0]) * timestep_ps

    T = data["temp"][mask]
    P = data["press_GPa"][mask]
    V = data["vol_A3"][mask]
    PE = data["pe"][mask]
    H = data["enthalpy_eV"][mask]

    V_mean = float(np.mean(V))
    H_span = max(float(np.max(H) - np.min(H)), 1.0e-12)
    PE_span = max(float(np.max(PE) - np.min(PE)), 1.0e-12)

    V_slope = linear_slope(tps, V)
    H_slope = linear_slope(tps, H)
    PE_slope = linear_slope(tps, PE)

    # Dimensionless-ish diagnostics.
    T_penalty = abs(float(np.mean(T)) - target_T) / max(target_T, 1.0)
    P_penalty = abs(float(np.mean(P))) / 5.0  # 5 GPa scale for small-cell NPT
    V_drift_frac = abs(V_slope) * max(float(tps[-1] - tps[0]), 1e-12) / max(V_mean, 1e-12)
    H_drift_frac = abs(H_slope) * max(float(tps[-1] - tps[0]), 1e-12) / H_span
    PE_drift_frac = abs(PE_slope) * max(float(tps[-1] - tps[0]), 1e-12) / PE_span

    score = (
        2.0 * T_penalty +
        1.0 * P_penalty +
        8.0 * V_drift_frac +
        4.0 * H_drift_frac +
        2.0 * PE_drift_frac
    )

    return {
        "score": float(score),
        "T_penalty": float(T_penalty),
        "P_penalty": float(P_penalty),
        "V_drift_frac": float(V_drift_frac),
        "H_drift_frac": float(H_drift_frac),
        "PE_drift_frac": float(PE_drift_frac),
        "V_slope_A3_per_ps": float(V_slope),
        "H_slope_eV_per_ps": float(H_slope),
        "PE_slope_eV_per_ps": float(PE_slope),
        "T_mean_K": float(np.mean(T)),
        "P_mean_GPa": float(np.mean(P)),
        "V_mean_A3": float(np.mean(V)),
        "H_mean_eV": float(np.mean(H)),
    }


def auto_select_window(data: dict[str, np.ndarray],
                       target_T: float,
                       timestep_ps: float,
                       min_window_ps: float = 20.0,
                       window_ps: Optional[float] = None,
                       stride_ps: float = 2.0,
                       discard_initial_ps: float = 0.0) -> tuple[np.ndarray, dict, list[dict]]:
    if window_ps is None:
        window_ps = min_window_ps
    if window_ps < min_window_ps:
        window_ps = min_window_ps

    step = data["step"]
    base_mask = np.ones(len(step), dtype=bool)
    if discard_initial_ps > 0:
        base_mask &= step >= step[0] + discard_initial_ps / timestep_ps

    sub = {k: v[base_mask] for k, v in data.items()}
    if len(sub["step"]) < 5:
        raise RuntimeError("No data left after initial discard.")

    candidates = make_candidate_windows(sub["step"], timestep_ps, window_ps, stride_ps)
    if not candidates:
        # fallback: use entire remaining region if it is at least min_window_ps
        total_ps = (sub["step"][-1] - sub["step"][0]) * timestep_ps
        if total_ps < min_window_ps:
            raise RuntimeError(f"Trajectory shorter than minimum window: {total_ps:.3f} ps < {min_window_ps:.3f} ps")
        mask = base_mask
        metrics = score_window(data, mask, target_T, timestep_ps)
        metrics.update({
            "selected_step_min": int(step[mask][0]),
            "selected_step_max": int(step[mask][-1]),
            "selected_time_ps": float((step[mask][-1] - step[mask][0]) * timestep_ps),
            "selection_method": "fallback_all_remaining",
        })
        return mask, metrics, [metrics]

    scored = []
    for left, right, local_mask in candidates:
        # convert local mask on sub arrays to global mask
        global_mask = np.zeros(len(step), dtype=bool)
        global_indices = np.where(base_mask)[0]
        global_mask[global_indices[local_mask]] = True

        metrics = score_window(data, global_mask, target_T, timestep_ps)
        metrics.update({
            "selected_step_min": left,
            "selected_step_max": right,
            "selected_time_ps": float((right - left) * timestep_ps),
            "selection_method": "auto_min_score",
        })
        scored.append((metrics["score"], global_mask, metrics))

    scored.sort(key=lambda x: x[0])
    best_score, best_mask, best_metrics = scored[0]
    window_table = [x[2] for x in scored]
    return best_mask, best_metrics, window_table


def select_tail_window(data: dict[str, np.ndarray],
                       target_T: float,
                       timestep_ps: float,
                       min_window_ps: float = 20.0,
                       window_ps: Optional[float] = None,
                       discard_initial_ps: float = 0.0) -> tuple[np.ndarray, dict, list[dict]]:
    if window_ps is None:
        window_ps = min_window_ps
    if window_ps < min_window_ps:
        window_ps = min_window_ps

    step = data["step"]
    base_mask = np.ones(len(step), dtype=bool)
    if discard_initial_ps > 0:
        base_mask &= step >= step[0] + discard_initial_ps / timestep_ps
    sub_step = step[base_mask]
    if len(sub_step) < 5:
        raise RuntimeError("No data left after initial discard.")

    end_step = float(sub_step[-1])
    left_step = end_step - window_ps / timestep_ps
    mask = base_mask & (step >= left_step) & (step <= end_step)
    if mask.sum() < 5:
        raise RuntimeError("Tail window contains fewer than 5 thermo rows.")
    selected_time_ps = float((step[mask][-1] - step[mask][0]) * timestep_ps)
    if selected_time_ps + 1.0e-9 < min_window_ps:
        raise RuntimeError(
            f"Trajectory shorter than minimum tail window: {selected_time_ps:.3f} ps < {min_window_ps:.3f} ps"
        )

    metrics = score_window(data, mask, target_T, timestep_ps)
    metrics.update({
        "selected_step_min": int(step[mask][0]),
        "selected_step_max": int(step[mask][-1]),
        "selected_time_ps": selected_time_ps,
        "selection_method": "tail_last_window",
    })
    return mask, metrics, [metrics]


# -----------------------------
# thermodynamic summary for a window
# -----------------------------

def summarize_selected_window(data: dict[str, np.ndarray],
                              mask: np.ndarray,
                              target_T: float,
                              timestep_ps: float,
                              natoms: int,
                              atoms_per_formula_unit: int,
                              nblocks: int = 5) -> dict:
    nfu = natoms / atoms_per_formula_unit
    sel = {k: v[mask] for k, v in data.items()}

    T = sel["temp"]
    P_bar = sel["press_bar"]
    P_GPa = sel["press_GPa"]
    V = sel["vol_A3"]
    PE = sel["pe"]
    Etot = sel["etot"]
    H = sel["enthalpy_eV"]
    a = sel["a_proxy_A"]
    rho = sel["density_g_cm3"]

    T_for_fluct = target_T if target_T is not None else float(np.mean(T))

    H_var = float(np.var(H, ddof=1)) if len(H) > 1 else float("nan")
    Cp_cell_eV_per_K = H_var / (KB_EV_PER_K * T_for_fluct**2)
    Cp_J_per_mol_UO2_K = Cp_cell_eV_per_K * EV_TO_J * NA / nfu

    # Block-wise Cp estimates from enthalpy fluctuations.
    # This is an MD statistical uncertainty proxy, not an experimental error bar.
    cp_block_vals = []
    for b in split_indices_into_blocks(len(H), nblocks):
        if len(b) > 2:
            H_var_b = float(np.var(H[b], ddof=1))
            Cp_b = H_var_b / (KB_EV_PER_K * T_for_fluct**2) * EV_TO_J * NA / nfu
            cp_block_vals.append(Cp_b)
    cp_block_vals = np.array(cp_block_vals, dtype=float)
    Cp_sem = float(np.std(cp_block_vals, ddof=1) / math.sqrt(len(cp_block_vals))) if len(cp_block_vals) > 1 else float("nan")

    V_var = float(np.var(V, ddof=1)) if len(V) > 1 else float("nan")
    if V_var and V_var > 0:
        KT_eV_A3 = KB_EV_PER_K * T_for_fluct * float(np.mean(V)) / V_var
        KT_GPa = KT_eV_A3 * EV_A3_TO_GPA
    else:
        KT_GPa = float("nan")

    time_ps = (sel["step"] - sel["step"][0]) * timestep_ps

    V_mean, V_sem = block_average_sem(V, nblocks)
    a_mean, a_sem = block_average_sem(a, nblocks)
    H_mean, H_sem = block_average_sem(H, nblocks)
    PE_mean, PE_sem = block_average_sem(PE, nblocks)

    return {
        "target_T_K": float(target_T),
        "natoms": int(natoms),
        "atoms_per_formula_unit": int(atoms_per_formula_unit),
        "n_formula_units": float(nfu),
        "n_used_points": int(mask.sum()),
        "step_min": int(sel["step"][0]),
        "step_max": int(sel["step"][-1]),
        "time_ps": float((sel["step"][-1] - sel["step"][0]) * timestep_ps),
        "T_mean_K": float(np.mean(T)),
        "T_std_K": float(np.std(T, ddof=1)),
        "P_mean_bar": float(np.mean(P_bar)),
        "P_std_bar": float(np.std(P_bar, ddof=1)),
        "P_mean_GPa": float(np.mean(P_GPa)),
        "P_std_GPa": float(np.std(P_GPa, ddof=1)),
        "V_mean_A3": V_mean,
        "V_sem_A3": V_sem,
        "V_std_A3": float(np.std(V, ddof=1)),
        "a_mean_A": a_mean,
        "a_sem_A": a_sem,
        "density_mean_g_cm3": float(np.mean(rho)),
        "density_sem_g_cm3": block_average_sem(rho, nblocks)[1],
        "density_std_g_cm3": float(np.std(rho, ddof=1)),
        "PE_mean_eV_cell": PE_mean,
        "PE_sem_eV_cell": PE_sem,
        "Etot_mean_eV_cell": float(np.mean(Etot)),
        "PE_mean_kJ_per_mol_UO2": float(PE_mean * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
        "PE_sem_kJ_per_mol_UO2": float(PE_sem * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
        "Etot_mean_kJ_per_mol_UO2": float(np.mean(Etot) * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
        "H_mean_eV_cell": H_mean,
        "H_sem_eV_cell": H_sem,
        "H_mean_kJ_per_mol_UO2": float(H_mean * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
        "H_sem_kJ_per_mol_UO2": float(H_sem * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
        "Cp_fluct_J_per_mol_UO2_K": float(Cp_J_per_mol_UO2_K),
        "Cp_fluct_sem_J_per_mol_UO2_K": float(Cp_sem),
        "KT_GPa_from_V_fluct": float(KT_GPa),
        "V_slope_A3_per_ps": linear_slope(time_ps, V),
        "PE_slope_eV_per_ps": linear_slope(time_ps, PE),
        "H_slope_eV_per_ps": linear_slope(time_ps, H),
    }


def write_selected_timeseries(path: Path, data: dict[str, np.ndarray], mask: np.ndarray, timestep_ps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sel = {k: v[mask] for k, v in data.items()}
    time_ps = (sel["step"] - sel["step"][0]) * timestep_ps

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time_ps", "T_K", "P_bar", "P_GPa", "V_A3", "a_A", "density_g_cm3", "PE_eV", "Etot_eV", "H_eV"])
        for i in range(len(time_ps)):
            w.writerow([
                int(sel["step"][i]), float(time_ps[i]), float(sel["temp"][i]),
                float(sel["press_bar"][i]), float(sel["press_GPa"][i]),
                float(sel["vol_A3"][i]), float(sel["a_proxy_A"][i]),
                float(sel["density_g_cm3"][i]), float(sel["pe"][i]),
                float(sel["etot"][i]), float(sel["enthalpy_eV"][i]),
            ])


# -----------------------------
# plotting
# -----------------------------

def decimate_xy(x: np.ndarray, y: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Return decimated x/y arrays for plotting raw scatter points only.

    This does not affect any analysis, window selection, binned means, fits, or UQ.
    It only reduces the number of raw points sent to matplotlib.
    """
    stride = max(1, int(stride))
    return x[::stride], y[::stride]


def binned_mean_curve(x: np.ndarray, y: np.ndarray, bin_ps: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
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
    xb, yb = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi if hi != edges[-1] else x <= hi)
        if np.any(mask):
            xb.append(float(np.mean(x[mask])))
            yb.append(float(np.mean(y[mask])))
    return np.array(xb), np.array(yb)


def plot_raw_binned_with_window(outpath: Path,
                                data: dict[str, np.ndarray],
                                selected_mask: np.ndarray,
                                timestep_ps: float,
                                plot_bin_ps: Optional[float],
                                title: str,
                                raw_alpha: float = 0.20,
                                raw_size: float = 6.0,
                                raw_decimate: int = 1) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    time_total = (data["step"] - data["step"][0]) * timestep_ps
    sel_time_min = float(np.min(time_total[selected_mask]))
    sel_time_max = float(np.max(time_total[selected_mask]))

    series = [
        ("temp", "T (K)"),
        ("press_GPa", "P (GPa)"),
        ("vol_A3", "V (Å$^3$)"),
        ("enthalpy_eV", "H (eV)"),
        ("pe", "PE (eV)"),
    ]

    fig, ax = plt.subplots(len(series), 1, figsize=(8.5, 11), sharex=True)
    for i, (key, ylabel) in enumerate(series):
        y = data[key]
        xd, yd = decimate_xy(time_total, y, raw_decimate)
        ax[i].scatter(xd, yd, s=raw_size, alpha=raw_alpha, color="0.45", linewidths=0)
        xb, yb = binned_mean_curve(time_total, y, plot_bin_ps)
        ax[i].plot(xb, yb, linewidth=2.3)
        ax[i].axvspan(sel_time_min, sel_time_max, alpha=0.15)
        ax[i].set_ylabel(ylabel)
    ax[-1].set_xlabel("Total production time (ps)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_all_temperature_diagnostics(outpath: Path,
                                     records: list[dict],
                                     timestep_ps: float,
                                     plot_bin_ps: Optional[float],
                                     raw_alpha: float = 0.10,
                                     raw_size: float = 4.0,
                                     raw_decimate: int = 1) -> None:
    """
    Plot all production MD results in one figure.

    Each temperature/stage is plotted as:
      - transparent gray raw points
      - thick binned mean curve
      - transparent shaded region for the selected analysis window

    To make multiple temperatures comparable in one figure, each stage's time axis
    is shifted so the first logged step starts at 0 ps.
    """
    if not records:
        return

    outpath.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("temp", "T (K)"),
        ("press_GPa", "P (GPa)"),
        ("vol_A3", "V (Å$^3$)"),
        ("pe", "PE (eV)"),
        ("enthalpy_eV", "H (eV)"),
    ]

    fig, ax = plt.subplots(len(panels), 1, figsize=(9.5, 12.5), sharex=True)

    for rec in records:
        data = rec["data"]
        selected_mask = rec["selected_mask"]
        label = rec["label"]

        time_ps = (data["step"] - data["step"][0]) * timestep_ps
        sel_time_min = float(np.min(time_ps[selected_mask]))
        sel_time_max = float(np.max(time_ps[selected_mask]))

        for i, (key, ylabel) in enumerate(panels):
            y = data[key]
            xd, yd = decimate_xy(time_ps, y, raw_decimate)
            ax[i].scatter(xd, yd, s=raw_size, alpha=raw_alpha, color="0.55", linewidths=0)
            xb, yb = binned_mean_curve(time_ps, y, plot_bin_ps)
            ax[i].plot(xb, yb, linewidth=2.0, label=label if i == 0 else None)
            ax[i].axvspan(sel_time_min, sel_time_max, alpha=0.08)
            ax[i].set_ylabel(ylabel)

    ax[-1].set_xlabel("Production time from start of each log (ps)")
    ax[0].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("All production NPT trajectories with selected analysis windows")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def plot_all_temperature_diagnostics_mixed_dt(outpath: Path,
                                              records: list[dict],
                                              plot_bin_ps: Optional[float],
                                              raw_alpha: float = 0.10,
                                              raw_size: float = 4.0,
                                              raw_decimate: int = 1) -> None:
    if not records:
        return
    outpath.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        ("temp", "T (K)"),
        ("press_GPa", "P (GPa)"),
        ("vol_A3", "V (Å$^3$)"),
        ("pe", "PE (eV)"),
        ("enthalpy_eV", "H (eV)"),
    ]
    fig, ax = plt.subplots(len(panels), 1, figsize=(9.5, 12.5), sharex=True)
    for rec in records:
        data = rec["data"]
        selected_mask = rec["selected_mask"]
        label = rec["label"]
        dt = float(rec.get("timestep_ps", 0.0001))
        time_ps = (data["step"] - data["step"][0]) * dt
        sel_time_min = float(np.min(time_ps[selected_mask]))
        sel_time_max = float(np.max(time_ps[selected_mask]))
        for i, (key, ylabel) in enumerate(panels):
            y = data[key]
            xd, yd = decimate_xy(time_ps, y, raw_decimate)
            ax[i].scatter(xd, yd, s=raw_size, alpha=raw_alpha, color="0.55", linewidths=0)
            xb, yb = binned_mean_curve(time_ps, y, plot_bin_ps)
            ax[i].plot(xb, yb, linewidth=2.0, label=label if i == 0 else None)
            ax[i].axvspan(sel_time_min, sel_time_max, alpha=0.08)
            ax[i].set_ylabel(ylabel)
    ax[-1].set_xlabel("Production time from start of each log (ps)")
    ax[0].legend(ncol=3, fontsize=8, frameon=False)
    fig.suptitle("All production NPT trajectories with selected analysis windows")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

def plot_xy(outpath: Path, x, y, xlabel, ylabel, title=None, y2=None, y2label=None) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="0.45", alpha=0.6)
    ax.plot(x, y, linewidth=2.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if y2 is not None:
        ax2 = ax.twinx()
        ax2.plot(x, y2, linewidth=2.0, linestyle="--")
        ax2.set_ylabel(y2label or "secondary")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def read_csv_dicts(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_float(row: dict, names: list[str]) -> Optional[float]:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def infer_series_formula_units(series_dir: Path, override: Optional[float] = None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--compare-formula-units values must be positive")
        return float(override)

    for csv_name in ("all_T_summary.csv", "thermo_functions_grid.csv"):
        path = series_dir / csv_name
        if not path.exists():
            continue
        rows = read_csv_dicts(path)
        if not rows:
            continue
        nfu = first_float(rows[0], ["n_formula_units"])
        if nfu and nfu > 0:
            return nfu

    metadata_path = series_dir / "temperature_range_metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        nfu = metadata.get("n_formula_units")
        if nfu and float(nfu) > 0:
            return float(nfu)

    raise ValueError(
        f"Cannot infer formula units for {series_dir}. Re-run thermo_lammps with a recent "
        "Atomi version or pass a matching --compare-formula-units value."
    )


def read_compare_series(
    series_dir: Path,
    label: str,
    n_formula_units: float,
    target_z: float,
    energy_basis: str,
) -> dict:
    grid_path = series_dir / "thermo_functions_grid.csv"
    if not grid_path.exists():
        raise ValueError(f"Missing thermo_functions_grid.csv in {series_dir}")
    rows = read_csv_dicts(grid_path)
    if not rows:
        raise ValueError(f"No rows in {grid_path}")

    multiplier = target_z if energy_basis == "target-cell" else 1.0
    data = {
        "label": label,
        "dir": series_dir,
        "n_formula_units": n_formula_units,
        "target_z": target_z,
        "T_K": [],
        "V_target_cell_A3": [],
        "a_A": [],
        "Cp": [],
        "S": [],
        "H_kJ": [],
        "G_kJ": [],
        "alpha_V_micro": [],
        "alpha_L_micro": [],
        "qha_md_blend_weight": [],
    }

    for row in rows:
        T = first_float(row, ["T_K"])
        if T is None:
            continue
        V_target = first_float(row, ["V_target_cell_A3"])
        if V_target is None:
            V_cell = first_float(row, ["V_fit_A3"])
            V_target = None if V_cell is None else V_cell * target_z / n_formula_units
        values = {
            "T_K": T,
            "V_target_cell_A3": V_target,
            "a_A": first_float(row, ["a_fit_A"]),
            "Cp": first_float(row, ["Cp_used_for_integration_J_per_mol_UO2_K"]),
            "S": first_float(row, ["S_rel_J_per_mol_UO2_K"]),
            "H_kJ": first_float(row, ["H_rel_J_per_mol_UO2"]),
            "G_kJ": first_float(row, ["G_rel_J_per_mol_UO2"]),
            "alpha_V_micro": first_float(row, ["alpha_V_micro_per_K"]),
            "alpha_L_micro": first_float(row, ["alpha_L_micro_per_K"]),
            "qha_md_blend_weight": first_float(row, ["qha_md_blend_weight"]),
        }
        for key, value in values.items():
            if key in ("H_kJ", "G_kJ") and value is not None:
                value = value * multiplier / 1000.0
            elif key in ("Cp", "S") and value is not None:
                value = value * multiplier
            data[key].append(value)

    return data


def plot_compare_series(
    outpath: Path,
    series: list[dict],
    key: str,
    ylabel: str,
    title: str,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> bool:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    plotted = False
    for item in series:
        T = np.array(item["T_K"], dtype=float)
        y = np.array([np.nan if value is None else value for value in item[key]], dtype=float)
        mask = np.isfinite(T) & np.isfinite(y)
        if t_min is not None:
            mask &= T >= t_min
        if t_max is not None:
            mask &= T <= t_max
        if not np.any(mask):
            continue
        order = np.argsort(T[mask])
        ax.plot(T[mask][order], y[mask][order], marker="o", linewidth=1.9, markersize=3.8, label=item["label"])
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("T (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return True


def compare_existing_lammps_series(
    series_dirs: list[Path],
    outdir: Path,
    labels: Optional[list[str]] = None,
    formula_units: Optional[list[float]] = None,
    target_z: float = 4.0,
    energy_basis: str = "per-formula",
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> None:
    if len(series_dirs) < 2:
        raise ValueError("--compare-series requires at least two thermo_lammps output directories")
    if target_z <= 0:
        raise ValueError("--target-z must be positive")
    if labels and len(labels) != len(series_dirs):
        raise ValueError("--compare-label must be repeated once per --compare-series directory")
    if formula_units and len(formula_units) != len(series_dirs):
        raise ValueError("--compare-formula-units must be repeated once per --compare-series directory")

    outdir.mkdir(parents=True, exist_ok=True)
    labels = labels or [path.name for path in series_dirs]
    formula_units = formula_units or [None] * len(series_dirs)

    series = []
    metadata = {
        "target_z_formula_units": target_z,
        "energy_basis": energy_basis,
        "temperature_window": {"t_min": t_min, "t_max": t_max},
        "series": [],
        "note": (
            "Comparison reads existing thermo_lammps thermo_functions_grid.csv files. "
            "If each source was generated with the same --qha-low-t-splice/QHA directory "
            "and blend flags, these overlays compare the corresponding hybrid QHA+MD curves."
        ),
    }
    for path, label, nfu_override in zip(series_dirs, labels, formula_units):
        resolved = path.resolve()
        nfu = infer_series_formula_units(resolved, nfu_override)
        item = read_compare_series(resolved, label, nfu, target_z, energy_basis)
        series.append(item)
        qha_meta_path = resolved / "qha_low_t_splice_metadata.json"
        metadata["series"].append({
            "label": label,
            "dir": str(resolved),
            "n_formula_units": nfu,
            "qha_low_t_splice_metadata": str(qha_meta_path) if qha_meta_path.exists() else None,
        })

    energy_label = "kJ/mol-target-cell" if energy_basis == "target-cell" else "kJ/mol-formula"
    cp_label = "J/mol-target-cell/K" if energy_basis == "target-cell" else "J/mol-formula/K"
    plot_defs = [
        ("V_target_cell_A3", "V (A3 per target cell)", "compare_volume_target_cell.png", "Normalized Volume"),
        ("a_A", "a (A)", "compare_lattice_a.png", "Lattice a"),
        ("Cp", f"Cp ({cp_label})", "compare_Cp.png", "QHA/MD Hybrid Cp"),
        ("S", f"S ({cp_label})", "compare_S.png", "Hybrid Entropy"),
        ("H_kJ", f"H ({energy_label})", "compare_H.png", "Hybrid Enthalpy"),
        ("G_kJ", f"G ({energy_label})", "compare_G.png", "Hybrid Gibbs Energy"),
        ("alpha_V_micro", "alpha_V (10^-6 K^-1)", "compare_alpha_V.png", "Volumetric CTE"),
        ("alpha_L_micro", "alpha_L (10^-6 K^-1)", "compare_alpha_L.png", "Linear CTE"),
        ("qha_md_blend_weight", "MD blend weight", "compare_qha_md_blend_weight.png", "QHA/MD Blend Weight"),
    ]

    index_rows = []
    for key, ylabel, filename, title in plot_defs:
        written = plot_compare_series(outdir / filename, series, key, ylabel, title, t_min, t_max)
        index_rows.append({"quantity": key, "file": filename, "written": written})

    with (outdir / "compare_index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["quantity", "file", "written"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)
    dump_json(outdir / "compare_metadata.json", metadata)


# -----------------------------
# stage/log discovery
# -----------------------------

def is_production_stage(stage: dict) -> bool:
    return stage.get("production_run", False) or stage.get("name", "").startswith("npt_prod")


def stage_temperature(stage: dict) -> Optional[float]:
    name = stage.get("name", "")
    for key in ("temperature", "temperature_end", "temperature_start"):
        value = stage.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return infer_temperature_from_name(name)


def is_npt_analysis_stage(stage: dict) -> bool:
    name = str(stage.get("name", "")).lower()
    stage_type = str(stage.get("type", "")).lower()
    if "nvt" in name or stage_type == "nvt":
        return False
    if is_production_stage(stage):
        return True
    return stage_type == "npt" or "npt" in name


def find_stage_log(root: Path, stage: dict) -> Path:
    name = stage["name"]
    chunk_name = stage.get("chunk_name", "chunk_production")
    chunk_dir = root / "stages" / name / chunk_name

    expected = chunk_dir / f"log.in.{name}_production"
    if expected.exists():
        return expected

    candidates = sorted(chunk_dir.glob("log.in.*"))
    if candidates:
        return candidates[-1]

    candidates = sorted(chunk_dir.glob("log.*"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(f"No LAMMPS log found for {name} in {chunk_dir}")


def preferred_stage_log(root: Path, stage: dict) -> Path:
    try:
        return find_stage_log(root, stage)
    except FileNotFoundError:
        pass

    stage_dir = root / "stages" / stage["name"]
    candidates = candidate_lammps_logs(stage_dir)
    if candidates:
        if any(path.parent.name == "chunk_production" for path in candidates):
            return next(path for path in candidates if path.parent.name == "chunk_production")
        return candidates[-1]
    raise FileNotFoundError(f"No LAMMPS log found for {stage['name']} in {stage_dir}")


def preferred_discovered_log(stage_dir: Path) -> Path | None:
    candidates = candidate_lammps_logs(stage_dir)
    if not candidates:
        return None
    if any(path.parent.name == "chunk_production" for path in candidates):
        return next(path for path in candidates if path.parent.name == "chunk_production")
    return candidates[-1]


def merge_duplicate_temperature_records(records: list[dict], duplicate_policy: str) -> list[dict]:
    merged = {}
    for record in records:
        temperature = record["temperature"]
        if temperature in merged:
            if duplicate_policy == "error":
                raise RuntimeError(
                    f"Duplicate T={temperature} K:\n"
                    f"  {merged[temperature]['log_path']}\n"
                    f"  {record['log_path']}"
                )
            if duplicate_policy == "first":
                continue
            old = merged[temperature]
            if record["config_index"] >= old["config_index"]:
                print(
                    f"WARNING: duplicate T={temperature} K; keeping later record:\n"
                    f"  old: {old['log_path']}\n"
                    f"  new: {record['log_path']}"
                )
                merged[temperature] = record
        else:
            merged[temperature] = record
    return [merged[temperature] for temperature in sorted(merged)]


# -----------------------------
# per-T processing
# -----------------------------


def collect_config_paths(configs: list[str] | None, config_dir: str | None, config_glob: str) -> list[Path]:
    paths: list[Path] = []
    if configs:
        for c in configs:
            paths.append(Path(c).resolve())
    if config_dir:
        paths.extend(sorted(Path(config_dir).resolve().glob(config_glob)))
    # de-duplicate
    out=[]; seen=set()
    for x in paths:
        if not x.exists():
            raise FileNotFoundError(f"Config not found: {x}")
        if x not in seen:
            out.append(x); seen.add(x)
    if not out:
        raise RuntimeError("No config files provided. Use --config and/or --config-dir.")
    return out


def discover_production_records(config_paths: list[Path], duplicate_policy: str = "highest_config_order") -> list[dict]:
    """Discover fixed-temperature NPT logs across one or more configs."""
    records=[]
    for ci, config_path in enumerate(config_paths):
        cfg=load_json(config_path)
        root=config_path.resolve().parent
        for stage in cfg.get("stages", []):
            if not is_npt_analysis_stage(stage):
                continue
            name=stage["name"]
            T=stage_temperature(stage)
            if T is None:
                print(f"WARNING: could not infer temperature for {name} in {config_path}; skipping")
                continue
            T=float(T)
            try:
                log_path=preferred_stage_log(root, stage)
            except FileNotFoundError as exc:
                print(f"WARNING: {exc}; skipping")
                continue
            records.append({
                "temperature": T,
                "stage": stage,
                "stage_name": name,
                "config_path": config_path,
                "config_root": root,
                "config_index": ci,
                "log_path": log_path,
                "timestep_ps": float(cfg.get("timestep", 0.0001)),
            })
    if not records:
        raise RuntimeError("No NPT logs found across provided config files.")

    return merge_duplicate_temperature_records(records, duplicate_policy)


def candidate_lammps_logs(stage_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    preferred_dirs = [stage_dir / "chunk_production"]
    preferred_dirs.extend(sorted(p for p in stage_dir.glob("chunk_*") if p.is_dir() and p.name != "chunk_production"))
    preferred_dirs.append(stage_dir)

    seen: set[Path] = set()
    for log_dir in preferred_dirs:
        if not log_dir.exists():
            continue
        for pattern in ("log.in.*", "log.*"):
            for path in sorted(log_dir.glob(pattern)):
                if path.is_file() and path not in seen:
                    candidates.append(path)
                    seen.add(path)
    return candidates


def find_md_root_stage_dirs(md_root: Path) -> list[Path]:
    search_roots = [md_root / "stages"] if (md_root / "stages").is_dir() else [md_root]
    stage_dirs: list[Path] = []
    seen: set[Path] = set()

    for search_root in search_roots:
        for path in [search_root] + sorted(search_root.rglob("*")):
            if not path.is_dir():
                continue
            name = path.name.lower()
            if "nvt" in name or "npt" not in name:
                continue
            if "chunk" in name:
                path = path.parent
            if path in seen:
                continue
            if infer_temperature_from_name(path.name) is None:
                continue
            if not candidate_lammps_logs(path):
                continue
            stage_dirs.append(path)
            seen.add(path)
    return stage_dirs


def discover_npt_records_from_md_root(
    md_root: Path,
    duplicate_policy: str = "highest_config_order",
    timestep_ps: Optional[float] = None,
) -> list[dict]:
    """Discover fixed-temperature NPT MD logs without relying on config JSON."""
    md_root = md_root.resolve()
    if not md_root.exists():
        raise FileNotFoundError(f"MD root not found: {md_root}")
    if not md_root.is_dir():
        raise NotADirectoryError(f"MD root is not a directory: {md_root}")

    records: list[dict] = []
    for stage_dir in find_md_root_stage_dirs(md_root):
        stage_name = stage_dir.name
        temperature = infer_temperature_from_name(stage_name)
        if temperature is None:
            continue
        log_path = preferred_discovered_log(stage_dir)
        if log_path is None:
            continue
        chunk_dir = log_path.parent
        records.append({
            "temperature": float(temperature),
            "stage": {
                "name": stage_name,
                "type": "npt",
                "temperature": float(temperature),
                "chunk_name": chunk_dir.name,
                "production_run": True,
                "discovered_from_md_root": True,
            },
            "stage_name": stage_name,
            "config_path": None,
            "config_root": md_root,
            "config_index": 0,
            "log_path": log_path,
            "timestep_ps": float(timestep_ps) if timestep_ps is not None else 0.0001,
            "md_root": md_root,
        })

    if not records:
        raise RuntimeError(
            f"No NPT LAMMPS logs found under {md_root}. Expected folders such as "
            "stages/npt_prod_300K/chunk_production/log.in.*; NVT folders are ignored."
        )

    return merge_duplicate_temperature_records(records, duplicate_policy)


def filter_records_by_T(records: list[dict], T_min: Optional[float], T_max: Optional[float]) -> list[dict]:
    T_all=np.array([r["temperature"] for r in records], dtype=float)
    if T_max is None:
        T_high=float(np.max(T_all))
    else:
        below=T_all[T_all <= T_max]
        if len(below):
            T_high=float(np.max(below))
        else:
            T_high=float(T_all[np.argmin(np.abs(T_all - T_max))])
    T_low=float(T_min) if T_min is not None else float(np.min(T_all))
    out=[r for r in records if r["temperature"] >= T_low - 1e-9 and r["temperature"] <= T_high + 1e-9]
    if not out:
        raise RuntimeError(f"No production stages remain after filtering T_min={T_min}, T_max={T_max}")
    print(f"Temperature-stage filter: requested min={T_min}, requested max={T_max}; processing {out[0]['temperature']}–{out[-1]['temperature']} K ({len(out)} stages).", flush=True)
    return out


def process_records(records: list[dict],
                    outdir: Path,
                    natoms: int,
                    atoms_per_formula_unit: int,
                    min_window_ps: float,
                    window_ps: Optional[float],
                    window_mode: str,
                    window_stride_ps: float,
                    discard_initial_ps: float,
                    plot_bin_ps: Optional[float],
                    nblocks: int,
                    timestep_override_ps: Optional[float] = None,
                    skip_per_T_plots: bool = False,
                    skip_combined_MD_plot: bool = False,
                    skip_selected_timeseries: bool = False,
                    raw_decimate: int = 1) -> list[dict]:
    summaries=[]
    diagnostic_records=[]

    for rec in records:
        stage=rec["stage"]
        name=rec["stage_name"]
        T=float(rec["temperature"])
        log_path=rec["log_path"]
        timestep_ps = float(timestep_override_ps) if timestep_override_ps is not None else float(rec.get("timestep_ps", 0.0001))
        config_path = rec.get("config_path")
        config_label = config_path.name if isinstance(config_path, Path) else "md-root"

        print(f"Processing {name}: T={T} K, source={config_label}, log={log_path}")

        stage_out = outdir / name
        stage_out.mkdir(parents=True, exist_ok=True)

        data = parse_lammps_thermo(log_path)
        data = add_derived_series(data, natoms, atoms_per_formula_unit)

        if window_mode == "tail":
            sel_mask, sel_metrics, window_table = select_tail_window(
                data=data,
                target_T=T,
                timestep_ps=timestep_ps,
                min_window_ps=min_window_ps,
                window_ps=window_ps,
                discard_initial_ps=discard_initial_ps,
            )
        else:
            sel_mask, sel_metrics, window_table = auto_select_window(
                data=data,
                target_T=T,
                timestep_ps=timestep_ps,
                min_window_ps=min_window_ps,
                window_ps=window_ps,
                stride_ps=window_stride_ps,
                discard_initial_ps=discard_initial_ps,
            )

        summary = summarize_selected_window(
            data=data,
            mask=sel_mask,
            target_T=T,
            timestep_ps=timestep_ps,
            natoms=natoms,
            atoms_per_formula_unit=atoms_per_formula_unit,
            nblocks=nblocks,
        )

        summary.update({
            "stage_name": name,
            "config_file": str(config_path.resolve()) if isinstance(config_path, Path) else "",
            "md_root": str(rec["md_root"].resolve()) if isinstance(rec.get("md_root"), Path) else "",
            "log_file": str(log_path.resolve()),
            "timestep_ps": timestep_ps,
            "selection_score": sel_metrics["score"],
            "selection_metrics": sel_metrics,
        })

        dump_json(stage_out / "thermo_summary.json", summary)
        dump_json(stage_out / "window_candidates.json", window_table)

        if not skip_selected_timeseries:
            write_selected_timeseries(stage_out / "selected_timeseries.csv", data, sel_mask, timestep_ps)

        if not skip_per_T_plots:
            plot_raw_binned_with_window(
                outpath=stage_out / "total_time_with_selected_window.png",
                data=data,
                selected_mask=sel_mask,
                timestep_ps=timestep_ps,
                plot_bin_ps=plot_bin_ps,
                title=f"{name}: auto-selected window {summary['time_ps']:.2f} ps",
                raw_decimate=raw_decimate,
            )

        summaries.append(summary)
        diagnostic_records.append({
            "label": f"{int(T) if float(T).is_integer() else T} K",
            "data": data,
            "selected_mask": sel_mask,
            "timestep_ps": timestep_ps,
        })

    summaries.sort(key=lambda d: d["target_T_K"])

    if not skip_combined_MD_plot:
        # If mixed timesteps, combined plot uses individual timestep per record.
        # The original plotting function assumes a common timestep. Use first one if all identical.
        unique_dt=sorted(set(round(r.get("timestep_ps", records[0].get("timestep_ps", 0.0001)), 12) for r in records))
        if len(unique_dt) == 1:
            plot_all_temperature_diagnostics(
                outpath=outdir / "all_T_MD_diagnostics_with_selected_windows.png",
                records=diagnostic_records,
                timestep_ps=unique_dt[0],
                plot_bin_ps=plot_bin_ps,
                raw_decimate=raw_decimate,
            )
        else:
            plot_all_temperature_diagnostics_mixed_dt(
                outpath=outdir / "all_T_MD_diagnostics_with_selected_windows.png",
                records=diagnostic_records,
                plot_bin_ps=plot_bin_ps,
            )

    return summaries


def process_from_config(config_path: Path,
                        outdir: Path,
                        natoms: int,
                        atoms_per_formula_unit: int,
                        min_window_ps: float,
                        window_ps: Optional[float],
                        window_stride_ps: float,
                        discard_initial_ps: float,
                        plot_bin_ps: Optional[float],
                        nblocks: int) -> list[dict]:
    records = discover_production_records([config_path])
    return process_records(
        records,
        outdir,
        natoms,
        atoms_per_formula_unit,
        min_window_ps,
        window_ps,
        "tail",
        window_stride_ps,
        discard_initial_ps,
        plot_bin_ps,
        nblocks,
    )

def process_from_manual(manual_root: Path) -> list[dict]:
    files = sorted(manual_root.glob("**/thermo_summary.json"))
    if not files:
        raise RuntimeError(f"No thermo_summary.json files found under {manual_root}")

    summaries = []
    for f in files:
        d = load_json(f)
        if "target_T_K" not in d:
            T = d.get("target_temperature_K", infer_temperature_from_name(f))
            d["target_T_K"] = T
        d["manual_summary_file"] = str(f.resolve())
        summaries.append(d)

    summaries.sort(key=lambda d: d["target_T_K"])
    return summaries


# -----------------------------
# combined thermodynamics
# -----------------------------

def polynomial_fit_and_derivative(T: np.ndarray, y: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = min(degree, max(len(T) - 1, 1))
    coeff = np.polyfit(T, y, degree)
    p = np.poly1d(coeff)
    dp = np.polyder(p)
    yfit = p(T)
    dydT = dp(T)
    return coeff, yfit, dydT



def insert_anchor_point_for_cp_fit(T: np.ndarray,
                                   Cp: np.ndarray,
                                   T_anchor: Optional[float],
                                   Cp_anchor: Optional[float],
                                   replace_tol_K: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    """
    Add or replace a Cp(T_anchor) point for fitting/integration.

    This is useful when Cp(300 K) comes from phonopy/QHA or literature,
    while high-T Cp comes from MLIP-MD dH/dT.
    """
    if T_anchor is None or Cp_anchor is None:
        return T, Cp

    T_anchor = float(T_anchor)
    Cp_anchor = float(Cp_anchor)

    T_new = np.array(T, dtype=float).copy()
    Cp_new = np.array(Cp, dtype=float).copy()

    idx = np.where(np.abs(T_new - T_anchor) <= replace_tol_K)[0]
    if len(idx) > 0:
        Cp_new[idx[0]] = Cp_anchor
    else:
        T_new = np.append(T_new, T_anchor)
        Cp_new = np.append(Cp_new, Cp_anchor)

    order = np.argsort(T_new)
    return T_new[order], Cp_new[order]


def integrate_from_reference(T_grid: np.ndarray,
                             Cp_grid: np.ndarray,
                             T_ref: float,
                             S_ref_J_mol_K: Optional[float] = None,
                             H_ref_J_mol: Optional[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate Cp from a reference temperature.

    Returns H(T), S(T), G(T), where:
      S(T_ref) = S_ref if provided, otherwise 0
      H(T_ref) = H_ref if provided, otherwise 0

    The integration is done on T_grid. This is intended for a splice workflow:
      0--T_ref: phonopy/QHA/literature
      T_ref--high T: MLIP-MD Cp
    """
    T_grid = np.asarray(T_grid, dtype=float)
    Cp_grid = np.asarray(Cp_grid, dtype=float)

    S0 = 0.0 if S_ref_J_mol_K is None else float(S_ref_J_mol_K)
    H0 = 0.0 if H_ref_J_mol is None else float(H_ref_J_mol)
    T_ref = float(T_ref)

    H_out = np.full_like(T_grid, np.nan, dtype=float)
    S_out = np.full_like(T_grid, np.nan, dtype=float)

    # Need an integration grid that includes T_ref exactly.
    grid = np.array(T_grid, dtype=float)
    if not np.any(np.isclose(grid, T_ref, rtol=0.0, atol=1.0e-12)):
        grid = np.sort(np.append(grid, T_ref))

    Cp_interp = np.interp(grid, T_grid, Cp_grid)

    ref_idx = int(np.argmin(np.abs(grid - T_ref)))

    H_grid = np.full_like(grid, np.nan, dtype=float)
    S_grid = np.full_like(grid, np.nan, dtype=float)
    H_grid[ref_idx] = H0
    S_grid[ref_idx] = S0

    # Integrate upward from T_ref.
    for i in range(ref_idx + 1, len(grid)):
        dT = grid[i] - grid[i - 1]
        H_grid[i] = H_grid[i - 1] + 0.5 * (Cp_interp[i] + Cp_interp[i - 1]) * dT
        cpt_i = Cp_interp[i] / grid[i] if grid[i] > 1.0e-12 else 0.0
        cpt_j = Cp_interp[i - 1] / grid[i - 1] if grid[i - 1] > 1.0e-12 else 0.0
        S_grid[i] = S_grid[i - 1] + 0.5 * (cpt_i + cpt_j) * dT

    # Integrate downward from T_ref, if requested grid goes below T_ref.
    for i in range(ref_idx - 1, -1, -1):
        dT = grid[i + 1] - grid[i]
        H_grid[i] = H_grid[i + 1] - 0.5 * (Cp_interp[i + 1] + Cp_interp[i]) * dT
        cpt_i = Cp_interp[i] / grid[i] if grid[i] > 1.0e-12 else 0.0
        cpt_j = Cp_interp[i + 1] / grid[i + 1] if grid[i + 1] > 1.0e-12 else 0.0
        S_grid[i] = S_grid[i + 1] - 0.5 * (cpt_i + cpt_j) * dT

    H_out = np.interp(T_grid, grid, H_grid)
    S_out = np.interp(T_grid, grid, S_grid)
    G_out = H_out - T_grid * S_out
    return H_out, S_out, G_out


def read_qha_temperature_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    if not path.exists():
        raise FileNotFoundError(f"QHA file not found: {path}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric temperature/value rows found in {path}")
    rows.sort(key=lambda item: item[0])
    return (
        np.array([row[0] for row in rows], dtype=float),
        np.array([row[1] for row in rows], dtype=float),
    )


def read_optional_qha_temperature_dat(
    qha_dir: Path,
    names: list[str],
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    for name in names:
        path = qha_dir / name
        if not path.exists():
            continue
        try:
            T, y = read_qha_temperature_dat(path)
        except (FileNotFoundError, ValueError):
            continue
        return T, y, str(path.resolve())
    return None, None, None


def qha_cp_scale_to_j_mol_formula(unit: str, qha_formula_units: float) -> float:
    if qha_formula_units <= 0:
        raise ValueError("--qha-anchor-formula-units must be positive")
    if unit == "J/mol-cell/K":
        return 1.0 / qha_formula_units
    if unit == "kJ/mol-cell/K":
        return 1000.0 / qha_formula_units
    if unit == "J/mol-formula/K":
        return 1.0
    if unit == "kJ/mol-formula/K":
        return 1000.0
    if unit == "eV-cell/K":
        return EV_TO_J * NA / qha_formula_units
    raise ValueError(f"Unsupported QHA Cp unit: {unit}")


def qha_cp_thermo_curve(
    qha_dir: Path,
    qha_formula_units: float,
    qha_cp_unit: str = "J/mol-cell/K",
) -> dict:
    T_qha, Cp_raw = read_qha_temperature_dat(qha_dir / "Cp-temperature.dat")
    Cp_qha = Cp_raw * qha_cp_scale_to_j_mol_formula(qha_cp_unit, qha_formula_units)
    H_qha = trapz_cumulative(T_qha, Cp_qha)
    Cp_over_T = np.zeros_like(Cp_qha, dtype=float)
    mask = T_qha > 1.0e-12
    Cp_over_T[mask] = Cp_qha[mask] / T_qha[mask]
    S_qha = trapz_cumulative(T_qha, Cp_over_T)
    T_volume, V_qha, qha_volume_file = read_optional_qha_temperature_dat(
        qha_dir,
        ["volume-temperature.dat"],
    )
    lattice_parameters = {}
    for spec in LATTICE_PARAMETER_SPECS:
        param_T, param_y, param_file = read_optional_qha_temperature_dat(qha_dir, spec["qha_names"])
        if param_T is None or param_y is None:
            continue
        lattice_parameters[spec["key"]] = {
            "T": param_T,
            "values": param_y,
            "file": param_file,
            "source": "file",
            "ylabel": spec["ylabel"],
        }
    if (
        "a" not in lattice_parameters
        and T_volume is not None
        and V_qha is not None
        and qha_formula_units > 0
    ):
        conventional_volume = V_qha * (4.0 / qha_formula_units)
        valid = conventional_volume > 0.0
        if np.any(valid):
            lattice_parameters["a"] = {
                "T": T_volume[valid],
                "values": conventional_volume[valid] ** (1.0 / 3.0),
                "file": qha_volume_file,
                "source": "derived_from_volume_cubic_z4",
                "ylabel": "Lattice a (Å)",
            }
    a_param = lattice_parameters.get("a", {})
    return {
        "T": T_qha,
        "Cp": Cp_qha,
        "H": H_qha,
        "S": S_qha,
        "V_T": T_volume,
        "V": V_qha,
        "a_T": a_param.get("T"),
        "a": a_param.get("values"),
        "lattice_parameters": lattice_parameters,
        "qha_dir": str(qha_dir),
        "qha_cp_file": str((qha_dir / "Cp-temperature.dat").resolve()),
        "qha_volume_file": qha_volume_file,
        "qha_lattice_file": a_param.get("file"),
        "qha_cp_unit": qha_cp_unit,
        "qha_formula_units": qha_formula_units,
        "temperature_min_K": float(T_qha[0]),
        "temperature_max_K": float(T_qha[-1]),
    }


def smoothstep_weights(T: np.ndarray, blend_start: float, blend_end: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if blend_end <= blend_start:
        return np.where(T >= blend_end, 1.0, 0.0)
    x = np.clip((T - blend_start) / (blend_end - blend_start), 0.0, 1.0)
    return 3.0 * x * x - 2.0 * x * x * x


def neel_activation_start(neel_t: float, apply_above_t: float) -> float:
    return max(0.0, min(float(neel_t), float(apply_above_t) - 30.0))


def neel_activation_weights(T: np.ndarray, neel_t: float, apply_above_t: float) -> np.ndarray:
    return smoothstep_weights(T, neel_activation_start(neel_t, apply_above_t), apply_above_t)


def default_qha_md_blend_interval(
    switch_T: float,
    qha_T: np.ndarray,
    md_T: np.ndarray,
) -> tuple[float, float]:
    overlap_min = max(float(np.min(qha_T)), float(np.min(md_T)))
    overlap_max = min(float(np.max(qha_T)), float(np.max(md_T)))
    if overlap_min <= overlap_max:
        half_width = min(50.0, max((overlap_max - overlap_min) / 4.0, 1.0))
        return max(overlap_min, switch_T - half_width), min(overlap_max, switch_T + half_width)
    return float(switch_T), float(switch_T)


def blend_qha_md_on_grid(
    T_grid: np.ndarray,
    qha_T: np.ndarray,
    qha_y: np.ndarray,
    md_y_grid: np.ndarray,
    blend_start: float,
    blend_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = smoothstep_weights(T_grid, blend_start, blend_end)
    qha_interp = np.interp(T_grid, qha_T, qha_y)
    out = md_y_grid.copy()
    out[T_grid < blend_start] = qha_interp[T_grid < blend_start]
    blend_mask = (T_grid >= blend_start) & (T_grid <= blend_end)
    out[blend_mask] = (
        (1.0 - weights[blend_mask]) * qha_interp[blend_mask]
        + weights[blend_mask] * md_y_grid[blend_mask]
    )
    return out, weights


def calibrate_blend_start_for_entropy_grid(
    T_grid: np.ndarray,
    qha_T: np.ndarray,
    qha_Cp: np.ndarray,
    md_Cp_grid: np.ndarray,
    original_start: float,
    blend_end: float,
    entropy_temperature: Optional[float],
    entropy_target: Optional[float],
    minimum_start: float,
) -> tuple[float, dict]:
    metadata = {
        "enabled": False,
        "reason": "no experimental entropy anchor",
        "original_blend_start_K": original_start,
        "calibrated_blend_start_K": original_start,
        "blend_end_K": blend_end,
        "minimum_allowed_blend_start_K": minimum_start,
        "entropy_anchor_temperature_K": entropy_temperature,
        "entropy_anchor_target_J_mol_formula_K": entropy_target,
    }
    if entropy_temperature is None or entropy_target is None:
        return original_start, metadata
    if entropy_temperature < float(np.min(T_grid)) or entropy_temperature > float(np.max(T_grid)):
        metadata["reason"] = "entropy anchor is outside the plotting/integration grid"
        return original_start, metadata
    if entropy_temperature >= blend_end:
        metadata["reason"] = "entropy anchor is not below the blend end"
        return original_start, metadata
    lower_bound = max(float(minimum_start), float(np.min(qha_T)), float(np.min(T_grid)))
    upper_bound = min(float(original_start), float(entropy_temperature))
    if upper_bound <= lower_bound:
        metadata["reason"] = "no allowed lower blend-start interval reaches the entropy anchor"
        return original_start, metadata

    def entropy_with_start(start: float) -> float:
        cp_grid, _weights = blend_qha_md_on_grid(
            T_grid,
            qha_T,
            qha_Cp,
            md_Cp_grid,
            start,
            blend_end,
        )
        Cp_over_T = np.divide(
            cp_grid,
            T_grid,
            out=np.zeros_like(cp_grid),
            where=T_grid > 0.0,
        )
        S_grid = trapz_cumulative(T_grid, Cp_over_T)
        return float(np.interp(entropy_temperature, T_grid, S_grid))

    current = entropy_with_start(original_start)
    low = entropy_with_start(lower_bound)
    metadata["S_at_original_blend_start_J_mol_formula_K"] = current
    metadata["S_at_minimum_blend_start_J_mol_formula_K"] = low
    lo_value, hi_value = sorted((current, low))
    if entropy_target < lo_value or entropy_target > hi_value:
        metadata["reason"] = "entropy anchor is outside the reachable calibration range"
        metadata["target_clipped_to_reachable_range"] = True
        entropy_target = min(max(entropy_target, lo_value), hi_value)
    direction = 1.0 if low >= current else -1.0
    lo_t, hi_t = lower_bound, upper_bound
    for _idx in range(40):
        mid = 0.5 * (lo_t + hi_t)
        mid_value = entropy_with_start(mid)
        if direction * (mid_value - entropy_target) >= 0.0:
            lo_t = mid
        else:
            hi_t = mid
    calibrated = 0.5 * (lo_t + hi_t)
    metadata.update(
        {
            "enabled": True,
            "reason": "blend_start calibrated to experimental/database entropy anchor",
            "calibrated_blend_start_K": calibrated,
            "S_at_calibrated_blend_start_J_mol_formula_K": entropy_with_start(calibrated),
        }
    )
    return calibrated, metadata


def neel_enthalpy_j_mol(neel_t: float, neel_entropy: float, neel_enthalpy: str) -> float:
    if neel_enthalpy == "auto":
        return float(neel_t) * float(neel_entropy)
    return float(neel_enthalpy) * 1000.0


def neel_entropy_reference_300(anchor_metadata: Optional[dict], thermo_formula: Optional[str]) -> dict:
    db_anchor = (anchor_metadata or {}).get("thermo_db_anchor")
    if db_anchor and str(db_anchor.get("formula", "")).upper() == "UO2":
        return {
            "source": db_anchor["database"],
            "formula": db_anchor["formula"],
            "T_K": db_anchor["temperature_value_K"],
            "S_J_mol_formula_K": db_anchor["S_J_mol_formula_K"],
        }
    if thermo_formula and thermo_formula.upper() == "UO2":
        return {
            "source": "JAEA-literature-default",
            "formula": "UO2",
            "T_K": 300.0,
            "S_J_mol_formula_K": 77.81270,
        }
    return {}


def apply_neel_correction_grid(
    *,
    T_grid: np.ndarray,
    S_grid: np.ndarray,
    H_grid: np.ndarray,
    neel_correction: str,
    neel_t: float,
    neel_entropy: float,
    neel_enthalpy: str,
    neel_apply_above_t: float,
    entropy_anchor_is_direct: bool,
    anchor_metadata: Optional[dict],
    thermo_formula: Optional[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    weights = np.zeros_like(T_grid, dtype=float)
    metadata = {
        "enabled": neel_correction == "on",
        "applied": False,
        "reason": "disabled",
    }
    if neel_correction != "on":
        return S_grid, H_grid, H_grid - T_grid * S_grid, weights, metadata
    if entropy_anchor_is_direct:
        metadata["reason"] = "skipped because direct experimental entropy anchoring is active"
        return S_grid, H_grid, H_grid - T_grid * S_grid, weights, metadata
    activation_start = neel_activation_start(neel_t, neel_apply_above_t)
    weights = neel_activation_weights(T_grid, neel_t, neel_apply_above_t)
    delta_s = float(neel_entropy)
    delta_h = neel_enthalpy_j_mol(neel_t, neel_entropy, neel_enthalpy)
    S_corrected = S_grid + delta_s * weights
    H_corrected = H_grid + delta_h * weights
    G_corrected = H_corrected - T_grid * S_corrected
    ref = neel_entropy_reference_300(anchor_metadata, thermo_formula)
    s_before_300 = float(np.interp(300.0, T_grid, S_grid)) if T_grid[0] <= 300.0 <= T_grid[-1] else None
    s_after_300 = (
        float(np.interp(300.0, T_grid, S_corrected))
        if T_grid[0] <= 300.0 <= T_grid[-1]
        else None
    )
    gap_before = None
    gap_after = None
    can_explain = None
    if ref and s_before_300 is not None and s_after_300 is not None:
        gap_before = ref["S_J_mol_formula_K"] - s_before_300
        gap_after = ref["S_J_mol_formula_K"] - s_after_300
        can_explain = abs(gap_after) < abs(gap_before) and abs(gap_after) <= max(1.0, 0.25 * abs(gap_before))
    metadata.update(
        {
            "applied": True,
            "reason": "explicit phonon-QHA plus Neel entropy correction",
            "neel_T_K": neel_t,
            "activation_start_K": activation_start,
            "apply_full_above_T_K": neel_apply_above_t,
            "delta_S_J_mol_formula_K": delta_s,
            "delta_H_J_mol_formula": delta_h,
            "enthalpy_mode": neel_enthalpy,
            "S_300_before_J_mol_formula_K": s_before_300,
            "S_300_after_J_mol_formula_K": s_after_300,
            "S_reference_300": ref,
            "delta_S_gap_before_J_mol_formula_K": gap_before,
            "delta_S_gap_after_J_mol_formula_K": gap_after,
            "neel_entropy_can_explain_gap": can_explain,
            "double_counting_guard": (
                "database S is treated as a benchmark only when explicit Neel correction is enabled"
            ),
        }
    )
    return S_corrected, H_corrected, G_corrected, weights, metadata


def apply_entropy_anchor_grid(
    *,
    T_grid: np.ndarray,
    S_grid: np.ndarray,
    H_grid: np.ndarray,
    entropy_anchor_T: Optional[float],
    entropy_anchor_S: Optional[float],
    source: Optional[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    metadata = {
        "applied": False,
        "source": source,
        "T_K": entropy_anchor_T,
        "target_J_mol_formula_K": entropy_anchor_S,
    }
    if entropy_anchor_T is None or entropy_anchor_S is None:
        metadata["reason"] = "no direct entropy anchor"
        return S_grid, H_grid - T_grid * S_grid, metadata
    if entropy_anchor_T < float(np.min(T_grid)) or entropy_anchor_T > float(np.max(T_grid)):
        metadata["reason"] = "entropy anchor temperature is outside the integration grid"
        return S_grid, H_grid - T_grid * S_grid, metadata
    current = float(np.interp(entropy_anchor_T, T_grid, S_grid))
    shift = float(entropy_anchor_S) - current
    S_out = S_grid + shift
    G_out = H_grid - T_grid * S_out
    metadata.update(
        {
            "applied": True,
            "S_before_anchor_J_mol_formula_K": current,
            "shift_J_mol_formula_K": shift,
            "S_after_anchor_J_mol_formula_K": float(entropy_anchor_S),
            "note": "Direct entropy anchor shifts S and recomputes G = H - TS.",
        }
    )
    return S_out, G_out, metadata


def neel_adjusted_entropy_benchmark(
    db_anchor: dict,
    neel_t: float,
    neel_entropy: float,
    neel_apply_above_t: float,
) -> tuple[float, dict]:
    neel_weight = float(
        neel_activation_weights(
            np.array([db_anchor["temperature_value_K"]], dtype=float),
            neel_t,
            neel_apply_above_t,
        )[0]
    )
    neel_offset = float(neel_entropy) * neel_weight
    target = db_anchor["S_J_mol_formula_K"] - neel_offset
    return target, {
        "source": db_anchor["database"],
        "formula": db_anchor["formula"],
        "T_K": db_anchor["temperature_value_K"],
        "used_as_entropy_anchor": False,
        "used_for_blend_calibration": True,
        "benchmark_S_J_mol_formula_K": db_anchor["S_J_mol_formula_K"],
        "neel_entropy_subtracted_J_mol_formula_K": neel_offset,
        "value_J_mol_formula_K": target,
        "reason": (
            "explicit Neel correction enabled: database S is used as a "
            "benchmark after subtracting the user Neel entropy contribution"
        ),
    }


def cp_overlap_diagnostics_np(
    qha_T: np.ndarray,
    qha_Cp: np.ndarray,
    md_T: np.ndarray,
    md_Cp: np.ndarray,
    blend_start: float,
    blend_end: float,
) -> dict:
    temps = sorted(
        {
            float(t)
            for t in np.concatenate([qha_T, md_T, np.array([blend_start, blend_end])])
            if blend_start <= float(t) <= blend_end
        }
    )
    rows = []
    deltas = []
    rels = []
    signs = []
    for temp in temps:
        if temp < qha_T[0] or temp > qha_T[-1] or temp < md_T[0] or temp > md_T[-1]:
            continue
        qha_value = float(np.interp(temp, qha_T, qha_Cp))
        md_value = float(np.interp(temp, md_T, md_Cp))
        delta = md_value - qha_value
        rel = abs(delta) / max(abs(qha_value), 1.0e-12)
        rows.append({
            "T_K": temp,
            "Cp_QHA": qha_value,
            "Cp_MD": md_value,
            "Cp_MD_minus_QHA": delta,
            "relative_mismatch": rel,
        })
        deltas.append(delta)
        rels.append(rel)
        signs.append(0 if abs(delta) <= 1.0e-12 else (1 if delta > 0 else -1))
    crossing = any(a * b < 0 for a, b in zip(signs, signs[1:])) or any(sign == 0 for sign in signs)
    mean_abs = float(np.mean(np.abs(deltas))) if deltas else float("nan")
    rms = float(np.sqrt(np.mean(np.square(deltas)))) if deltas else float("nan")
    mean_rel = float(np.mean(rels)) if rels else float("nan")
    return {
        "rows": rows,
        "mean_absolute_cp_mismatch": mean_abs,
        "rms_cp_mismatch": rms,
        "mean_relative_cp_mismatch": mean_rel,
        "relative_mismatch_at_blend_start": rows[0]["relative_mismatch"] if rows else None,
        "relative_mismatch_at_blend_end": rows[-1]["relative_mismatch"] if rows else None,
        "cp_curves_cross_in_blend_interval": bool(crossing),
        "warning_cp_mismatch_exceeds_10_percent": bool(any(rel > 0.10 for rel in rels)),
    }


def plot_overlap_mismatch_np(outpath: Path, diagnostics: dict) -> None:
    rows = diagnostics.get("rows", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(
        [row["T_K"] for row in rows],
        [row["relative_mismatch"] * 100.0 for row in rows],
        "o-",
        color="#111111",
        linewidth=1.8,
        markersize=3.5,
    )
    ax.axhline(10.0, color="#b00020", linestyle="--", linewidth=1.2, label="10% warning")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("|Cp_MD - Cp_QHA| / |Cp_QHA| (%)")
    ax.set_title("QHA/MD Cp Mismatch In Blend Region")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def integrate_qha_cp_anchor(
    qha_dir: Path,
    anchor_T: float,
    qha_formula_units: float,
    qha_cp_unit: str = "J/mol-cell/K",
) -> dict:
    """
    Derive Cp, S, and relative H anchor values from QHA Cp(T).

    Outputs are per mole of formula units, matching lammps-thermo-series.
    H is relative to the first QHA temperature. S is integrated as Cp/T from
    the first QHA temperature, using 0 for the endpoint integrand at T=0.
    """
    curve = qha_cp_thermo_curve(qha_dir, qha_formula_units, qha_cp_unit)
    T_qha = curve["T"]
    if anchor_T < T_qha[0] or anchor_T > T_qha[-1]:
        raise ValueError(
            f"QHA anchor T={anchor_T:g} K is outside Cp-temperature.dat range "
            f"{T_qha[0]:g}-{T_qha[-1]:g} K"
        )
    anchor = {
        "source": "qha-cp-integration",
        "qha_dir": curve["qha_dir"],
        "qha_cp_file": curve["qha_cp_file"],
        "qha_cp_unit": curve["qha_cp_unit"],
        "qha_formula_units": curve["qha_formula_units"],
        "T_K": float(anchor_T),
        "Cp_J_mol_formula_K": float(np.interp(anchor_T, curve["T"], curve["Cp"])),
        "H_J_mol_formula": float(np.interp(anchor_T, curve["T"], curve["H"])),
        "S_J_mol_formula_K": float(np.interp(anchor_T, curve["T"], curve["S"])),
        "integration_reference_T_K": float(T_qha[0]),
        "temperature_min_K": curve["temperature_min_K"],
        "temperature_max_K": curve["temperature_max_K"],
        "note": "S and H were numerically integrated from QHA Cp(T).",
    }
    if T_qha[0] > 1.0e-12:
        anchor["note"] += (
            " QHA Cp data did not start at 0 K, so S/H are relative "
            "to the first QHA T."
        )
    return anchor


def choose_qha_md_cp_switch(
    qha_T: np.ndarray,
    qha_Cp: np.ndarray,
    md_T: np.ndarray,
    md_Cp: np.ndarray,
    requested: Optional[float] = None,
    minimum: Optional[float] = 50.0,
) -> tuple[Optional[float], str]:
    if requested is not None:
        return float(requested), "manual"
    qha_mask = np.isfinite(qha_T) & np.isfinite(qha_Cp)
    md_mask = np.isfinite(md_T) & np.isfinite(md_Cp)
    qha_T = qha_T[qha_mask]
    qha_Cp = qha_Cp[qha_mask]
    md_T = md_T[md_mask]
    md_Cp = md_Cp[md_mask]
    if len(qha_T) == 0 or len(md_T) == 0:
        return None, "missing-cp-source"
    qha_order = np.argsort(qha_T)
    md_order = np.argsort(md_T)
    qha_T = qha_T[qha_order]
    qha_Cp = qha_Cp[qha_order]
    md_T = md_T[md_order]
    md_Cp = md_Cp[md_order]
    overlap_min = max(float(qha_T[0]), float(md_T[0]))
    overlap_max = min(float(qha_T[-1]), float(md_T[-1]))
    if minimum is not None:
        overlap_min = max(overlap_min, float(minimum))
    if overlap_min <= overlap_max:
        candidates = sorted(
            {
                float(temp)
                for temp in np.concatenate([qha_T, md_T])
                if overlap_min <= temp <= overlap_max
            }
        )
        if not candidates:
            candidates = [overlap_min, overlap_max]

        def delta(temp: float) -> float:
            return abs(np.interp(temp, qha_T, qha_Cp) - np.interp(temp, md_T, md_Cp))

        return min(candidates, key=delta), "overlap-closest-cp"
    if qha_T[-1] < md_T[0]:
        switch = 0.5 * (float(qha_T[-1]) + float(md_T[0]))
        if minimum is not None:
            switch = max(switch, float(minimum))
        return switch, "gap-midpoint-qha-low-md-high"
    if md_T[-1] < qha_T[0]:
        switch = 0.5 * (float(md_T[-1]) + float(qha_T[0]))
        if minimum is not None and switch < float(minimum):
            return None, "gap-switch-below-minimum"
        return switch, "gap-midpoint-md-low-qha-high"
    return None, "no-switch-found"


def summary_series(summaries: list[dict], keys: list[str]) -> tuple[np.ndarray | None, str]:
    for key in keys:
        values = np.array([s.get(key, np.nan) for s in summaries], dtype=float)
        if np.isfinite(values).sum() >= 2:
            return values, key
    return None, ""


def fit_grid_from_series(T: np.ndarray, y: np.ndarray, T_grid: np.ndarray, degree: int) -> np.ndarray:
    finite = np.isfinite(T) & np.isfinite(y)
    if finite.sum() >= 3:
        deg = min(degree, finite.sum() - 1)
        return np.poly1d(np.polyfit(T[finite], y[finite], deg))(T_grid)
    if finite.sum() >= 2:
        return np.interp(T_grid, T[finite], y[finite])
    return np.full_like(T_grid, np.nan, dtype=float)


def parse_lattice_references(items: list[str] | None) -> dict[str, float]:
    refs = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Reference must look like key=value, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Reference key is empty in: {item}")
        refs[key] = float(value)
    return refs


def should_correct_source(source: str, apply_to: str) -> bool:
    return apply_to in ("both", source)


def correct_array_to_reference(
    T_values: np.ndarray,
    y_values: np.ndarray,
    *,
    source: str,
    reference_temperature: Optional[float],
    reference_value: Optional[float],
    correction: str,
    apply_to: str,
) -> tuple[np.ndarray, dict]:
    metadata = {
        "source": source,
        "correction": correction,
        "apply_to": apply_to,
        "reference_T_K": reference_temperature,
        "reference_value": reference_value,
        "applied": False,
    }
    y_values = np.array(y_values, dtype=float).copy()
    if (
        correction == "none"
        or reference_temperature is None
        or reference_value is None
        or not should_correct_source(source, apply_to)
        or len(T_values) == 0
    ):
        return y_values, metadata
    finite = np.isfinite(T_values) & np.isfinite(y_values)
    if finite.sum() < 2:
        metadata["note"] = "Not enough finite values for correction"
        return y_values, metadata
    t_finite = T_values[finite]
    y_finite = y_values[finite]
    if reference_temperature < np.min(t_finite) or reference_temperature > np.max(t_finite):
        metadata["note"] = "Reference temperature is outside curve range"
        return y_values, metadata
    raw_ref = float(np.interp(reference_temperature, t_finite, y_finite))
    metadata["raw_value_at_reference"] = raw_ref
    if correction == "shift":
        shift = reference_value - raw_ref
        metadata["shift"] = shift
        metadata["applied"] = True
        return y_values + shift, metadata
    if correction == "scale":
        if abs(raw_ref) <= 1.0e-12:
            metadata["note"] = "Cannot scale because raw reference value is zero"
            return y_values, metadata
        scale = reference_value / raw_ref
        metadata["scale"] = scale
        metadata["applied"] = True
        return y_values * scale, metadata
    raise ValueError(f"Unsupported structural correction: {correction}")


def fill_missing_anchors_from_qha(
    *,
    thermo_anchor_T: Optional[float],
    thermo_anchor_S_J_mol_K: Optional[float],
    thermo_anchor_Cp_J_mol_K: Optional[float],
    thermo_anchor_H_J_mol: Optional[float],
    qha_anchor_dir: Optional[Path],
    qha_anchor_formula_units: Optional[float],
    qha_anchor_cp_unit: str,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], dict]:
    if qha_anchor_dir is None:
        return (
            thermo_anchor_T,
            thermo_anchor_S_J_mol_K,
            thermo_anchor_Cp_J_mol_K,
            thermo_anchor_H_J_mol,
            {},
        )
    if qha_anchor_formula_units is None:
        raise ValueError("--qha-anchor-formula-units is required with --qha-anchor-dir")
    anchor_T = 300.0 if thermo_anchor_T is None else float(thermo_anchor_T)
    qha_anchor = integrate_qha_cp_anchor(
        qha_dir=qha_anchor_dir,
        anchor_T=anchor_T,
        qha_formula_units=qha_anchor_formula_units,
        qha_cp_unit=qha_anchor_cp_unit,
    )
    filled = {
        "source": qha_anchor["source"],
        "manual_values_take_priority": True,
        "qha_anchor": qha_anchor,
        "filled_fields": [],
    }
    if thermo_anchor_T is None:
        thermo_anchor_T = qha_anchor["T_K"]
        filled["filled_fields"].append("thermo_anchor_T")
    if thermo_anchor_Cp_J_mol_K is None:
        thermo_anchor_Cp_J_mol_K = qha_anchor["Cp_J_mol_formula_K"]
        filled["filled_fields"].append("thermo_anchor_Cp_J_mol_K")
    if thermo_anchor_S_J_mol_K is None:
        thermo_anchor_S_J_mol_K = qha_anchor["S_J_mol_formula_K"]
        filled["filled_fields"].append("thermo_anchor_S_J_mol_K")
    if thermo_anchor_H_J_mol is None:
        thermo_anchor_H_J_mol = qha_anchor["H_J_mol_formula"]
        filled["filled_fields"].append("thermo_anchor_H_J_mol")
    return (
        thermo_anchor_T,
        thermo_anchor_S_J_mol_K,
        thermo_anchor_Cp_J_mol_K,
        thermo_anchor_H_J_mol,
        filled,
    )


def fill_enthalpy_anchor_from_thermo_db(
    *,
    thermo_db: Optional[str],
    thermo_formula: Optional[str],
    thermo_phase: str,
    thermo_db_temperature: Optional[float],
    thermo_anchor_T: Optional[float],
    thermo_anchor_H_J_mol: Optional[float],
    anchor_metadata: Optional[dict],
) -> tuple[Optional[float], Optional[float], dict]:
    if thermo_db is None:
        return thermo_anchor_T, thermo_anchor_H_J_mol, anchor_metadata or {}
    if thermo_db != "jaea":
        raise ValueError(f"Unsupported thermodynamic database: {thermo_db}")
    if not thermo_formula:
        raise ValueError("--thermo-formula is required with --thermo-db")
    anchor_t = thermo_anchor_T
    if anchor_t is None:
        anchor_t = thermo_db_temperature if thermo_db_temperature is not None else 300.0
    db_anchor = jaea_anchor(thermo_formula, anchor_t, phase=thermo_phase)
    filled_fields = []
    qha_filled_h = "thermo_anchor_H_J_mol" in (anchor_metadata or {}).get("filled_fields", [])
    if thermo_anchor_H_J_mol is None or qha_filled_h:
        thermo_anchor_H_J_mol = db_anchor["H_J_mol_formula"]
        filled_fields.append("thermo_anchor_H_J_mol")
    metadata = dict(anchor_metadata or {})
    metadata["thermo_db_anchor"] = db_anchor
    metadata["thermo_db_filled_fields"] = filled_fields
    return anchor_t, thermo_anchor_H_J_mol, metadata


def build_combined_thermo(summaries: list[dict],
                          outdir: Path,
                          fit_degree: int = 3,
                          cp_source: str = "fluct",
                          plot_T_min: Optional[float] = None,
                          plot_T_max: Optional[float] = None,
                          plot_T_step: float = 10.0,
                          anchor_zero: bool = False,
                          n_bootstrap: int = 300,
                          bootstrap_seed: int = 12345,
                          thermo_anchor_T: Optional[float] = None,
                          thermo_anchor_S_J_mol_K: Optional[float] = None,
                          thermo_anchor_Cp_J_mol_K: Optional[float] = None,
                          thermo_anchor_H_J_mol: Optional[float] = None,
                          use_anchor_for_integration: bool = False,
                          use_anchor_Cp_in_fit: bool = False,
                          anchor_metadata: Optional[dict] = None,
                          qha_low_t_curve: Optional[dict] = None,
                          qha_splice_switch_temperature: Optional[float] = None,
                          qha_splice_min_switch_temperature: float = 50.0,
                          qha_splice_blend_start: Optional[float] = None,
                          qha_splice_blend_end: Optional[float] = None,
                          entropy_anchor_blend_T: Optional[float] = None,
                          entropy_anchor_blend_S_J_mol_K: Optional[float] = None,
                          entropy_anchor_min_blend_start: float = 200.0,
                          entropy_anchor_is_direct: bool = False,
                          neel_correction: str = "off",
                          neel_t: float = 30.8,
                          neel_entropy: float = 8.4,
                          neel_enthalpy: str = "auto",
                          neel_apply_above_t: float = 50.0,
                          thermo_formula: Optional[str] = None,
                          structure_reference_temperature: Optional[float] = None,
                          volume_reference: Optional[float] = None,
                          lattice_references: Optional[dict[str, float]] = None,
                          structure_correction: str = "none",
                          structure_correction_apply_to: str = "both",
                          plot_thermo_db_points: bool = False,
                          target_z: float = 4.0) -> list[dict]:
    outdir.mkdir(parents=True, exist_ok=True)

    T_data_all = np.array([s["target_T_K"] for s in summaries], dtype=float)

    # Temperature range selection.
    # If plot_T_max is requested, use the closest available completed production T
    # at or below the requested high-T limit. If none is below, use the nearest available.
    Tmax_available = float(np.max(T_data_all))
    Tmin_available = float(np.min(T_data_all))

    if plot_T_max is None:
        T_high_dataset = Tmax_available
    else:
        below = T_data_all[T_data_all <= plot_T_max]
        if len(below) > 0:
            T_high_dataset = float(np.max(below))
        else:
            T_high_dataset = float(T_data_all[np.argmin(np.abs(T_data_all - plot_T_max))])

    # For fitting, only use data up to the selected highest completed T.
    use_mask = T_data_all <= T_high_dataset + 1.0e-9
    summaries = [s for s, keep in zip(summaries, use_mask) if keep]

    # Number of UO2 formula units in the MD simulation cell.
    # For 2x2x2 fluorite UO2: 96 atoms / 3 atoms per UO2 = 32 formula units.
    # This must be defined before any eV/cell -> kJ/mol-UO2 conversion.
    nfu = float(summaries[0].get("n_formula_units", 32.0))
    if nfu <= 0:
        raise ValueError("n_formula_units must be positive")
    if target_z <= 0:
        raise ValueError("--target-z must be positive")

    T = np.array([s["target_T_K"] for s in summaries], dtype=float)
    V = np.array([s["V_mean_A3"] for s in summaries], dtype=float)
    a = np.array([s.get("a_mean_A", s.get("a_proxy_mean_A")) for s in summaries], dtype=float)
    rho = np.array([s["density_mean_g_cm3"] for s in summaries], dtype=float)
    H_eV = np.array([s["H_mean_eV_cell"] for s in summaries], dtype=float)
    H_kJ_mol = H_eV * EV_CELL_TO_KJ_PER_MOL_UO2 / summaries[0].get("n_formula_units", 32.0)
    Cp_fluct = np.array([s["Cp_fluct_J_per_mol_UO2_K"] for s in summaries], dtype=float)
    KT = np.array([s["KT_GPa_from_V_fluct"] for s in summaries], dtype=float)

    V_sem = np.array([s.get("V_sem_A3", np.nan) for s in summaries], dtype=float)
    a_sem = np.array([s.get("a_sem_A", np.nan) for s in summaries], dtype=float)
    H_sem = np.array([s.get("H_sem_eV_cell", np.nan) for s in summaries], dtype=float)
    Cp_sem = np.array([s.get("Cp_fluct_sem_J_per_mol_UO2_K", np.nan) for s in summaries], dtype=float)
    rho_sem = np.array([s.get("density_sem_g_cm3", np.nan) for s in summaries], dtype=float)

    # Plot/evaluation grid. This can start from 0 K even if no MD data exist at 0 K.
    if plot_T_min is None:
        T_grid_min = 0.0 if anchor_zero else float(np.min(T))
    else:
        T_grid_min = float(plot_T_min)

    T_grid_max = T_high_dataset
    if plot_T_step <= 0:
        raise ValueError("plot_T_step must be positive.")
    T_grid = np.arange(T_grid_min, T_grid_max + 0.5 * plot_T_step, plot_T_step)
    if len(T_grid) == 0 or T_grid[-1] < T_grid_max:
        T_grid = np.append(T_grid, T_grid_max)

    # Fit V and a using available MD data; evaluate both at MD T and on user grid.
    deg_fit = min(fit_degree, max(len(T) - 1, 1))
    coeff_V = np.polyfit(T, V, deg_fit)
    pV = np.poly1d(coeff_V)
    dpV = np.polyder(pV)
    V_fit = pV(T)
    dVdT = dpV(T)
    V_grid = pV(T_grid)
    dVdT_grid = dpV(T_grid)

    coeff_a = np.polyfit(T, a, deg_fit)
    pa = np.poly1d(coeff_a)
    dpa = np.polyder(pa)
    a_fit = pa(T)
    dadT = dpa(T)
    a_grid = pa(T_grid)
    dadT_grid = dpa(T_grid)
    lattice_md = {
        "a": {
            "T": T,
            "values": a,
            "grid": a_grid.copy(),
            "summary_key": "a_mean_A",
            "ylabel": "Lattice a (Å)",
        }
    }
    for spec in LATTICE_PARAMETER_SPECS:
        if spec["key"] == "a":
            continue
        values, summary_key = summary_series(summaries, spec["summary_keys"])
        if values is None:
            continue
        lattice_md[spec["key"]] = {
            "T": T,
            "values": values,
            "grid": fit_grid_from_series(T, values, T_grid, deg_fit),
            "summary_key": summary_key,
            "ylabel": spec["ylabel"],
        }

    alpha_V = dVdT / V_fit
    alpha_L = dadT / a_fit
    alpha_V_grid = dVdT_grid / V_grid
    alpha_L_grid = dadT_grid / a_grid

    # H derivative gives Cp estimate in J/mol-UO2/K.
    coeff_H = np.polyfit(T, H_eV, deg_fit)
    pH = np.poly1d(coeff_H)
    dpH = np.polyder(pH)
    H_fit_eV = pH(T)
    H_grid_eV = pH(T_grid)
    H_abs_kJ_per_mol_UO2_grid = H_grid_eV * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu
    dHdT_eV_per_K_cell = dpH(T)
    dHdT_grid_eV_per_K_cell = dpH(T_grid)

    nfu = float(nfu)
    Cp_from_H = dHdT_eV_per_K_cell * EV_TO_J * NA / nfu
    Cp_from_H_grid = dHdT_grid_eV_per_K_cell * EV_TO_J * NA / nfu

    if cp_source == "dH":
        Cp_for_integral_data = Cp_from_H
    else:
        Cp_for_integral_data = Cp_fluct

    # Optional external Cp anchor, e.g. Cp(300 K) from phonopy/QHA or literature.
    # This constrains the Cp fit/integration but does not change the MD raw summaries.
    T_for_Cp_fit = T
    Cp_for_Cp_fit = Cp_for_integral_data
    if use_anchor_Cp_in_fit and thermo_anchor_T is not None and thermo_anchor_Cp_J_mol_K is not None:
        T_for_Cp_fit, Cp_for_Cp_fit = insert_anchor_point_for_cp_fit(
            T, Cp_for_integral_data, thermo_anchor_T, thermo_anchor_Cp_J_mol_K
        )

    # Smooth Cp for integration using polynomial fit if enough finite points.
    finite = np.isfinite(Cp_for_Cp_fit)
    if finite.sum() >= 3:
        cp_deg = min(3, finite.sum() - 1)
        coeff_Cp = np.polyfit(T_for_Cp_fit[finite], Cp_for_Cp_fit[finite], cp_deg)
        pCp = np.poly1d(coeff_Cp)
        Cp_smooth = pCp(T)
        Cp_grid = pCp(T_grid)
    else:
        Cp_smooth = Cp_for_integral_data
        Cp_grid = np.interp(T_grid, T_for_Cp_fit, Cp_for_Cp_fit)

    # Optional physical anchor at 0 K for integrals: Cp(0)=0, S(0)=0, H_rel(0)=0, G_rel(0)=0.
    # This does not add a fake MD data point; it only anchors the integration grid.
    if anchor_zero and T_grid[0] > 0:
        T_grid = np.insert(T_grid, 0, 0.0)
        V_grid = np.insert(V_grid, 0, pV(0.0))
        a_grid = np.insert(a_grid, 0, pa(0.0))
        alpha_V_grid = np.insert(alpha_V_grid, 0, dpV(0.0) / pV(0.0))
        alpha_L_grid = np.insert(alpha_L_grid, 0, dpa(0.0) / pa(0.0))
        H_grid_eV = np.insert(H_grid_eV, 0, pH(0.0))
        Cp_from_H_grid = np.insert(Cp_from_H_grid, 0, dpH(0.0) * EV_TO_J * NA / nfu)
        Cp_grid = np.insert(Cp_grid, 0, 0.0)
    elif anchor_zero and abs(T_grid[0]) < 1e-12:
        Cp_grid[0] = 0.0

    lattice_md["a"]["grid"] = a_grid.copy()
    for key, item in lattice_md.items():
        if key == "a":
            continue
        item["grid"] = fit_grid_from_series(T, item["values"], T_grid, deg_fit)

    Cp_md_grid = Cp_grid.copy()
    V_md_grid = V_grid.copy()
    qha_blend_weights = np.ones_like(T_grid, dtype=float)
    lattice_hybrid = {}
    lattice_references = lattice_references or {}
    thermo_db_anchor = (anchor_metadata or {}).get("thermo_db_anchor")
    thermo_db_points = {}
    if plot_thermo_db_points and thermo_db_anchor:
        label = f"{thermo_db_anchor['database'].upper()} {thermo_db_anchor['formula']}"
        temp = thermo_db_anchor["temperature_value_K"]
        thermo_db_points = {
            "S": [{"T_K": temp, "value": thermo_db_anchor["S_J_mol_formula_K"], "label": label}],
            "H": [{"T_K": temp, "value": thermo_db_anchor["H_J_mol_formula"] / 1000.0, "label": label}],
            "G": [{"T_K": temp, "value": thermo_db_anchor["G_J_mol_formula"] / 1000.0, "label": label}],
        }
    structural_metadata = {
        "reference_T_K": structure_reference_temperature,
        "volume_reference": volume_reference,
        "lattice_references": lattice_references,
        "correction_type": structure_correction,
        "apply_to": structure_correction_apply_to,
        "note": "CTE is derived from corrected hybrid V/a curves; CTE is not blended directly.",
    }
    qha_splice_metadata = {}
    if qha_low_t_curve:
        switch_T, switch_method = choose_qha_md_cp_switch(
            qha_low_t_curve["T"],
            qha_low_t_curve["Cp"],
            T,
            Cp_for_integral_data,
            requested=qha_splice_switch_temperature,
            minimum=qha_splice_min_switch_temperature,
        )
        if switch_T is not None:
            if (qha_splice_blend_start is None) != (qha_splice_blend_end is None):
                raise ValueError(
                    "--qha-splice-blend-start and --qha-splice-blend-end must be used together"
                )
            if qha_splice_blend_start is None:
                blend_start, blend_end = default_qha_md_blend_interval(
                    switch_T,
                    qha_low_t_curve["T"],
                    T,
                )
            else:
                blend_start = float(qha_splice_blend_start)
                blend_end = float(qha_splice_blend_end)
            if blend_end < blend_start:
                raise ValueError("--qha-splice-blend-end must be >= --qha-splice-blend-start")
            blend_start, entropy_blend_calibration = calibrate_blend_start_for_entropy_grid(
                T_grid,
                qha_low_t_curve["T"],
                qha_low_t_curve["Cp"],
                Cp_md_grid,
                blend_start,
                blend_end,
                entropy_anchor_blend_T,
                entropy_anchor_blend_S_J_mol_K,
                entropy_anchor_min_blend_start,
            )
            Cp_grid, qha_blend_weights = blend_qha_md_on_grid(
                T_grid,
                qha_low_t_curve["T"],
                qha_low_t_curve["Cp"],
                Cp_md_grid,
                blend_start,
                blend_end,
            )
            qha_volume_mode = "md-only"
            qha_lattice_modes = {}
            lattice_hybrid = {}
            if qha_low_t_curve.get("V") is not None and qha_low_t_curve.get("V_T") is not None:
                qha_V_scaled = qha_low_t_curve["V"] * (nfu / qha_low_t_curve["qha_formula_units"])
                qha_V_corrected, qha_V_correction = correct_array_to_reference(
                    qha_low_t_curve["V_T"],
                    qha_V_scaled,
                    source="qha",
                    reference_temperature=structure_reference_temperature,
                    reference_value=volume_reference,
                    correction=structure_correction,
                    apply_to=structure_correction_apply_to,
                )
                V_md_corrected, md_V_correction = correct_array_to_reference(
                    T_grid,
                    V_md_grid,
                    source="md",
                    reference_temperature=structure_reference_temperature,
                    reference_value=volume_reference,
                    correction=structure_correction,
                    apply_to=structure_correction_apply_to,
                )
                V_grid, _weights = blend_qha_md_on_grid(
                    T_grid,
                    qha_low_t_curve["V_T"],
                    qha_V_corrected,
                    V_md_corrected,
                    blend_start,
                    blend_end,
                )
                dVdT_grid = np.gradient(V_grid, T_grid)
                alpha_V_grid = dVdT_grid / V_grid
                qha_volume_mode = "hybrid"
                structural_metadata["volume"] = {
                    "qha_correction": qha_V_correction,
                    "md_correction": md_V_correction,
                    "source_mode": qha_volume_mode,
                }
            for key, md_item in lattice_md.items():
                qha_item = qha_low_t_curve.get("lattice_parameters", {}).get(key)
                qha_lattice_modes[key] = "md-only"
                md_grid_corrected, md_correction = correct_array_to_reference(
                    T_grid,
                    md_item["grid"],
                    source="md",
                    reference_temperature=structure_reference_temperature,
                    reference_value=lattice_references.get(key),
                    correction=structure_correction,
                    apply_to=structure_correction_apply_to,
                )
                hybrid_grid = md_grid_corrected.copy()
                qha_correction = {}
                qha_values_corrected = None
                if qha_item is not None:
                    qha_values_corrected, qha_correction = correct_array_to_reference(
                        qha_item["T"],
                        qha_item["values"],
                        source="qha",
                        reference_temperature=structure_reference_temperature,
                        reference_value=lattice_references.get(key),
                        correction=structure_correction,
                        apply_to=structure_correction_apply_to,
                    )
                    hybrid_grid, _weights = blend_qha_md_on_grid(
                        T_grid,
                        qha_item["T"],
                        qha_values_corrected,
                        md_grid_corrected,
                        blend_start,
                        blend_end,
                    )
                    qha_lattice_modes[key] = "hybrid"
                lattice_hybrid[key] = {
                    "hybrid_grid": hybrid_grid,
                    "md_grid": md_grid_corrected,
                    "md_grid_raw": md_item["grid"],
                    "md_T": md_item["T"],
                    "md_values": md_item["values"],
                    "md_values_corrected": np.interp(md_item["T"], T_grid, md_grid_corrected),
                    "qha": qha_item,
                    "qha_values_corrected": qha_values_corrected,
                    "mode": qha_lattice_modes[key],
                    "ylabel": md_item.get("ylabel", f"Lattice {key}"),
                }
                structural_metadata.setdefault("lattice_parameters", {})[key] = {
                    "qha_correction": qha_correction,
                    "md_correction": md_correction,
                    "source_mode": qha_lattice_modes[key],
                }
            if "a" in lattice_hybrid:
                a_grid = lattice_hybrid["a"]["hybrid_grid"]
                dadT_grid = np.gradient(a_grid, T_grid)
                alpha_L_grid = dadT_grid / a_grid
            diagnostics = cp_overlap_diagnostics_np(
                qha_low_t_curve["T"],
                qha_low_t_curve["Cp"],
                T,
                Cp_for_integral_data,
                blend_start,
                blend_end,
            )
            plot_overlap_mismatch_np(outdir / "overlap_mismatch_Cp.png", diagnostics)
            qha_splice_metadata = {
                "enabled": True,
                "switch_temperature_K": float(switch_T),
                "switch_method": switch_method,
                "minimum_switch_temperature_K": qha_splice_min_switch_temperature,
                "blend_start_K": float(blend_start),
                "blend_end_K": float(blend_end),
                "blend_function": "smoothstep w=3x^2-2x^3",
                "entropy_anchor_blend_calibration": entropy_blend_calibration,
                "cp_overlap_diagnostics": {
                    key: value
                    for key, value in diagnostics.items()
                    if key != "rows"
                },
                "cp_overlap_diagnostics_rows": diagnostics["rows"],
                "structural_hybrid": structural_metadata,
                "qha_volume_mode": qha_volume_mode,
                "qha_lattice_modes": qha_lattice_modes,
                "qha_source": {
                    "qha_dir": qha_low_t_curve["qha_dir"],
                    "qha_cp_file": qha_low_t_curve["qha_cp_file"],
                    "qha_volume_file": qha_low_t_curve.get("qha_volume_file"),
                    "qha_lattice_files": {
                        key: item.get("file")
                        for key, item in qha_low_t_curve.get("lattice_parameters", {}).items()
                    },
                    "qha_lattice_sources": {
                        key: item.get("source", "file")
                        for key, item in qha_low_t_curve.get("lattice_parameters", {}).items()
                    },
                    "qha_cp_unit": qha_low_t_curve["qha_cp_unit"],
                    "qha_formula_units": qha_low_t_curve["qha_formula_units"],
                },
                "uq_note": (
                    "Cp UQ band is tapered by smoothstep weight: zero-width in the QHA-only "
                    "region, partial MD statistical width in the blend, full MD statistical "
                    "width above blend_end."
                ),
                "note": (
                    "QHA supplies low-T Cp; QHA and MD Cp are smoothstep-blended "
                    "before integrating S/H/G."
                ),
            }
        else:
            qha_splice_metadata = {
                "enabled": False,
                "switch_method": switch_method,
                "minimum_switch_temperature_K": qha_splice_min_switch_temperature,
                "note": "QHA low-T splice skipped because no acceptable Cp switch was found.",
            }

    # Avoid division by zero in Cp/T integral. At T=0, use integrand 0 for the endpoint.
    Cp_over_T = np.zeros_like(Cp_grid, dtype=float)
    nonzero_T = T_grid > 1.0e-12
    Cp_over_T[nonzero_T] = Cp_grid[nonzero_T] / T_grid[nonzero_T]

    # Relative/anchored thermodynamic integrations.
    # Default: integrate from first grid point, usually 0 K if --anchor-zero is used.
    # Anchor mode: use S(T_anchor), H(T_anchor), and optionally Cp(T_anchor)
    # from phonopy/QHA or literature, then integrate MLIP-MD Cp away from T_anchor.
    if use_anchor_for_integration and thermo_anchor_T is not None:
        H_rel_J_mol_grid, S_rel_J_mol_K_grid, G_rel_J_mol_grid = integrate_from_reference(
            T_grid=T_grid,
            Cp_grid=Cp_grid,
            T_ref=thermo_anchor_T,
            S_ref_J_mol_K=thermo_anchor_S_J_mol_K,
            H_ref_J_mol=thermo_anchor_H_J_mol,
        )
    else:
        H_rel_J_mol_grid = trapz_cumulative(T_grid, Cp_grid)
        S_rel_J_mol_K_grid = trapz_cumulative(T_grid, Cp_over_T)
        G_rel_J_mol_grid = H_rel_J_mol_grid - T_grid * S_rel_J_mol_K_grid

    if qha_splice_metadata.get("enabled"):
        reference_T = qha_splice_metadata["blend_start_K"]
        switch_H = float(np.interp(reference_T, qha_low_t_curve["T"], qha_low_t_curve["H"]))
        H_rel_J_mol_grid, S_rel_J_mol_K_grid, G_rel_J_mol_grid = integrate_from_reference(
            T_grid=T_grid,
            Cp_grid=Cp_grid,
            T_ref=reference_T,
            S_ref_J_mol_K=0.0,
            H_ref_J_mol=switch_H,
        )
        if len(T_grid) and abs(T_grid[0]) <= 1.0e-12:
            S_rel_J_mol_K_grid = trapz_cumulative(T_grid, Cp_over_T)
        enthalpy_shift = 0.0
        if thermo_anchor_T is not None and thermo_anchor_H_J_mol is not None:
            current_anchor_H = float(np.interp(thermo_anchor_T, T_grid, H_rel_J_mol_grid))
            enthalpy_shift = float(thermo_anchor_H_J_mol) - current_anchor_H
            H_rel_J_mol_grid = H_rel_J_mol_grid + enthalpy_shift
        G_rel_J_mol_grid = H_rel_J_mol_grid - T_grid * S_rel_J_mol_K_grid
        qha_splice_metadata["entropy_reference"] = {
            "T_K": 0.0 if len(T_grid) and abs(T_grid[0]) <= 1.0e-12 else float(reference_T),
            "S_J_mol_formula_K": 0.0,
            "source": "S(0 K)=0 when grid includes 0 K; otherwise entropy is relative",
        }
        qha_splice_metadata["enthalpy_reference"] = {
            "T_K": float(reference_T),
            "H_J_mol_formula": switch_H,
            "source": "QHA H integrated from QHA Cp at blend_start",
        }
        qha_splice_metadata["enthalpy_anchor_shift"] = {
            "anchor_T_K": thermo_anchor_T,
            "anchor_H_J_mol_formula": thermo_anchor_H_J_mol,
            "shift_J_mol_formula": enthalpy_shift,
            "source": "manual --thermo-anchor-H" if thermo_anchor_H_J_mol is not None else None,
        }

    entropy_anchor_shift_metadata = {
        "applied": False,
        "reason": "direct entropy anchor inactive",
    }
    if entropy_anchor_is_direct:
        entropy_source = "manual --thermo-anchor-S"
        if (anchor_metadata or {}).get("thermo_db_anchor") and neel_correction != "on":
            entropy_source = (anchor_metadata or {})["thermo_db_anchor"]["database"]
        S_rel_J_mol_K_grid, G_rel_J_mol_grid, entropy_anchor_shift_metadata = apply_entropy_anchor_grid(
            T_grid=T_grid,
            S_grid=S_rel_J_mol_K_grid,
            H_grid=H_rel_J_mol_grid,
            entropy_anchor_T=entropy_anchor_blend_T,
            entropy_anchor_S=entropy_anchor_blend_S_J_mol_K,
            source=entropy_source,
        )
        if qha_splice_metadata.get("enabled"):
            qha_splice_metadata["entropy_anchor_shift"] = entropy_anchor_shift_metadata

    S_before_neel_grid = S_rel_J_mol_K_grid.copy()
    H_before_neel_grid = H_rel_J_mol_grid.copy()
    G_before_neel_grid = G_rel_J_mol_grid.copy()
    (
        S_rel_J_mol_K_grid,
        H_rel_J_mol_grid,
        G_rel_J_mol_grid,
        neel_weights,
        neel_metadata,
    ) = apply_neel_correction_grid(
        T_grid=T_grid,
        S_grid=S_rel_J_mol_K_grid,
        H_grid=H_rel_J_mol_grid,
        neel_correction=neel_correction,
        neel_t=neel_t,
        neel_entropy=neel_entropy,
        neel_enthalpy=neel_enthalpy,
        neel_apply_above_t=neel_apply_above_t,
        entropy_anchor_is_direct=entropy_anchor_is_direct,
        anchor_metadata=anchor_metadata,
        thermo_formula=thermo_formula,
    )
    if qha_splice_metadata.get("enabled"):
        qha_splice_metadata["neel_correction"] = neel_metadata

    # Interpolate grid relative functions back to MD T for the summary table.
    H_rel_J_mol = np.interp(T, T_grid, H_rel_J_mol_grid)
    S_rel_J_mol_K = np.interp(T, T_grid, S_rel_J_mol_K_grid)
    G_rel_J_mol = np.interp(T, T_grid, G_rel_J_mol_grid)
    S_before_neel = np.interp(T, T_grid, S_before_neel_grid)
    H_before_neel = np.interp(T, T_grid, H_before_neel_grid)
    G_before_neel = np.interp(T, T_grid, G_before_neel_grid)
    neel_weights_at_T = np.interp(T, T_grid, neel_weights)
    Cp_grid_at_T = np.interp(T, T_grid, Cp_grid)
    Cp_from_H_at_T = np.interp(T, T_grid, Cp_from_H_grid)

    uq_bands = {}
    if n_bootstrap and n_bootstrap > 0 and len(T) >= 3:
        uq_bands = bootstrap_temperature_functions(
            T=T, V=V, V_sem=V_sem, a=a, a_sem=a_sem,
            H=H_eV, H_sem=H_sem, Cp=Cp_fluct, Cp_sem=Cp_sem,
            T_grid=T_grid, fit_degree=fit_degree, cp_source=cp_source,
            nfu=nfu, anchor_zero=anchor_zero,
            n_boot=n_bootstrap, seed=bootstrap_seed,
        )
    if qha_splice_metadata.get("enabled") and "Cp_grid" in uq_bands:
        lo, hi = uq_bands["Cp_grid"]
        half_width = 0.5 * np.abs(hi - lo)
        half_width = qha_blend_weights * half_width
        uq_bands["Cp_grid"] = (Cp_grid - half_width, Cp_grid + half_width)

    combined = []
    for i, s in enumerate(summaries):
        row = dict(s)
        row.update({
            "target_z_formula_units": float(target_z),
            "V_fit_A3": float(V_fit[i]),
            "V_per_formula_A3": float(V_fit[i] / nfu),
            "V_target_cell_A3": float(V_fit[i] * target_z / nfu),
            "a_fit_A": float(a_fit[i]),
            "alpha_V_1_per_K": float(alpha_V[i]),
            "alpha_L_1_per_K": float(alpha_L[i]),
            "alpha_V_micro_per_K": float(alpha_V[i] * 1e6),
            "alpha_L_micro_per_K": float(alpha_L[i] * 1e6),
            "H_fit_eV_cell": float(H_fit_eV[i]),
            "H_abs_kJ_per_mol_UO2": float(H_fit_eV[i] * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu),
            "Cp_from_dH_J_per_mol_UO2_K": float(Cp_from_H_at_T[i]),
            "Cp_used_for_integration_J_per_mol_UO2_K": float(Cp_grid_at_T[i]),
            "S_rel_J_per_mol_UO2_K": float(S_rel_J_mol_K[i]),
            "H_rel_J_per_mol_UO2": float(H_rel_J_mol[i]),
            "G_rel_J_per_mol_UO2": float(G_rel_J_mol[i]),
            "neel_weight": float(neel_weights_at_T[i]),
            "S_before_neel_J_per_mol_UO2_K": float(S_before_neel[i]),
            "H_before_neel_J_per_mol_UO2": float(H_before_neel[i]),
            "G_before_neel_J_per_mol_UO2": float(G_before_neel[i]),
        })
        combined.append(row)

    # write combined outputs
    keys = list(combined[0].keys())
    with (outdir / "all_T_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in combined:
            w.writerow(row)
    dump_json(outdir / "all_T_summary.json", combined)

    grid_rows = []
    # Density on grid from V_grid.
    mass_g = nfu * MOLAR_MASS_UO2_G_MOL / NA
    rho_grid = mass_g / (V_grid * 1.0e-24)
    for i in range(len(T_grid)):
        grid_rows.append({
            "T_K": float(T_grid[i]),
            "n_formula_units": float(nfu),
            "target_z_formula_units": float(target_z),
            "V_fit_A3": float(V_grid[i]),
            "V_per_formula_A3": float(V_grid[i] / nfu),
            "V_target_cell_A3": float(V_grid[i] * target_z / nfu),
            "a_fit_A": float(a_grid[i]),
            "density_fit_g_cm3": float(rho_grid[i]),
            "alpha_V_1_per_K": float(alpha_V_grid[i]),
            "alpha_L_1_per_K": float(alpha_L_grid[i]),
            "alpha_V_micro_per_K": float(alpha_V_grid[i] * 1e6),
            "alpha_L_micro_per_K": float(alpha_L_grid[i] * 1e6),
            "H_fit_eV_cell": float(H_grid_eV[i]),
            "H_abs_kJ_per_mol_UO2": float(H_abs_kJ_per_mol_UO2_grid[i]),
            "Cp_from_dH_J_per_mol_UO2_K": float(Cp_from_H_grid[i]),
            "Cp_used_for_integration_J_per_mol_UO2_K": float(Cp_grid[i]),
            "qha_md_blend_weight": float(qha_blend_weights[i]),
            "H_rel_J_per_mol_UO2": float(H_rel_J_mol_grid[i]),
            "S_rel_J_per_mol_UO2_K": float(S_rel_J_mol_K_grid[i]),
            "G_rel_J_per_mol_UO2": float(G_rel_J_mol_grid[i]),
            "neel_weight": float(neel_weights[i]),
            "S_before_neel_J_per_mol_UO2_K": float(S_before_neel_grid[i]),
            "H_before_neel_J_per_mol_UO2": float(H_before_neel_grid[i]),
            "G_before_neel_J_per_mol_UO2": float(G_before_neel_grid[i]),
        })
        for key, band in uq_bands.items():
            lo, hi = band
            grid_rows[-1][f"{key}_p16"] = float(lo[i])
            grid_rows[-1][f"{key}_p84"] = float(hi[i])

    grid_keys = list(grid_rows[0].keys())
    with (outdir / "thermo_functions_grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=grid_keys)
        w.writeheader()
        for row in grid_rows:
            w.writerow(row)
    dump_json(outdir / "thermo_functions_grid.json", grid_rows)

    range_meta = {
        "available_T_min_K": Tmin_available,
        "available_T_max_K": Tmax_available,
        "requested_plot_T_min_K": plot_T_min,
        "requested_plot_T_max_K": plot_T_max,
        "used_highest_dataset_T_K": T_high_dataset,
        "plot_T_grid_min_K": float(T_grid[0]),
        "plot_T_grid_max_K": float(T_grid[-1]),
        "plot_T_step_K": plot_T_step,
        "anchor_zero": anchor_zero,
        "cp_source": cp_source,
        "fit_degree": deg_fit,
        "n_formula_units": nfu,
        "target_z_formula_units": target_z,
        "thermo_anchor_T": thermo_anchor_T,
        "thermo_anchor_S_J_mol_K": thermo_anchor_S_J_mol_K,
        "thermo_anchor_Cp_J_mol_K": thermo_anchor_Cp_J_mol_K,
        "thermo_anchor_H_J_mol": thermo_anchor_H_J_mol,
        "use_anchor_for_integration": use_anchor_for_integration,
        "use_anchor_Cp_in_fit": use_anchor_Cp_in_fit,
        "anchor_metadata": anchor_metadata or {},
        "qha_low_t_splice": qha_splice_metadata,
        "entropy_anchor_shift": entropy_anchor_shift_metadata,
        "neel_correction": neel_metadata,
    }
    dump_json(outdir / "temperature_range_metadata.json", range_meta)
    if anchor_metadata:
        dump_json(outdir / "thermo_anchor_metadata.json", anchor_metadata)
    if qha_splice_metadata:
        dump_json(outdir / "qha_low_t_splice_metadata.json", qha_splice_metadata)

    # plots
    plot_xy(outdir / "V_vs_T.png", T, V, "T (K)", "V (Å$^3$)", "Volume vs T", y2=V_fit, y2label="V fit")
    plot_xy(outdir / "a_vs_T.png", T, a, "T (K)", "a proxy (Å)", "Lattice proxy vs T", y2=a_fit, y2label="a fit")
    plot_xy(outdir / "density_vs_T.png", T, rho, "T (K)", "density (g/cm$^3$)", "Density vs T")
    plot_xy(outdir / "H_vs_T.png", T, H_eV, "T (K)", "H (eV/cell)", "Enthalpy vs T", y2=H_fit_eV, y2label="H fit")
    plot_xy(outdir / "H_abs_kJ_per_mol_UO2_vs_T.png", T, H_kJ_mol, "T (K)", "H (kJ/mol-UO$_2$)", "Absolute MD enthalpy scale vs T", y2=H_fit_eV * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu, y2label="H fit")
    plot_xy(outdir / "Cp_vs_T.png", T, Cp_fluct, "T (K)", "Cp (J/mol-UO$_2$/K)", "Cp vs T", y2=Cp_from_H, y2label="Cp from dH/dT")
    plot_xy(outdir / "KT_vs_T.png", T, KT, "T (K)", "K$_T$ (GPa)", "Bulk modulus from V fluctuations")
    plot_xy(outdir / "CTE_vs_T.png", T, alpha_V * 1e6, "T (K)", r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)", "Volumetric CTE", y2=alpha_L * 1e6, y2label=r"$\alpha_L$ ($10^{-6}$ K$^{-1}$)")
    plot_xy(outdir / "S_rel_vs_T.png", T, S_rel_J_mol_K, "T (K)", "S-S$_0$ (J/mol/K)", "Relative entropy")
    plot_xy(outdir / "G_rel_vs_T.png", T, G_rel_J_mol / 1000.0, "T (K)", "G-G$_0$ (kJ/mol)", "Relative Gibbs energy")

    # Smooth/grid function plots over user-requested range.
    plot_xy(outdir / "V_function_grid.png", T_grid, V_grid, "T (K)", "V fit (Å$^3$)", "Volume function")
    plot_xy(outdir / "a_function_grid.png", T_grid, a_grid, "T (K)", "a fit (Å)", "Lattice function")
    plot_xy(outdir / "CTE_function_grid.png", T_grid, alpha_V_grid * 1e6, "T (K)", r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)", "CTE function", y2=alpha_L_grid * 1e6, y2label=r"$\alpha_L$ ($10^{-6}$ K$^{-1}$)")
    plot_xy(outdir / "Cp_function_grid.png", T_grid, Cp_grid, "T (K)", "Cp used (J/mol-UO$_2$/K)", "Cp function")
    plot_xy(outdir / "S_function_grid.png", T_grid, S_rel_J_mol_K_grid, "T (K)", "S-S$_0$ (J/mol/K)", "Relative entropy function")
    plot_xy(outdir / "G_function_grid.png", T_grid, G_rel_J_mol_grid / 1000.0, "T (K)", "G-G$_0$ (kJ/mol)", "Relative Gibbs function")

    if qha_splice_metadata.get("enabled"):
        neel_region = None
        if neel_metadata.get("applied"):
            neel_region = (
                neel_metadata["activation_start_K"],
                neel_metadata["apply_full_above_T_K"],
            )
        Cp_over_T_md = np.zeros_like(Cp_md_grid, dtype=float)
        md_nonzero = T_grid > 1.0e-12
        Cp_over_T_md[md_nonzero] = Cp_md_grid[md_nonzero] / T_grid[md_nonzero]
        H_md_rel_grid = trapz_cumulative(T_grid, Cp_md_grid)
        S_md_rel_grid = trapz_cumulative(T_grid, Cp_over_T_md)
        if thermo_anchor_T is not None and thermo_anchor_H_J_mol is not None:
            md_anchor_h = float(np.interp(thermo_anchor_T, T_grid, H_md_rel_grid))
            H_md_rel_grid = H_md_rel_grid + (float(thermo_anchor_H_J_mol) - md_anchor_h)
        G_md_rel_grid = H_md_rel_grid - T_grid * S_md_rel_grid
        qha_V_scaled = None
        qha_V_corrected = None
        V_md_corrected = V_md_grid
        if qha_low_t_curve.get("V") is not None and qha_low_t_curve.get("V_T") is not None:
            qha_V_scaled = qha_low_t_curve["V"] * (nfu / qha_low_t_curve["qha_formula_units"])
            qha_V_corrected, _meta = correct_array_to_reference(
                qha_low_t_curve["V_T"],
                qha_V_scaled,
                source="qha",
                reference_temperature=structure_reference_temperature,
                reference_value=volume_reference,
                correction=structure_correction,
                apply_to=structure_correction_apply_to,
            )
            V_md_corrected, _meta = correct_array_to_reference(
                T_grid,
                V_md_grid,
                source="md",
                reference_temperature=structure_reference_temperature,
                reference_value=volume_reference,
                correction=structure_correction,
                apply_to=structure_correction_apply_to,
            )
            plot_qha_md_overlap(
                outdir / "volume_QHA_MD_overlap.png",
                qha_low_t_curve["V_T"],
                qha_V_scaled,
                T_grid,
                V_md_grid,
                "V (Å$^3$)",
                "QHA/MD Volume Overlap",
                qha_splice_metadata["blend_start_K"],
                qha_splice_metadata["blend_end_K"],
            )
        plot_hybrid_grid(
            outdir / "hybrid_Cp_QHA_MD.png",
            T_grid,
            Cp_grid,
            "Cp (J/mol-UO$_2$/K)",
            "Hybrid QHA+MD Cp",
            qha_low_t_curve["T"],
            qha_low_t_curve["Cp"],
            T_grid,
            Cp_md_grid,
            uq_bands.get("Cp_grid"),
            qha_splice_metadata["blend_start_K"],
            qha_splice_metadata["blend_end_K"],
        )
        plot_hybrid_grid(
            outdir / "hybrid_S_QHA_MD.png",
            T_grid,
            S_rel_J_mol_K_grid,
            "S (J/mol/K)",
            "Hybrid QHA+MD Entropy",
            qha_low_t_curve["T"],
            qha_low_t_curve["S"],
            T_grid,
            S_md_rel_grid,
            None,
            qha_splice_metadata["blend_start_K"],
            qha_splice_metadata["blend_end_K"],
            db_points=thermo_db_points.get("S"),
            neel_region=neel_region,
        )
        if neel_metadata.get("applied"):
            plot_hybrid_grid(
                outdir / "hybrid_S_QHA_MD_phonon_only.png",
                T_grid,
                S_before_neel_grid,
                "S (J/mol/K)",
                "Phonon-Only Hybrid QHA+MD Entropy",
                qha_low_t_curve["T"],
                qha_low_t_curve["S"],
                T_grid,
                S_md_rel_grid,
                None,
                qha_splice_metadata["blend_start_K"],
                qha_splice_metadata["blend_end_K"],
                db_points=thermo_db_points.get("S"),
                neel_region=neel_region,
            )
        plot_hybrid_grid(
            outdir / "hybrid_H_QHA_MD.png",
            T_grid,
            (H_before_neel_grid if neel_metadata.get("applied") else H_rel_J_mol_grid) / 1000.0,
            "H (kJ/mol)",
            "Integrated Hybrid QHA+MD Enthalpy",
            qha_low_t_curve["T"],
            qha_low_t_curve["H"] / 1000.0,
            T_grid,
            H_md_rel_grid / 1000.0,
            None,
            qha_splice_metadata["blend_start_K"],
            qha_splice_metadata["blend_end_K"],
            db_points=thermo_db_points.get("H"),
            neel_region=neel_region,
        )
        if neel_metadata.get("applied"):
            plot_hybrid_grid(
                outdir / "hybrid_H_QHA_MD_neel_corrected.png",
                T_grid,
                H_rel_J_mol_grid / 1000.0,
                "H (kJ/mol)",
                "Neel-Corrected Hybrid QHA+MD Enthalpy",
                qha_low_t_curve["T"],
                qha_low_t_curve["H"] / 1000.0,
                T_grid,
                H_md_rel_grid / 1000.0,
                None,
                qha_splice_metadata["blend_start_K"],
                qha_splice_metadata["blend_end_K"],
                db_points=thermo_db_points.get("H"),
                neel_region=neel_region,
            )
        plot_hybrid_grid(
            outdir / "hybrid_G_QHA_MD.png",
            T_grid,
            (G_before_neel_grid if neel_metadata.get("applied") else G_rel_J_mol_grid) / 1000.0,
            "G (kJ/mol)",
            "Hybrid G = H - TS",
            qha_low_t_curve["T"],
            (qha_low_t_curve["H"] - qha_low_t_curve["T"] * qha_low_t_curve["S"]) / 1000.0,
            T_grid,
            G_md_rel_grid / 1000.0,
            None,
            qha_splice_metadata["blend_start_K"],
            qha_splice_metadata["blend_end_K"],
            db_points=thermo_db_points.get("G"),
            neel_region=neel_region,
        )
        if neel_metadata.get("applied"):
            plot_hybrid_grid(
                outdir / "hybrid_G_QHA_MD_neel_corrected.png",
                T_grid,
                G_rel_J_mol_grid / 1000.0,
                "G (kJ/mol)",
                "Neel-Corrected Hybrid G = H - TS",
                qha_low_t_curve["T"],
                (qha_low_t_curve["H"] - qha_low_t_curve["T"] * qha_low_t_curve["S"]) / 1000.0,
                T_grid,
                G_md_rel_grid / 1000.0,
                None,
                qha_splice_metadata["blend_start_K"],
                qha_splice_metadata["blend_end_K"],
                db_points=thermo_db_points.get("G"),
                neel_region=neel_region,
            )
        plot_structural_hybrid_detail(
            outdir / "hybrid_V_QHA_MD.png",
            T_grid=T_grid,
            hybrid_y=V_grid,
            ylabel="V (Å$^3$)",
            title="Corrected Hybrid QHA+MD Volume" if qha_V_scaled is not None else "MD Volume",
            qha_T_raw=qha_low_t_curve.get("V_T"),
            qha_y_raw=qha_V_scaled,
            md_T_raw=T_grid,
            md_y_raw=V_md_grid,
            qha_T_corrected=qha_low_t_curve.get("V_T"),
            qha_y_corrected=qha_V_corrected,
            md_T_corrected=T_grid,
            md_y_corrected=V_md_corrected,
            blend_start=qha_splice_metadata["blend_start_K"],
            blend_end=qha_splice_metadata["blend_end_K"],
        )
        plot_hybrid_grid(
            outdir / "hybrid_alpha_V_QHA_MD.png",
            T_grid,
            alpha_V_grid * 1e6,
            r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)",
            "Hybrid Volumetric CTE From V(T)",
            None,
            None,
            None,
            None,
            None,
            qha_splice_metadata["blend_start_K"],
            qha_splice_metadata["blend_end_K"],
        )
        for key, item in lattice_hybrid.items():
            qha_item = item.get("qha")
            qha_T = qha_item.get("T") if qha_item is not None else None
            qha_values = qha_item.get("values") if qha_item is not None else None
            plot_structural_hybrid_detail(
                outdir / f"hybrid_{key}_QHA_MD.png",
                T_grid=T_grid,
                hybrid_y=item["hybrid_grid"],
                ylabel=item["ylabel"],
                title=f"Corrected Hybrid QHA+MD Lattice {key}",
                qha_T_raw=qha_T,
                qha_y_raw=qha_values,
                md_T_raw=T_grid,
                md_y_raw=item["md_grid_raw"],
                qha_T_corrected=qha_T,
                qha_y_corrected=item.get("qha_values_corrected"),
                md_T_corrected=T_grid,
                md_y_corrected=item["md_grid"],
                blend_start=qha_splice_metadata["blend_start_K"],
                blend_end=qha_splice_metadata["blend_end_K"],
            )
            if qha_T is not None and qha_values is not None:
                plot_qha_md_overlap(
                    outdir / f"lattice_{key}_QHA_MD_overlap.png",
                    qha_T,
                    qha_values,
                    T_grid,
                    item["md_grid"],
                    item["ylabel"],
                    f"QHA/MD Lattice {key} Overlap",
                    qha_splice_metadata["blend_start_K"],
                    qha_splice_metadata["blend_end_K"],
                )
            if key == "a":
                plot_hybrid_grid(
                    outdir / "hybrid_alpha_L_QHA_MD.png",
                    T_grid,
                    alpha_L_grid * 1e6,
                    r"$\alpha_L$ ($10^{-6}$ K$^{-1}$)",
                    "Hybrid Linear CTE From a(T)",
                    None,
                    None,
                    None,
                    None,
                    None,
                    qha_splice_metadata["blend_start_K"],
                    qha_splice_metadata["blend_end_K"],
                )


    # UQ-band plots. Bands are MD processing/statistical bands from block SEM + bootstrap.
    plot_function_with_band(
        outdir / "V_function_grid_UQ.png", T, V, V_sem, T_grid, V_grid,
        uq_bands.get("V_grid"), "T (K)", "V (Å$^3$)", "Volume function with MD UQ band"
    )
    plot_function_with_band(
        outdir / "a_function_grid_UQ.png", T, a, a_sem, T_grid, a_grid,
        uq_bands.get("a_grid"), "T (K)", "a (Å)", "Lattice function with MD UQ band"
    )
    plot_function_with_band(
        outdir / "density_function_grid_UQ.png", T, rho, rho_sem, T_grid, rho_grid,
        uq_bands.get("density_grid"), "T (K)", "density (g/cm$^3$)", "Density function with MD UQ band"
    )
    plot_function_with_band(
        outdir / "H_function_grid_UQ.png", T, H_eV, H_sem, T_grid, H_grid_eV,
        uq_bands.get("H_grid_eV"), "T (K)", "H (eV/cell)", "Enthalpy function with MD UQ band"
    )
    plot_function_with_band(
        outdir / "H_abs_kJ_per_mol_UO2_function_grid_UQ.png",
        T, H_kJ_mol, H_sem * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu,
        T_grid, H_abs_kJ_per_mol_UO2_grid,
        (uq_bands["H_grid_eV"][0] * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu, uq_bands["H_grid_eV"][1] * EV_CELL_TO_KJ_PER_MOL_UO2 / nfu) if "H_grid_eV" in uq_bands else None,
        "T (K)", "H (kJ/mol-UO$_2$)", "Absolute MD enthalpy scale with MD UQ band"
    )
    # Cp plotting:
    # If cp_source == "dH", the thermodynamic Cp function comes from dH/dT.
    # Do NOT use the NPT fluctuation Cp points to set the y-axis in that case,
    # because small-cell NPT fluctuation Cp can be orders of magnitude too large.
    if cp_source == "dH":
        plot_function_with_band(
            outdir / "Cp_function_grid_UQ.png",
            T,
            Cp_from_H_at_T,
            None,
            T_grid,
            Cp_grid,
            uq_bands.get("Cp_grid"),
            "T (K)",
            "Cp from dH/dT (J/mol-UO$_2$/K)",
            "Cp from dH/dT with MD UQ band",
        )

        # Keep the fluctuation estimator as a separate diagnostic only.
        plot_function_with_band(
            outdir / "Cp_fluctuation_diagnostic_UQ.png",
            T,
            Cp_fluct,
            Cp_sem,
            T_grid,
            Cp_fluct if len(Cp_fluct) == len(T_grid) else np.interp(T_grid, T, Cp_fluct),
            None,
            "T (K)",
            "NPT fluctuation Cp (J/mol-UO$_2$/K)",
            "Diagnostic only: NPT fluctuation Cp",
        )
    else:
        plot_function_with_band(
            outdir / "Cp_function_grid_UQ.png", T, Cp_fluct, Cp_sem, T_grid, Cp_grid,
            uq_bands.get("Cp_grid"), "T (K)", "Cp (J/mol-UO$_2$/K)", "Cp function with MD UQ band"
        )
    plot_function_with_band(
        outdir / "CTE_alphaV_function_grid_UQ.png", T, alpha_V * 1e6, None, T_grid, alpha_V_grid * 1e6,
        (uq_bands["alpha_V_grid"][0] * 1e6, uq_bands["alpha_V_grid"][1] * 1e6) if "alpha_V_grid" in uq_bands else None,
        "T (K)", r"$\alpha_V$ ($10^{-6}$ K$^{-1}$)", "Volumetric CTE with MD UQ band"
    )
    plot_function_with_band(
        outdir / "S_function_grid_UQ.png", T, S_rel_J_mol_K, None, T_grid, S_rel_J_mol_K_grid,
        uq_bands.get("S_rel_J_mol_K_grid"), "T (K)", "S-S$_0$ (J/mol/K)", "Relative entropy with MD UQ band"
    )
    plot_function_with_band(
        outdir / "G_function_grid_UQ.png", T, G_rel_J_mol / 1000.0, None, T_grid, G_rel_J_mol_grid / 1000.0,
        (uq_bands["G_rel_J_mol_grid"][0] / 1000.0, uq_bands["G_rel_J_mol_grid"][1] / 1000.0) if "G_rel_J_mol_grid" in uq_bands else None,
        "T (K)", "G-G$_0$ (kJ/mol)", "Relative Gibbs energy with MD UQ band"
    )

    dump_json(outdir / "uncertainty_metadata.json", {
        "uncertainty_type": "MD processing/statistical uncertainty from selected-window block SEM plus parametric bootstrap",
        "band_percentiles": [16, 84],
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "not_included": [
            "MLIP model-form uncertainty",
            "DFT reference-data uncertainty",
            "finite-size systematic error",
            "thermostat/barostat systematic bias"
        ]
    })

    return combined


# -----------------------------
# main
# -----------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lammps-thermo-series")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", nargs="+", help="One or more production config JSON files")
    src.add_argument(
        "--md-root",
        type=Path,
        help=(
            "Folder containing MD run folders. The scanner reads NPT folders only "
            "(for example stages/npt_prod_300K/...) and ignores NVT ramp folders."
        ),
    )
    src.add_argument("--manual-analysis-root", help="Root containing existing thermo_summary.json files")
    src.add_argument(
        "--compare-series",
        nargs="+",
        type=Path,
        help="Compare two or more existing thermo_lammps output directories.",
    )
    ap.add_argument("--config-dir", default=None, help="Optional directory containing more config JSON files")
    ap.add_argument("--config-glob", default="*.json", help="Glob pattern used with --config-dir")
    ap.add_argument("--duplicate-policy", choices=["highest_config_order", "first", "error"], default="highest_config_order")
    ap.add_argument("--timestep-ps", type=float, default=None, help="Override timestep for all configs; otherwise each config timestep is used")

    ap.add_argument("--outdir", default="analysis/temperature_series_thermo")
    ap.add_argument("--natoms", type=int, default=96)
    ap.add_argument("--atoms-per-formula-unit", type=int, default=3)
    ap.add_argument("--target-z", type=float, default=4.0, help="Formula units in the normalized structural target cell, e.g. 4 for fluorite UO2.")
    ap.add_argument("--compare-label", action="append", default=[], help="Label for a --compare-series directory. Repeat once per series.")
    ap.add_argument("--compare-formula-units", action="append", type=float, default=[], help="Formula units in a compared MD series if old outputs do not record it. Repeat once per series.")
    ap.add_argument("--compare-energy-basis", choices=["per-formula", "target-cell"], default="per-formula", help="Basis for compared Cp/S/H/G overlays.")
    ap.add_argument("--min-window-ps", type=float, default=20.0)
    ap.add_argument("--window-ps", type=float, default=None, help="Window length. Default=min-window-ps")
    ap.add_argument(
        "--window-mode",
        choices=["tail", "auto"],
        default="tail",
        help="Window selection mode. Default tail uses the last --min-window-ps/--window-ps of each NPT run; auto uses the older stationarity score search.",
    )
    ap.add_argument("--window-stride-ps", type=float, default=2.0)
    ap.add_argument("--discard-initial-ps", type=float, default=0.0, help="Ignore this early part before auto window search")
    ap.add_argument("--plot-bin-ps", type=float, default=0.5)
    ap.add_argument("--nblocks", type=int, default=5)
    ap.add_argument("--fit-degree", type=int, default=3)
    ap.add_argument("--cp-source", choices=["fluct", "dH"], default="fluct", help="Cp source for S/G integration")
    ap.add_argument("--plot-T-min", type=float, default=None, help="Lowest T for fitted function grid. Can be 0 even without MD data at 0 K")
    ap.add_argument("--plot-T-max", type=float, default=None, help="Requested high T. Script uses closest completed production T at/below this value")
    ap.add_argument("--plot-T-step", type=float, default=10.0, help="Temperature grid step for fitted functions")
    ap.add_argument("--anchor-zero", action="store_true", help="Anchor integration at T=0 with Cp(0)=S(0)=Hrel(0)=Grel(0)=0")
    ap.add_argument("--n-bootstrap", type=int, default=300, help="Bootstrap samples for MD UQ bands. Use 0 to disable")
    ap.add_argument("--bootstrap-seed", type=int, default=12345)

    # Optional thermodynamic anchor for phonopy/QHA/literature splice.
    ap.add_argument("--thermo-anchor-T", type=float, default=None,
                    help="Reference temperature for external Cp/S/H anchor, e.g. 300")
    ap.add_argument("--thermo-anchor-S", type=float, default=None,
                    help="Entropy at anchor T in J/mol-UO2/K, e.g. literature or phonopy S(300 K)")
    ap.add_argument("--thermo-anchor-Cp", type=float, default=None,
                    help="Cp at anchor T in J/mol-UO2/K, e.g. phonopy/literature Cp(300 K)")
    ap.add_argument("--thermo-anchor-H", type=float, default=None,
                    help="Relative enthalpy at anchor T in J/mol-UO2. Optional; default 0 at anchor")
    ap.add_argument("--use-anchor-for-integration", action="store_true",
                    help="Use the external anchor for S/H/G integration instead of integrating from 0 K")
    ap.add_argument("--use-anchor-Cp-in-fit", action="store_true",
                    help="Add/replace Cp(T_anchor) in the Cp fit used for S/H/G integration")
    ap.add_argument(
        "--thermo-db",
        choices=["jaea"],
        default=None,
        help="Thermodynamic database used to fill --thermo-anchor-H automatically.",
    )
    ap.add_argument(
        "--thermo-formula",
        default=None,
        help="Formula key for --thermo-db, e.g. UO2 for the JAEA UO2 page.",
    )
    ap.add_argument(
        "--thermo-phase",
        default="solid",
        help="Phase label recorded in metadata for database anchors.",
    )
    ap.add_argument(
        "--thermo-db-temperature",
        type=float,
        default=None,
        help="Database lookup temperature. Defaults to --thermo-anchor-T, or 300 K.",
    )
    ap.add_argument(
        "--plot-thermo-db-points",
        action="store_true",
        help="Overlay thermodynamic database points on hybrid S/H/G plots.",
    )
    ap.add_argument(
        "--qha-anchor-dir",
        type=Path,
        default=None,
        help="QHA output folder with Cp-temperature.dat for deriving missing Cp/S/H anchors",
    )
    ap.add_argument(
        "--qha-anchor-formula-units",
        type=float,
        default=None,
        help="Formula units in the QHA cell, used to normalize QHA Cp to per formula unit",
    )
    ap.add_argument(
        "--qha-anchor-cp-unit",
        choices=[
            "J/mol-cell/K",
            "kJ/mol-cell/K",
            "J/mol-formula/K",
            "kJ/mol-formula/K",
            "eV-cell/K",
        ],
        default="J/mol-cell/K",
        help="Unit of QHA Cp-temperature.dat values",
    )
    ap.add_argument(
        "--qha-low-t-splice",
        action="store_true",
        help="Build smooth QHA+MD hybrid Cp/S/H/G using QHA as the low-T branch.",
    )
    ap.add_argument(
        "--qha-splice-switch-temperature",
        type=float,
        default=None,
        help="Override the automatic QHA-to-MD low-T splice switch temperature in K.",
    )
    ap.add_argument(
        "--qha-splice-min-switch-temperature",
        type=float,
        default=50.0,
        help="Reject automatic QHA-to-MD low-T splice switches below this temperature in K.",
    )
    ap.add_argument(
        "--qha-splice-blend-start",
        type=float,
        default=None,
        help="Start temperature for smooth QHA-to-MD Cp blending in K.",
    )
    ap.add_argument(
        "--qha-splice-blend-end",
        type=float,
        default=None,
        help="End temperature for smooth QHA-to-MD Cp blending in K.",
    )
    ap.add_argument(
        "--entropy-anchor-min-blend-start",
        type=float,
        default=200.0,
        help=(
            "Lowest allowed QHA-to-MD blend-start temperature when experimental/database "
            "S is used to calibrate the integrated hybrid entropy."
        ),
    )
    ap.add_argument(
        "--neel-correction",
        choices=["off", "on"],
        default="off",
        help="Apply an optional low-temperature magnetic/Neel entropy correction.",
    )
    ap.add_argument("--neel-T", type=float, default=30.8, help="Neel transition temperature in K.")
    ap.add_argument(
        "--neel-entropy",
        type=float,
        default=8.4,
        help="Magnetic/Neel entropy contribution in J/mol-formula/K.",
    )
    ap.add_argument(
        "--neel-enthalpy",
        default="auto",
        help="Neel enthalpy offset in kJ/mol-formula, or 'auto' = T_N * DeltaS / 1000.",
    )
    ap.add_argument(
        "--neel-apply-above-T",
        type=float,
        default=50.0,
        help="Temperature where the full Neel entropy/enthalpy offset is active.",
    )
    ap.add_argument(
        "--structure-reference-temperature",
        type=float,
        default=None,
        help="Reference temperature for optional V/a baseline correction.",
    )
    ap.add_argument(
        "--volume-reference",
        type=float,
        default=None,
        help="Reference volume in the MD simulation-cell basis for V(T) correction.",
    )
    ap.add_argument(
        "--lattice-reference",
        action="append",
        default=[],
        help="Reference lattice parameter as key=value, e.g. a=5.47. Repeat for b/c.",
    )
    ap.add_argument(
        "--structure-correction",
        choices=["none", "shift", "scale"],
        default="none",
        help="Baseline correction applied to V/a before structural hybrid and CTE derivation.",
    )
    ap.add_argument(
        "--structure-correction-apply-to",
        choices=["qha", "md", "both"],
        default="both",
        help="Which structural source to correct to the reference value.",
    )

    # Speed controls for large log sets.
    ap.add_argument("--skip-per-T-plots", action="store_true", help="Skip individual per-temperature raw/binned diagnostic PNGs")
    ap.add_argument("--skip-combined-MD-plot", action="store_true", help="Skip the very large all-temperature raw MD diagnostic PNG")
    ap.add_argument("--skip-selected-timeseries", action="store_true", help="Do not write selected_timeseries.csv for each temperature")
    ap.add_argument("--raw-decimate", type=int, default=10, help="Plot only every Nth raw MD point in diagnostic scatter plots. Does not affect analysis. Use 1 for no decimation")
    hidden_hybrid_flags = {
        "--thermo-db",
        "--thermo-formula",
        "--thermo-phase",
        "--thermo-db-temperature",
        "--plot-thermo-db-points",
        "--qha-anchor-dir",
        "--qha-anchor-formula-units",
        "--qha-anchor-cp-unit",
        "--qha-low-t-splice",
        "--qha-splice-switch-temperature",
        "--qha-splice-min-switch-temperature",
        "--qha-splice-blend-start",
        "--qha-splice-blend-end",
        "--entropy-anchor-min-blend-start",
        "--neel-correction",
        "--neel-T",
        "--neel-entropy",
        "--neel-enthalpy",
        "--neel-apply-above-T",
        "--structure-reference-temperature",
        "--volume-reference",
        "--lattice-reference",
        "--structure-correction",
        "--structure-correction-apply-to",
    }
    for action in ap._actions:
        if hidden_hybrid_flags.intersection(action.option_strings):
            action.help = argparse.SUPPRESS
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    if args.target_z <= 0:
        ap.error("--target-z must be positive")

    moved_to_qha_md_flags = []
    if args.qha_anchor_dir is not None:
        moved_to_qha_md_flags.append("--qha-anchor-dir")
    if args.qha_anchor_formula_units is not None:
        moved_to_qha_md_flags.append("--qha-anchor-formula-units")
    if args.qha_low_t_splice:
        moved_to_qha_md_flags.append("--qha-low-t-splice")
    if args.qha_splice_switch_temperature is not None:
        moved_to_qha_md_flags.append("--qha-splice-switch-temperature")
    if args.qha_splice_blend_start is not None:
        moved_to_qha_md_flags.append("--qha-splice-blend-start")
    if args.qha_splice_blend_end is not None:
        moved_to_qha_md_flags.append("--qha-splice-blend-end")
    if args.thermo_db is not None:
        moved_to_qha_md_flags.append("--thermo-db")
    if args.plot_thermo_db_points:
        moved_to_qha_md_flags.append("--plot-thermo-db-points")
    if args.neel_correction == "on":
        moved_to_qha_md_flags.append("--neel-correction")
    if args.structure_reference_temperature is not None:
        moved_to_qha_md_flags.append("--structure-reference-temperature")
    if args.volume_reference is not None:
        moved_to_qha_md_flags.append("--volume-reference")
    if args.lattice_reference:
        moved_to_qha_md_flags.append("--lattice-reference")
    if args.structure_correction != "none":
        moved_to_qha_md_flags.append("--structure-correction")
    if args.structure_correction_apply_to != "both":
        moved_to_qha_md_flags.append("--structure-correction-apply-to")
    if moved_to_qha_md_flags and not args.compare_series:
        ap.error(
            "thermo_lammps is now MD-only. Remove these QHA/hybrid flags and run "
            f"thermo_qha_md afterward for QHA+MD integration: {', '.join(moved_to_qha_md_flags)}"
        )

    if args.compare_series:
        try:
            compare_existing_lammps_series(
                series_dirs=args.compare_series,
                outdir=outdir,
                labels=args.compare_label or None,
                formula_units=args.compare_formula_units or None,
                target_z=args.target_z,
                energy_basis=args.compare_energy_basis,
                t_min=args.plot_T_min,
                t_max=args.plot_T_max,
            )
        except ValueError as exc:
            ap.error(str(exc))
        print(f"Done. Comparison outputs written to: {outdir.resolve()}")
        return

    if args.config:
        config_paths = collect_config_paths(args.config, args.config_dir, args.config_glob)
        print("Config files:")
        for p in config_paths:
            print(f"  - {p}")
        records_all = discover_production_records(config_paths, duplicate_policy=args.duplicate_policy)
        records = filter_records_by_T(records_all, args.plot_T_min, args.plot_T_max)
        outdir.mkdir(parents=True, exist_ok=True)
        dump_json(outdir / "discovered_stage_records.json", [
            {"temperature": r["temperature"], "stage_name": r["stage_name"], "config_path": str(r["config_path"]), "log_path": str(r["log_path"])}
            for r in records_all
        ])
        dump_json(outdir / "used_stage_records.json", [
            {"temperature": r["temperature"], "stage_name": r["stage_name"], "config_path": str(r["config_path"]), "log_path": str(r["log_path"])}
            for r in records
        ])
        summaries = process_records(
            records=records,
            outdir=outdir / "per_T_analysis",
            natoms=args.natoms,
            atoms_per_formula_unit=args.atoms_per_formula_unit,
            min_window_ps=args.min_window_ps,
            window_ps=args.window_ps,
            window_mode=args.window_mode,
            window_stride_ps=args.window_stride_ps,
            discard_initial_ps=args.discard_initial_ps,
            plot_bin_ps=args.plot_bin_ps,
            nblocks=args.nblocks,
            timestep_override_ps=args.timestep_ps,
            skip_per_T_plots=args.skip_per_T_plots,
            skip_combined_MD_plot=args.skip_combined_MD_plot,
            skip_selected_timeseries=args.skip_selected_timeseries,
            raw_decimate=args.raw_decimate,
        )
    elif args.md_root:
        if args.config_dir:
            ap.error("--config-dir can only be used with --config")
        records_all = discover_npt_records_from_md_root(
            args.md_root,
            duplicate_policy=args.duplicate_policy,
            timestep_ps=args.timestep_ps,
        )
        records = filter_records_by_T(records_all, args.plot_T_min, args.plot_T_max)
        outdir.mkdir(parents=True, exist_ok=True)
        dump_json(outdir / "discovered_stage_records.json", [
            {
                "temperature": r["temperature"],
                "stage_name": r["stage_name"],
                "md_root": str(r.get("md_root", "")),
                "log_path": str(r["log_path"]),
            }
            for r in records_all
        ])
        dump_json(outdir / "used_stage_records.json", [
            {
                "temperature": r["temperature"],
                "stage_name": r["stage_name"],
                "md_root": str(r.get("md_root", "")),
                "log_path": str(r["log_path"]),
            }
            for r in records
        ])
        summaries = process_records(
            records=records,
            outdir=outdir / "per_T_analysis",
            natoms=args.natoms,
            atoms_per_formula_unit=args.atoms_per_formula_unit,
            min_window_ps=args.min_window_ps,
            window_ps=args.window_ps,
            window_mode=args.window_mode,
            window_stride_ps=args.window_stride_ps,
            discard_initial_ps=args.discard_initial_ps,
            plot_bin_ps=args.plot_bin_ps,
            nblocks=args.nblocks,
            timestep_override_ps=args.timestep_ps,
            skip_per_T_plots=args.skip_per_T_plots,
            skip_combined_MD_plot=args.skip_combined_MD_plot,
            skip_selected_timeseries=args.skip_selected_timeseries,
            raw_decimate=args.raw_decimate,
        )
    else:
        summaries = process_from_manual(Path(args.manual_analysis_root))

    try:
        (
            thermo_anchor_T,
            thermo_anchor_S,
            thermo_anchor_Cp,
            thermo_anchor_H,
            anchor_metadata,
        ) = fill_missing_anchors_from_qha(
            thermo_anchor_T=args.thermo_anchor_T,
            thermo_anchor_S_J_mol_K=args.thermo_anchor_S,
            thermo_anchor_Cp_J_mol_K=args.thermo_anchor_Cp,
            thermo_anchor_H_J_mol=args.thermo_anchor_H,
            qha_anchor_dir=args.qha_anchor_dir,
            qha_anchor_formula_units=args.qha_anchor_formula_units,
            qha_anchor_cp_unit=args.qha_anchor_cp_unit,
        )
    except (FileNotFoundError, ValueError) as exc:
        ap.error(str(exc))
    try:
        thermo_anchor_T, thermo_anchor_H, anchor_metadata = fill_enthalpy_anchor_from_thermo_db(
            thermo_db=args.thermo_db,
            thermo_formula=args.thermo_formula,
            thermo_phase=args.thermo_phase,
            thermo_db_temperature=args.thermo_db_temperature,
            thermo_anchor_T=thermo_anchor_T,
            thermo_anchor_H_J_mol=thermo_anchor_H,
            anchor_metadata=anchor_metadata,
        )
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if (
        args.neel_correction == "on"
        and args.thermo_anchor_H is None
        and "thermo_anchor_H_J_mol" in (anchor_metadata or {}).get("thermo_db_filled_fields", [])
    ):
        qha_anchor = (anchor_metadata or {}).get("qha_anchor")
        if qha_anchor and "thermo_anchor_H_J_mol" in (anchor_metadata or {}).get("filled_fields", []):
            thermo_anchor_H = qha_anchor["H_J_mol_formula"]
        else:
            thermo_anchor_H = None
        anchor_metadata["thermo_db_H_anchor_suppressed_for_neel"] = True
    if anchor_metadata:
        print("Thermodynamic anchor:")
        qha_anchor = anchor_metadata.get("qha_anchor")
        if qha_anchor:
            print(f"  T = {qha_anchor['T_K']:.6g} K")
            print(f"  Cp = {qha_anchor['Cp_J_mol_formula_K']:.6g} J/mol-formula/K")
            print(f"  S = {qha_anchor['S_J_mol_formula_K']:.6g} J/mol-formula/K")
            print(f"  H = {qha_anchor['H_J_mol_formula']:.6g} J/mol-formula")
        db_anchor = anchor_metadata.get("thermo_db_anchor")
        if db_anchor:
            print(f"  DB = {db_anchor['database']} {db_anchor['formula']} {db_anchor['phase']}")
            print(f"  DB H = {db_anchor['H_J_mol_formula']:.6g} J/mol-formula")
        if not args.use_anchor_for_integration:
            print("  note: add --use-anchor-for-integration to use QHA S/H in S/H/G curves")
        if not args.use_anchor_Cp_in_fit:
            print("  note: add --use-anchor-Cp-in-fit to add QHA Cp to the Cp fit")

    qha_low_t_curve = None
    if args.qha_low_t_splice:
        if args.qha_anchor_dir is None:
            ap.error("--qha-low-t-splice requires --qha-anchor-dir")
        if args.qha_anchor_formula_units is None:
            ap.error("--qha-low-t-splice requires --qha-anchor-formula-units")
        try:
            qha_low_t_curve = qha_cp_thermo_curve(
                args.qha_anchor_dir,
                args.qha_anchor_formula_units,
                args.qha_anchor_cp_unit,
            )
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))
    if args.entropy_anchor_min_blend_start < 0.0:
        ap.error("--entropy-anchor-min-blend-start must be non-negative")
    if args.neel_T <= 0.0:
        ap.error("--neel-T must be positive")
    if args.neel_entropy < 0.0:
        ap.error("--neel-entropy must be non-negative")
    if args.neel_apply_above_T <= args.neel_T:
        ap.error("--neel-apply-above-T must be greater than --neel-T")
    if args.neel_enthalpy != "auto":
        try:
            float(args.neel_enthalpy)
        except ValueError:
            ap.error("--neel-enthalpy must be 'auto' or a numeric kJ/mol-formula value")

    entropy_anchor_blend_T = None
    entropy_anchor_blend_S = None
    entropy_anchor_is_direct = False
    if args.thermo_anchor_T is not None and args.thermo_anchor_S is not None:
        entropy_anchor_blend_T = args.thermo_anchor_T
        entropy_anchor_blend_S = args.thermo_anchor_S
        entropy_anchor_is_direct = True
    db_anchor = (anchor_metadata or {}).get("thermo_db_anchor")
    if db_anchor and args.neel_correction != "on":
        entropy_anchor_blend_T = db_anchor["temperature_value_K"]
        entropy_anchor_blend_S = db_anchor["S_J_mol_formula_K"]
        entropy_anchor_is_direct = True
    elif db_anchor and args.neel_correction == "on":
        entropy_anchor_blend_T = db_anchor["temperature_value_K"]
        entropy_anchor_blend_S, entropy_benchmark_metadata = neel_adjusted_entropy_benchmark(
            db_anchor,
            args.neel_T,
            args.neel_entropy,
            args.neel_apply_above_T,
        )
        entropy_anchor_is_direct = False
        anchor_metadata["neel_adjusted_entropy_benchmark"] = entropy_benchmark_metadata

    try:
        lattice_references = parse_lattice_references(args.lattice_reference)
        build_combined_thermo(
            summaries=summaries,
            outdir=outdir,
            fit_degree=args.fit_degree,
            cp_source=args.cp_source,
            plot_T_min=args.plot_T_min,
            plot_T_max=args.plot_T_max,
            plot_T_step=args.plot_T_step,
            anchor_zero=args.anchor_zero,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
            thermo_anchor_T=thermo_anchor_T,
            thermo_anchor_S_J_mol_K=thermo_anchor_S,
            thermo_anchor_Cp_J_mol_K=thermo_anchor_Cp,
            thermo_anchor_H_J_mol=thermo_anchor_H,
            use_anchor_for_integration=args.use_anchor_for_integration,
            use_anchor_Cp_in_fit=args.use_anchor_Cp_in_fit,
            anchor_metadata=anchor_metadata,
            qha_low_t_curve=qha_low_t_curve,
            qha_splice_switch_temperature=args.qha_splice_switch_temperature,
            qha_splice_min_switch_temperature=args.qha_splice_min_switch_temperature,
            qha_splice_blend_start=args.qha_splice_blend_start,
            qha_splice_blend_end=args.qha_splice_blend_end,
            entropy_anchor_blend_T=entropy_anchor_blend_T,
            entropy_anchor_blend_S_J_mol_K=entropy_anchor_blend_S,
            entropy_anchor_min_blend_start=args.entropy_anchor_min_blend_start,
            entropy_anchor_is_direct=entropy_anchor_is_direct,
            neel_correction=args.neel_correction,
            neel_t=args.neel_T,
            neel_entropy=args.neel_entropy,
            neel_enthalpy=args.neel_enthalpy,
            neel_apply_above_t=args.neel_apply_above_T,
            thermo_formula=args.thermo_formula,
            structure_reference_temperature=args.structure_reference_temperature,
            volume_reference=args.volume_reference,
            lattice_references=lattice_references,
            structure_correction=args.structure_correction,
            structure_correction_apply_to=args.structure_correction_apply_to,
            plot_thermo_db_points=args.plot_thermo_db_points,
            target_z=args.target_z,
        )
    except ValueError as exc:
        ap.error(str(exc))

    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
