"""LAMMPS MD box diagnostics and symmetry guards."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _finite(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def _stats(values: Iterable[float], prefix: str) -> dict[str, float | int]:
    arr = _finite(values)
    if len(arr) == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_sem": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_n": 0,
        }
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": std,
        f"{prefix}_sem": std / math.sqrt(len(arr)) if len(arr) > 1 else 0.0,
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_n": int(len(arr)),
    }


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom == 0.0:
        return math.nan
    cosine = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def cell_parameters(cell: np.ndarray) -> dict[str, float]:
    """Return conventional cell lengths, angles, and volume for an ASE/LAMMPS cell."""
    a_vec, b_vec, c_vec = np.asarray(cell, dtype=float)
    return {
        "a_A": float(np.linalg.norm(a_vec)),
        "b_A": float(np.linalg.norm(b_vec)),
        "c_A": float(np.linalg.norm(c_vec)),
        "alpha_deg": _angle_deg(b_vec, c_vec),
        "beta_deg": _angle_deg(a_vec, c_vec),
        "gamma_deg": _angle_deg(a_vec, b_vec),
        "volume_A3": float(abs(np.linalg.det(cell))),
    }


def restricted_triclinic_cell(lx: float, ly: float, lz: float, xy: float = 0.0, xz: float = 0.0, yz: float = 0.0) -> np.ndarray:
    """Build a LAMMPS restricted-triclinic cell matrix with row vectors."""
    return np.asarray(
        [
            [lx, 0.0, 0.0],
            [xy, ly, 0.0],
            [xz, yz, lz],
        ],
        dtype=float,
    )


def infer_box_symmetry(
    a_A: float,
    b_A: float,
    c_A: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    length_rel_tol: float = 0.01,
    angle_tol_deg: float = 1.0,
) -> str:
    """Infer a lattice-family label from metric symmetry.

    This is intentionally a box-shape diagnostic, not a full atomic-space-group
    finder. It catches the NPT cell-shape issues that matter for post-analysis.
    """
    lengths = np.asarray([a_A, b_A, c_A], dtype=float)
    angles = np.asarray([alpha_deg, beta_deg, gamma_deg], dtype=float)
    if not np.all(np.isfinite(lengths)) or not np.all(np.isfinite(angles)) or np.any(lengths <= 0.0):
        return "unknown"

    scale = max(float(np.max(np.abs(lengths))), 1.0)

    def same(x: float, y: float) -> bool:
        return abs(x - y) / scale <= length_rel_tol

    def near_angle(x: float, target: float) -> bool:
        return abs(x - target) <= angle_tol_deg

    right = all(near_angle(x, 90.0) for x in angles)
    if right:
        if same(a_A, b_A) and same(a_A, c_A):
            return "cubic"
        if same(a_A, b_A) or same(a_A, c_A) or same(b_A, c_A):
            return "tetragonal"
        return "orthorhombic"

    if same(a_A, b_A) and near_angle(alpha_deg, 90.0) and near_angle(beta_deg, 90.0) and near_angle(gamma_deg, 120.0):
        return "hexagonal"
    if same(a_A, b_A) and same(a_A, c_A) and max(angles) - min(angles) <= angle_tol_deg:
        return "rhombohedral"
    n_right = sum(near_angle(x, 90.0) for x in angles)
    if n_right == 2:
        return "monoclinic"
    return "triclinic"


def summarize_cell_parameters(
    params: list[dict[str, float]],
    length_rel_tol: float = 0.01,
    angle_tol_deg: float = 1.0,
    tilt_source: str = "cell",
) -> dict[str, object]:
    if not params:
        return {
            "box_status": "missing",
            "box_symmetry": "unknown",
            "box_warning": "No cell or thermo box vectors were available.",
            "n_box_samples": 0,
            "length_rel_tol": length_rel_tol,
            "angle_tol_deg": angle_tol_deg,
            "tilt_source": tilt_source,
        }

    summary: dict[str, object] = {
        "n_box_samples": len(params),
        "length_rel_tol": float(length_rel_tol),
        "angle_tol_deg": float(angle_tol_deg),
        "tilt_source": tilt_source,
    }
    for key in ("volume_A3", "a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"):
        summary.update(_stats([p[key] for p in params], key))

    symmetry = infer_box_symmetry(
        float(summary["a_A_mean"]),
        float(summary["b_A_mean"]),
        float(summary["c_A_mean"]),
        float(summary["alpha_deg_mean"]),
        float(summary["beta_deg_mean"]),
        float(summary["gamma_deg_mean"]),
        length_rel_tol=length_rel_tol,
        angle_tol_deg=angle_tol_deg,
    )
    summary["box_symmetry"] = symmetry
    summary["box_status"] = "ok" if symmetry != "unknown" else "warning"
    summary["box_warning"] = "" if symmetry != "unknown" else "Could not infer MD box symmetry from finite cell metrics."
    return summary


def summarize_cells(cells: Iterable[np.ndarray], length_rel_tol: float = 0.01, angle_tol_deg: float = 1.0) -> dict[str, object]:
    params = [cell_parameters(np.asarray(cell, dtype=float)) for cell in cells]
    return summarize_cell_parameters(params, length_rel_tol=length_rel_tol, angle_tol_deg=angle_tol_deg, tilt_source="trajectory_cell")


def summarize_lammps_box_arrays(
    lx: Iterable[float],
    ly: Iterable[float],
    lz: Iterable[float],
    volume: Iterable[float] | None = None,
    xy: Iterable[float] | None = None,
    xz: Iterable[float] | None = None,
    yz: Iterable[float] | None = None,
    length_rel_tol: float = 0.01,
    angle_tol_deg: float = 1.0,
) -> dict[str, object]:
    lx_arr = np.asarray(list(lx), dtype=float)
    ly_arr = np.asarray(list(ly), dtype=float)
    lz_arr = np.asarray(list(lz), dtype=float)
    n = min(len(lx_arr), len(ly_arr), len(lz_arr))
    if n == 0:
        return summarize_cell_parameters([], length_rel_tol=length_rel_tol, angle_tol_deg=angle_tol_deg, tilt_source="thermo_lx_ly_lz")

    def optional(values: Iterable[float] | None) -> np.ndarray:
        if values is None:
            return np.zeros(n, dtype=float)
        arr = np.asarray(list(values), dtype=float)
        if len(arr) < n:
            padded = np.zeros(n, dtype=float)
            padded[: len(arr)] = arr
            return padded
        return arr[:n]

    xy_arr = optional(xy)
    xz_arr = optional(xz)
    yz_arr = optional(yz)
    vol_arr = np.asarray(list(volume), dtype=float)[:n] if volume is not None else None
    params: list[dict[str, float]] = []
    for i in range(n):
        values = [lx_arr[i], ly_arr[i], lz_arr[i], xy_arr[i], xz_arr[i], yz_arr[i]]
        if not np.all(np.isfinite(values)) or min(lx_arr[i], ly_arr[i], lz_arr[i]) <= 0.0:
            continue
        item = cell_parameters(restricted_triclinic_cell(*values))
        if vol_arr is not None and i < len(vol_arr) and np.isfinite(vol_arr[i]) and vol_arr[i] > 0.0:
            item["volume_A3"] = float(vol_arr[i])
        params.append(item)
    tilt_source = "thermo_lx_ly_lz_tilt" if xy is not None or xz is not None or yz is not None else "thermo_lx_ly_lz_orthogonal_assumed"
    return summarize_cell_parameters(params, length_rel_tol=length_rel_tol, angle_tol_deg=angle_tol_deg, tilt_source=tilt_source)


def flatten_box_summary(summary: dict[str, object], prefix: str = "") -> dict[str, object]:
    keys = [
        "box_status",
        "box_symmetry",
        "box_warning",
        "n_box_samples",
        "length_rel_tol",
        "angle_tol_deg",
        "tilt_source",
        "volume_A3_mean",
        "volume_A3_std",
        "volume_A3_sem",
        "a_A_mean",
        "a_A_std",
        "a_A_sem",
        "b_A_mean",
        "b_A_std",
        "b_A_sem",
        "c_A_mean",
        "c_A_std",
        "c_A_sem",
        "alpha_deg_mean",
        "alpha_deg_std",
        "alpha_deg_sem",
        "beta_deg_mean",
        "beta_deg_std",
        "beta_deg_sem",
        "gamma_deg_mean",
        "gamma_deg_std",
        "gamma_deg_sem",
    ]
    return {prefix + key: summary.get(key, "") for key in keys}


def format_box_summary(summary: dict[str, object], label: str = "MD box") -> str:
    symmetry = summary.get("box_symmetry", "unknown")
    status = summary.get("box_status", "unknown")

    def fmt(key: str, digits: int = 4) -> str:
        value = summary.get(key)
        try:
            value_f = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "nan"
        if not math.isfinite(value_f):
            return "nan"
        return f"{value_f:.{digits}f}"

    return (
        f"{label}: symmetry={symmetry} status={status}; "
        f"a,b,c={fmt('a_A_mean')},{fmt('b_A_mean')},{fmt('c_A_mean')} A; "
        f"alpha,beta,gamma={fmt('alpha_deg_mean', 2)},{fmt('beta_deg_mean', 2)},{fmt('gamma_deg_mean', 2)} deg; "
        f"V={fmt('volume_A3_mean')} A^3; samples={summary.get('n_box_samples', 0)}; "
        f"tol={float(summary.get('length_rel_tol', 0.01)) * 100:.2f}%/{float(summary.get('angle_tol_deg', 1.0)):.2f} deg"
    )
