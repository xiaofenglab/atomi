"""Reusable scientific plotting style helpers for Atomi reports.

The helpers here intentionally keep Matplotlib imports inside plotting
functions so analysis modules can import this file in non-plotting contexts.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


COLORS: dict[str, str] = {
    "model": "#1f77b4",
    "reference": "#4a4a4a",
    "experiment": "#d95f02",
    "simulation": "#2ca02c",
    "diagnostic": "#7f7f7f",
    "warning": "#d62728",
    "posterior": "#1f77b4",
    "accepted": "#2ca02c",
    "attempted": "#111111",
    "pending": "#9a9a9a",
}

LINESTYLES: dict[str, str] = {
    "model": "-",
    "reference": "--",
    "experiment": "none",
    "simulation": "-",
    "diagnostic": ":",
    "posterior": "-",
}

MARKERS: dict[str, str] = {
    "experiment": "o",
    "accepted": "o",
    "attempted": "x",
    "pending": "o",
    "benchmark": "*",
}


def matplotlib_cache_dir() -> Path:
    """Return a writable Matplotlib cache directory for headless jobs."""

    cache = Path(tempfile.gettempdir()) / "atomi-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def configure_matplotlib(
    *,
    backend: str = "Agg",
    font_size: float = 11.0,
    title_size: float = 13.0,
    label_size: float = 11.0,
    tick_size: float = 10.0,
    legend_size: float = 9.0,
    dpi: int = 220,
) -> Any:
    """Configure Matplotlib for clean scientific report figures."""

    cache = matplotlib_cache_dir()
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use(backend)
    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": label_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )
    return matplotlib


def require_pyplot(*, backend: str = "Agg") -> Any:
    """Return Matplotlib pyplot after applying Atomi's headless defaults."""

    configure_matplotlib(backend=backend)
    import matplotlib.pyplot as plt

    return plt


def apply_axes_style(
    ax: Any,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    grid: bool = True,
    legend: bool = False,
    legend_kwargs: dict[str, Any] | None = None,
) -> None:
    """Apply common axis labels, grid, spines, and optional legend."""

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", length=4.0, width=0.8)
    if legend:
        kwargs = {"frameon": False}
        if legend_kwargs:
            kwargs.update(legend_kwargs)
        ax.legend(**kwargs)


def save_figure(
    fig: Any,
    path: Path | str,
    *,
    dpi: int = 220,
    extra_formats: Iterable[str] = (),
) -> list[Path]:
    """Save a figure and optional sibling formats, returning written paths."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = [target]
    fig.savefig(target, dpi=dpi)
    for fmt in extra_formats:
        suffix = "." + fmt.lower().lstrip(".")
        sibling = target.with_suffix(suffix)
        fig.savefig(sibling, dpi=dpi)
        written.append(sibling)
    return written


def _finite_xy(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.shape != yy.shape:
        raise ValueError("x and y must have the same shape")
    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size == 0:
        return xx, yy
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    unique_x: list[float] = []
    unique_y: list[float] = []
    for value in np.unique(xx):
        same = xx == value
        unique_x.append(float(value))
        unique_y.append(float(np.mean(yy[same])))
    return np.asarray(unique_x), np.asarray(unique_y)


def smooth_xy(
    x: Sequence[float],
    y: Sequence[float],
    *,
    points: int = 240,
    method: str = "pchip",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth, shape-preserving curve for model/reference trends.

    PCHIP is used when SciPy is available because it avoids the overshoot that
    can make sparse thermodynamic curves look more certain than they are.
    """

    xx, yy = _finite_xy(x, y)
    if xx.size < 3 or points <= xx.size or float(xx[0]) == float(xx[-1]):
        return xx, yy
    grid = np.linspace(float(xx[0]), float(xx[-1]), int(points))
    if method == "linear":
        return grid, np.interp(grid, xx, yy)
    if method != "pchip":
        raise ValueError(f"Unsupported smoothing method: {method}")
    try:
        from scipy.interpolate import PchipInterpolator

        interpolator = PchipInterpolator(xx, yy, extrapolate=False)
        return grid, np.asarray(interpolator(grid), dtype=float)
    except Exception:
        return grid, np.interp(grid, xx, yy)


def plot_curve(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str | None = None,
    role: str = "model",
    color: str | None = None,
    linestyle: str | None = None,
    linewidth: float = 2.2,
    alpha: float = 1.0,
    smooth: bool = True,
    smooth_points: int = 240,
    marker: str | None = None,
    marker_size: float = 4.8,
    marker_every: int | None = None,
    zorder: float | None = None,
) -> Any:
    """Plot a model/reference curve with optional shape-preserving smoothing."""

    color = color or COLORS.get(role, COLORS["model"])
    linestyle = linestyle or LINESTYLES.get(role, "-")
    xx, yy = smooth_xy(x, y, points=smooth_points) if smooth else _finite_xy(x, y)
    line = ax.plot(
        xx,
        yy,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )[0]
    if marker:
        raw_x, raw_y = _finite_xy(x, y)
        if marker_every and marker_every > 1:
            raw_x = raw_x[::marker_every]
            raw_y = raw_y[::marker_every]
        ax.plot(
            raw_x,
            raw_y,
            linestyle="none",
            marker=marker,
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=color,
            alpha=alpha,
            zorder=(zorder + 0.2) if zorder is not None else None,
        )
    return line


def plot_model_curve(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot a primary model curve."""

    return plot_curve(ax, x, y, label=label, role="model", **kwargs)


def plot_reference_curve(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str | None = None,
    **kwargs: Any,
) -> Any:
    """Plot an assessed/literature/reference curve."""

    kwargs.setdefault("linewidth", 2.0)
    return plot_curve(ax, x, y, label=label, role="reference", **kwargs)


def plot_literature_points(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    label: str | None = None,
    yerr: Sequence[float] | Sequence[Sequence[float]] | None = None,
    color: str | None = None,
    marker: str = "o",
    size: float = 48.0,
    edgecolor: str = "white",
    linewidth: float = 0.7,
    zorder: float = 4.0,
) -> Any:
    """Plot measured or tabulated literature points as visible markers."""

    color = color or COLORS["experiment"]
    xx, yy = _finite_xy(x, y)
    if yerr is not None:
        return ax.errorbar(
            xx,
            yy,
            yerr=yerr,
            fmt=marker,
            ms=math.sqrt(size),
            color=color,
            markerfacecolor=color,
            markeredgecolor=edgecolor,
            markeredgewidth=linewidth,
            capsize=3,
            linestyle="none",
            label=label,
            zorder=zorder,
        )
    return ax.scatter(
        xx,
        yy,
        s=size,
        marker=marker,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )


def plot_status_markers(
    ax: Any,
    x: Sequence[float],
    y: Sequence[float],
    *,
    status: str,
    label: str | None = None,
    size: float = 54.0,
    zorder: float = 5.0,
) -> Any:
    """Plot accepted/attempted/pending calculation status points."""

    status_key = status.lower().strip()
    color = COLORS.get(status_key, COLORS["diagnostic"])
    marker = MARKERS.get(status_key, "o")
    xx, yy = _finite_xy(x, y)
    if marker in {"x", "+", "1", "2", "3", "4", "|", "_"}:
        return ax.scatter(
            xx,
            yy,
            s=size,
            marker=marker,
            color=color,
            linewidth=1.4,
            label=label,
            zorder=zorder,
        )
    face = "none" if status_key == "pending" else color
    return ax.scatter(
        xx,
        yy,
        s=size,
        marker=marker,
        facecolors=face,
        edgecolors=color,
        linewidth=1.2,
        label=label,
        zorder=zorder,
    )


def plot_posterior_band(
    ax: Any,
    x: Sequence[float],
    lower: Sequence[float],
    median: Sequence[float],
    upper: Sequence[float],
    *,
    label: str = "posterior median",
    band_label: str = "posterior interval",
    color: str | None = None,
    alpha: float = 0.18,
    smooth: bool = True,
) -> tuple[Any, Any]:
    """Plot a CALPHAD-style uncertainty band plus median curve."""

    color = color or COLORS["posterior"]
    xx, lo = smooth_xy(x, lower) if smooth else _finite_xy(x, lower)
    xm, med = smooth_xy(x, median) if smooth else _finite_xy(x, median)
    xu, hi = smooth_xy(x, upper) if smooth else _finite_xy(x, upper)
    common = xx
    if not (np.array_equal(xx, xm) and np.array_equal(xx, xu)):
        lo = np.interp(common, xx, lo)
        med = np.interp(common, xm, med)
        hi = np.interp(common, xu, hi)
    band = ax.fill_between(common, lo, hi, color=color, alpha=alpha, label=band_label)
    line = ax.plot(common, med, color=color, linewidth=2.3, label=label)[0]
    return band, line


__all__ = [
    "COLORS",
    "LINESTYLES",
    "MARKERS",
    "apply_axes_style",
    "configure_matplotlib",
    "matplotlib_cache_dir",
    "plot_curve",
    "plot_literature_points",
    "plot_model_curve",
    "plot_posterior_band",
    "plot_reference_curve",
    "plot_status_markers",
    "require_pyplot",
    "save_figure",
    "smooth_xy",
]
