from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atomi.viz.style import (
    COLORS,
    apply_axes_style,
    plot_literature_points,
    plot_model_curve,
    plot_posterior_band,
    plot_status_markers,
    require_pyplot,
    save_figure,
    smooth_xy,
)


def test_smooth_xy_sorts_duplicates_and_preserves_endpoints() -> None:
    x, y = smooth_xy([300, 100, 200, 200], [9, 1, 4, 6], points=25)

    assert x[0] == pytest.approx(100)
    assert x[-1] == pytest.approx(300)
    assert y[0] == pytest.approx(1)
    assert y[-1] == pytest.approx(9)
    assert np.all(np.diff(x) > 0)


def test_smooth_xy_linear_fallback_shape() -> None:
    x, y = smooth_xy([0, 1, 2], [0, 1, 0], points=11, method="linear")

    assert len(x) == 11
    assert y[0] == pytest.approx(0)
    assert y[5] == pytest.approx(1)
    assert y[-1] == pytest.approx(0)


def test_scientific_plot_helpers_write_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    plt = require_pyplot()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))

    plot_model_curve(ax, [300, 600, 900], [40, 70, 91], label="QHA", marker="o")
    plot_literature_points(ax, [300, 500], [45, 75], label="literature")
    plot_status_markers(ax, [900], [88], status="attempted", label="SLUSCHI attempted")
    plot_posterior_band(
        ax,
        [300, 600, 900],
        [35, 62, 84],
        [40, 70, 91],
        [46, 78, 99],
        label="posterior median",
    )
    apply_axes_style(ax, xlabel="T (K)", ylabel="S (J mol-UC2-1 K-1)", legend=True)

    paths = save_figure(fig, tmp_path / "style_smoke.png", extra_formats=("svg",))
    plt.close(fig)

    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert COLORS["model"].startswith("#")
