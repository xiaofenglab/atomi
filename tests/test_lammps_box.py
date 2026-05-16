from __future__ import annotations

import numpy as np

from atomi.lammps.box import infer_box_symmetry, summarize_cells, summarize_lammps_box_arrays


def test_infer_box_symmetry_basic_metric_families() -> None:
    assert infer_box_symmetry(5.47, 5.471, 5.469, 90.0, 90.0, 90.0) == "cubic"
    assert infer_box_symmetry(5.47, 5.47, 6.10, 90.0, 90.0, 90.0) == "tetragonal"
    assert infer_box_symmetry(5.47, 5.80, 6.10, 90.0, 90.0, 90.0) == "orthorhombic"
    assert infer_box_symmetry(5.47, 5.47, 6.10, 90.0, 90.0, 120.0) == "hexagonal"


def test_summarize_lammps_box_arrays_reports_cubic() -> None:
    summary = summarize_lammps_box_arrays(
        [5.47, 5.48],
        [5.47, 5.48],
        [5.47, 5.48],
        volume=[5.47**3, 5.48**3],
    )

    assert summary["box_symmetry"] == "cubic"
    assert summary["box_status"] == "ok"
    assert summary["n_box_samples"] == 2
    assert summary["tilt_source"] == "thermo_lx_ly_lz_orthogonal_assumed"


def test_summarize_cells_detects_triclinic_tilt() -> None:
    cell = np.asarray(
        [
            [5.0, 0.0, 0.0],
            [0.8, 5.0, 0.0],
            [0.2, 0.4, 5.0],
        ],
        dtype=float,
    )
    summary = summarize_cells([cell])

    assert summary["box_symmetry"] == "triclinic"
    assert summary["box_status"] == "ok"
