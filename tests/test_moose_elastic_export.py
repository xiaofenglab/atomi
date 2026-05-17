from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from atomi.lammps.elastic import voigt_reuss_hill
from atomi.moose.elastic_export import main


def cubic_tensor() -> np.ndarray:
    c = np.zeros((6, 6), dtype=float)
    c[:3, :3] = 100.0
    np.fill_diagonal(c[:3, :3], 300.0)
    c[3, 3] = c[4, 4] = c[5, 5] = 80.0
    return c


def test_moose_elastic_export_writes_tensor_table_and_include(tmp_path: Path) -> None:
    c = cubic_tensor()
    elastic_dir = tmp_path / "elastic"
    elastic_dir.mkdir()
    (elastic_dir / "elastic_tensors.json").write_text(
        json.dumps(
            {
                "300.0": {
                    "temperature_K": 300.0,
                    "C_symmetry_reduced_GPa": c.tolist(),
                    "moduli": voigt_reuss_hill(c),
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "moose"
    main(["--elastic-dir", str(elastic_dir), "--outdir", str(out), "--material", "UO2"])

    rows = list(csv.DictReader((out / "moose_elastic_tensors.csv").open(encoding="utf-8")))
    assert float(rows[0]["C11_Pa"]) == 300.0e9
    include = (out / "UO2_elasticity.i").read_text(encoding="utf-8")
    assert "ComputeElasticityTensor" in include
    assert "PiecewiseLinear" in include
    assert "C_ijkl" in include
