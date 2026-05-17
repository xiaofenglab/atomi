from __future__ import annotations

import csv
from pathlib import Path

from atomi.lammps.thermal_conductivity import main


def test_thermal_k_lammps_exports_elastic_lower_bound(tmp_path: Path) -> None:
    summary = tmp_path / "elastic_thermophysical_summary.csv"
    summary.write_text(
        "temperature_K,k_min_cahill_W_mK,k_min_clarke_W_mK\n"
        "300,2.5,2.0\n"
        "600,2.0,1.7\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(["--elastic-summary", str(summary), "--elastic-select", "average", "--outdir", str(out), "--no-plot"])

    rows = list(csv.DictReader((out / "thermal_conductivity_T.csv").open(encoding="utf-8")))
    assert rows[0]["source"] == "elastic_min_average"
    assert float(rows[0]["k_W_mK"]) == 2.25


def test_thermal_k_lammps_integrates_scaled_green_kubo_tail(tmp_path: Path) -> None:
    gk = tmp_path / "gk.csv"
    gk.write_text(
        "time_ps,HCACF_x,HCACF_y,HCACF_z\n"
        "0,1,2,3\n"
        "1,1,2,3\n"
        "2,1,2,3\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(
        [
            "--green-kubo-csv",
            str(gk),
            "--green-kubo-temperature-K",
            "300",
            "--green-kubo-scale",
            "0.5",
            "--plateau-start-ps",
            "2",
            "--outdir",
            str(out),
            "--no-plot",
        ]
    )
    rows = list(csv.DictReader((out / "thermal_conductivity_T.csv").open(encoding="utf-8")))
    assert float(rows[0]["k_W_mK"]) == 2.0
    assert float(rows[0]["k_x_W_mK"]) == 1.0
