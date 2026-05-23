from __future__ import annotations

import csv
import json
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
    main(
        [
            "--elastic-summary",
            str(summary),
            "--elastic-select",
            "average",
            "--outdir",
            str(out),
            "--formula",
            "UO2",
            "--natoms",
            "96",
            "--atoms-per-formula-unit",
            "3",
            "--no-plot",
        ]
    )

    rows = list(csv.DictReader((out / "thermal_conductivity_T.csv").open(encoding="utf-8")))
    assert rows[0]["source"] == "elastic_min_average"
    assert float(rows[0]["k_W_mK"]) == 2.25
    assert float(rows[0]["n_formula_units"]) == 32.0
    metadata = json.loads((out / "thermal_conductivity_metadata.json").read_text(encoding="utf-8"))
    assert metadata["cell_metadata"]["formula"] == "UO2"


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


def test_thermal_k_lammps_summarizes_nma_modes_and_compares_gk(tmp_path: Path) -> None:
    gk = tmp_path / "gk.csv"
    gk.write_text("T_K,k_W_mK,k_std_W_mK\n900,2.0,0.1\n", encoding="utf-8")
    nma = tmp_path / "nma_modes.csv"
    nma.write_text(
        "T_K,frequency_THz,branch,lifetime_ps,mode_heat_capacity_J_m3K,vg_x_m_s,vg_y_m_s,vg_z_m_s\n"
        "900,3.0,TA,10,1000000,100,100,100\n"
        "900,12.0,optical,5,2000000,200,100,100\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"

    main(
        [
            "--green-kubo-csv",
            str(gk),
            "--green-kubo-label",
            "GK_900K",
            "--nma-csv",
            str(nma),
            "--nma-label",
            "NMA_900K",
            "--nma-optical-cutoff-THz",
            "9",
            "--outdir",
            str(out),
            "--no-plot",
        ]
    )

    rows = list(csv.DictReader((out / "thermal_conductivity_T.csv").open(encoding="utf-8")))
    nma_rows = [row for row in rows if row["source"] == "NMA_900K"]
    assert len(nma_rows) == 1
    assert float(nma_rows[0]["k_W_mK"]) > 0.0
    assert int(nma_rows[0]["nma_mode_count"]) == 2
    assert float(nma_rows[0]["nma_optical_k_W_mK"]) > 0.0
    comparison = list(csv.DictReader((out / "gk_nma_comparison.csv").open(encoding="utf-8")))
    assert comparison[0]["diagnostic"] == "large_gap_nonphonon_or_disorder_transport"
    metadata = json.loads((out / "thermal_conductivity_metadata.json").read_text(encoding="utf-8"))
    assert metadata["gk_nma_comparison"]["n_rows"] == 1
