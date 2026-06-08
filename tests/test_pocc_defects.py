from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy.pocc_defects import (
    DefectConfiguration,
    effective_charge,
    gduo2_observables,
    solve_static_zentropy,
)


def test_gduo2_charge_and_composition_observables() -> None:
    gd_u5 = {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    gd2_vo = {"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1}

    assert effective_charge(gd_u5) == 0
    assert effective_charge(gd2_vo) == 0
    assert gduo2_observables(gd_u5)["x_Gd"] == gduo2_observables(gd2_vo)["x_Gd"]
    assert gduo2_observables(gd_u5)["h_U5"] == 2 / 32
    assert gduo2_observables(gd2_vo)["delta"] == 1 / 32


def test_static_zentropy_population_keeps_degeneracy_separate_from_energy() -> None:
    configs = [
        DefectConfiguration(
            config_id="gd_u5_nn",
            phase="fluorite",
            species_counts={"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0},
            sublattice_counts={"cation": 32, "anion": 64},
            degeneracy=2,
            degeneracy_type="motif_embedding",
            E_static_eV=-100.00,
            motif_labels=["Gd_U5_NN"],
            metadata={"oxidation_assignment": "manual_review"},
        ),
        DefectConfiguration(
            config_id="gd2_vo_nn",
            phase="fluorite",
            species_counts={"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1},
            sublattice_counts={"cation": 32, "anion": 64},
            degeneracy=10,
            degeneracy_type="motif_embedding",
            E_static_eV=-99.95,
            motif_labels=["Gd2_VO_NN"],
        ),
    ]

    population, surface, motifs = solve_static_zentropy(
        configs,
        temperatures=[1200.0],
        mu_o_values=[None],
        group_by_x_gd=False,
    )

    assert len(population) == 2
    assert len(surface) == 1
    assert len(motifs) == 2
    probs = {row["config_id"]: row["probability"] for row in population}
    assert probs["gd2_vo_nn"] > probs["gd_u5_nn"]
    assert surface[0]["S_population_J_molK"] > 0
    assert "S_site_ideal_J_molK" in surface[0]
    assert "S_excess_conf_J_molK" in surface[0]


def test_cli_solve_static_outputs_tables(tmp_path: Path) -> None:
    from atomi.zentropy.pocc_defects import main

    ensemble = tmp_path / "ensemble.jsonl"
    records = [
        {
            "config_id": "gd_u5",
            "species_counts": {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0},
            "degeneracy": 1,
            "E_static_eV": -10.0,
            "motif_labels": ["Gd_U5"],
            "oxidation_assignment": "manual_review",
        },
        {
            "config_id": "gd2_vo",
            "species_counts": {"U4": 30, "U5": 0, "Gd3": 2, "O": 63, "VaO": 1},
            "degeneracy": 4,
            "E_static_eV": -9.9,
            "motif_labels": ["Gd2_VO"],
        },
    ]
    ensemble.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    outdir = tmp_path / "out"

    main(
        [
            "solve-static",
            "--ensemble",
            str(ensemble),
            "--outdir",
            str(outdir),
            "--temperature",
            "1000",
            "--mu-o",
            "-5",
            "--no-group-by-x-gd",
        ]
    )

    assert (outdir / "configuration_audit.csv").exists()
    assert (outdir / "population_vector.csv").exists()
    with (outdir / "zentropy_surface.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert rows[0]["dominant_config_id"] in {"gd_u5", "gd2_vo"}
