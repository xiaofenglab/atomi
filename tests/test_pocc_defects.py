from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy.pocc_defects import (
    DefectConfiguration,
    build_degeneracy_table,
    effective_charge,
    gduo2_observables,
    ingest_vasp_runs,
    raw_supercell_degeneracy,
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


def test_degeneracy_table_uses_raw_supercell_counts(tmp_path: Path) -> None:
    rows, metadata = build_degeneracy_table(
        [
            {
                "motif_id": "2Gd_2U5_sc32",
                "N_cation": 32,
                "N_anion_sites": 64,
                "Gd3": 2,
                "U5": 2,
                "VaO": 0,
            },
            {
                "motif_id": "2Gd_VaO_sc32",
                "N_cation": 32,
                "N_anion_sites": 64,
                "Gd3": 2,
                "U5": 0,
                "VaO": 1,
            },
        ]
    )

    assert metadata["n_motifs"] == 2
    by_id = {row["motif_id"]: row for row in rows}
    assert by_id["2Gd_2U5_sc32"]["g_raw_supercell"] == raw_supercell_degeneracy(
        {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    )
    assert by_id["2Gd_2U5_sc32"]["g_raw_supercell"] == 215760
    assert by_id["2Gd_VaO_sc32"]["g_raw_supercell"] == 31744

    from atomi.zentropy.pocc_defects import main

    motif_csv = tmp_path / "motifs.csv"
    motif_csv.write_text(
        "motif_id,N_cation,N_anion_sites,Gd3,U5,VaO\n"
        "2Gd_2U5_sc32,32,64,2,2,0\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "deg"
    main(["degeneracy-table", "--motif-csv", str(motif_csv), "--outdir", str(outdir)])
    assert (outdir / "motif_degeneracy_gk.csv").exists()


def test_vasp_ingest_builds_auditable_pocc_ensemble(tmp_path: Path) -> None:
    from atomi.zentropy.pocc_defects import main

    run = tmp_path / "gd_u5_run"
    run.mkdir()
    (run / "CONTCAR").write_text(
        """Gd-UO2 defect motif
1.0
4 0 0
0 4 0
0 0 4
U Gd O
30 2 64
Direct
""",
        encoding="utf-8",
    )
    (run / "OUTCAR").write_text(
        """NIONS =     96 ions
 free  energy   TOTEN  =      -123.456789 eV
 volume of cell :       64.0000
""",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "run,config_id,U5,degeneracy,motif_labels,oxidation_assignment\n"
        f"{run},gd_u5_nn,2,3,Gd_U5_NN,manual_review\n",
        encoding="utf-8",
    )

    configs, audit = ingest_vasp_runs([run], metadata={str(run): {"U5": "2", "oxidation_assignment": "manual"}})
    assert configs[0].species_counts == {"U4": 28, "U5": 2, "Gd3": 2, "O": 64, "VaO": 0}
    assert audit[0]["effective_charge"] == 0

    outdir = tmp_path / "ingest"
    main(["ingest-vasp", "--run-dir", str(run), "--metadata-csv", str(metadata), "--outdir", str(outdir)])

    assert (outdir / "ensemble.jsonl").exists()
    record = json.loads((outdir / "ensemble.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["config_id"] == "gd_u5_nn"
    assert record["E_static_eV"] == -123.456789
    assert record["species_counts"]["U4"] == 28
    with (outdir / "vasp_ingest_audit.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["warnings"] == ""
