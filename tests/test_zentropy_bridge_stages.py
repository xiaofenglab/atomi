from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy import active_learning, export, free_energy, motif_cluster, solid_solution_scan, solve


def write_motif_db(path: Path) -> None:
    payload = {
        "schema": "test",
        "records": [
            {
                "motif_id": "gd_u5_near",
                "motif_family": "Gd3_U5",
                "defect_label": "charge_compensated",
                "degeneracy": 2,
                "formula": "GdU17O36",
                "energy_per_formula_unit_eV": -10.0,
                "volume_per_formula_unit_A3": 41.0,
                "size_normalization": {
                    "formula_units": 18,
                    "guest_cation_fraction": 0.0555556,
                    "oxygen_delta_per_formula_unit": 0.0,
                },
                "motif_metadata": {"spin_order_host": "AFM-like"},
                "run_dir": "/runs/gd_u5_near",
            },
            {
                "motif_id": "gd_vo_near",
                "motif_family": "Gd3_VO",
                "defect_label": "oxygen_vacancy",
                "degeneracy": 1,
                "formula": "GdU17O35",
                "energy_per_formula_unit_eV": -9.95,
                "volume_per_formula_unit_A3": 40.8,
                "size_normalization": {
                    "formula_units": 18,
                    "guest_cation_fraction": 0.0555556,
                    "oxygen_delta_per_formula_unit": 0.0555556,
                },
                "motif_metadata": {"spin_order_host": "FM"},
                "run_dir": "/runs/gd_vo_near",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_zentropy_free_energy_solve_export_and_active_learning(tmp_path: Path) -> None:
    motif_db = tmp_path / "defect_motif_db.json"
    write_motif_db(motif_db)

    stage2 = tmp_path / "stage2"
    free_energy.main(["--motif-db", str(motif_db), "--temperature", "300:500:200", "--outdir", str(stage2)])
    free_rows = read_csv(stage2 / "microstate_free_energy.csv")
    assert len(free_rows) == 4
    assert {row["motif_id"] for row in free_rows} == {"gd_u5_near", "gd_vo_near"}

    stage3 = tmp_path / "stage3"
    solve.main(["--free-energy-csv", str(stage2 / "microstate_free_energy.csv"), "--outdir", str(stage3)])
    prob_rows = read_csv(stage3 / "ensemble_probabilities.csv")
    thermo_rows = read_csv(stage3 / "zentropy_thermo_functions.csv")
    assert len(prob_rows) == 4
    assert len(thermo_rows) == 2
    by_temp = {}
    for row in prob_rows:
        by_temp.setdefault(row["T_K"], 0.0)
        by_temp[row["T_K"]] += float(row["probability"])
    assert all(abs(value - 1.0) < 1.0e-10 for value in by_temp.values())

    stage4 = tmp_path / "stage4"
    export.main(
        [
            "--thermo-csv",
            str(stage3 / "zentropy_thermo_functions.csv"),
            "--probability-csv",
            str(stage3 / "ensemble_probabilities.csv"),
            "--outdir",
            str(stage4),
            "--material",
            "GdUO2",
            "--phase",
            "defect-fluorite",
        ]
    )
    assert (stage4 / "calphad_pseudodata.csv").exists()
    assert (stage4 / "moose_material_table.csv").exists()

    stage5 = tmp_path / "stage5"
    active_learning.main(
        [
            "--probability-csv",
            str(stage3 / "ensemble_probabilities.csv"),
            "--motif-db",
            str(motif_db),
            "--outdir",
            str(stage5),
            "--top-n",
            "2",
        ]
    )
    candidates = read_csv(stage5 / "active_learning_candidates.csv")
    assert len(candidates) == 2
    assert candidates[0]["rank"] == "1"


def test_mlip_scan_and_motif_cluster(tmp_path: Path) -> None:
    seed_root = tmp_path / "seeds" / "motif_a"
    seed_root.mkdir(parents=True)
    (seed_root / "POSCAR").write_text("placeholder\n", encoding="utf-8")

    scan_out = tmp_path / "scan"
    solid_solution_scan.main(
        [
            "--seed-root",
            str(tmp_path / "seeds"),
            "--guest-fraction",
            "0:0.25:0.25",
            "--oxygen-delta",
            "0",
            "--compensation",
            "host_valence",
            "--motif-family",
            "paired_near",
            "--outdir",
            str(scan_out),
        ]
    )
    scan_rows = read_csv(scan_out / "solid_solution_scan_manifest.csv")
    assert len(scan_rows) == 2
    assert scan_rows[0]["seed_structure"].endswith("POSCAR")

    motif_db = tmp_path / "defect_motif_db.json"
    write_motif_db(motif_db)
    cluster_out = tmp_path / "clusters"
    motif_cluster.main(["--motif-db", str(motif_db), "--outdir", str(cluster_out)])
    reps = read_csv(cluster_out / "motif_cluster_representatives.csv")
    assert len(reps) == 2
    assert {row["representative"] for row in reps} == {"yes"}
