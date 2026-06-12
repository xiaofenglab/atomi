from __future__ import annotations

import csv
from pathlib import Path

from atomi.zentropy.backends.base import CETrainingRecord, CETrainingSet, write_ce_training_jsonl
from atomi.zentropy.gnn_active_learning import build_candidate_pool, main, score_candidates, select_candidates


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_candidate_pool_enforces_charge_neutral_redox_relation() -> None:
    rows = build_candidate_pool(x_grid=[0.125], delta_grid=[0.0, 0.03125, 0.1])

    assert {row["compensation_family"] for row in rows} == {"u5_compensated", "mixed_u5_vacancy"}
    assert all(abs(float(row["charge_residual"])) < 1.0e-10 for row in rows)
    assert all(float(row["h_U5"]) >= 0.0 for row in rows)


def test_score_candidates_uses_predictions_and_ranks_uncertain_low_energy() -> None:
    candidates = build_candidate_pool(x_grid=[0.125], delta_grid=[0.0, 0.03125])
    prediction_rows = {
        candidates[0]["candidate_id"]: {
            "candidate_id": candidates[0]["candidate_id"],
            "predicted_G_eV_per_fu": "-1.0",
            "surrogate_uncertainty_eV": "0.02",
            "source": "unit_gnn",
        },
        candidates[1]["candidate_id"]: {
            "candidate_id": candidates[1]["candidate_id"],
            "predicted_G_eV_per_fu": "-0.9",
            "surrogate_uncertainty_eV": "0.30",
            "source": "unit_gnn",
        },
    }

    scored = score_candidates(candidates, prediction_rows=prediction_rows, exploration_weight=1.0)
    selected = select_candidates(scored, top_n=1)

    assert selected[0]["candidate_id"] == candidates[1]["candidate_id"]
    assert selected[0]["surrogate_source"] == "unit_gnn"


def test_gnn_active_learning_cli_round_trip(tmp_path: Path) -> None:
    training = CETrainingSet(
        system_name="Gd-UO2",
        parent_structure_path="fluorite",
        records=[
            CETrainingRecord(
                record_id="low_gd_seed",
                structure_path="seed/POSCAR",
                composition={"x_Gd": 0.0625, "delta_VO": 0.0, "h_U5": 0.0625},
                motif_features={"Gd_U5_nn": 2.0},
                source="seed",
                metadata={"motif_family": "Gd_U5"},
            )
        ],
    )
    seed_jsonl = tmp_path / "seed.jsonl"
    write_ce_training_jsonl(seed_jsonl, training)

    pool_dir = tmp_path / "pool"
    main(
        [
            "build-candidates",
            "--seed-training-jsonl",
            str(seed_jsonl),
            "--outdir",
            str(pool_dir),
            "--x-grid",
            "0.125",
            "--delta-grid",
            "0,0.03125",
        ]
    )
    pool = read_csv(pool_dir / "candidate_pool.csv")
    assert len(pool) == 2
    assert pool[0]["seed_motif_id"] == "low_gd_seed"

    prediction = tmp_path / "prediction.csv"
    prediction.write_text(
        "candidate_id,predicted_G_eV_per_fu,surrogate_uncertainty_eV,source\n"
        f"{pool[0]['candidate_id']},-1.0,0.1,mace_committee\n"
        f"{pool[1]['candidate_id']},-0.8,0.5,mace_committee\n",
        encoding="utf-8",
    )
    score_dir = tmp_path / "score"
    main(
        [
            "score-candidates",
            "--candidate-csv",
            str(pool_dir / "candidate_pool.csv"),
            "--prediction-csv",
            str(prediction),
            "--seed-training-jsonl",
            str(seed_jsonl),
            "--outdir",
            str(score_dir),
        ]
    )
    scored = read_csv(score_dir / "scored_candidates.csv")
    assert scored[0]["surrogate_source"] == "mace_committee"

    select_dir = tmp_path / "select"
    main(["select-dft", "--scored-csv", str(score_dir / "scored_candidates.csv"), "--outdir", str(select_dir), "--top-n", "1"])
    assert (select_dir / "selected_dft_candidates.csv").exists()
    assert (select_dir / "selected_candidates_mode4_prior.jsonl").exists()
