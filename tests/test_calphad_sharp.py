import csv
import json
from pathlib import Path

from atomi.calphad import sharp
from atomi.cli.main import main as atomi_main


def write_search_csv(path: Path) -> None:
    rows = [
        {
            "liquid_family": "right",
            "liquid_lambda": -0.2,
            "compound_label": "Na3U5Cl18",
            "gform_kj_mol": -8.5,
            "dcp_b_j_mol_k": -60,
            "sexcess_j_mol_k": -5,
            "hmix_rmse_kj_mol": 0.4,
            "eutectic_x": 0.340,
            "eutectic_t_k": 790,
            "highu_liq_rmse_k": 18,
            "total_score": 1.0,
        },
        {
            "liquid_family": "right",
            "liquid_lambda": -0.1,
            "compound_label": "Na3U5Cl18",
            "gform_kj_mol": -8.0,
            "dcp_b_j_mol_k": -40,
            "sexcess_j_mol_k": -5,
            "hmix_rmse_kj_mol": 0.5,
            "eutectic_x": 0.345,
            "eutectic_t_k": 795,
            "highu_liq_rmse_k": 25,
            "total_score": 1.7,
        },
        {
            "liquid_family": "broad",
            "liquid_lambda": 0.4,
            "compound_label": "NaU2Cl7",
            "gform_kj_mol": -6.0,
            "dcp_b_j_mol_k": 0,
            "sexcess_j_mol_k": 10,
            "hmix_rmse_kj_mol": 1.1,
            "eutectic_x": 0.270,
            "eutectic_t_k": 725,
            "highu_liq_rmse_k": 80,
            "total_score": 6.0,
        },
        {
            "liquid_family": "broad",
            "liquid_lambda": 0.6,
            "compound_label": "NaU2Cl7",
            "gform_kj_mol": -5.5,
            "dcp_b_j_mol_k": 20,
            "sexcess_j_mol_k": 15,
            "hmix_rmse_kj_mol": 1.4,
            "eutectic_x": 0.250,
            "eutectic_t_k": 700,
            "highu_liq_rmse_k": 110,
            "total_score": 8.0,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_sharp_analyze_writes_ranking_and_target_residuals(tmp_path: Path) -> None:
    search = tmp_path / "search.csv"
    write_search_csv(search)
    outdir = tmp_path / "sharp"

    result = sharp.main(
        [
            "analyze",
            "--search-csv",
            str(search),
            "--outdir",
            str(outdir),
            "--target",
            "eutectic_x=0.341",
            "--target",
            "eutectic_t_k=784",
            "--top-n",
            "2",
        ]
    )

    assert result["schema"] == sharp.SCHEMA
    assert "abs_eutectic_x_minus_target" in result["metric_columns"]
    ranking = list(csv.DictReader((outdir / "sharp_parameter_ranking.csv").open(encoding="utf-8")))
    assert ranking
    assert ranking[0]["parameter"]
    sensitivity = list(csv.DictReader((outdir / "sharp_sensitivity.csv").open(encoding="utf-8")))
    assert any(row["metric"] == "abs_eutectic_t_k_minus_target" for row in sensitivity)
    report = (outdir / "sharp_report.md").read_text(encoding="utf-8")
    assert "SHARP CALPHAD parameter attribution report" in report
    metadata = json.loads((outdir / "sharp_metadata.json").read_text(encoding="utf-8"))
    assert metadata["top_subset_n"] == 2


def test_atomi_cli_forwards_calphad_sharp(tmp_path: Path) -> None:
    search = tmp_path / "search.csv"
    write_search_csv(search)
    outdir = tmp_path / "forwarded"

    atomi_main(["calphad-sharp", "analyze", "--search-csv", str(search), "--outdir", str(outdir)])

    assert (outdir / "sharp_report.md").exists()
    assert (outdir / "sharp_parameter_ranking.csv").exists()
