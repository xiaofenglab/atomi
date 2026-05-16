from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.lammps.elastic_qha_md_compare import main


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_elastic_qha_md_compare_reports_structure_only_qha(tmp_path: Path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "volume-temperature.dat").write_text("300 800\n500 832\n", encoding="utf-8")
    md = tmp_path / "elastic_fit"
    write_csv(
        md / "elastic_moduli_T.csv",
        [
            {
                "temperature_K": 300,
                "V_mean_A3": 400,
                "a_mean_A": 5.47,
                "C11_GPa": 300,
                "C12_GPa": 100,
                "C44_GPa": 80,
                "K_H_GPa": 166.7,
                "G_H_GPa": 85.0,
                "E_H_GPa": 223.0,
                "nu_H": 0.276,
            }
        ],
    )
    out = tmp_path / "compare"

    metadata = main(
        [
            "--qha-dir",
            str(qha),
            "--elastic-md-dir",
            str(md),
            "--outdir",
            str(out),
            "--qha-formula-units",
            "8",
            "--md-formula-units",
            "4",
            "--target-z",
            "4",
            "--no-plots",
        ]
    )

    assert metadata["readiness"]["qha_has_structure"] is True
    assert metadata["readiness"]["qha_has_elastic"] is False
    assert metadata["readiness"]["can_compare_elastic_constants"] is False
    assert (out / "structure_qha_md_overlay.csv").exists()
    elastic_rows = list(csv.DictReader((out / "elastic_qha_md_overlay.csv").open(encoding="utf-8")))
    assert {row["source"] for row in elastic_rows} == {"MD elastic"}


def test_elastic_qha_md_compare_uses_qha_elastic_table(tmp_path: Path) -> None:
    qha = tmp_path / "qha"
    qha.mkdir()
    (qha / "a-temperature.dat").write_text("300 5.46\n", encoding="utf-8")
    write_csv(
        qha / "elastic_moduli_T.csv",
        [{"temperature_K": 300, "C11_GPa": 310, "C12_GPa": 110, "C44_GPa": 85, "K_H_GPa": 176.0}],
    )
    md = tmp_path / "elastic_fit"
    write_csv(
        md / "elastic_moduli_T.csv",
        [{"temperature_K": 300, "a_mean_A": 5.47, "C11_GPa": 300, "C12_GPa": 100, "C44_GPa": 80, "K_H_GPa": 166.7}],
    )
    out = tmp_path / "compare"

    metadata = main(
        [
            "--qha-dir",
            str(qha),
            "--elastic-md-dir",
            str(md),
            "--outdir",
            str(out),
            "--qha-formula-units",
            "4",
            "--md-formula-units",
            "4",
            "--no-plots",
        ]
    )

    assert metadata["readiness"]["can_compare_elastic_constants"] is True
    rows = list(csv.DictReader((out / "elastic_qha_md_overlay.csv").open(encoding="utf-8")))
    c11 = [row for row in rows if row["component"] == "C11"]
    assert {row["source"] for row in c11} == {"QHA/static", "MD elastic"}
    written = json.loads((out / "elastic_qha_md_metadata.json").read_text(encoding="utf-8"))
    assert written["qha_elastic"]["elasticity_ready"] is True
