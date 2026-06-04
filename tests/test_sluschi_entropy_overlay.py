import csv
import json
from pathlib import Path

from atomi.cli.main import main as atomi_main


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_lammps_sconfig_qha_overlay_normalizes_entropy_with_uq(tmp_path: Path):
    qha = tmp_path / "entropy-temperature.dat"
    qha.write_text("300 180\n900 240\n1500 310\n", encoding="utf-8")

    sconfig = tmp_path / "lammps_sconfig_summary.csv"
    sconfig.write_text(
        "\n".join(
            [
                "temperature_K,mean_pair_sconfig_J_mol_atom_K,sem_pair_sconfig_J_mol_atom_K",
                "900,2.5,0.25",
                "1500,3.1,0.30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "overlay"

    atomi_main(
        [
            "lammps-sconfig-qha-overlay",
            "--qha-entropy",
            str(qha),
            "--qha-entropy-unit",
            "J/mol-cell/K",
            "--qha-formula-units",
            "2",
            "--sluschi-csv",
            str(sconfig),
            "--atoms-per-formula",
            "3",
            "--quantity",
            "sconfig",
            "--label",
            "SLUSCHI Sconf",
            "--outdir",
            str(outdir),
            "--no-plot",
        ]
    )

    rows = read_rows(outdir / "sluschi_qha_entropy_overlay.csv")
    qha_rows = [row for row in rows if row["source"] == "QHA"]
    sluschi_rows = [row for row in rows if row["source"] == "SLUSCHI"]
    assert qha_rows[0]["entropy_J_mol_formula_K"] == "90.0"
    assert sluschi_rows[0]["label"] == "SLUSCHI Sconf"
    assert sluschi_rows[0]["entropy_J_mol_formula_K"] == "7.5"
    assert sluschi_rows[0]["yerr_low_J_mol_formula_K"] == "0.75"
    metadata = json.loads((outdir / "sluschi_qha_entropy_overlay_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "atomi.lammps.sluschi_qha_entropy_overlay.v1"
    assert metadata["n_qha_points"] == 3
    assert metadata["n_sluschi_points"] == 2


def test_lammps_sconfig_summary_includes_uq_columns(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "collect.stdout").write_text(
        "\n".join(
            [
                "The pair between element 1-1 appears to be solid. I suggest that you take the mean:  1.0",
                "The pair between element 1-2 appears to be liquid. I suggest that you take the mean:  2.0",
                "The pair between element 2-2 appears to be liquid. I suggest that you take the mean:  3.0",
            ]
        ),
        encoding="utf-8",
    )
    outdir = tmp_path / "sconfig"

    atomi_main(
        [
            "lammps-sconfig",
            "--root",
            str(root),
            "--outdir",
            str(outdir),
            "--system",
            "UO2",
            "--formula",
            "UO2",
            "--temperature-k",
            "900",
        ]
    )

    rows = read_rows(outdir / "lammps_sconfig_summary.csv")
    assert rows[0]["mean_pair_sconfig_J_mol_atom_K"] == "2.0"
    assert rows[0]["std_pair_sconfig_J_mol_atom_K"] == "1.0"
    assert rows[0]["sem_pair_sconfig_J_mol_atom_K"].startswith("0.577")
    assert rows[0]["min_pair_sconfig_J_mol_atom_K"] == "1.0"
    assert rows[0]["max_pair_sconfig_J_mol_atom_K"] == "3.0"
