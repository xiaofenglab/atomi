from __future__ import annotations

import csv
import json
from pathlib import Path

from atomi.zentropy import sd_dd_thermo


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_sd_dd_thermo_writes_populations_summary_and_cef_seed(tmp_path: Path) -> None:
    defects = tmp_path / "defects.csv"
    defects.write_text(
        "defect_id,model,formation_energy_eV,degeneracy,capacity_per_formula,charge,delta_O,sublattice,site_species\n"
        "Gd_U,SD,1.0,2,1.0,-1,0,cation,Gd_U\n"
        "V_O,SD,2.0,1,2.0,2,-1,anion,V_O\n",
        encoding="utf-8",
    )
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "pair_id,defect_a,defect_b,binding_energy_eV,capacity_per_formula\n"
        "GdU_VO_pair,Gd_U,V_O,-0.4,1.0\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "out"

    sd_dd_thermo.main(
        [
            "--defect-csv",
            str(defects),
            "--pair-csv",
            str(pairs),
            "--outdir",
            str(outdir),
            "--temperature",
            "600,1200",
            "--chemical-potential",
            "O=-5.0",
            "--material",
            "Gd_UO2",
            "--formula",
            "UO2",
        ]
    )

    pop = rows(outdir / "sd_dd_defect_populations.csv")
    pair_rows = [row for row in pop if row["defect_id"] == "GdU_VO_pair"]
    assert pair_rows
    assert float(pair_rows[0]["formation_energy_eV"]) == 2.6
    summary = rows(outdir / "sd_dd_summary.csv")
    assert float(summary[1]["single_defect_concentration_per_formula"]) > float(
        summary[0]["single_defect_concentration_per_formula"]
    )
    cef = rows(outdir / "sd_dd_cef_seed.csv")
    assert cef[0]["cef_role"] == "seed_site_fraction_or_endmember_energy_for_future_CEF_assessment"
    metadata = json.loads((outdir / "sd_dd_metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == sd_dd_thermo.SCHEMA


def test_sd_dd_thermo_cli_json_reports_outputs(tmp_path: Path, capsys) -> None:
    defects = tmp_path / "defects.csv"
    defects.write_text(
        "defect_id,formation_energy_eV,degeneracy\n"
        "V_O,1.5,1\n",
        encoding="utf-8",
    )

    sd_dd_thermo.main(["--defect-csv", str(defects), "--outdir", str(tmp_path / "out"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert "populations" in payload["outputs"]


def test_sd_dd_init_prepare_build_and_fit_workflow(tmp_path: Path) -> None:
    workflow = tmp_path / "workflow"
    sd_dd_thermo.main(["init", "--outdir", str(workflow), "--system", "(Gd,U)O2"])
    assert (workflow / "sd_dd_workflow.json").exists()
    assert (workflow / "01_seed_structures" / "sd_dd_seed_index_template.csv").exists()

    template = tmp_path / "VASP_TEMPLATE"
    template.mkdir()
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        (template / name).write_text(f"{name}\n", encoding="utf-8")
    seed = tmp_path / "seed_POSCAR"
    seed.write_text(
        "\n".join(
            [
                "U O",
                "1.0",
                "4 0 0",
                "0 4 0",
                "0 0 4",
                "U O",
                "1 2",
                "Direct",
                "0 0 0",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "",
            ]
        ),
        encoding="utf-8",
    )
    seed_csv = tmp_path / "seed_index.csv"
    seed_csv.write_text(
        "case_id,model,seed_poscar,template,charge,delta_O\n"
        f"V_O_seed,SD,{seed},{template},2,-1\n",
        encoding="utf-8",
    )
    sd_dd_thermo.main(["prepare-runs", "--seed-csv", str(seed_csv), "--outdir", str(tmp_path / "runs")])
    assert (tmp_path / "runs" / "V_O_seed" / "POSCAR").exists()
    assert (tmp_path / "runs" / "runlist.txt").exists()

    motif_db = tmp_path / "defect_motif_db.json"
    motif_db.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "motif_id": "V_O_seed",
                        "motif_family": "oxygen_vacancy",
                        "defect_label": "O_V1",
                        "energy_eV": -28.0,
                        "counts": {"U": 1, "O": 1},
                        "degeneracy": 1,
                        "size_normalization": {
                            "formula_units": 1.0,
                            "oxygen_delta_per_formula_unit": -1.0,
                        },
                        "motif_metadata": {"spin_order_all": "nonmagnetic"},
                        "run_dir": str(tmp_path / "runs" / "V_O_seed"),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    defects = tmp_path / "defects.csv"
    sd_dd_thermo.main(
        [
            "build-defects",
            "--motif-db-json",
            str(motif_db),
            "--out",
            str(defects),
            "--parent-reference-energy-eV",
            "-30",
            "--chemical-potential",
            "O=-5",
        ]
    )
    defect_rows = rows(defects)
    assert defect_rows[0]["defect_id"] == "V_O_seed"
    assert float(defect_rows[0]["formation_energy_eV"]) == -3.0

    solution = tmp_path / "solution.csv"
    solution.write_text(
        "x,G_mix_eV_per_formula\n"
        "0.25,0.03\n"
        "0.5,0.04\n"
        "0.75,0.03\n",
        encoding="utf-8",
    )
    sd_dd_thermo.main(["fit-solution", "--solution-csv", str(solution), "--outdir", str(tmp_path / "solution_fit")])
    fit_rows = rows(tmp_path / "solution_fit" / "solution_model_parameters.csv")
    assert fit_rows[0]["parameter"] == "L0_eV_per_formula"
