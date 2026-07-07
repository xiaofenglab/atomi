from __future__ import annotations

import csv
from pathlib import Path

from atomi.qchem import molcas_postanalysis


def _write_transitions(path: Path, energies: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["state_from", "state_to", "energy_ev", "oscillator_strength"])
        writer.writeheader()
        for idx, energy in enumerate(energies, start=2):
            writer.writerow(
                {
                    "state_from": 1,
                    "state_to": idx,
                    "energy_ev": energy,
                    "oscillator_strength": 0.001 * idx,
                }
            )


def test_workflow_record_names_core_tools() -> None:
    record = molcas_postanalysis.workflow_record()
    assert record["schema"] == "atomi.molcas_postanalysis_workflow.v1"
    commands = "\n".join(item["command"] for item in record["commands"])
    assert "molcas-root-helper audit" in commands
    assert "molcas-xanes-spectrum" in commands
    assert "molcas-postanalysis m45-two-panel" in commands
    assert any("Sarah" in tool or "project report" in tool for tool in record["toolset"])


def test_m45_two_panel_writes_spectra_and_summary(tmp_path: Path) -> None:
    m5 = tmp_path / "m5.csv"
    m4 = tmp_path / "m4.csv"
    _write_transitions(m5, [3579.1, 3579.6, 3580.0])
    _write_transitions(m4, [3749.2, 3750.0, 3751.1])
    outdir = tmp_path / "out"
    rc = molcas_postanalysis.main(
        [
            "m45-two-panel",
            "--m5-transitions-csv",
            str(m5),
            "--m4-transitions-csv",
            str(m4),
            "--no-xraydb",
            "--m5-emin",
            "3576",
            "--m5-emax",
            "3583",
            "--m4-emin",
            "3746",
            "--m4-emax",
            "3754",
            "--outdir",
            str(outdir),
            "--prefix",
            "u_test",
        ]
    )
    assert rc == 0
    assert (outdir / "u_test_m5_xanes.csv").exists()
    assert (outdir / "u_test_m4_xanes.csv").exists()
    assert (outdir / "molcas_u_m45_xanes_2panel_tall_sticks.png").exists()
    text = (outdir / "molcas_u_m45_xanes_2panel_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_m45_two_panel.v1" in text
    assert "As-computed transition energies" in text



def test_rank_transitions_and_orbital_handoff(tmp_path: Path) -> None:
    transitions = tmp_path / "transitions.csv"
    _write_transitions(transitions, [100.0, 101.0, 102.0, 103.0])
    outdir = tmp_path / "ranked"
    rc = molcas_postanalysis.main(
        [
            "rank-transitions",
            "--transitions-csv",
            str(transitions),
            "--top",
            "2",
            "--plot",
            "--outdir",
            str(outdir),
        ]
    )
    assert rc == 0
    assert (outdir / "important_dipole_transitions.csv").exists()
    assert (outdir / "important_dipole_transitions.png").exists()
    summary = (outdir / "important_dipole_transitions_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_important_dipole_transitions.v1" in summary

    molcas_dir = tmp_path / "molcas"
    molcas_dir.mkdir()
    (molcas_dir / "cluster.rasscf.h5").write_text("h5 placeholder", encoding="utf-8")
    (molcas_dir / "cluster.rassi.h5").write_text("h5 placeholder", encoding="utf-8")
    (molcas_dir / "cluster.rasscf.molden").write_text("[Molden Format]\n", encoding="utf-8")
    handoff = tmp_path / "handoff"
    rc = molcas_postanalysis.main(["orbital-handoff", "--molcas-dir", str(molcas_dir), "--outdir", str(handoff)])
    assert rc == 0
    text = (handoff / "MOLCAS_ORBITAL_HANDOFF.md").read_text(encoding="utf-8")
    assert "rasscf_h5_active_orbitals" in text
    assert "rassi_h5_state_interaction" in text



def test_mo_diagram(tmp_path: Path) -> None:
    orbitals = tmp_path / "mo.csv"
    with orbitals.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block", "label", "energy_ev", "occupation", "character"])
        writer.writeheader()
        writer.writerow({"block": "ground", "label": "5f delta", "energy_ev": 0.0, "occupation": 1, "character": "nonbonding"})
        writer.writerow({"block": "core-excited", "label": "5f pi*", "energy_ev": 2.0, "occupation": 1, "character": "antibonding"})
        writer.writerow({"block": "core-excited", "label": "5f sigma*", "energy_ev": 4.0, "occupation": 0, "character": "antibonding"})
    transitions = tmp_path / "transitions.csv"
    _write_transitions(transitions, [3579.1, 3580.4])
    outdir = tmp_path / "mo"
    rc = molcas_postanalysis.main(
        [
            "mo-diagram",
            "--orbitals-csv",
            str(orbitals),
            "--transitions-csv",
            str(transitions),
            "--outdir",
            str(outdir),
        ]
    )
    assert rc == 0
    assert (outdir / "molcas_schematic_mo_diagram_summary.json").exists()
    assert (outdir / "molcas_schematic_mo_diagram.png").exists()
    assert "atomi.molcas_mo_diagram.v1" in (outdir / "molcas_schematic_mo_diagram_summary.json").read_text(encoding="utf-8")



def test_u5f_splitting(tmp_path: Path) -> None:
    xyz = tmp_path / "uo8.xyz"
    xyz.write_text(
        "\n".join(
            [
                "9",
                "test UO8 CN8 cluster",
                "U 0.0 0.0 0.0",
                "O 1.0 1.0 1.0",
                "O -1.0 1.0 1.0",
                "O 1.0 -1.0 1.0",
                "O 1.0 1.0 -1.0",
                "O -1.0 -1.0 1.0",
                "O -1.0 1.0 -1.0",
                "O 1.0 -1.0 -1.0",
                "O -1.0 -1.0 -1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outdir = tmp_path / "split"
    rc = molcas_postanalysis.main(["u5f-splitting", "--structure", str(xyz), "--outdir", str(outdir)])
    assert rc == 0
    assert (outdir / "u5f_so_lf_splitting.png").exists()
    text = (outdir / "u5f_so_lf_splitting_summary.json").read_text(encoding="utf-8")
    assert "atomi.u5f_so_lf_splitting_diagram.v1" in text
    assert "Polly and Bagus" in text
    assert "local-cluster" in text
    assert '"coordination_number": 8' in text
