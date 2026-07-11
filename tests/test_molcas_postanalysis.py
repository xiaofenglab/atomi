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
    assert "molcas-postanalysis ao-composition" in commands
    assert "molcas-postanalysis orbital-splitting" in commands
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
    assert '"broadening": "voigt"' in text
    assert '"stick_relative_threshold": 0.95' in text



def test_extract_m45_transitions_from_output(tmp_path: Path) -> None:
    out = tmp_path / "molcas.out"
    out.write_text(
        """
 SO State    Total energy (au)    Spin-free states, spin, and weights
       1      0.0000000000       1 0.5 1.0
       2      0.0010000000       2 0.5 1.0
       3    131.4100000000       3 0.5 1.0
       4    131.4110000000       4 0.5 1.0
       5    137.7800000000       5 0.5 1.0
       6    137.7810000000       6 0.5 1.0
------------------------------

++ Dipole transition strengths (SO states):
 From To Osc. strength
 1 3 1.0E-03
 2 3 3.0E-03
 1 4 2.0E-03
 2 4 4.0E-03
 1 5 5.0E-03
 2 5 7.0E-03
 1 6 6.0E-03
 2 6 8.0E-03
""",
        encoding="utf-8",
    )
    outdir = tmp_path / "extract"
    rc = molcas_postanalysis.main(
        [
            "extract-m45-transitions",
            "--molcas-out",
            str(out),
            "--initial-states",
            "1,2",
            "--outdir",
            str(outdir),
            "--prefix",
            "fake",
        ]
    )
    assert rc == 0
    assert (outdir / "fake_m45_all_transitions_for_atomi.csv").exists()
    assert (outdir / "fake_m5_transitions_for_atomi.csv").exists()
    assert (outdir / "fake_m4_transitions_for_atomi.csv").exists()
    summary = (outdir / "fake_m45_extract_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_m45_transition_extract.v1" in summary
    assert '"n_m5": 2' in summary
    assert '"n_m4": 2' in summary



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
    assert '"relative_threshold": 0.95' in summary
    assert '"n_ranked": 1' in summary

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
        writer.writerow({"block": "source", "label": "Ga 1s", "energy_ev": 0.0, "occupation": 1, "character": "core"})
        writer.writerow({"block": "acceptor", "label": "Ga 4px", "energy_ev": 2.0, "occupation": 0, "character": "Ga 4p"})
        writer.writerow({"block": "acceptor", "label": "Ga 4py", "energy_ev": 4.0, "occupation": 0, "character": "Ga 4p"})
    transitions = tmp_path / "transitions.csv"
    with transitions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["state_from", "state_to", "energy_ev", "oscillator_strength", "label", "source_label", "target_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_from": 1,
                "state_to": 2,
                "energy_ev": 10370.1,
                "oscillator_strength": 0.01,
                "label": "SO 1->2",
                "source_label": "Ga 1s",
                "target_label": "Ga 4px",
            }
        )
        writer.writerow(
            {
                "state_from": 1,
                "state_to": 3,
                "energy_ev": 10371.4,
                "oscillator_strength": 0.02,
                "label": "SO 1->3",
                "source_label": "Ga 1s",
                "target_label": "Ga 4py",
            }
        )
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
    summary = (outdir / "molcas_schematic_mo_diagram_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_mo_diagram.v1" in summary
    assert '"source_label": "Ga 1s"' in summary
    assert '"target_label": "Ga 4py"' in summary


def test_orbital_splitting(tmp_path: Path) -> None:
    orbitals = tmp_path / "ga_k_orbitals.csv"
    with orbitals.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["block", "label", "energy_ev", "occupation", "character"])
        writer.writeheader()
        writer.writerow({"block": "core", "label": "Ga 1s", "energy_ev": -390.0, "occupation": 1, "character": "source"})
        writer.writerow({"block": "acceptor", "label": "Ga 4px", "energy_ev": 0.0, "occupation": 0, "character": "Ga 4p/Cl 3p"})
        writer.writerow({"block": "acceptor", "label": "Ga 4py", "energy_ev": 0.4, "occupation": 0, "character": "Ga 4p/Cl 3p"})
        writer.writerow({"block": "acceptor", "label": "Ga 4pz", "energy_ev": 0.8, "occupation": 0, "character": "Ga 4p/Cl 3p"})
    outdir = tmp_path / "split"
    rc = molcas_postanalysis.main(
        [
            "orbital-splitting",
            "--orbitals-csv",
            str(orbitals),
            "--outdir",
            str(outdir),
            "--title",
            "Ga K-edge acceptor splitting",
        ]
    )
    assert rc == 0
    assert (outdir / "molcas_orbital_splitting.png").exists()
    assert (outdir / "molcas_orbital_splitting_levels.csv").exists()
    text = (outdir / "molcas_orbital_splitting_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_orbital_splitting_diagram.v1" in text
    assert '"acceptor": 0.8' in text


def test_ao_composition_from_pseudonatural_block(tmp_path: Path) -> None:
    molcas_out = tmp_path / "molcas.out"
    molcas_out.write_text(
        """
 Pseudonatural active orbitals and approximate occupation numbers

      26    -0.0522      0.3333       7 GA1      4py   (-1.0078)       8 GA1      5py   ( 0.1300)
                                      21 CL2      3py   ( 0.3726)      32 O3       2s   (-0.1118)
      27    -0.0473      0.3333       3 GA1      4px   ( 1.0899)       4 GA1      5px   (-0.1282)
                                      18 CL2      3px   (-0.3731)      32 O3       2s   (-0.7837)
      28    -0.0085      0.3333       3 GA1      4px   ( 1.2671)      32 O3       2s   ( 0.3826)
                                      33 O3       3s    ( 0.1638)      34 O3       2px  ( 0.7577)

 Mulliken Population Analysis
""",
        encoding="utf-8",
    )
    outdir = tmp_path / "ao"
    rc = molcas_postanalysis.main(
        [
            "ao-composition",
            "--molcas-out",
            str(molcas_out),
            "--section-index",
            "0",
            "--section-label",
            "ga_4h2o",
            "--mo-range",
            "26-28",
            "--ao-coeff-cutoff",
            "0.1",
            "--outdir",
            str(outdir),
        ]
    )
    assert rc == 0
    csv_text = (outdir / "molcas_ao_composition.csv").read_text(encoding="utf-8")
    assert "GA1,Ga,4py" in csv_text
    assert "GA1,Ga,4px" in csv_text
    assert "CL2,Cl,3py" in csv_text
    assert (outdir / "molcas_ao_composition.png").exists()
    summary = (outdir / "molcas_ao_composition_summary.json").read_text(encoding="utf-8")
    assert "atomi.molcas_ao_composition.v1" in summary
    assert '"n_mo_rows": 3' in summary



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
