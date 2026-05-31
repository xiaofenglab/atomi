from __future__ import annotations

from pathlib import Path

from atomi.cp2k.aimd_status import summarize_run
from atomi.cp2k.blue_moon import integrate_windows, summarize_window
from atomi.cp2k.ligand_exchange import summarize_ligand_exchange


def write_cp2k_window(
    root: Path,
    *,
    target_cl_A: float = 2.4,
    target_o_A: float = 3.0,
    cl_tail_A: float = 3.25,
    o_tail_A: float = 2.10,
    finished: bool = True,
) -> None:
    root.mkdir()
    (root / "ga_cl4.inp").write_text(
        "\n".join(
            [
                "&GLOBAL",
                "  PROJECT ga_cl4_test",
                "  RUN_TYPE MD",
                "&END GLOBAL",
                "&FORCE_EVAL",
                "  &DFT",
                "    CHARGE 0",
                "    MULTIPLICITY 1",
                "    BASIS_SET_FILE_NAME BASIS_MOLOPT",
                "    POTENTIAL_FILE_NAME POTENTIAL",
                "    &XC",
                "      &XC_FUNCTIONAL PBE",
                "      &END XC_FUNCTIONAL",
                "      &VDW_POTENTIAL",
                "        &PAIR_POTENTIAL",
                "          TYPE DFTD3",
                "        &END PAIR_POTENTIAL",
                "      &END VDW_POTENTIAL",
                "    &END XC",
                "  &END DFT",
                "  &SUBSYS",
                "    &CELL",
                "      ABC 22 22 22",
                "    &END CELL",
                "    &TOPOLOGY",
                "      COORD_FILE_NAME start.xyz",
                "    &END TOPOLOGY",
                "  &END SUBSYS",
                "&END FORCE_EVAL",
                "&MOTION",
                "  &MD",
                "    ENSEMBLE NVT",
                "    STEPS 4",
                "    TIMESTEP 0.5",
                "    TEMPERATURE 300",
                "  &END MD",
                "  &PRINT",
                "    &TRAJECTORY",
                "      FILENAME ga_cl4-pos.xyz",
                "      &EACH",
                "        MD 1",
                "      &END EACH",
                "    &END TRAJECTORY",
                "  &END PRINT",
                "&END MOTION",
                "&COLVAR",
                "  &DISTANCE",
                "    ATOMS 1 2",
                "  &END DISTANCE",
                "&END COLVAR",
                "&COLVAR",
                "  &DISTANCE",
                "    ATOMS 1 3",
                "  &END DISTANCE",
                "&END COLVAR",
                "&COLLECTIVE",
                "  COLVAR 1",
                "  &RESTRAINT",
                f"    TARGET [angstrom] {target_cl_A}",
                "    K [kcalmol] 50",
                "  &END RESTRAINT",
                "&END COLLECTIVE",
                "&COLLECTIVE",
                "  COLVAR 2",
                "  &RESTRAINT",
                f"    TARGET [angstrom] {target_o_A}",
                "    K [kcalmol] 40",
                "  &END RESTRAINT",
                "&END COLLECTIVE",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "ga_cl4-1.ener").write_text(
        "\n".join(
            [
                "# Step Time Kin Temp Pot Cons CPU",
                "0 0.0 0.0 300.0 -10.00 -9.50 0.0",
                "1 0.5 0.0 301.0 -10.10 -9.55 2.0",
                "4 2.0 0.0 299.5 -10.20 -9.60 2.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log_lines = [
        "MD| Step number 4",
        "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -10.20",
    ]
    if finished:
        log_lines.append("PROGRAM ENDED")
    (root / "ga_cl4.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    frames = [
        (1, 2.45, 3.20),
        (2, target_cl_A, target_o_A),
        (3, cl_tail_A, o_tail_A),
        (4, cl_tail_A + 0.02, o_tail_A - 0.02),
    ]
    xyz_lines: list[str] = []
    for step, cl_x, o_x in frames:
        xyz_lines.extend(
            [
                "5",
                f"i = {step}",
                "Ga 0.000 0.000 0.000",
                f"Cl {cl_x:.3f} 0.000 0.000",
                f"O {o_x:.3f} 0.000 0.000",
                f"H {o_x + 0.96:.3f} 0.000 0.000",
                f"H {o_x:.3f} 0.960 0.000",
            ]
        )
    (root / "ga_cl4-pos.xyz").write_text("\n".join(xyz_lines) + "\n", encoding="utf-8")


def test_cp2k_aimd_status_extracts_progress_and_restraints(tmp_path: Path) -> None:
    run = tmp_path / "win"
    write_cp2k_window(run)

    summary = summarize_run(run)

    assert summary["status"] == "finished"
    assert summary["latest_step"] == 4
    assert summary["target_steps"] == 4
    assert summary["percent_complete"] == 100.0
    assert summary["input"]["ensemble"] == "NVT"
    assert summary["input"]["colvars"][0]["atoms"] == [1, 2]
    assert summary["input"]["restraints"][1]["target_A"] == 3.0
    assert summary["trajectory"]["composition"] == {"Cl": 1, "Ga": 1, "H": 2, "O": 1}

    quick = summarize_run(run, max_trajectory_frames=2)
    assert quick["trajectory"]["frame_count"] == 2
    assert quick["trajectory"]["frame_count_truncated"] is True


def test_ligand_exchange_summary_labels_product_state(tmp_path: Path) -> None:
    run = tmp_path / "win"
    write_cp2k_window(run)

    summary = summarize_ligand_exchange(run / "ga_cl4-pos.xyz", input_file=run / "ga_cl4.inp")

    assert summary["leaving_index"] == 2
    assert summary["entering_index"] == 3
    assert summary["product_state_label"] == "water-bound/chloride-dissociated"
    assert summary["entering_ligand_identity"] == "water-like"
    assert summary["last_coordination_counts"]["O"] == 1


def test_blue_moon_window_integration_uses_cp2k_restraints(tmp_path: Path) -> None:
    w1 = tmp_path / "w1"
    w2 = tmp_path / "w2"
    write_cp2k_window(w1, target_cl_A=2.4, cl_tail_A=2.45)
    write_cp2k_window(w2, target_cl_A=3.2, cl_tail_A=3.25)

    windows = [
        summarize_window(w1, colvar=1, tail_fraction=0.5),
        summarize_window(w2, colvar=1, tail_fraction=0.5),
    ]
    profile = integrate_windows(windows)

    assert [row["target_A"] for row in profile] == [2.4, 3.2]
    assert profile[0]["sample_count"] == 4
    assert profile[1]["pmf_relative_kcal_mol"] >= 0.0
