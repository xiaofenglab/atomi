from pathlib import Path

from atomi.cp2k.bond_analysis import analyze_one_file, main, parse_cp2k_input


def write_tiny_traj(path: Path) -> None:
    frame0 = (
        "5\n"
        "MD step = 0\n"
        "Ga 0 0 0\n"
        "Cl 2.20 0 0\n"
        "Cl -2.20 0 0\n"
        "Cl 0 2.20 0\n"
        "Cl 0 -2.20 0\n"
    )
    frame1 = (
        "5\n"
        "MD step = 10\n"
        "Ga 0 0 0\n"
        "Cl 2.30 0 0\n"
        "Cl -2.25 0 0\n"
        "Cl 0 2.35 0\n"
        "Cl 0 -2.20 0\n"
    )
    path.write_text(frame0 + frame1, encoding="utf-8")


def test_analyze_one_file_tracks_initial_ligands(tmp_path: Path) -> None:
    traj = tmp_path / "traj.xyz"
    write_tiny_traj(traj)

    summary = analyze_one_file(
        traj,
        metal_index=1,
        ligand_elements={"Cl"},
        n_nearest=4,
        tail_fraction=0.5,
        timestep_fs=0.25,
    )

    assert summary["metal_symbol"] == "Ga"
    assert summary["tracked_indices_1based"] == [2, 3, 4, 5]
    assert summary["tail_nframes"] == 1
    assert summary["total_time_ps"] == 0.0025
    assert round(summary["shell_mean_tail"], 4) == 2.275


def test_parse_cp2k_input_reads_md_and_restraint_metadata(tmp_path: Path) -> None:
    inp = tmp_path / "run.inp"
    inp.write_text(
        "&GLOBAL\n"
        "  PROJECT ga_cl4\n"
        "&END GLOBAL\n"
        "&FORCE_EVAL\n"
        "  &SUBSYS\n"
        "    &COLVAR\n"
        "      &DISTANCE\n"
        "        ATOMS 1 2\n"
        "      &END DISTANCE\n"
        "    &END COLVAR\n"
        "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
        "&MOTION\n"
        "  &MD\n"
        "    STEPS 100\n"
        "    TIMESTEP 0.25\n"
        "    TEMPERATURE 300\n"
        "  &END MD\n"
        "  &CONSTRAINT\n"
        "    &COLLECTIVE\n"
        "      TARGET [angstrom] 2.2\n"
        "      &RESTRAINT\n"
        "        K [kcalmol] 5.0\n"
        "      &END RESTRAINT\n"
        "    &END COLLECTIVE\n"
        "  &END CONSTRAINT\n"
        "&END MOTION\n",
        encoding="utf-8",
    )

    info = parse_cp2k_input(inp)

    assert info["project"] == "ga_cl4"
    assert info["timestep_fs"] == 0.25
    assert info["md_steps"] == 100
    assert info["temperature"] == 300
    assert info["colvar_atoms"] == [(1, 2)]
    assert info["target_angstrom"] == 2.2
    assert info["k_kcalmol"] == 5.0


def test_main_can_write_summary_csv_without_plotting(tmp_path: Path) -> None:
    traj = tmp_path / "traj.xyz"
    csv = tmp_path / "summary.csv"
    write_tiny_traj(traj)

    main(
        [
            str(traj),
            "--metal-index",
            "1",
            "--ligand-elements",
            "Cl",
            "--n-nearest",
            "4",
            "--tail-fraction",
            "0.5",
            "--no-plot",
            "--summary-csv",
            str(csv),
        ]
    )

    text = csv.read_text(encoding="utf-8")
    assert "tracked_mean_tail" in text
    assert "traj.xyz" in text

