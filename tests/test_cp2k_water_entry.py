from pathlib import Path

from atomi.cp2k.water_entry import main, parse_cp2k_input, read_xyz_trajectory


def write_trajectory(path: Path) -> None:
    frame = (
        "8\n"
        "MD step = {step}\n"
        "Ga 0.0 0.0 0.0\n"
        "Cl 2.8 0.0 0.0\n"
        "Cl -2.2 0.0 0.0\n"
        "O {water_x} 0.0 0.0\n"
        "H {h1_x} 0.0 0.0\n"
        "H {water_x} 0.8 0.0\n"
        "O -2.8 0.0 0.0\n"
        "H -3.6 0.0 0.0\n"
    )
    path.write_text(
        frame.format(step=0, water_x=4.0, h1_x=4.8)
        + frame.format(step=1000, water_x=2.9, h1_x=3.7)
        + frame.format(step=2000, water_x=2.75, h1_x=3.55),
        encoding="utf-8",
    )


def write_input(path: Path) -> None:
    path.write_text(
        "&GLOBAL\n"
        "  PROJECT old_project\n"
        "&END GLOBAL\n"
        "&EXT_RESTART\n"
        "  RESTART_FILE_NAME old.restart\n"
        "&END EXT_RESTART\n"
        "&FORCE_EVAL\n"
        "  &DFT\n"
        "    &SCF\n"
        "      SCF_GUESS RESTART\n"
        "    &END SCF\n"
        "  &END DFT\n"
        "  &SUBSYS\n"
        "    &TOPOLOGY\n"
        "      COORD_FILE_NAME old.xyz\n"
        "    &END TOPOLOGY\n"
        "    &COLVAR\n"
        "      &DISTANCE\n"
        "        ATOMS 1 2\n"
        "      &END DISTANCE\n"
        "    &END COLVAR\n"
        "    &KIND Ga\n"
        "    &END KIND\n"
        "  &END SUBSYS\n"
        "&END FORCE_EVAL\n"
        "&MOTION\n"
        "  &CONSTRAINT\n"
        "    &COLLECTIVE\n"
        "      COLVAR 1\n"
        "      TARGET [angstrom] 2.550\n"
        "      &RESTRAINT\n"
        "        K [kcalmol] 50.0\n"
        "      &END RESTRAINT\n"
        "    &END COLLECTIVE\n"
        "  &END CONSTRAINT\n"
        "  &MD\n"
        "    TIMESTEP 1.0\n"
        "    STEPS 3000\n"
        "  &END MD\n"
        "&END MOTION\n",
        encoding="utf-8",
    )


def test_water_entry_writes_candidate_inputs(tmp_path: Path) -> None:
    traj = tmp_path / "run-pos.xyz"
    inp = tmp_path / "run.inp"
    outdir = tmp_path / "water_entry"
    write_trajectory(traj)
    write_input(inp)

    main(
        [
            str(traj),
            "--inp",
            str(inp),
            "--outdir",
            str(outdir),
            "--last-ps",
            "1.5",
            "--max-candidates",
            "2",
            "--min-frame-gap",
            "1",
            "--cl-target",
            "2.80",
            "--water-target",
            "2.80",
        ]
    )

    frames = read_xyz_trajectory(traj)
    assert len(frames) == 3
    summary = outdir / "water_entry_candidates.csv"
    assert summary.is_file()
    candidate_inputs = sorted(outdir.glob("*_water_entry.inp"))
    assert candidate_inputs
    text = candidate_inputs[0].read_text(encoding="utf-8")
    assert "PROJECT cand_" in text
    assert "COORD_FILE_NAME cand_" in text
    assert "SCF_GUESS ATOMIC" in text
    assert "&EXT_RESTART" not in text
    assert "ATOMS 1 2" in text
    assert "ATOMS 1 4" in text
    assert "TARGET [angstrom] 2.800" in text

    info = parse_cp2k_input(candidate_inputs[0])
    assert info["colvar_atoms"] == [(1, 2), (1, 4)]
