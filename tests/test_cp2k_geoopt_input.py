from pathlib import Path

from atomi.cp2k.geoopt_input import (
    extract_step_start_val,
    main,
    resolve_max_iter,
)


def test_extract_step_start_val_uses_last_value(tmp_path: Path) -> None:
    restart = tmp_path / "run.restart"
    restart.write_text(
        "&MOTION\n"
        "  STEP_START_VAL 120\n"
        "  STEP_START_VAL 135\n"
        "&END MOTION\n",
        encoding="utf-8",
    )

    assert extract_step_start_val(restart) == 135


def test_restart_max_iter_adds_requested_steps(tmp_path: Path) -> None:
    restart = tmp_path / "run.restart"
    restart.write_text("  STEP_START_VAL 50\n", encoding="utf-8")

    assert resolve_max_iter("restart", 150, restart) == (200, 50)
    assert resolve_max_iter("start", 150, restart) == (150, 0)


def test_geoopt_input_can_inline_restraints(tmp_path: Path) -> None:
    xyz = tmp_path / "ga_cl4.xyz"
    xyz.write_text(
        "5\n"
        "seed\n"
        "Ga 0 0 0\n"
        "Cl 2.2 0 0\n"
        "Cl -2.2 0 0\n"
        "Cl 0 2.2 0\n"
        "Cl 0 -2.2 0\n",
        encoding="utf-8",
    )
    colvar = tmp_path / "colvar.inc"
    colvar.write_text(
        "    &COLVAR\n"
        "      &DISTANCE\n"
        "        ATOMS 1 2\n"
        "      &END DISTANCE\n"
        "    &END COLVAR\n",
        encoding="utf-8",
    )
    constraint = tmp_path / "constraint.inc"
    constraint.write_text(
        "  &CONSTRAINT\n"
        "    CONSTRAINT_INIT T\n"
        "  &END CONSTRAINT\n",
        encoding="utf-8",
    )
    out = tmp_path / "refine.inp"

    main(
        [
            "--xyz",
            str(xyz),
            "--stage",
            "refine",
            "--mode",
            "start",
            "--charge",
            "0",
            "--box",
            "26",
            "--project",
            "ga_cl4_refine",
            "--colvar-file",
            str(colvar),
            "--constraint-file",
            str(constraint),
            "--out",
            str(out),
        ]
    )

    text = out.read_text(encoding="utf-8")
    assert "PROJECT ga_cl4_refine" in text
    assert "CUTOFF 350" in text
    assert "EXTRAPOLATION ASPC" in text
    assert "SCF_GUESS ATOMIC" in text
    assert "ATOMS 1 2" in text
    assert "&CONSTRAINT" in text

