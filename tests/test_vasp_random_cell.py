from pathlib import Path

from atomi.vasp import random_cell


def test_rand_cell_builds_direct_projection_randomization_args(tmp_path: Path) -> None:
    parser = random_cell.build_parser()
    args = parser.parse_args(
        [
            "--poscar",
            "POSCAR",
            "--incar",
            "INCAR",
            "--outdir",
            str(tmp_path / "rand"),
            "--cation-elements",
            "Gd,U",
            "--anion-elements",
            "O",
            "--oxidation-state",
            "Gd=3,U=5,O=-2",
            "--magmom-oxidation",
            "Gd:7=3,U:1=5",
            "--seed",
            "15",
        ]
    )

    project_args = random_cell.route_project_args(args)

    assert project_args[:4] == ["--element-poscar", "POSCAR", "--structure-poscar", "POSCAR"]
    assert project_args[project_args.index("--cation-elements") + 1] == "Gd,U"
    assert project_args[project_args.index("--species-order") + 1] == "Gd,U,O"
    assert project_args[project_args.index("--randomize-candidates") + 1] == "3"
    assert project_args[project_args.index("--randomize-pool-size") + 1] == "200"
    assert project_args[project_args.index("--randomize-seed") + 1] == "15"
    assert project_args[project_args.index("--randomize-sublattice") + 1] == "cation"
    assert "Gd:7=3,U:1=5" in project_args


def test_rand_cell_submit_atat_uses_generated_sbatch(tmp_path: Path, monkeypatch) -> None:
    atat_dir = tmp_path / "atat_random"
    atat_dir.mkdir()
    submit = atat_dir / "submit_mcsqs.sbatch"
    submit.write_text("#!/bin/bash\n", encoding="utf-8")
    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr(random_cell.subprocess, "run", fake_run)

    random_cell.submit_atat(tmp_path)

    assert calls == [(["sbatch", str(submit)], atat_dir.resolve(), True)]
