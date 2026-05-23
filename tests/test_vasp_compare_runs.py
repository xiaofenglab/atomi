from pathlib import Path

from atomi.vasp.compare_runs import main


def write_poscar(path: Path, *, scale_x: float = 1.0, shift_second: float = 0.0) -> None:
    path.write_text(
        "\n".join(
            [
                "compare",
                "1.0",
                f"{scale_x:.8f} 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "U O",
                "1 1",
                "Direct",
                "0.0 0.0 0.0",
                f"{0.5 + shift_second:.8f} 0.5 0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_outcar(path: Path, energy: float, moments: list[float]) -> None:
    rows = []
    for idx, moment in enumerate(moments, start=1):
        rows.append(f" {idx:4d} 0.000 0.000 0.000 0.000 {moment:.6f}")
    path.write_text(
        " NIONS =      2 ions\n"
        f" free  energy   TOTEN  =       {energy:.6f} eV\n"
        " magnetization (x)\n"
        " # of ion       s       p       d       f       tot\n"
        + "\n".join(rows)
        + "\n tot\n\n",
        encoding="utf-8",
    )


def test_vasp_compare_runs_prints_side_by_side_diagnostics(tmp_path: Path, capsys) -> None:
    run_a = tmp_path / "spin_a"
    run_b = tmp_path / "spin_b"
    run_a.mkdir()
    run_b.mkdir()
    write_poscar(run_a / "CONTCAR")
    write_poscar(run_b / "CONTCAR", scale_x=1.02, shift_second=0.02)
    write_outcar(run_a / "OUTCAR", -10.0, [2.0, 0.0])
    write_outcar(run_b / "OUTCAR", -9.9, [-2.0, 0.0])

    main([str(run_a), str(run_b), "--label-a", "script", "--label-b", "atat", "--top-atoms", "1"])

    out = capsys.readouterr().out
    assert "VASP Run Comparison" in out
    assert "script" in out
    assert "atat" in out
    assert "Energy" in out
    assert "Unit Cell" in out
    assert "Spin / Magnetization" in out
    assert "Atom-By-Atom Structural Difference" in out
    assert "Energy difference B-A: +0.10000000 eV" in out
    assert "Total moment difference B-A: -4.000000 mu_B" in out
    assert "Max displacement" in out
