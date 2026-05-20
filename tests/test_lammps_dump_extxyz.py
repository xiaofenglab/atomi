from __future__ import annotations

import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.lammps.dump_extxyz import main


def write_dump(path: Path) -> None:
    frames = []
    for step, shift in [(0, 0.0), (10, 0.1), (20, 0.2)]:
        frames.append(
            "\n".join(
                [
                    "ITEM: TIMESTEP",
                    str(step),
                    "ITEM: NUMBER OF ATOMS",
                    "2",
                    "ITEM: BOX BOUNDS pp pp pp",
                    "0 5",
                    "0 5",
                    "0 5",
                    "ITEM: ATOMS id type x y z",
                    f"1 1 {0.0 + shift:.3f} 0.000 0.000",
                    f"2 2 {1.0 + shift:.3f} 1.000 1.000",
                ]
            )
        )
    path.write_text("\n".join(frames) + "\n", encoding="utf-8")


def test_lammps2extxyz_writes_last_window_outputs(tmp_path: Path, capsys) -> None:
    dump = tmp_path / "dump.lammpstrj"
    write_dump(dump)
    outprefix = tmp_path / "uo2_1500K"

    main(
        [
            "--dump",
            str(dump),
            "--type-map",
            "1=U",
            "2=O",
            "--dt",
            "0.001",
            "--dump-every",
            "10",
            "--window-ps",
            "0.01",
            "--outprefix",
            str(outprefix),
        ]
    )

    output = capsys.readouterr().out
    assert "Selected frames" in output
    summary = json.loads((tmp_path / "uo2_1500K_summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "atomi.lammps.dump_extxyz.v1"
    assert summary["n_total_frames"] == 3
    assert summary["n_selected_frames"] == 2
    assert Path(summary["outputs"]["multi_frame_extxyz"]).exists()
    extxyz = Path(summary["outputs"]["last_frame_extxyz"]).read_text(encoding="utf-8")
    assert "U" in extxyz
    assert "O" in extxyz


def test_lammps2extxyz_reads_poscar2lammps_type_map_json(tmp_path: Path) -> None:
    dump = tmp_path / "dump.lammpstrj"
    write_dump(dump)
    type_map_json = tmp_path / "structure.data.json"
    type_map_json.write_text(json.dumps({"lammps_type_map": {"U": 1, "O": 2}}), encoding="utf-8")

    atomi_main(
        [
            "lammps2extxyz",
            "--dump",
            str(dump),
            "--type-map-json",
            str(type_map_json),
            "--dt",
            "0.001",
            "--dump-every",
            "10",
            "--outprefix",
            str(tmp_path / "from_atomi"),
        ]
    )

    assert (tmp_path / "from_atomi_lastwindow.extxyz").exists()
