import gzip
import json
from pathlib import Path

from atomi.cp2k.clean_run import main, thin_xyz_trajectory


def write_xyz_traj(path: Path, nframes: int = 5) -> None:
    chunks = []
    for i in range(nframes):
        chunks.append(
            "2\n"
            f"frame {i}\n"
            f"Ga {i}.0 0.0 0.0\n"
            f"Cl {i + 1}.0 0.0 0.0\n"
        )
    path.write_text("".join(chunks), encoding="utf-8")


def test_thin_xyz_trajectory_keeps_stride_and_final(tmp_path: Path) -> None:
    src = tmp_path / "run-pos.xyz"
    dst = tmp_path / "run-pos_stride2.xyz"
    write_xyz_traj(src, nframes=6)

    kept = thin_xyz_trajectory(src, dst, stride=2, keep_last=True)

    text = dst.read_text(encoding="utf-8")
    assert kept == 4
    assert "frame 0" in text
    assert "frame 2" in text
    assert "frame 4" in text
    assert "frame 5" in text


def test_clean_run_dry_run_writes_manifest_without_changes(tmp_path: Path) -> None:
    (tmp_path / "run.inp").write_text("&GLOBAL\n&END\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("log\n", encoding="utf-8")
    (tmp_path / "run.restart").write_text("restart\n", encoding="utf-8")
    (tmp_path / "run-RESTART.wfn").write_text("wfn\n", encoding="utf-8")
    write_xyz_traj(tmp_path / "run-pos.xyz", nframes=3)

    main([str(tmp_path)])

    manifest = json.loads((tmp_path / "atomi_clean_manifest.json").read_text(encoding="utf-8"))
    assert manifest["executed"] is False
    assert (tmp_path / "run.log").is_file()
    assert any(action["kind"] == "gzip" for action in manifest["actions"])


def test_clean_run_execute_compresses_and_moves_extras(tmp_path: Path) -> None:
    (tmp_path / "run.inp").write_text("&GLOBAL\n&END\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("log\n", encoding="utf-8")
    (tmp_path / "run.restart").write_text("restart\n", encoding="utf-8")
    (tmp_path / "run-RESTART.wfn").write_text("wfn\n", encoding="utf-8")
    (tmp_path / "run-vel.xyz").write_text("vel\n", encoding="utf-8")
    write_xyz_traj(tmp_path / "run-pos.xyz", nframes=5)

    main(
        [
            str(tmp_path),
            "--execute",
            "--reduce-trajectory-stride",
            "2",
            "--replace-trajectory",
        ]
    )

    assert not (tmp_path / "run.log").exists()
    assert gzip.open(tmp_path / "run.log.gz", "rt").read() == "log\n"
    assert (tmp_path / "run-pos_stride2.xyz").is_file()
    assert (tmp_path / "_atomi_removed" / "run-pos.xyz").is_file()
    assert (tmp_path / "_atomi_removed" / "run-vel.xyz").is_file()
    assert (tmp_path / "run.inp").is_file()
    assert (tmp_path / "run.restart").is_file()
    assert (tmp_path / "run-RESTART.wfn").is_file()

