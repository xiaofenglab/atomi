from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomi.qchem import molcas_postanalysis


def test_x2c_so_parent_natural_orbital_plan(tmp_path: Path) -> None:
    jobiph = tmp_path / "run.JobIph_9"
    coord = tmp_path / "cluster.xyz"
    jobiph.write_text("placeholder\n", encoding="utf-8")
    coord.write_text("1\ncluster\nU 0 0 0\n", encoding="utf-8")
    outdir = tmp_path / "plan"

    rc = molcas_postanalysis.main(
        [
            "natural-orbital-plan",
            "--jobiph",
            str(jobiph),
            "--root",
            "166",
            "--coord",
            str(coord),
            "--state-role",
            "so-parent",
            "--spin-free-state",
            "1125",
            "--so-state",
            "1395",
            "--parent-weight-percent",
            "10.97",
            "--orbital",
            "3:19-21",
            "--orbital",
            "4:25-28",
            "--label",
            "m5_so1395_sf1125",
            "--outdir",
            str(outdir),
        ]
    )

    assert rc == 0
    replay = (outdir / "m5_so1395_sf1125.inp").read_text(encoding="utf-8")
    grid = (outdir / "m5_so1395_sf1125_grid.inp").read_text(encoding="utf-8")
    manifest = json.loads(
        (outdir / "m5_so1395_sf1125_natural_orbital_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert "RX2C\nAMFI" in replay
    assert "EJOB" in replay
    assert "NATORB\n1\nTRD1" in replay
    assert "1 1\n166" in replay
    assert f">>COPY {jobiph} JOB001" in replay
    assert "ORBITAL\n7" in grid
    assert "3 19" in grid
    assert "4 28" in grid
    assert "m5_so1395_sf1125.SiOrb.1" in grid
    assert manifest["schema"] == "atomi.molcas_natural_orbital_plan.v1"
    assert manifest["relativity"]["scalar_hamiltonian"] == "X2C"
    assert manifest["spin_free_state"] == 1125
    assert manifest["so_state"] == 1395
    assert "not a unique spin-orbit orbital" in manifest["state_interpretation"]
    assert len(manifest["selected_orbitals"]) == 7


def test_dkh2_plan_requests_property_picture_change(tmp_path: Path) -> None:
    outdir = tmp_path / "dkh"
    rc = molcas_postanalysis.main(
        [
            "natural-orbital-plan",
            "--jobiph",
            str(tmp_path / "run.JobIph_1"),
            "--root",
            "1",
            "--coord",
            str(tmp_path / "cluster.xyz"),
            "--state-role",
            "ground",
            "--relativity",
            "dkh2",
            "--orbital",
            "1:1",
            "--label",
            "ground_dkh2",
            "--outdir",
            str(outdir),
        ]
    )

    assert rc == 0
    replay = (outdir / "ground_dkh2.inp").read_text(encoding="utf-8")
    manifest = json.loads(
        (outdir / "ground_dkh2_natural_orbital_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert "RELATIVISTIC\nR02O02\nAMFI" in replay
    assert manifest["relativity"]["scalar_hamiltonian"] == "DKH2"
    assert manifest["relativity"]["property_picture_change"] == "DKH2 R02O02"


def test_so_parent_requires_state_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --so-state and --spin-free-state"):
        molcas_postanalysis.main(
            [
                "natural-orbital-plan",
                "--jobiph",
                str(tmp_path / "run.JobIph_1"),
                "--root",
                "1",
                "--coord",
                str(tmp_path / "cluster.xyz"),
                "--state-role",
                "so-parent",
                "--orbital",
                "1:1",
                "--label",
                "invalid",
                "--outdir",
                str(tmp_path / "invalid"),
            ]
        )
