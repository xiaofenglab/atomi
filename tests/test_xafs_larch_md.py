from __future__ import annotations

import json

import numpy as np
from ase import Atoms
from ase.io import write

from atomi.xafs.larch_md import (
    build_compare_parser,
    build_larch_run_parser,
    build_prepare_parser,
    larch_xftf,
    run_compare,
    run_larch,
    run_prepare,
)
from atomi.xafs.status import build_xafs_status


def test_xafs_prepare_writes_metal_absorber_feff_inputs(tmp_path) -> None:
    traj = tmp_path / "uo2.extxyz"
    atoms = Atoms(
        symbols=["U", "O", "O"],
        positions=[[0.0, 0.0, 0.0], [2.35, 0.0, 0.0], [0.0, 2.35, 0.0]],
        cell=[6.0, 6.0, 6.0],
        pbc=True,
    )
    write(traj, [atoms, atoms.copy()], format="extxyz")

    outdir = tmp_path / "xafs_prepare"
    args = build_prepare_parser().parse_args(
        [
            "--traj",
            str(traj),
            "--outdir",
            str(outdir),
            "--cluster-radius",
            "3.0",
            "--max-frames",
            "1",
            "--max-absorber-sites",
            "1",
        ]
    )
    summary = run_prepare(args)

    assert summary["absorber"] == "U"
    assert summary["n_clusters"] == 1
    feff_inp = next((outdir / "clusters").glob("frame_*/site_*/feff.inp"))
    text = feff_inp.read_text(encoding="utf-8")
    assert "EDGE L3" in text
    assert "POTENTIALS" in text
    assert "ATOMS" in text
    assert "U_absorber" in text
    metadata = json.loads((outdir / "xafs_prepare_metadata.json").read_text(encoding="utf-8"))
    assert metadata["roadmap"]["pdf_xafs_joint_analysis"]


def test_xafs_larch_run_collects_existing_chi_without_larch(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    cluster = prepared / "clusters" / "frame_000000" / "site_000001_U"
    cluster.mkdir(parents=True)
    (prepared / "cluster_dirs.txt").write_text(str(cluster) + "\n", encoding="utf-8")
    k = np.arange(2.0, 6.1, 0.5)
    chi = np.sin(k) / k
    with (cluster / "chi.dat").open("w", encoding="utf-8") as handle:
        for kv, cv in zip(k, chi):
            handle.write(f"{kv:.6f} {cv:.8f}\n")

    outdir = tmp_path / "larch_run"
    args = build_larch_run_parser().parse_args(
        [
            "--prepared-dir",
            str(prepared),
            "--outdir",
            str(outdir),
            "--no-run-feff",
            "--no-archive-output",
        ]
    )
    summary = run_larch(args)

    assert summary["n_chi_curves_used"] == 1
    assert (outdir / "ensemble_chi_k.dat").exists()
    assert "larch_transform_status" in summary


def test_xafs_status_reports_missing_larch_without_failure(monkeypatch) -> None:
    monkeypatch.delenv("ATOMI_XAFS_LARCH_PYTHON", raising=False)
    monkeypatch.delenv("ATOMI_XAFS_LARCH_ENV", raising=False)

    status = build_xafs_status()

    assert "active_environment" in status
    assert "external_larch_environment" in status
    assert status["larch_mode"] in {"active-python", "external-python", "missing"}
    assert status["suggestions"]


def test_larch_xftf_reports_external_probe_when_active_larch_missing(monkeypatch, tmp_path) -> None:
    fake_python = tmp_path / "missing-python"
    monkeypatch.setenv("ATOMI_XAFS_LARCH_PYTHON", str(fake_python))

    transform, status = larch_xftf(
        np.linspace(1.0, 4.0, 6),
        np.linspace(0.0, 1.0, 6),
        1.0,
        4.0,
        2,
        1.0,
        "kaiser",
    )

    assert transform is None
    assert "xraylarch import failed" in status or "external Larch probe failed" in status


def test_xafs_md_compare_fits_scale_and_writes_metrics(tmp_path) -> None:
    xafs_dir = tmp_path / "xafs"
    xafs_dir.mkdir()
    k = np.arange(2.0, 10.1, 0.25)
    model = np.sin(k) / k
    exp = 2.0 * model
    with (xafs_dir / "ensemble_chi_k.dat").open("w", encoding="utf-8") as handle:
        for kv, cv in zip(k, model):
            handle.write(f"{kv:.8f} {cv:.10f}\n")
    exp_path = tmp_path / "exp.chik"
    with exp_path.open("w", encoding="utf-8") as handle:
        for kv, cv in zip(k, exp):
            handle.write(f"{kv:.8f} {cv:.10f}\n")

    outdir = tmp_path / "compare"
    args = build_compare_parser().parse_args(
        [
            "--xafs-dir",
            str(xafs_dir),
            "--exp-chi",
            str(exp_path),
            "--outdir",
            str(outdir),
            "--baseline-order",
            "-1",
            "--no-archive-output",
        ]
    )
    summary = run_compare(args)

    assert abs(summary["fit"]["scale"] - 2.0) < 1e-8
    assert summary["metrics"]["rmse"] < 1e-8
    assert (outdir / "xafs_compare_curve.csv").exists()
