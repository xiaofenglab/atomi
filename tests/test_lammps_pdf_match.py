from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from atomi.lammps import pdf_match


def write_xy(path, x, y):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# x y\n")
        for xi, yi in zip(x, y):
            handle.write(f"{xi} {yi}\n")


def make_series(tmp_path):
    series_dir = tmp_path / "pdf_series"
    series_dir.mkdir()
    x = np.linspace(1.0, 6.0, 101)
    exp = np.exp(-0.5 * ((x - 3.0) / 0.35) ** 2)
    model_good = 0.8 * exp + 0.1
    model_bad = np.exp(-0.5 * ((x - 4.0) / 0.45) ** 2)
    items = []
    for temp, y in ((300.0, model_good), (600.0, model_bad)):
        tdir = series_dir / f"T_{int(temp)}K"
        tdir.mkdir()
        g_path = tdir / f"T_{int(temp)}K_GofR_from_FQ.dat"
        write_xy(g_path, x, y)
        items.append(
            {
                "temperature": temp,
                "stage_name": f"npt_prod_{int(temp)}K",
                "GofR_from_FQ": str(g_path),
                "GofR_direct": str(g_path),
                "SofQ": str(g_path),
                "FofQ": str(g_path),
                "FofQ_windowed": str(g_path),
            }
        )
    (series_dir / "series_summary.json").write_text(json.dumps({"series": items}), encoding="utf-8")
    exp_path = tmp_path / "experiment.gr"
    write_xy(exp_path, x, exp)
    return series_dir, exp_path


def common_args(tmp_path, series_dir, exp_path, outdir_name):
    return SimpleNamespace(
        pdf_series=series_dir,
        config=None,
        md_root=None,
        exp=exp_path,
        quantity="auto",
        g_source="from-fq",
        fq_source="raw",
        x_min=1.0,
        x_max=6.0,
        fit_scale=True,
        baseline_order=0,
        outdir=tmp_path / outdir_name,
        config_dir=None,
        config_glob="*.json",
        duplicate_policy="highest_config_order",
        dump_format="lammps-dump-text",
        type_map=[],
        dt=None,
        dump_every=None,
        window_ps=20.0,
        t_min=None,
        t_max=None,
        frame_step=None,
        rmax=12.0,
        dr=0.02,
        qmax=25.0,
        dq=0.05,
        gr_rmax=None,
        gr_dr=None,
        scattering="xray",
        weights=[],
        window_function="lorch",
        write_selected_extxyz=False,
        archive_path=None,
        no_archive_output=True,
    )


def test_compare_ranks_best_pdf_window(tmp_path) -> None:
    series_dir, exp_path = make_series(tmp_path)
    args = common_args(tmp_path, series_dir, exp_path, "compare")
    quantity = pdf_match.validate_args(pdf_match.build_compare_parser(), args)
    summary = pdf_match.write_compare_outputs(args, series_dir, quantity)

    assert summary["best"]["temperature"] == 300.0
    assert (args.outdir / "compare_rank.csv").exists()
    assert (args.outdir / "best_compare_curve.csv").exists()
    assert (args.outdir / "best_compare_overlay.png").exists()


def test_reweight_outputs_weights_and_curve(tmp_path) -> None:
    series_dir, exp_path = make_series(tmp_path)
    args = common_args(tmp_path, series_dir, exp_path, "reweight")
    args.kl_strength = 1e-4
    args.max_iter = 200
    args.learning_rate = 0.2
    quantity = pdf_match.validate_args(pdf_match.build_reweight_parser(), args)
    summary = pdf_match.write_reweight_outputs(args, series_dir, quantity)

    assert summary["n_candidates"] == 2
    assert summary["corrected_reweighted_metrics"]["rmse"] < summary["uniform_metrics"]["rmse"]
    assert (args.outdir / "frame_window_weights.csv").exists()
    assert (args.outdir / "reweighted_curve.csv").exists()
    assert (args.outdir / "reweighted_overlay.png").exists()
