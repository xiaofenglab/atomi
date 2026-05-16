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


def test_compare_prepares_pdfgetx3_raw_chi_from_md_density(tmp_path) -> None:
    series_dir, _ = make_series(tmp_path)
    tdir = series_dir / "T_300K"
    summary_path = tdir / "T_300K_summary.json"
    summary_path.write_text(
        json.dumps({"avg_counts": {"U": 1, "O": 2}, "avg_volume_A3": 40.0}),
        encoding="utf-8",
    )
    metadata = json.loads((series_dir / "series_summary.json").read_text(encoding="utf-8"))
    metadata["series"][0]["summary_json"] = str(summary_path)
    metadata["series"][0]["avg_volume_A3"] = 40.0
    (series_dir / "series_summary.json").write_text(json.dumps(metadata), encoding="utf-8")
    sample = tmp_path / "sample.chi"
    write_xy(sample, np.linspace(0.5, 5.0, 10), np.ones(10))
    outdir = tmp_path / "raw_compare"

    pdf_match.compare_main(
        [
            "--pdf-series",
            str(series_dir),
            "--exp-raw-sample",
            str(sample),
            "--outdir",
            str(outdir),
            "--quantity",
            "G",
            "--md-temperature",
            "300",
            "--density-source",
            "md",
            "--prepare-pdfgetx3-only",
        ]
    )

    cfg = outdir / "pdfgetx3_exp" / "pdfgetx3.cfg"
    assert cfg.exists()
    text = cfg.read_text(encoding="utf-8")
    assert "composition = U 1 O 2" in text
    assert "density =" in text
    assert "outputtype = iq sq fq gr" in text
    assert (outdir / "pdfgetx3_exp" / "run_pdfgetx3.sh").exists()
    prep = json.loads((outdir / "pdfgetx3_exp" / "pdfgetx3_prep_metadata.json").read_text(encoding="utf-8"))
    assert prep["prep"]["density_source"] == "md"
    assert prep["output_ready"] is False


def test_compare_accepts_pdfgetx_iq_alias_with_warning(tmp_path) -> None:
    series_dir, exp_path = make_series(tmp_path)
    iq_path = tmp_path / "experiment.iq"
    write_xy(iq_path, np.linspace(1.0, 6.0, 101), np.zeros(101))
    args = common_args(tmp_path, series_dir, iq_path, "compare_iq")
    args.quantity = "IQ"
    quantity = pdf_match.validate_args(pdf_match.build_compare_parser(), args)
    summary = pdf_match.write_compare_outputs(args, series_dir, quantity)

    assert summary["quantity"] == "IQ"
    assert "pseudo-I(Q)" in summary["quantity_note"]
    assert (args.outdir / "best_compare_overlay.png").exists()
