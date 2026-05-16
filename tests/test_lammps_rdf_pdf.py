from __future__ import annotations

import tarfile
from types import SimpleNamespace

import numpy as np

from atomi.lammps import rdf_pdf


class FakeCell:
    def __init__(self, length: float):
        self.array = np.eye(3) * length


class FakeAtoms:
    def __init__(self, symbols: list[str], positions: list[tuple[float, float, float]], length: float):
        self._symbols = list(symbols)
        self._positions = np.asarray(positions, dtype=float)
        self.cell = FakeCell(length)
        self._volume = length**3

    def get_positions(self):
        return self._positions.copy()

    def get_chemical_symbols(self):
        return list(self._symbols)

    def get_volume(self):
        return self._volume

    def __len__(self):
        return len(self._symbols)


def test_partial_rdf_and_structure_factor_core_are_finite() -> None:
    frames = [
        FakeAtoms(
            ["U", "U", "O", "O"],
            [(0, 0, 0), (3, 0, 0), (1.5, 1.5, 0), (4.5, 1.5, 0)],
            8.0,
        ),
        FakeAtoms(
            ["U", "U", "O", "O"],
            [(0, 0, 0), (3.1, 0, 0), (1.5, 1.4, 0), (4.5, 1.6, 0)],
            8.0,
        ),
    ]
    species = ["O", "U"]
    r_edges = np.arange(0.0, 6.0, 0.1)
    pair_hist, avg_counts, avg_volume, n_frames = rdf_pdf.compute_partial_histograms(
        frames,
        species,
        r_edges,
    )
    r, partial = rdf_pdf.normalize_partial_rdfs(pair_hist, avg_counts, avg_volume, n_frames, r_edges)
    rho0 = sum(avg_counts.values()) / avg_volume
    weights = {"O": 8.0, "U": 92.0}
    g_total, conc = rdf_pdf.weighted_total_gr_constant(species, partial, avg_counts, weights)
    q_values = np.arange(0.2, 6.0, 0.2)
    sq, sq_conc = rdf_pdf.partials_to_sq_constant(
        species,
        partial,
        avg_counts,
        weights,
        rho0,
        r,
        q_values,
    )
    fq = q_values * (sq - 1.0)
    fq_windowed, window = rdf_pdf.apply_window(q_values, fq, "lorch", qmax=6.0)
    gr = rdf_pdf.fq_to_gr(q_values, fq_windowed, r)

    assert set(pair_hist) == {("O", "O"), ("O", "U"), ("U", "U")}
    assert avg_counts == {"O": 2.0, "U": 2.0}
    assert conc == sq_conc == {"O": 0.5, "U": 0.5}
    assert np.isfinite(g_total).all()
    assert np.isfinite(sq).all()
    assert np.isfinite(fq_windowed).all()
    assert np.isfinite(gr).all()
    assert window[0] < 1.0


def test_trapz_compat_uses_trapezoid_when_available(monkeypatch) -> None:
    called = {}

    def fake_trapezoid(y, x):
        called["used"] = True
        return 42.0

    monkeypatch.setattr(rdf_pdf.np, "trapezoid", fake_trapezoid, raising=False)

    value = rdf_pdf.trapz_compat(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0]))

    assert value == 42.0
    assert called["used"]


def test_run_from_existing_traj_can_use_custom_reader(tmp_path, monkeypatch) -> None:
    frames = [
        FakeAtoms(
            ["U", "U", "O", "O"],
            [(0, 0, 0), (3, 0, 0), (1.5, 1.5, 0), (4.5, 1.5, 0)],
            8.0,
        )
    ]

    monkeypatch.setattr(rdf_pdf, "read_frames_from_traj", lambda *args, **kwargs: frames)
    monkeypatch.setattr(rdf_pdf, "write_selected_frames", lambda *args, **kwargs: {})

    args = SimpleNamespace(
        dump=None,
        traj=tmp_path / "traj.extxyz",
        start=None,
        stop=None,
        step=None,
        outdir=tmp_path,
        prefix="uo2",
        rmax=6.0,
        dr=0.1,
        qmax=6.0,
        dq=0.2,
        gr_rmax=None,
        gr_dr=None,
        scattering="custom",
        weights=["U=92", "O=8"],
        window_function="lorch",
        fitting_exports="auto",
        pdfgui_dr_uncertainty=0.0,
        pdfgui_dgr=1.0,
        frame_overlays=True,
        frame_overlay_step=1,
        frame_overlay_max=0,
        adp=True,
        no_plots=True,
        archive_path=None,
        no_archive_output=False,
        write_selected_extxyz=False,
    )
    summary = rdf_pdf.run(args)

    assert summary["outputs"]["pdfgui_GofR"].endswith("uo2_pdfgui_GofR.gr")
    assert (tmp_path / "uo2_partial_rdfs.csv").exists()
    assert (tmp_path / "uo2_SofQ.dat").exists()
    assert (tmp_path / "uo2_FofQ_windowed.dat").exists()
    assert (tmp_path / "uo2_pdfgui_GofR_direct_4col.gr").exists()
    assert (tmp_path / "uo2_pdfgui_GofR_from_FQ_4col.gr").exists()
    assert (tmp_path / "uo2_rmcprofile_iQ_Sminus1.dat").exists()
    assert (tmp_path / "uo2_rmcprofile_pdfgetx_FQ_QSminus1.dat").exists()
    assert (tmp_path / "uo2_frame_overlay_gtot.csv").exists()
    assert (tmp_path / "uo2_frame_overlay_SofQ.csv").exists()
    assert (tmp_path / "uo2_adp_atoms.csv").exists()
    assert (tmp_path / "uo2_adp_species.csv").exists()
    assert summary["outputs"]["adp"]["species_adp_csv"].endswith("uo2_adp_species.csv")
    assert summary["outputs"]["frame_overlays"]["n_overlay_frames"] == 1
    assert summary["outputs"]["fitting_exports"]["pdfgui_GofR_direct_4col"].endswith(
        "uo2_pdfgui_GofR_direct_4col.gr"
    )
    assert (tmp_path / "uo2_summary.json").exists()
    archive = tmp_path.with_name(f"{tmp_path.name}.tar.gz")
    assert summary["archive"] == str(archive.resolve())
    assert archive.exists()
    with tarfile.open(archive, "r:gz") as handle:
        names = set(handle.getnames())
    assert f"{tmp_path.name}/uo2_summary.json" in names
    assert f"{tmp_path.name}/uo2_pdfgui_GofR.gr" in names
    assert f"{tmp_path.name}/uo2_pdfgui_GofR_direct_4col.gr" in names
    assert f"{tmp_path.name}/uo2_adp_species.csv" in names


def test_series_mode_uses_npt_records_and_writes_overlays(tmp_path, monkeypatch) -> None:
    frames = [
        FakeAtoms(
            ["U", "U", "O", "O"],
            [(0, 0, 0), (3, 0, 0), (1.5, 1.5, 0), (4.5, 1.5, 0)],
            8.0,
        ),
        FakeAtoms(
            ["U", "U", "O", "O"],
            [(0.05, 0, 0), (3.05, 0, 0), (1.5, 1.55, 0), (4.5, 1.55, 0)],
            8.1,
        )
    ]
    records = []
    for temp in (300.0, 600.0):
        chunk = tmp_path / f"stages/npt_prod_{int(temp)}K/chunk_production"
        chunk.mkdir(parents=True)
        log = chunk / f"log.in.npt_prod_{int(temp)}K_production"
        dump = chunk / f"dump.npt_prod_{int(temp)}K_production.lammpstrj"
        log.write_text("log\n", encoding="utf-8")
        dump.write_text("dump\n", encoding="utf-8")
        records.append(
            {
                "temperature": temp,
                "stage": {"name": f"npt_prod_{int(temp)}K", "type": "npt", "temperature": temp},
                "stage_name": f"npt_prod_{int(temp)}K",
                "config_path": None,
                "config_root": tmp_path,
                "config_index": 0,
                "log_path": log,
                "timestep_ps": 0.001,
                "md_root": tmp_path,
            }
        )

    monkeypatch.setattr(rdf_pdf, "discover_npt_records_from_md_root", lambda *args, **kwargs: records)
    monkeypatch.setattr(rdf_pdf, "read_frames_from_dump", lambda *args, **kwargs: (frames, {"window_ps_used": 5.0}))
    monkeypatch.setattr(rdf_pdf, "write_selected_frames", lambda *args, **kwargs: {})

    args = SimpleNamespace(
        config=None,
        md_root=tmp_path,
        config_dir=None,
        config_glob="*.json",
        duplicate_policy="highest_config_order",
        t_min=None,
        t_max=None,
        dump_format="lammps-dump-text",
        type_map=["1=O", "2=U"],
        dt=None,
        dump_every=500,
        window_ps=5.0,
        frame_step=None,
        outdir=tmp_path / "series",
        rmax=6.0,
        dr=0.1,
        qmax=6.0,
        dq=0.2,
        gr_rmax=None,
        gr_dr=None,
        scattering="custom",
        weights=["U=92", "O=8"],
        window_function="lorch",
        fitting_exports="auto",
        pdfgui_dr_uncertainty=0.0,
        pdfgui_dgr=1.0,
        frame_overlays=False,
        frame_overlay_step=1,
        frame_overlay_max=0,
        adp=True,
        no_plots=False,
        archive_path=None,
        no_archive_output=True,
        write_selected_extxyz=False,
    )

    summary = rdf_pdf.run_series(args)

    assert len(summary["series"]) == 2
    assert (args.outdir / "series_index.csv").exists()
    assert (args.outdir / "series_summary.json").exists()
    assert (args.outdir / "series_structure_vs_T.csv").exists()
    assert (args.outdir / "series_adp_Uiso_vs_T.csv").exists()
    assert (args.outdir / "series_volume_vs_T_UQ.png").exists()
    assert (args.outdir / "series_Uiso_U_vs_T_UQ.png").exists()
    assert (args.outdir / "overlay_weighted_gr.png").exists()
    assert (args.outdir / "T_300K" / "T_300K_pdfgui_GofR.gr").exists()
    assert (args.outdir / "T_300K" / "T_300K_pdfgui_GofR_direct_4col.gr").exists()
    assert (args.outdir / "T_300K" / "T_300K_rmcprofile_iQ_Sminus1.dat").exists()
    assert (args.outdir / "T_600K" / "T_600K_rmcprofile_SofQ.sq").exists()


def test_series_mode_writes_sbatch_without_running(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    outdir = tmp_path / "pdf_jobs"

    rdf_pdf.main(
        [
            "--config",
            str(config),
            "--outdir",
            str(outdir),
            "--type-map",
            "1=O",
            "2=U",
            "--window-ps",
            "20",
            "--adp",
            "--write-sbatch",
            "--job-name",
            "uo2_pdf",
            "--time",
            "24:00:00",
            "--cpus",
            "16",
            "--mem",
            "128G",
            "--module",
            "chem/python",
        ]
    )

    run_script = outdir / "run_pdf_lammps_series.sh"
    sbatch_script = outdir / "submit_pdf_lammps_series.sbatch"
    assert run_script.exists()
    assert sbatch_script.exists()
    run_text = run_script.read_text(encoding="utf-8")
    assert "python -m atomi.cli.main pdf_lammps_series" in run_text
    assert "--window-ps 20" in run_text
    assert "--adp" in run_text
    assert "--write-sbatch" not in run_text
    assert "module load chem/python" in run_text
    sbatch_text = sbatch_script.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=uo2_pdf" in sbatch_text
    assert "#SBATCH --time=24:00:00" in sbatch_text
    assert "#SBATCH --cpus-per-task=16" in sbatch_text
    assert "#SBATCH --mem=128G" in sbatch_text
    assert (outdir / "series_sbatch_summary.json").exists()
