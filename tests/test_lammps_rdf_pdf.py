from __future__ import annotations

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
        no_plots=True,
        write_selected_extxyz=False,
    )
    summary = rdf_pdf.run(args)

    assert summary["outputs"]["pdfgui_GofR"].endswith("uo2_pdfgui_GofR.gr")
    assert (tmp_path / "uo2_partial_rdfs.csv").exists()
    assert (tmp_path / "uo2_SofQ.dat").exists()
    assert (tmp_path / "uo2_FofQ_windowed.dat").exists()
    assert (tmp_path / "uo2_summary.json").exists()
