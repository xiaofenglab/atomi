from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from atomi.analysis.plot_dataset import validate_bundle
from atomi.qchem import molcas_workflow


SYNTHETIC_OUTPUT = r"""
 Pseudonatural active orbitals and approximate occupation numbers
    1   -0.5000   2.0000      1 U1 5f0 ( 0.800)   9 O2 2pz ( 0.300)
    2   -0.1000   1.0000      2 U1 5f2+ ( 0.700)  10 O2 2px (-0.400)
    3    0.2000   0.0000      3 U1 7s ( 0.750)    11 O2 2py ( 0.250)

  SPIN-FREE ENERGIES:
  (Shifted by EMIN (a.u.) = -100.0)

 SF State       Relative EMIN(au)   Rel lowest level(eV)    D:o, cm**(-1)      Abs_M
    1             0.0000000000        0.0000000000              0.0000         1.0
    2             0.0100000000        0.2721138625           2194.7460         1.0
    3             0.4000000000       10.8845544984          87789.8400         1.0

 Weights of the five most important spin-orbit-free states for each spin-orbit state.

 SO State  Total energy (au)           Spin-free states, spin, and weights
 -------------------------------------------------------------------------------------------------------
    1         0.000000       1 0.5  0.8000    2 0.5  0.2000
    2         0.001000       2 0.5  0.7000    1 0.5  0.3000
    3         0.400000       3 0.5  0.9000    2 0.5  0.1000
 -------------------------------------------------------------------------------------------------------

++ Dipole transition strengths (SO states):
   From To Osc. strength
    1  3  0.0100
    2  3  0.0200

 Happy landing!
"""


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    inp = tmp_path / "synthetic.inp"
    out = tmp_path / "synthetic.out"
    inp.write_text(
        """
&RASSCF
Symmetry = 1
Spin = 2
Nactel = 1 0 0
CIroot = 3 3 1
TDM

&RASSI
SpinOrbit
""",
        encoding="utf-8",
    )
    out.write_text(SYNTHETIC_OUTPUT, encoding="utf-8")
    return inp, out


def test_mocparse_freezes_reusable_tables(tmp_path: Path) -> None:
    inp, out = _write_inputs(tmp_path)
    bundle = tmp_path / "bundle-r1"
    assert molcas_workflow.mocparse_main(
        [str(out), "--inp", str(inp), "--outdir", str(bundle), "--project", "synthetic"]
    ) == 0
    manifest = bundle / "plot_dataset.json"
    report = validate_bundle(manifest)
    assert report["valid"] is True
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["molcas_parse_schema"] == molcas_workflow.MOLCAS_BUNDLE_SCHEMA
    assert payload["metadata"]["table_row_counts"]["transitions"] == 2
    assert payload["metadata"]["table_row_counts"]["so_parent_weights"] == 6
    table_ids = {row["id"] for row in payload["tables"]}
    assert {"transitions", "spin_free_states", "spin_orbit_states", "orbitals", "ao_components"} <= table_ids
    sources = {Path(row["path"]).name: row for row in payload["sources"]}
    assert sources[out.name]["sha256"]
    assert sources[inp.name]["sha256"]


def test_mocxanes_and_mocmo_render_only_from_bundle(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    inp, out = _write_inputs(tmp_path)
    bundle = tmp_path / "bundle-r1"
    molcas_workflow.mocparse_main([str(out), "--inp", str(inp), "--outdir", str(bundle)])
    manifest = bundle / "plot_dataset.json"

    xanes = tmp_path / "xanes"
    assert molcas_workflow.mocxanes_main(
        [
            str(manifest),
            "--element", "U",
            "--edge", "M5:10:12",
            "--initial-states", "1,2",
            "--no-xraydb",
            "--outdir", str(xanes),
        ]
    ) == 0
    xanes_summary = json.loads((xanes / "mocxanes_render.json").read_text(encoding="utf-8"))
    assert xanes_summary["initial_states"] == [1, 2]
    assert xanes_summary["broadening_profile"] == "u-m45-polly-high-resolution"
    assert xanes_summary["edges"][0]["broadening_profile"]["gaussian_fwhm_ev"] == pytest.approx(1.2894947)
    assert xanes_summary["edges"][0]["broadening_profile"]["lorentzian_fwhm_ev"] == pytest.approx(0.3621724)
    assert xanes_summary["edges"][0]["n_transitions"] == 1
    assert (xanes / "molcas_xanes.png").is_file()
    assert (xanes / "molcas_xanes.pdf").is_file()
    assert (xanes / "molcas_xanes.svg").is_file()
    with Path(xanes_summary["edges"][0]["sticks_csv"]).open(newline="", encoding="utf-8") as handle:
        sticks = list(csv.DictReader(handle))
    assert float(sticks[0]["oscillator_strength"]) == pytest.approx(0.015)

    mo = tmp_path / "mo"
    assert molcas_workflow.mocmo_main(
        [
            str(manifest),
            "--state-window", "0:12",
            "--outdir", str(mo),
        ]
    ) == 0
    mo_summary = json.loads((mo / "mocmo_render.json").read_text(encoding="utf-8"))
    assert mo_summary["n_links"] >= 3
    assert "separate spin-free and spin-orbit manifold minima" in mo_summary["state_energy_reference"]
    assert "many-electron spin-free states" in mo_summary["interpretation_guard"]
    assert (mo / "molcas_mo_levels.png").is_file()
    assert (mo / "molcas_spin_free_to_so_states.png").is_file()


def test_renderers_reject_a_generic_non_molcas_bundle(tmp_path: Path) -> None:
    manifest = tmp_path / "plot_dataset.json"
    manifest.write_text(json.dumps({"schema": "atomi.analysis.plot_dataset.v1"}), encoding="utf-8")
    with pytest.raises(ValueError):
        molcas_workflow.mocxanes_main(
            [
                str(manifest),
                "--element", "Ce",
                "--edge", "L3:5700:5750",
                "--initial-states", "1",
                "--outdir", str(tmp_path / "xanes"),
            ]
        )


def test_xanes_normalization_labels_are_explicit() -> None:
    assert molcas_workflow._xanes_ylabel("max") == "Normalized intensity (a.u.)"
    assert molcas_workflow._xanes_ylabel("area") == "Area-normalized intensity (eV$^{-1}$)"
    assert molcas_workflow._xanes_ylabel("none") == "Unnormalized intensity (a.u.)"
