import csv
import json
from pathlib import Path

import pytest

from atomi.calphad.mivm import main as mivm_main
from atomi.cli.main import main as atomi_main
from atomi.thermo_prior import (
    line_compound_spec_from_prior,
    read_prior,
    solve_pseudobinary_coefficients,
    write_line_compound_prior,
)


def test_pseudobinary_coefficients_for_na3u5cl18():
    coeffs = solve_pseudobinary_coefficients("Na3U5Cl18", "NaCl", "UCl3")

    assert coeffs["coeff_a"] == pytest.approx(3.0)
    assert coeffs["coeff_b"] == pytest.approx(5.0)
    assert coeffs["x_B"] == pytest.approx(0.625)


def test_thermo_prior_cli_writes_line_compound_prior(tmp_path: Path):
    out = tmp_path / "Na3U5Cl18.prior.json"

    atomi_main(
        [
            "thermo-prior",
            "line-compound",
            "--formula",
            "Na3U5Cl18",
            "--component-a",
            "NaCl",
            "--component-b",
            "UCl3",
            "--gform-kj-mol",
            "-10",
            "--dcp-form",
            "1.5",
            "--tref-k",
            "793",
            "--out",
            str(out),
        ]
    )

    prior = read_prior(out)
    assert prior["schema"] == "atomi.thermo_prior.v1"
    assert prior["pseudo_binary"]["x_B"] == pytest.approx(0.625)
    assert prior["thermo"]["gform_ref_kJ_mol"] == pytest.approx(-10.0)
    compound = line_compound_spec_from_prior(prior, default_tref_k=793)
    assert compound["label"] == "Na3U5Cl18"
    assert compound["dCp_form_J_mol_K"] == pytest.approx(1.5)


def test_write_line_compound_prior_from_elemental_basis(tmp_path: Path):
    out = tmp_path / "compound.prior.json"
    prior = write_line_compound_prior(
        out=out,
        formula="A3B5",
        component_a="A",
        component_b="B",
        gform_ref_kj_mol=-12.5,
        tref_k=800,
    )

    assert prior["calphad_mivm"]["line_compound_spec"].startswith("A3B5:0.625:-12.5")
    assert out.exists()


def test_benchmark_uq_phase_accepts_line_compound_prior(tmp_path: Path):
    curves = tmp_path / "curves.csv"
    curves.write_text("x_B,hmix\n0.1,-1\n0.3,-3\n0.5,-2\n0.7,-1\n0.9,0\n", encoding="utf-8")
    prior = tmp_path / "A3B5.prior.json"
    write_line_compound_prior(
        out=prior,
        formula="A3B5",
        component_a="A",
        component_b="B",
        gform_ref_kj_mol=-10,
        tref_k=800,
    )
    outdir = tmp_path / "bench_prior"

    metadata = mivm_main(
        [
            "benchmark-uq-phase",
            "--curve-csv",
            str(curves),
            "--x-column",
            "x_B",
            "--curve-columns",
            "hmix",
            "--component-a",
            "A",
            "--component-b",
            "B",
            "--x-component",
            "B",
            "--tm-a",
            "1000",
            "--tm-b",
            "1100",
            "--dhfus-a",
            "20",
            "--dhfus-b",
            "22",
            "--line-compound-prior",
            str(prior),
            "--eutectic-x",
            "0.35",
            "--eutectic-t",
            "800",
            "--outdir",
            str(outdir),
        ]
    )

    assert metadata is not None
    assert metadata["line_compounds"][0]["label"] == "A3B5"
    assert metadata["line_compound_prior_paths"] == [str(prior.resolve())]
    rows = list(csv.DictReader((outdir / "candidate_phase_diagrams.csv").open(encoding="utf-8")))
    assert "line_compound_A3B5_K" in rows[0]


def test_cp_placeholder_cli_writes_prior(tmp_path: Path):
    out = tmp_path / "UCl3.cp.prior.json"

    atomi_main(
        [
            "thermo-prior",
            "cp-solid",
            "--formula",
            "UCl3",
            "--cp-j-mol-k",
            "130",
            "--temperature-min-k",
            "300",
            "--temperature-max-k",
            "1200",
            "--out",
            str(out),
        ]
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "cp_solid"
    assert data["thermo"]["Cp_J_mol_K"] == pytest.approx(130)
