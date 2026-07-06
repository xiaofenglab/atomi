import csv
import json
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.sluschi import route_c


def test_route_c_sharp_solid_sconf_near_zero():
    sconf = route_c.zentropy_sconf_from_probabilities({12: 0.995, 11: 0.005}, same_species=True)
    assert sconf < 0.3


def test_route_c_broad_liquid_sconf_positive():
    sconf = route_c.zentropy_sconf_from_probabilities({8: 0.1, 9: 0.2, 10: 0.4, 11: 0.2, 12: 0.1}, same_species=True)
    assert sconf > 5.0


def test_route_c_pair_selection_prefers_same_species():
    dists = [
        route_c.CoordinationDistribution("K", "Cl", {6: 1.0}),
        route_c.CoordinationDistribution("K", "K", {10: 0.5, 11: 0.5}),
        route_c.CoordinationDistribution("Cl", "Cl", {10: 0.5, 11: 0.5}),
    ]
    selected, note, warnings = route_c.select_route_c_pairs(dists, phase="liquid")
    assert {dist.pair for dist in selected} == {"K-K", "Cl-Cl"}
    assert "same" in note
    assert not warnings


def test_route_c_analyze_writes_expected_summary_columns(tmp_path: Path):
    coord = tmp_path / "coord.csv"
    with coord.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "T_K", "central", "neighbor", "cn", "probability"])
        writer.writeheader()
        for cn, p in [(8, 0.1), (9, 0.2), (10, 0.4), (11, 0.2), (12, 0.1)]:
            writer.writerow({"phase": "liquid", "T_K": 1200, "central": "K", "neighbor": "K", "cn": cn, "probability": p})
            writer.writerow({"phase": "liquid", "T_K": 1200, "central": "Cl", "neighbor": "Cl", "cn": cn, "probability": p})
    out = tmp_path / "out"
    atomi_main(
        [
            "sluschi-route-c",
            "analyze",
            "--phase",
            "liquid",
            "--temperature-k",
            "1200",
            "--formula",
            "KCl",
            "--coordination-csv",
            str(coord),
            "--svib-j-mol-atom-k",
            "80",
            "--h-kj-mol-atom",
            "-100",
            "--outdir",
            str(out),
        ]
    )
    rows = list(csv.DictReader((out / "route_c_summary.csv").open()))
    assert rows
    for field in route_c.SUMMARY_FIELDS:
        assert field in rows[0]
    assert float(rows[0]["Sconf_J_mol_atom_K"]) > 5.0
    assert (out / "phase_health_route_c.json").exists()


def test_route_c_parse_sluschi_summary_formula_basis(tmp_path: Path):
    summary = tmp_path / "sluschi_entropy_summary.csv"
    summary.write_text(
        "phase,T_K,atoms_per_formula,Svib_J_mol_formula_K,Sconf_J_mol_formula_K\n"
        "liquid,1200,2,160,14\n",
        encoding="utf-8",
    )
    parsed = route_c.parse_sluschi_mds_outputs(summary, formula="KCl")
    assert parsed["Svib_J_mol_atom_K"] == pytest.approx(80.0)
    assert parsed["Sconf_J_mol_atom_K"] == pytest.approx(7.0)


def test_route_c_kcl_demo_cli(tmp_path: Path):
    out = tmp_path / "demo"
    atomi_main(["sluschi-route-c", "kcl-demo", "--outdir", str(out)])
    manifest = json.loads((out / "kcl_route_c_demo_manifest.json").read_text(encoding="utf-8"))
    assert "coordination_csv" in manifest
    assert (out / "route_c_plan.json").exists()
