from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from atomi.qchem import molcas_bagus_covalency


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_general_example(tmp_path: Path) -> Path:
    orbitals = tmp_path / "orbitals.csv"
    _write_csv(
        orbitals,
        [
            "state",
            "orbital",
            "occupation",
            "energy",
            "radius",
            "primary",
            "secondary",
            "ligand",
            "lz",
            "sz",
            "jz",
        ],
        [
            {
                "state": "ground raw",
                "orbital": "a",
                "occupation": 1.0,
                "energy": -1.0,
                "radius": 0.9,
                "primary": 0.90,
                "secondary": 0.05,
                "ligand": 0.05,
                "lz": 1.0,
                "sz": 0.5,
                "jz": 1.5,
            },
            {
                "state": "ground raw",
                "orbital": "b",
                "occupation": 1.0,
                "energy": -0.6,
                "radius": 1.1,
                "primary": 0.90,
                "secondary": 0.05,
                "ligand": 0.05,
                "lz": -1.0,
                "sz": 0.5,
                "jz": -0.5,
            },
            {
                "state": "excited raw",
                "orbital": "a",
                "occupation": 1.0,
                "energy": -0.8,
                "radius": 3.0,
                "primary": 0.20,
                "secondary": 0.10,
                "ligand": 0.70,
                "lz": "",
                "sz": "",
                "jz": "",
            },
            {
                "state": "excited raw",
                "orbital": "b",
                "occupation": 1.0,
                "energy": -0.2,
                "radius": 3.2,
                "primary": 0.20,
                "secondary": 0.10,
                "ligand": 0.70,
                "lz": "",
                "sz": "",
                "jz": "",
            },
        ],
    )
    configurations = tmp_path / "configurations.csv"
    _write_csv(
        configurations,
        [
            "state",
            "class",
            "weight_percent",
            "core_holes",
            "ligand_holes",
            "primary_electrons",
            "secondary_electrons",
        ],
        [
            {
                "state": "gs",
                "class": "nominal",
                "weight_percent": 70,
                "core_holes": 0,
                "ligand_holes": 0,
                "primary_electrons": 2,
                "secondary_electrons": 0,
            },
            {
                "state": "gs",
                "class": "ligand_hole",
                "weight_percent": 30,
                "core_holes": 0,
                "ligand_holes": 1,
                "primary_electrons": 3,
                "secondary_electrons": 0,
            },
            {
                "state": "es",
                "class": "nominal",
                "weight_percent": 50,
                "core_holes": 1,
                "ligand_holes": 0,
                "primary_electrons": 2,
                "secondary_electrons": 0,
            },
            {
                "state": "es",
                "class": "ligand_hole_4d",
                "weight_percent": 50,
                "core_holes": 1,
                "ligand_holes": 1,
                "primary_electrons": 2,
                "secondary_electrons": 1,
            },
        ],
    )
    transitions = tmp_path / "transitions.csv"
    _write_csv(
        transitions,
        [
            "final_state",
            "edge",
            "energy",
            "strength",
            "class",
            "assignment_status",
        ],
        [
            {
                "final_state": "es",
                "edge": "L3",
                "energy": 6200.0,
                "strength": 1.0,
                "class": "main_line",
                "assignment_status": "accepted",
            },
            {
                "final_state": "es",
                "edge": "L3",
                "energy": 6204.0,
                "strength": 0.2,
                "class": "lmct",
                "assignment_status": "accepted",
            },
        ],
    )
    spec = {
        "schema": molcas_bagus_covalency.SCHEMA_SPEC,
        "system": {
            "label": "generic_4d_ligand_cluster",
            "central_element": "M",
            "central_atom_index": 1,
        },
        "method": {
            "program": "OpenMolcas",
            "basis": "matched synthetic basis",
            "relativity": "state-matched",
            "wavefunction": "state-resolved synthetic CI",
        },
        "shells": {
            "primary": "4d",
            "secondary": "5s",
            "core": "2p3/2",
            "ligand": "ligand p",
        },
        "reference": {
            "label": "matched isolated ion",
            "radial_extent_angstrom": {"4d": 1.0, "5s": 1.5},
            "basis_policy": "same basis and projection grid",
        },
        "states": [
            {
                "id": "gs",
                "label": "Ground",
                "role": "ground",
                "edge": "",
                "scientific_status": "accepted",
            },
            {
                "id": "es",
                "label": "L3 parent",
                "role": "excited",
                "edge": "L3",
                "scientific_status": "accepted",
            },
        ],
        "inputs": {
            "orbitals": {
                "id": "matched_orbitals",
                "path": orbitals.name,
                "scope": "active",
                "quality": "production",
                "identity_guard": True,
                "columns": {
                    "state_id": "state",
                    "orbital_id": "orbital",
                    "occupation": "occupation",
                    "orbital_energy_ev": "energy",
                    "radial_extent_angstrom": "radius",
                    "primary_projection": "primary",
                    "secondary_projection": "secondary",
                    "ligand_projection": "ligand",
                    "orbital_lz_hbar": "lz",
                    "spin_sz_hbar": "sz",
                    "total_jz_hbar": "jz",
                },
                "value_maps": {
                    "state_id": {
                        "ground raw": "gs",
                        "excited raw": "es",
                    }
                },
            },
            "configurations": {
                "id": "state_ci",
                "path": configurations.name,
                "quality": "production",
                "weight_scale": 0.01,
                "filters": {"state_id": ["gs", "es"]},
                "columns": {
                    "state_id": "state",
                    "configuration_class": "class",
                    "weight": "weight_percent",
                    "core_holes": "core_holes",
                    "ligand_holes": "ligand_holes",
                    "primary_electrons": "primary_electrons",
                    "secondary_electrons": "secondary_electrons",
                },
            },
            "transitions": {
                "id": "l3_sticks",
                "path": transitions.name,
                "quality": "production",
                "assignment_basis": "state_resolved_configuration",
                "satellite_values": ["lmct"],
                "columns": {
                    "state_id": "final_state",
                    "edge": "edge",
                    "energy_ev": "energy",
                    "oscillator_strength": "strength",
                    "satellite_class": "class",
                    "assignment_status": "assignment_status",
                },
            },
        },
        "spectral_sectors": [
            {
                "edge": "L3",
                "label": "high_energy_sideband",
                "kind": "descriptor",
                "emin_ev": 6203.0,
                "emax_ev": 6205.0,
            }
        ],
        "thresholds": dict(molcas_bagus_covalency.DEFAULT_THRESHOLDS),
        "required_measures": dict(
            molcas_bagus_covalency.DEFAULT_REQUIRED_MEASURES
        ),
        "sources": [],
        "caveats": ["Synthetic regression fixture."],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def test_template_is_element_neutral_and_guarded() -> None:
    spec = molcas_bagus_covalency.template_spec()

    assert spec["system"]["central_element"] == "M"
    assert "U" not in json.dumps(spec)
    assert spec["inputs"]["configurations"][0]["weight_scale"] == 1.0
    assert spec["inputs"]["transitions"][0]["satellite_values"]
    assert any("universal percent-covalent" in item for item in spec["caveats"])


def test_investigation_writes_canonical_tables_and_rejects_identity_collapse(
    tmp_path: Path,
) -> None:
    spec_path = _write_general_example(tmp_path)
    outdir = tmp_path / "analysis"

    summary = molcas_bagus_covalency.investigate(
        spec_path,
        outdir,
        make_plot=False,
    )

    assert summary["overall_scientific_status"] == "rejected"
    assert Path(summary["outputs"]["canonical_orbitals_csv"]).is_file()
    assert Path(summary["outputs"]["canonical_configurations_csv"]).is_file()
    assert Path(summary["outputs"]["canonical_transitions_csv"]).is_file()
    assert Path(summary["outputs"]["provenance_json"]).is_file()
    assert "universal percent-covalent" in summary["interpretation_guard"]

    orbital_rows = summary["results"]["orbital_metrics"]
    ground = next(row for row in orbital_rows if row["state_id"] == "gs")
    excited = next(row for row in orbital_rows if row["state_id"] == "es")
    assert ground["scientific_status"] == "accepted"
    assert ground["primary_identity_fraction"] == pytest.approx(0.90)
    assert ground["radial_extent_ratio"] == pytest.approx(1.0)
    assert ground["orbital_lz_expectation_hbar"] == pytest.approx(0.0)
    assert excited["scientific_status"] == "rejected"
    assert excited["primary_identity_fraction"] == pytest.approx(0.20)
    assert excited["radial_extent_ratio"] == pytest.approx(3.1)
    assert {guard["status"] for guard in excited["guards"]} == {"rejected", "accepted"}

    configurations = summary["results"]["configuration_metrics"]
    ground_configuration = next(
        row for row in configurations if row["state_id"] == "gs"
    )
    assert ground_configuration["raw_weight_norm"] == pytest.approx(1.0)
    assert ground_configuration["ligand_hole_weight"] == pytest.approx(0.30)

    satellite = next(
        row
        for row in summary["results"]["spectral_metrics"]
        if row["metric"] == "satellite_intensity"
    )
    assert satellite["scientific_status"] == "accepted"
    assert satellite["fraction_of_edge_intensity"] == pytest.approx(1.0 / 6.0)

    persisted = json.loads(
        (outdir / "bagus_covalency_summary.json").read_text(encoding="utf-8")
    )
    assert persisted["outputs"]["provenance_json"].endswith(
        "bagus_covalency_provenance.json"
    )
    assert persisted["results"]["orbital_metrics"][1]["guards"]


def test_cli_writes_template_and_integrated_subcommand_is_available(
    tmp_path: Path,
) -> None:
    template = tmp_path / "bagus.json"

    assert (
        molcas_bagus_covalency.main(["--write-template", str(template)]) == 0
    )
    assert template.is_file()

    from atomi.qchem import molcas_postanalysis

    parser = molcas_postanalysis.build_parser()
    args = parser.parse_args(
        ["bagus-covalency", "--write-template", str(tmp_path / "nested.json")]
    )
    assert args.func is molcas_bagus_covalency.run_from_args
