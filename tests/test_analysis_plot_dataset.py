from __future__ import annotations

import json
from pathlib import Path

import pytest

from atomi.analysis.plot_dataset import (
    ColumnSpec,
    PlotDataset,
    SeriesSpec,
    TableSpec,
    ViewSpec,
    freeze_dataset,
    freeze_from_spec,
    resolve_view,
    validate_bundle,
)
from atomi.core.run_record import RunArtifact


def _dataset(table: Path, source: Path) -> PlotDataset:
    return PlotDataset(
        dataset_id="uo8-m5-spectrum",
        revision=1,
        project="UO2 MOLCAS XANES",
        chart_family="xanes",
        scientific_question="What is the calculated U M5 envelope?",
        science_owner="UO2 MOLCAS student",
        canonical_report="/project/UO2_MOLCAS_PROJECT_REPORT.local.md#xanes",
        scientific_status="accepted",
        quality="publication",
        tables=[
            TableSpec(
                id="m5_spectrum",
                path=str(table),
                columns=(
                    ColumnSpec(
                        "energy_ev",
                        "float",
                        "x",
                        quantity="excitation_energy",
                        unit="eV",
                    ),
                    ColumnSpec(
                        "intensity",
                        "float",
                        "y",
                        quantity="normalized_intensity",
                        unit="1",
                    ),
                ),
                row_key=("energy_ev",),
                row_scope="All broadened grid points.",
            )
        ],
        views=[
            ViewSpec(
                id="m5",
                series=(
                    SeriesSpec(
                        id="m5",
                        table="m5_spectrum",
                        x="energy_ev",
                        y="intensity",
                        label="U M5 edge XANES",
                    ),
                ),
                xlabel="Excitation energies (eV)",
                ylabel="Normalized intensity (a.u.)",
            )
        ],
        sources=[RunArtifact(path=str(source), role="raw-molcas-output")],
        transformations=["Voigt broadening, 0.70 eV effective FWHM"],
        normalization="Maximum intensity equals one.",
        uncertainty={"meaning": "none"},
        caveats=["Computed energies are not aligned to experiment."],
    )


def test_plot_dataset_freeze_validate_and_resolve(tmp_path: Path) -> None:
    table = tmp_path / "spectrum.csv"
    table.write_text(
        "energy_ev,intensity\n3579.0,0.2\n3580.0,1.0\n",
        encoding="utf-8",
    )
    source = tmp_path / "molcas.out"
    source.write_text("read-only source\n", encoding="utf-8")

    manifest = freeze_dataset(_dataset(table, source), tmp_path / "bundle")
    report = validate_bundle(manifest)
    resolved = resolve_view(manifest, "m5")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert report["valid"]
    assert payload["schema"] == "atomi.analysis.plot_dataset.v1"
    assert payload["tables"][0]["path"] == "tables/m5_spectrum.csv"
    assert payload["tables"][0]["rows"] == 2
    assert resolved["dataset_id"] == "uo8-m5-spectrum"
    assert resolved["series"][0]["table_path"].endswith(
        "tables/m5_spectrum.csv"
    )
    assert resolved["view"]["xlabel"] == "Excitation energies (eV)"


def test_plot_dataset_detects_table_tampering(tmp_path: Path) -> None:
    table = tmp_path / "spectrum.csv"
    table.write_text("energy_ev,intensity\n3579.0,0.2\n", encoding="utf-8")
    source = tmp_path / "molcas.out"
    source.write_text("source\n", encoding="utf-8")
    manifest = freeze_dataset(_dataset(table, source), tmp_path / "bundle")

    frozen = manifest.parent / "tables" / "m5_spectrum.csv"
    frozen.write_text("energy_ev,intensity\n3579.0,0.9\n", encoding="utf-8")
    report = validate_bundle(manifest)

    assert not report["valid"]
    assert "Table fingerprint mismatch: m5_spectrum" in report["errors"]


def test_plot_dataset_rejects_nonfinite_numeric_data(tmp_path: Path) -> None:
    table = tmp_path / "spectrum.csv"
    table.write_text("energy_ev,intensity\n3579.0,nan\n", encoding="utf-8")
    source = tmp_path / "molcas.out"
    source.write_text("source\n", encoding="utf-8")

    with pytest.raises(ValueError, match="NaN or infinity"):
        freeze_dataset(_dataset(table, source), tmp_path / "bundle")


def test_plot_dataset_rejects_unsafe_table_id(tmp_path: Path) -> None:
    table = tmp_path / "spectrum.csv"
    table.write_text("energy_ev,intensity\n3579.0,0.2\n", encoding="utf-8")
    source = tmp_path / "molcas.out"
    source.write_text("source\n", encoding="utf-8")
    dataset = _dataset(table, source)
    dataset.tables[0] = TableSpec(
        id="../escape",
        path=str(table),
        columns=dataset.tables[0].columns,
    )

    with pytest.raises(ValueError, match="Table ID"):
        freeze_dataset(dataset, tmp_path / "bundle")


def test_plot_dataset_rejects_manifest_table_path_escape(tmp_path: Path) -> None:
    table = tmp_path / "spectrum.csv"
    table.write_text("energy_ev,intensity\n3579.0,0.2\n", encoding="utf-8")
    source = tmp_path / "molcas.out"
    source.write_text("source\n", encoding="utf-8")
    manifest = freeze_dataset(_dataset(table, source), tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tables"][0]["path"] = "../spectrum.csv"
    payload.pop("integrity")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    report = validate_bundle(manifest)

    assert not report["valid"]
    assert any("escapes the bundle" in error for error in report["errors"])


def test_freeze_from_relative_json_spec(tmp_path: Path) -> None:
    (tmp_path / "curve.csv").write_text(
        "temperature_K,entropy\n300,50.0\n600,70.0\n",
        encoding="utf-8",
    )
    (tmp_path / "source.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "report.md").write_text("# Report\n", encoding="utf-8")
    spec = {
        "dataset": {
            "dataset_id": "entropy-v1",
            "revision": 1,
            "project": "KCl",
            "chart_family": "thermochemistry",
            "scientific_question": "What is the accepted entropy curve?",
            "science_owner": "KCl student",
            "canonical_report": "report.md#entropy",
            "scientific_status": "accepted",
            "quality": "production",
        },
        "sources": [{"path": "source.json", "role": "analysis-summary"}],
        "tables": [
            {
                "id": "entropy",
                "path": "curve.csv",
                "columns": [
                    {
                        "name": "temperature_K",
                        "dtype": "float",
                        "role": "x",
                        "quantity": "temperature",
                        "unit": "K",
                    },
                    {
                        "name": "entropy",
                        "dtype": "float",
                        "role": "y",
                        "quantity": "entropy",
                        "unit": "J mol-1 K-1",
                    },
                ],
            }
        ],
        "views": [
            {
                "id": "main",
                "series": [
                    {
                        "id": "entropy",
                        "table": "entropy",
                        "x": "temperature_K",
                        "y": "entropy",
                    }
                ],
            }
        ],
    }
    spec_path = tmp_path / "plot-data-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    manifest = freeze_from_spec(spec_path, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert validate_bundle(manifest)["valid"]
    assert payload["dataset"]["canonical_report"] == (
        f"{(tmp_path / 'report.md').resolve()}#entropy"
    )
    assert payload["sources"][0]["path"] == str(
        (tmp_path / "source.json").resolve()
    )
