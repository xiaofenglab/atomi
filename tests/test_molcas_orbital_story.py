from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from atomi.qchem import molcas_postanalysis


def _image(path: Path, phase: float = 0.0) -> None:
    x, y = np.mgrid[-1:1:40j, -1:1:40j]
    field = np.sin(3 * x + phase) * np.cos(3 * y - phase)
    plt.imsave(path, field, cmap="coolwarm")


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_orbital_atlas_writes_frozen_plot_packet(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _image(image_a)
    _image(image_b, phase=0.8)
    manifest = tmp_path / "atlas.csv"
    columns = [
        "label",
        "image_path",
        "family",
        "family_order",
        "order",
        "symmetry",
        "occupation",
        "status",
    ]
    _write_csv(
        manifest,
        columns,
        [
            {
                "label": "Au 19",
                "image_path": image_a.name,
                "family": "Au",
                "family_order": 1,
                "order": 1,
                "symmetry": "Au",
                "occupation": 0.55,
                "status": "accepted",
            },
            {
                "label": "Bu 25",
                "image_path": image_b.name,
                "family": "Bu",
                "family_order": 2,
                "order": 1,
                "symmetry": "Bu",
                "occupation": 0.45,
                "status": "provisional",
            },
        ],
    )
    outdir = tmp_path / "atlas"
    rc = molcas_postanalysis.main(
        [
            "orbital-atlas",
            "--manifest-csv",
            str(manifest),
            "--outdir",
            str(outdir),
            "--figure-stem",
            "test_atlas",
            "--columns",
            "2",
            "--dpi",
            "80",
        ]
    )
    assert rc == 0
    assert (outdir / "test_atlas.png").exists()
    assert (outdir / "test_atlas.pdf").exists()
    record = json.loads((outdir / "test_atlas_plot_dataset.json").read_text(encoding="utf-8"))
    assert record["schema"] == "atomi.molcas_orbital_atlas.v1"
    assert record["group_order"] == ["Au", "Bu"]
    assert record["n_orbitals"] == 2
    assert len(record["source_images"]) == 2


def test_orbital_pair_requires_nto_contribution_metadata(tmp_path: Path) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _image(image_a)
    _image(image_b)
    manifest = tmp_path / "pairs.csv"
    columns = [
        "pair_label",
        "left_image_path",
        "right_image_path",
        "left_label",
        "right_label",
    ]
    _write_csv(
        manifest,
        columns,
        [
            {
                "pair_label": "pair 1",
                "left_image_path": image_a.name,
                "right_image_path": image_b.name,
                "left_label": "particle",
                "right_label": "hole",
            }
        ],
    )
    with pytest.raises(ValueError, match="NTO pairs require"):
        molcas_postanalysis.main(
            [
                "orbital-pair",
                "--manifest-csv",
                str(manifest),
                "--plot-kind",
                "nto",
            ]
        )


def test_orbital_network_reserves_entanglement_for_dmrg(tmp_path: Path) -> None:
    nodes = tmp_path / "nodes.csv"
    edges = tmp_path / "edges.csv"
    _write_csv(
        nodes,
        ["node_id", "label", "family", "family_order", "order", "status"],
        [
            {
                "node_id": "a",
                "label": "Au19",
                "family": "Au orbitals",
                "family_order": 1,
                "order": 1,
                "status": "accepted",
            },
            {
                "node_id": "f",
                "label": "U 5f",
                "family": "AO families",
                "family_order": 2,
                "order": 1,
                "status": "accepted",
            },
        ],
    )
    _write_csv(
        edges,
        ["source", "target", "value"],
        [{"source": "a", "target": "f", "value": 0.91}],
    )
    with pytest.raises(ValueError, match="requires a DMRG/QCMaquis"):
        molcas_postanalysis.main(
            [
                "orbital-network",
                "--nodes-csv",
                str(nodes),
                "--edges-csv",
                str(edges),
                "--metric",
                "mutual_information",
                "--method",
                "RASSCF AO coefficients",
            ]
        )
    outdir = tmp_path / "network"
    rc = molcas_postanalysis.main(
        [
            "orbital-network",
            "--nodes-csv",
            str(nodes),
            "--edges-csv",
            str(edges),
            "--metric",
            "ao_composition",
            "--title",
            "AO composition network",
            "--outdir",
            str(outdir),
            "--figure-stem",
            "test_network",
            "--dpi",
            "80",
        ]
    )
    assert rc == 0
    record = json.loads((outdir / "test_network_plot_dataset.json").read_text(encoding="utf-8"))
    assert record["is_true_entanglement"] is False
    assert "not orbital entanglement" in record["scientific_guard"]


def test_orbital_state_metrics_rejects_mismatched_electron_count(tmp_path: Path) -> None:
    occupations = tmp_path / "occupations.csv"
    _write_csv(
        occupations,
        ["state", "orbital_id", "occupation", "status"],
        [
            {"state": "ground", "orbital_id": "1", "occupation": 1.0, "status": "accepted"},
            {"state": "ground", "orbital_id": "2", "occupation": 1.0, "status": "accepted"},
            {"state": "excited", "orbital_id": "1", "occupation": 1.5, "status": "provisional"},
            {"state": "excited", "orbital_id": "2", "occupation": 1.0, "status": "provisional"},
        ],
    )
    outdir = tmp_path / "metrics"
    rc = molcas_postanalysis.main(
        [
            "orbital-state-metrics",
            "--occupations-csv",
            str(occupations),
            "--pair",
            "ground:excited",
            "--outdir",
            str(outdir),
            "--figure-stem",
            "test_metrics",
            "--dpi",
            "80",
        ]
    )
    assert rc == 0
    record = json.loads((outdir / "test_metrics_summary.json").read_text(encoding="utf-8"))
    pair = record["pair_metrics"][0]
    assert pair["same_dimension_guard_pass"] is True
    assert pair["electron_count_guard_pass"] is False
    assert pair["comparison_accepted"] is False
