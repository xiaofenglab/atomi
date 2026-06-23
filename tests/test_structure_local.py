from __future__ import annotations

from pathlib import Path

import pytest

from atomi import local_structure as local
from atomi.local_structure import analyze_file, collect_neighbors, compare_environments, parse_center_selector

ase = pytest.importorskip("ase")


def _toy_uo_atoms():
    from ase import Atoms

    symbols = ["U", "U", "U", "O", "O", "O", "O"]
    positions = [
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (8.0, 0.0, 0.0),
        (1.8, 0.0, 0.0),
        (0.0, 2.0, 0.0),
        (5.0, 1.9, 0.0),
        (5.0, 0.0, 2.1),
    ]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    return atoms


def test_parse_center_selector_supports_element_relative_ranges() -> None:
    symbols = _toy_uo_atoms().get_chemical_symbols()

    assert parse_center_selector("U:2-3", symbols) == [1, 2]
    assert parse_center_selector("1,3", symbols) == [0, 2]
    assert parse_center_selector("U:1,O:2", symbols) == [0, 4]


def test_collect_neighbors_filters_by_element() -> None:
    atoms = _toy_uo_atoms()

    records = collect_neighbors(atoms, 0, radius=2.5, neighbor_elements={"O"})

    assert [record.atom_label for record in records] == ["O1", "O2"]
    assert [round(record.distance, 3) for record in records] == [1.8, 2.0]


def test_compare_environments_reports_cage_deltas() -> None:
    atoms = _toy_uo_atoms()
    left = collect_neighbors(atoms, 0, radius=3.0, neighbor_elements={"O"})
    right = collect_neighbors(atoms, 1, radius=3.0, neighbor_elements={"O"})

    rows = compare_environments({"U1": left, "U2": right}, first_shell_n=2)

    assert rows[0]["left"] == "U1"
    assert rows[0]["right"] == "U2"
    assert rows[0]["radial_max_abs_delta_A"] == pytest.approx(0.1)


def test_analyze_file_writes_summary_and_cluster_xyz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    atoms = _toy_uo_atoms()
    input_path = tmp_path / "toy.xyz"
    input_path.write_text("reader is monkeypatched in this unit test\n", encoding="utf-8")
    monkeypatch.setattr(local, "read_atoms", lambda *args, **kwargs: atoms)

    result = analyze_file(
        input_path,
        centers="U:1-2",
        outdir=tmp_path / "analysis",
        fmt="xyz",
        radius=3.0,
        neighbor_elements=["O"],
        first_shell_n=2,
        write_clusters=True,
        quiet=True,
    )

    assert Path(result["summary_csv"]).exists()
    assert Path(result["compare_csv"]).exists()
    assert len(result["cluster_paths"]) == 2
    assert all(Path(path).exists() for path in result["cluster_paths"])
