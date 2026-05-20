from __future__ import annotations

import json
from pathlib import Path

from atomi.cli.main import main as atomi_main
from atomi.lammps.poscar_data import main


def write_poscar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "UO2 test",
                "1.0",
                "5.47 0 0",
                "0 5.47 0",
                "0 0 5.47",
                "U O",
                "1 2",
                "Direct",
                "0 0 0",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_poscar2lammps_writes_data_metadata_and_updates_config(tmp_path: Path, capsys) -> None:
    poscar = tmp_path / "POSCAR"
    write_poscar(poscar)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"initial_structure": "old.data", "model_file": "model.pt"}), encoding="utf-8")
    out = tmp_path / "structures" / "uo2.data"

    main([str(poscar), "--out", str(out), "--update-config", str(config)])

    output = capsys.readouterr().out
    assert "Wrote LAMMPS data" in output
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "3 atoms" in text
    assert "2 atom types" in text
    metadata = json.loads((tmp_path / "structures" / "uo2.data.json").read_text(encoding="utf-8"))
    assert metadata["species_order"] == ["U", "O"]
    assert metadata["lammps_type_map"] == {"U": 1, "O": 2}
    updated = json.loads(config.read_text(encoding="utf-8"))
    assert updated["initial_structure"] == "structures/uo2.data"


def test_poscar2lammps_replicates_cell(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_poscar(poscar)
    out = tmp_path / "big.data"

    main([str(poscar), "--out", str(out), "--replicate", "2", "1", "1"])

    metadata = json.loads((tmp_path / "big.data.json").read_text(encoding="utf-8"))
    assert metadata["natoms"] == 6
    assert metadata["replicate"] == [2, 1, 1]


def test_poscar2lammps_can_sort_by_species(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_poscar(poscar)
    out = tmp_path / "sorted.data"

    main([str(poscar), "--out", str(out), "--species-order", "O", "U", "--sort-by-species"])

    metadata = json.loads((tmp_path / "sorted.data.json").read_text(encoding="utf-8"))
    assert metadata["species_order"] == ["O", "U"]
    assert metadata["sorted_by_species"] is True
    assert metadata["lammps_type_map"] == {"O": 1, "U": 2}


def test_atomi_dispatches_poscar2lammps(tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    write_poscar(poscar)
    out = tmp_path / "from_atomi.data"

    atomi_main(["poscar2lammps", str(poscar), "--out", str(out)])

    assert out.exists()
