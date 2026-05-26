import csv
import json
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.magmom import read_poscar_structure
from atomi.vasp.poscar_project import atom_symbols, main as project_main


def write_source_poscar(path: Path) -> None:
    path.write_text(
        "A element reference\n"
        "1.0\n"
        "4.0 0.0 0.0\n"
        "0.0 4.0 0.0\n"
        "0.0 0.0 4.0\n"
        "U Gd O\n"
        "1 1 4\n"
        "Direct\n"
        "0.52 0.50 0.50\n"
        "0.01 0.02 0.00\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n",
        encoding="utf-8",
    )


def write_relaxed_target_poscar(path: Path) -> None:
    path.write_text(
        "B relaxed structure with one O vacancy\n"
        "1.0\n"
        "4.1 0.0 0.0\n"
        "0.0 4.1 0.0\n"
        "0.0 0.0 4.1\n"
        "U O\n"
        "2 3\n"
        "Direct\n"
        "0.00 0.00 0.00\n"
        "0.50 0.50 0.50\n"
        "0.26 0.24 0.25\n"
        "0.74 0.76 0.75\n"
        "0.25 0.75 0.24\n",
        encoding="utf-8",
    )


def test_project_poscar_maps_cation_elements_by_site_and_rewrites_magmom(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected"
    write_source_poscar(source)
    write_relaxed_target_poscar(target)
    incar.write_text("ENCUT = 520\nMAGMOM = 2 7 4*0\n", encoding="utf-8")

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--incar-a",
            str(incar),
            "--outdir",
            str(out),
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert projected.species.symbols == ["U", "Gd", "O"]
    assert projected.species.counts == [1, 1, 3]
    assert atom_symbols(projected) == ["U", "Gd", "O", "O", "O"]
    assert projected.scaled_positions[0] == pytest.approx([0.50, 0.50, 0.50])
    assert projected.scaled_positions[1] == pytest.approx([0.00, 0.00, 0.00])
    assert "MAGMOM = 2 7 3*0" in (out / "INCAR").read_text(encoding="utf-8")

    with (out / "poscar_projection_map.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["target_atom"], row["source_element"]) for row in rows] == [("1", "Gd"), ("2", "U")]

    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    assert plan["source_cation_count"] == 2
    assert plan["target_cation_count"] == 2
    assert plan["species_counts"] == {"U": 1, "Gd": 1, "O": 3}


def test_materials_opt_project_poscar_alias(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    out = tmp_path / "projected_alias"
    write_source_poscar(source)
    write_relaxed_target_poscar(target)

    atomi_main(
        [
            "materials-opt",
            "project-poscar",
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--cation-elements",
            "U,Gd",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert projected.species.symbols == ["U", "Gd", "O"]
    assert projected.species.counts == [1, 1, 3]
