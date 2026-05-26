import csv
import json
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.magmom import existing_magmom_values, read_poscar_structure
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


def write_weird_order_source_poscar(path: Path) -> None:
    path.write_text(
        "A element reference with unusual species order\n"
        "1.0\n"
        "4.0 0.0 0.0\n"
        "0.0 4.0 0.0\n"
        "0.0 0.0 4.0\n"
        "Gd O U\n"
        "1 4 1\n"
        "Direct\n"
        "0.01 0.02 0.00\n"
        "0.25 0.25 0.25\n"
        "0.75 0.75 0.75\n"
        "0.25 0.75 0.25\n"
        "0.75 0.25 0.75\n"
        "0.52 0.50 0.50\n",
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


def write_2x3x3_cation_source(path: Path) -> None:
    positions: list[tuple[str, tuple[float, float, float]]] = []
    gd_position = (0.75, 0.5, 0.5)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                symbol = "Gd" if position == gd_position else "U"
                positions.append((symbol, position))
    grouped = [item for item in positions if item[0] == "U"] + [item for item in positions if item[0] == "Gd"]
    path.write_text(
        "A 2x3x3 cation pattern\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "U Gd\n"
        "17 1\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for _symbol, (x, y, z) in grouped),
        encoding="utf-8",
    )


def write_2x3x3_defect_cation_source(path: Path) -> list[float]:
    positions: list[tuple[str, tuple[float, float, float], float]] = []
    gd_cell = (1, 2, 2)
    u5_cell = (0, 2, 2)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                if (i, j, k) == gd_cell:
                    positions.append(("Gd", position, 7.0))
                elif (i, j, k) == u5_cell:
                    positions.append(("U", position, 5.0))
                else:
                    positions.append(("U", position, 2.0 if (i + j + k) % 2 == 0 else -2.0))
    grouped = [item for item in positions if item[0] == "U"] + [item for item in positions if item[0] == "Gd"]
    path.write_text(
        "A 2x3x3 cation pattern with defects outside origin crop\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "U Gd\n"
        "17 1\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for _symbol, (x, y, z), _moment in grouped),
        encoding="utf-8",
    )
    return [moment for _symbol, _position, moment in grouped]


def write_2x3x3_equal_cation_source(path: Path) -> None:
    positions: list[tuple[str, tuple[float, float, float]]] = []
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                symbol = "Gd" if len(positions) < 9 else "U"
                positions.append((symbol, position))
    grouped = [item for item in positions if item[0] == "U"] + [item for item in positions if item[0] == "Gd"]
    path.write_text(
        "A 2x3x3 equal cation pattern\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "U Gd\n"
        "9 9\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for _symbol, (x, y, z) in grouped),
        encoding="utf-8",
    )


def write_2x2x2_cation_target(path: Path) -> None:
    positions = [
        ((i + 0.5) / 2, (j + 0.5) / 2, (k + 0.5) / 2)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    path.write_text(
        "B Ia-3-like doubled cell cation skeleton\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 2.0 0.0\n"
        "0.0 0.0 2.0\n"
        "U\n"
        "8\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in positions),
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


def test_project_poscar_uses_target_centered_species_order_and_reorders_magmom(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_order"
    write_weird_order_source_poscar(source)
    write_relaxed_target_poscar(target)
    incar.write_text("ENCUT = 520\nMAGMOM = 7 4*0 2\n", encoding="utf-8")

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
            "--cation-elements",
            "U,Gd",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert projected.species.symbols == ["U", "Gd", "O"]
    assert projected.species.counts == [1, 1, 3]
    assert existing_magmom_values(out / "INCAR", 5) == pytest.approx([2, 7, 0, 0, 0])


def test_project_poscar_crops_2x3x3_source_to_2x2x2_target(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    out = tmp_path / "projected_crop"
    write_2x3x3_cation_source(source)
    write_2x2x2_cation_target(target)

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--source-supercell",
            "2x3x3",
            "--source-keep-cells",
            "2x2x2",
            "--cation-elements",
            "U,Gd",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    assert projected.species.symbols == ["U", "Gd"]
    assert projected.species.counts == [7, 1]
    assert prepared.species.symbols == ["U", "Gd"]
    assert prepared.species.counts == [7, 1]
    assert atom_symbols(projected).count("Gd") == 1
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    assert plan["source_cation_count"] == 8
    assert plan["target_cation_count"] == 8
    assert plan["prepared_source_poscar"] == str(out / "POSCAR_A_prepared")
    assert plan["source_operations"][0]["source_supercell"] == [2, 3, 3]
    assert plan["source_operations"][0]["keep_cells"] == [2, 2, 2]


def test_project_poscar_crop_prefers_minority_and_charge_coupled_cations(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_defect_crop"
    moments = write_2x3x3_defect_cation_source(source)
    write_2x2x2_cation_target(target)
    incar.write_text("MAGMOM = " + " ".join(f"{moment:g}" for moment in moments) + "\n", encoding="utf-8")

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
            "--source-supercell",
            "2x3x3",
            "--source-keep-cells",
            "2x2x2",
            "--cation-elements",
            "U,Gd",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert projected.species.symbols == ["U", "Gd"]
    assert projected.species.counts == [7, 1]
    output_moments = existing_magmom_values(out / "INCAR", 8)
    assert output_moments is not None
    assert output_moments.count(7.0) == 1
    assert output_moments.count(5.0) == 1
    assert sum(1 for moment in output_moments if abs(moment) == 2.0) == 6
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["selection_policy"] == "defect_preserving"
    assert operation["crop_origin_cells"] == [0, 1, 1]
    assert operation["minority_cations_kept"] == 1
    assert operation["charge_variant_cations_kept"] == 1
    assert operation["source_magnetic_signature"]["U"]["positive"] > 0
    assert operation["source_magnetic_signature"]["U"]["negative"] > 0
    assert operation["crop_magnetic_signature"]["U"]["positive"] > 0
    assert operation["crop_magnetic_signature"]["U"]["negative"] > 0
    assert operation["crop_magnetic_signature"]["Gd"]["positive"] == 1


def test_project_poscar_equal_cation_counts_use_regular_origin_crop(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    out = tmp_path / "projected_equal_crop"
    write_2x3x3_equal_cation_source(source)
    write_2x2x2_cation_target(target)

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--source-supercell",
            "2x3x3",
            "--source-keep-cells",
            "2x2x2",
            "--cation-elements",
            "U,Gd",
        ]
    )

    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["crop_origin_cells"] == [0, 0, 0]
    assert operation["selection_reason"] == "regular_origin_crop_no_minority_cation"
    assert operation["minority_cation_elements"] == []


def test_project_poscar_crop_repairs_relaxed_boundary_cation_count(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    out = tmp_path / "projected_boundary_repair"
    write_2x3x3_cation_source(source)
    write_2x2x2_cation_target(target)
    lines = source.read_text(encoding="utf-8").splitlines()
    lines[8] = "0.2500000000 -0.0001000000 0.0000000000"
    lines[9] = "0.2500000000 0.3333333333 0.6668000000"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--source-supercell",
            "2x3x3",
            "--source-keep-cells",
            "2x2x2",
            "--source-crop-policy",
            "origin",
            "--cation-elements",
            "U,Gd",
        ]
    )

    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    assert prepared.species.total_atoms == 8
    assert prepared.species.symbols == ["U", "Gd"]
    assert prepared.species.counts == [7, 1]
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["cation_boundary_repair"] is True
    assert operation["cation_count_before_repair"] == 6
    assert operation["cation_count_after_repair"] == 8
    assert operation["expected_cation_count"] == 8


def test_project_poscar_boundary_repair_preserves_balanced_cation_counts(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    out = tmp_path / "projected_balanced_repair"
    write_2x3x3_equal_cation_source(source)
    write_2x2x2_cation_target(target)
    lines = source.read_text(encoding="utf-8").splitlines()
    lines[8] = "0.2500000000 -0.0001000000 0.0000000000"
    lines[9] = "0.2500000000 0.3333333333 0.6668000000"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--source-supercell",
            "2x3x3",
            "--source-keep-cells",
            "2x2x2",
            "--cation-elements",
            "U,Gd",
        ]
    )

    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    assert prepared.species.symbols == ["U", "Gd"]
    assert prepared.species.counts == [4, 4]
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["selection_reason"] == "regular_origin_crop_no_minority_cation"
    assert operation["cation_boundary_repair"] is True
    assert operation["cation_species_target_counts"] == {"Gd": 4, "U": 4}


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
