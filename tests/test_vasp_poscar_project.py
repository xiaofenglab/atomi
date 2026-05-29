import csv
import json
from pathlib import Path

import pytest

from atomi.cli.main import main as atomi_main
from atomi.vasp.magmom import (
    existing_magmom_values,
    expand_magmom_tokens,
    find_magmom_line,
    read_poscar_structure,
    strip_incar_comment,
)
from atomi.vasp.poscar_project import atom_symbols, cell_volume, main as project_main, repaired_crop_cation_indices


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


def expanded_magmom_values(incar: Path) -> list[float]:
    _line_index, line = find_magmom_line(incar.read_text(encoding="utf-8", errors="replace").splitlines())
    assert line is not None
    body = strip_incar_comment(line).split("=", 1)[-1]
    return expand_magmom_tokens(body.split())


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_origin_shifted_target_poscar(path: Path) -> None:
    path.write_text(
        "B relaxed structure with shifted cation origin\n"
        "1.0\n"
        "4.1 0.0 0.0\n"
        "0.0 4.1 0.0\n"
        "0.0 0.0 4.1\n"
        "U O\n"
        "2 3\n"
        "Direct\n"
        "0.2100000000 0.2700000000 0.1000000000\n"
        "0.7200000000 0.7500000000 0.6000000000\n"
        "0.2600000000 0.2400000000 0.2500000000\n"
        "0.7400000000 0.7600000000 0.7500000000\n"
        "0.2500000000 0.7500000000 0.2400000000\n",
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


def write_2x3x3_gd_vo_source(path: Path) -> list[float]:
    cations: list[tuple[str, tuple[float, float, float], float]] = []
    anions: list[tuple[str, tuple[float, float, float], float]] = []
    gd_cells = {(0, 2, 2), (1, 2, 2)}
    vacancy_cell = (0, 1, 1)
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                if (i, j, k) in gd_cells:
                    cations.append(("Gd", position, 7.0))
                else:
                    cations.append(("U", position, 2.0 if (i + j + k) % 2 == 0 else -2.0))
                anion_sites = [
                    ((i + 0.25) / 2, (j + 0.25) / 3, (k + 0.25) / 3),
                    ((i + 0.75) / 2, (j + 0.75) / 3, (k + 0.75) / 3),
                ]
                for site_index, anion_position in enumerate(anion_sites):
                    if (i, j, k) == vacancy_cell and site_index == 0:
                        continue
                    anions.append(("O", anion_position, 0.0))
    grouped = (
        [item for item in cations if item[0] == "Gd"]
        + [item for item in cations if item[0] == "U"]
        + anions
    )
    path.write_text(
        "A 2x3x3 Gd-doped source with one oxygen vacancy\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "Gd U O\n"
        "2 16 35\n"
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


def write_2x3x3_balanced_fold_source(path: Path) -> list[float]:
    positions: list[tuple[str, tuple[float, float, float], float]] = []
    anions: list[tuple[str, tuple[float, float, float], float]] = []
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                if (i + j + k) % 2 == 0:
                    positions.append(("Gd", position, 7.0))
                else:
                    positions.append(("U", position, 2.0 if (i + k) % 2 == 0 else -2.0))
                anions.append(("O", ((i + 0.25) / 2, (j + 0.25) / 3, (k + 0.25) / 3), 0.0))
                anions.append(("O", ((i + 0.75) / 2, (j + 0.75) / 3, (k + 0.75) / 3), 0.0))
    grouped = [item for item in positions if item[0] == "Gd"] + [item for item in positions if item[0] == "U"] + anions
    path.write_text(
        "A 2x3x3 balanced cation and anion pattern for representative folding\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "Gd U O\n"
        "9 9 36\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for _symbol, (x, y, z), _moment in grouped),
        encoding="utf-8",
    )
    return [moment for _symbol, _position, moment in grouped]


def write_2x3x3_noisy_charge_coupled_source(path: Path) -> list[float]:
    positions: list[tuple[str, tuple[float, float, float], float]] = []
    gd_cells = {(1, 2, 2), (0, 2, 1)}
    u5_cell = (0, 2, 2)
    noisy_u4 = [
        -1.994,
        2.011,
        -2.005,
        1.990,
        -1.989,
        -1.997,
        1.996,
        -1.996,
        -2.001,
        2.015,
        1.986,
        2.000,
        2.015,
        1.990,
        -1.992,
    ]
    u4_index = 0
    for i in range(2):
        for j in range(3):
            for k in range(3):
                position = ((i + 0.5) / 2, (j + 0.5) / 3, (k + 0.5) / 3)
                if (i, j, k) in gd_cells:
                    positions.append(("Gd", position, 7.071 if (i + j + k) % 2 == 0 else -7.070))
                elif (i, j, k) == u5_cell:
                    positions.append(("U", position, -1.044))
                else:
                    positions.append(("U", position, noisy_u4[u4_index]))
                    u4_index += 1
    grouped = [item for item in positions if item[0] == "Gd"] + [item for item in positions if item[0] == "U"]
    path.write_text(
        "A 2x3x3 noisy Gd/U charge-coupled pattern\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 3.0 0.0\n"
        "0.0 0.0 3.0\n"
        "Gd U\n"
        "2 16\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for _symbol, (x, y, z), _moment in grouped),
        encoding="utf-8",
    )
    return [moment for _symbol, _position, moment in grouped]


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


def write_2x2x2_uo2_target(path: Path) -> None:
    cations = [
        ((i + 0.5) / 2, (j + 0.5) / 2, (k + 0.5) / 2)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    anions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                anions.append(((i + 0.25) / 2, (j + 0.25) / 2, (k + 0.25) / 2))
                anions.append(((i + 0.75) / 2, (j + 0.75) / 2, (k + 0.75) / 2))
    path.write_text(
        "B 2x2x2 full UO2 target\n"
        "1.0\n"
        "2.0 0.0 0.0\n"
        "0.0 2.0 0.0\n"
        "0.0 0.0 2.0\n"
        "U O\n"
        "8 16\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in [*cations, *anions]),
        encoding="utf-8",
    )


def write_large_volume_2x2x2_cation_target(path: Path) -> None:
    positions = [
        ((i + 0.5) / 2, (j + 0.5) / 2, (k + 0.5) / 2)
        for i in range(2)
        for j in range(2)
        for k in range(2)
    ]
    path.write_text(
        "B larger-volume doubled cell cation skeleton\n"
        "1.0\n"
        "4.0 0.0 0.0\n"
        "0.0 4.0 0.0\n"
        "0.0 0.0 4.0\n"
        "U\n"
        "8\n"
        "Direct\n"
        + "".join(f"{x:.10f} {y:.10f} {z:.10f}\n" for x, y, z in positions),
        encoding="utf-8",
    )


def test_project_poscar_maps_cation_elements_by_site_and_rewrites_magmom(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    assert plan["source_magmom_count"] == 6
    assert plan["output_magmom_count"] == 5
    assert plan["source_cation_count"] == 2
    assert plan["target_cation_count"] == 2
    assert plan["species_counts"] == {"U": 1, "Gd": 1, "O": 3}
    assert plan["source_cation_magmom_summary"]["U"]["count"] == 1
    assert plan["source_cation_magmom_summary"]["U"]["positive"] == 1
    assert plan["source_cation_magmom_summary"]["Gd"]["count"] == 1
    assert plan["source_cation_magmom_summary"]["Gd"]["positive"] == 1
    assert plan["cation_magmom_summary"]["U"]["count"] == 1
    assert plan["cation_magmom_summary"]["U"]["positive"] == 1
    assert plan["cation_magmom_summary"]["Gd"]["count"] == 1
    assert plan["cation_magmom_summary"]["Gd"]["positive"] == 1
    assert plan["cation_magmom_comparison"]["U"]["count_delta"] == 0
    assert plan["cation_magmom_comparison"]["Gd"]["unique_abs_moments_match"] is True
    assert "O" not in plan["cation_magmom_summary"]
    stdout = capsys.readouterr().out
    assert "Prepared A MAGMOM" in stdout
    assert "Projected C MAGMOM" in stdout
    assert "Cation MAGMOM check" in stdout
    assert "U: n=1 +1 -0 0=0" in stdout
    assert "Gd: n=1 +1 -0 0=0" in stdout
    assert "U: n 1->1 (delta +0)" in stdout


def test_project_poscar_rejects_mismatched_source_magmom_count(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_bad_magmom"
    write_source_poscar(source)
    write_relaxed_target_poscar(target)
    incar.write_text("ENCUT = 520\nMAGMOM = 2 7 4*0 99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MAGMOM count .* source POSCAR"):
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


def test_project_poscar_uses_cation_elements_order_for_output_cations(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_cation_order"
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
            "Gd,U",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert projected.species.symbols == ["Gd", "U", "O"]
    assert projected.species.counts == [1, 1, 3]
    assert existing_magmom_values(out / "INCAR", 5) == pytest.approx([7, 2, 0, 0, 0])
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    assert plan["cation_elements"] == ["Gd", "U"]
    assert plan["species_order"] == ["Gd", "U", "O"]


def test_project_poscar_aligns_global_cation_origin_before_matching(tmp_path: Path) -> None:
    source = tmp_path / "A_POSCAR"
    target = tmp_path / "B_POSCAR"
    out = tmp_path / "projected_shifted"
    write_source_poscar(source)
    write_origin_shifted_target_poscar(target)

    project_main(
        [
            "--element-poscar",
            str(source),
            "--structure-poscar",
            str(target),
            "--outdir",
            str(out),
            "--cation-elements",
            "U,Gd",
            "--max-cation-distance",
            "0.1",
        ]
    )

    projected = read_poscar_structure(out / "POSCAR")
    assert atom_symbols(projected)[:2] == ["U", "Gd"]
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    alignment = plan["cation_origin_alignment"]
    assert alignment["improved"] is True
    assert alignment["shift_fractional"] == pytest.approx([0.2, 0.25, 0.1])
    assert alignment["max_cation_distance_after_A"] < 0.1


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


def test_project_poscar_crop_preserves_noisy_charge_coupled_host(tmp_path: Path) -> None:
    source = tmp_path / "A_noisy_charge_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_noisy_charge_crop"
    moments = write_2x3x3_noisy_charge_coupled_source(source)
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
            "Gd,U",
        ]
    )

    output_moments = existing_magmom_values(out / "INCAR", 8)
    assert output_moments is not None
    assert sum(1 for moment in output_moments if abs(moment) > 6.0) == 2
    assert sum(1 for moment in output_moments if 0.8 < abs(moment) < 1.3) == 1
    assert sum(1 for moment in output_moments if 1.7 < abs(moment) < 2.3) == 5
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["selection_policy"] == "defect_preserving"
    assert operation["minority_cations_kept"] == 2
    assert operation["charge_variant_cations_kept"] == 1


def test_project_poscar_crop_preserves_oxygen_vacancy_and_charge_neutrality(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "A_gd_vo_POSCAR"
    target = tmp_path / "B_2x2x2_uo2_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_gd_vo"
    moments = write_2x3x3_gd_vo_source(source)
    write_2x2x2_uo2_target(target)
    incar.write_text("MAGMOM = " + " ".join(f"{moment:g}" for moment in moments) + "\n", encoding="utf-8")
    (tmp_path / "KPOINTS").write_text("Gamma\n", encoding="utf-8")
    (tmp_path / "POTCAR").write_text("fake-potcar\n", encoding="utf-8")
    out.mkdir()
    for stale_name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR"):
        (out / stale_name).write_text("stale\n", encoding="utf-8")

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
            "Gd,U",
            "--anion-elements",
            "O",
            "--oxidation-state",
            "Gd=3,U=4,O=-2",
            "--randomize-candidates",
            "3",
            "--randomize-pool-size",
            "8",
            "--randomize-seed",
            "17",
        ]
    )

    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    assert prepared.species.symbols == ["Gd", "U", "O"]
    assert prepared.species.counts[:2] == [2, 6]
    assert not (out / "POSCAR").exists()
    assert not (out / "INCAR").exists()
    assert not (out / "KPOINTS").exists()
    assert not (out / "POTCAR").exists()
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    assert plan["copied_static_vasp_inputs"] == []
    assert plan["anion_vacancy_summary"]["removed_anion_counts"] == {"O": 1}
    assert plan["anion_vacancy_summary"]["output_anion_counts"] == {"O": 15}
    locality = plan["anion_vacancy_summary"]["removed_anion_nearest_cations"]
    assert len(locality) == 1
    assert locality[0]["removed_element"] == "O"
    assert locality[0]["nearest_cation_element"] in {"Gd", "U"}
    assert locality[0]["nearest_cation_distance_A"] > 0
    assert plan["charge_summary"]["neutrality_ok"] is True
    assert plan["charge_summary"]["total_charge"] == pytest.approx(0.0)
    gd_distance = plan["guest_cation_distance_summary"]["symbols"]["Gd"]
    assert gd_distance["pair_count"] == 1
    assert gd_distance["source_min_distance_A"] > 0
    assert gd_distance["output_min_distance_A"] > 0
    assert gd_distance["nearest_distance_preserved"] is True
    direct = plan["direct_candidate_summary"]
    assert direct["enabled"] is True
    direct_dir = Path(direct["run_dir"])
    assert direct_dir.name == "direct_projected"
    assert direct_dir.parent.name == "candidates"
    direct_poscar = read_poscar_structure(Path(direct["poscar"]))
    projected = direct_poscar
    assert Path(plan["output_poscar"]) == Path(direct["poscar"])
    assert Path(plan["output_incar"]) == Path(direct["incar"])
    assert direct_poscar.species.symbols == ["Gd", "U", "O"]
    assert direct_poscar.species.counts == [2, 6, 15]
    assert len(expanded_magmom_values(Path(direct["incar"]))) == projected.species.total_atoms
    assert (direct_dir / "KPOINTS").read_text(encoding="utf-8") == "Gamma\n"
    assert (direct_dir / "POTCAR").read_text(encoding="utf-8") == "fake-potcar\n"
    assert sorted(Path(path).name for path in direct["copied_static_vasp_inputs"]) == ["KPOINTS", "POTCAR"]
    random_summary = plan["randomized_candidate_summary"]
    assert random_summary["enabled"] is True
    assert random_summary["candidate_count"] == 3
    assert random_summary["pool_size"] == 8
    assert random_summary["selected_count"] == 3
    assert Path(random_summary["candidates_dir"]).name == "candidates"
    assert Path(random_summary["atat_rndstr"]).is_file()
    assert Path(random_summary["atat_pseudo_species_map"]).is_file()
    atat_run = Path(random_summary["atat_run_script"])
    atat_submit = Path(random_summary["atat_submit_script"])
    atat_readme = Path(random_summary["atat_readme"])
    assert atat_run.is_file()
    assert atat_submit.is_file()
    assert atat_readme.is_file()
    run_text = atat_run.read_text(encoding="utf-8")
    assert "mcsqs -2=6" in run_text
    assert "mcsqs -n=" in run_text
    assert "tail_log mcsqs.err" in run_text
    assert "ATAT mcsqs search failed" in run_text
    submit_text = atat_submit.read_text(encoding="utf-8")
    assert "#SBATCH --time=04:00:00" in submit_text
    assert "SCRIPT_DIR=" in submit_text
    assert 'cd "${SCRIPT_DIR}"' in submit_text
    assert "sbatch submit_mcsqs.sbatch" in atat_readme.read_text(encoding="utf-8")
    candidate_index = rows(out / "randomized_candidate_index.csv")
    assert len(candidate_index) == 3
    pool_index = rows(out / "randomized_pool_rankings.csv")
    assert len(pool_index) == 8
    assert {row["stability_status"] for row in candidate_index} == {"ok"}
    scores = [float(row["stability_score"]) for row in candidate_index]
    assert scores == sorted(scores, reverse=True)
    for candidate in random_summary["candidates"]:
        candidate_poscar = read_poscar_structure(Path(candidate["poscar"]))
        assert candidate_poscar.species.symbols == ["Gd", "U", "O"]
        assert candidate_poscar.species.counts == [2, 6, 15]
        assert candidate["charge_summary"]["neutrality_ok"] is True
        assert candidate["stability_rank"]["status"] == "ok"
        assert len(expanded_magmom_values(Path(candidate["incar"]))) == candidate_poscar.species.total_atoms
        candidate_dir = Path(candidate["run_dir"])
        assert (candidate_dir / "KPOINTS").read_text(encoding="utf-8") == "Gamma\n"
        assert (candidate_dir / "POTCAR").read_text(encoding="utf-8") == "fake-potcar\n"
        assert sorted(Path(path).name for path in candidate["copied_static_vasp_inputs"]) == ["KPOINTS", "POTCAR"]
    captured = capsys.readouterr().out
    assert "Worst cation matches:" in captured
    assert "Removed anion vacancy locality:" in captured
    assert "source atoms [" in captured
    assert "Direct candidate :" in captured
    assert "Randomized candidates: 3" in captured
    assert "Pool       : 8" in captured
    assert "ATAT sbatch:" in captured


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


def test_project_poscar_can_reduce_source_to_representative_cation_cell(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_reduced"
    moments = write_2x3x3_balanced_fold_source(source)
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
            "--source-reduce-cells",
            "2x2x2",
            "--cation-elements",
            "Gd,U",
        ]
    )

    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    projected = read_poscar_structure(out / "POSCAR")
    assert prepared.species.symbols == ["Gd", "U", "O"]
    assert prepared.species.counts == [4, 4, 16]
    assert projected.species.symbols == ["Gd", "U"]
    assert projected.species.counts == [4, 4]
    output_moments = existing_magmom_values(out / "INCAR", 8)
    assert output_moments is not None
    assert output_moments[:4] == pytest.approx([7.0, 7.0, 7.0, 7.0])
    assert {abs(moment) for moment in output_moments[4:]} == {2.0}
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    assert plan["source_magmom_count"] == 54
    assert plan["output_magmom_count"] == 8
    assert len(expanded_magmom_values(out / "INCAR")) == projected.species.total_atoms
    operation = plan["source_operations"][0]
    assert operation["kind"] == "source_representative_reduce"
    assert operation["source_supercell"] == [2, 3, 3]
    assert operation["reduce_cells"] == [2, 2, 2]
    assert operation["source_cation_count"] == 18
    assert operation["folded_cation_site_count"] == 8
    assert operation["source_non_cation_count"] == 36
    assert operation["folded_non_cation_site_count"] == 16
    assert operation["selected_non_cation_count"] == 16
    assert operation["cation_species_target_counts"] == {"Gd": 4, "U": 4}
    assert operation["cation_species_selected_counts"] == {"Gd": 4, "U": 4}
    assert operation["non_cation_species_target_counts"] == {"O": 16}
    assert operation["non_cation_species_selected_counts"] == {"O": 16}
    assert plan["source_cation_magmom_summary"]["Gd"]["count"] == 4
    assert plan["cation_magmom_comparison"]["Gd"]["count_delta"] == 0
    assert plan["cation_magmom_comparison"]["U"]["unique_abs_moments_match"] is True


def test_project_poscar_reduction_preserves_noisy_charge_coupled_host(tmp_path: Path) -> None:
    source = tmp_path / "A_noisy_charge_POSCAR"
    target = tmp_path / "B_2x2x2_POSCAR"
    incar = tmp_path / "A_INCAR"
    out = tmp_path / "projected_noisy_charge"
    moments = write_2x3x3_noisy_charge_coupled_source(source)
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
            "--source-reduce-cells",
            "2x2x2",
            "--cation-elements",
            "Gd,U",
        ]
    )

    output_moments = existing_magmom_values(out / "INCAR", 8)
    assert output_moments is not None
    assert sum(1 for moment in output_moments if abs(moment) > 6.0) == 2
    assert sum(1 for moment in output_moments if 0.8 < abs(moment) < 1.3) == 1
    assert sum(1 for moment in output_moments if 1.7 < abs(moment) < 2.3) == 5
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["source_operations"][0]
    assert operation["cation_species_selected_counts"] == {"Gd": 2, "U": 6}
    assert operation["cation_abs_moment_target_counts"]["U|negative|1"] == 1
    assert operation["cation_abs_moment_selected_counts"]["U|negative|1"] == 1


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


def test_crop_boundary_repair_prefers_protected_cations() -> None:
    repaired = repaired_crop_cation_indices(
        positions=[
            [0.92, 0.92, 0.92],
            [0.10, 0.10, 0.10],
            [0.20, 0.20, 0.20],
            [0.30, 0.30, 0.30],
        ],
        symbols=["U", "U", "U", "U"],
        cation_indices=[0, 1, 2, 3],
        ranges=((0.0, 0.5), (0.0, 0.5), (0.0, 0.5)),
        expected_count=2,
        protected_indices={0},
    )

    assert 0 in repaired
    assert len(repaired) == 2


def test_project_poscar_can_scale_target_volume_to_prepared_source(tmp_path: Path) -> None:
    source = tmp_path / "A_2x3x3_POSCAR"
    target = tmp_path / "B_large_POSCAR"
    out = tmp_path / "projected_scaled_target"
    write_2x3x3_cation_source(source)
    write_large_volume_2x2x2_cation_target(target)

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
            "--scale-target-volume-to-source",
        ]
    )

    prepared = read_poscar_structure(out / "POSCAR_A_prepared")
    projected = read_poscar_structure(out / "POSCAR")
    assert cell_volume(projected.cell) == pytest.approx(cell_volume(prepared.cell))
    target_positions = read_poscar_structure(target).scaled_positions
    for projected_position, target_position in zip(projected.scaled_positions, target_positions):
        assert projected_position == pytest.approx(target_position)
    plan = json.loads((out / "poscar_projection_plan.json").read_text(encoding="utf-8"))
    operation = plan["target_operations"][-1]
    assert operation["kind"] == "scale_volume_to_source"
    assert operation["target_volume_before_A3"] == pytest.approx(64.0)
    assert operation["target_volume_after_A3"] == pytest.approx(cell_volume(prepared.cell))


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
