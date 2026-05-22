from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read

from atomi.vasp.md_snapshots import main


def write_poscar(path: Path, cell: float = 1.0) -> None:
    path.write_text(
        "\n".join(
            [
                "UO2 reference",
                "1.0",
                f"{cell} 0 0",
                f"0 {cell} 0",
                f"0 0 {cell}",
                "U O",
                "1 2",
                "Direct",
                "0.10 0.10 0.10",
                "0.40 0.40 0.40",
                "0.70 0.70 0.70",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_small_reference(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "small UO reference",
                "1.0",
                "1 0 0",
                "0 1 0",
                "0 0 1",
                "U O",
                "1 1",
                "Direct",
                "0.10 0.10 0.10",
                "0.60 0.60 0.60",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_dump(path: Path, frames: list[tuple[int, list[tuple[int, int, float, float, float]]]], cell: float) -> None:
    lines: list[str] = []
    for timestep, atoms in frames:
        lines.extend(
            [
                "ITEM: TIMESTEP",
                str(timestep),
                "ITEM: NUMBER OF ATOMS",
                str(len(atoms)),
                "ITEM: BOX BOUNDS pp pp pp",
                f"0 {cell}",
                f"0 {cell}",
                f"0 {cell}",
                "ITEM: ATOMS id type xs ys zs",
            ]
        )
        for atom_id, atom_type, x, y, z in atoms:
            lines.append(f"{atom_id} {atom_type} {x:.10f} {y:.10f} {z:.10f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_elastic_config(root: Path, stage: dict) -> Path:
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "timestep": 0.001,
                "atom_type_map": {"1": "U", "2": "O"},
                "stages": [stage],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def write_template(path: Path, magmom: str = "MAGMOM = 2 -2") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "INCAR").write_text(f"SYSTEM = test\n{magmom}\n", encoding="utf-8")
    (path / "KPOINTS").write_text("Gamma\n1\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")
    (path / "POTCAR").write_text("PAW_PBE U\nPAW_PBE O\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_elasticity_correction_selects_tail_frames_and_writes_stress_metadata(tmp_path: Path) -> None:
    reference = tmp_path / "POSCAR"
    write_poscar(reference)
    stage = {
        "name": "nvt_stress_300K_uniaxial_x_p005",
        "temperature": 300,
        "chunk_name": "chunk_01",
        "elastic_run": True,
        "deformation": {
            "mode": "xx",
            "strain": 0.005,
            "voigt_strain": [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    config = write_elastic_config(tmp_path, stage)
    chunk = tmp_path / "stages" / stage["name"] / "chunk_01"
    chunk.mkdir(parents=True)
    (chunk.parent / "PASS").write_text("ok\n", encoding="utf-8")
    frames = []
    for timestep in [0, 1000, 2000, 3000, 4000, 5000]:
        frames.append(
            (
                timestep,
                [
                    (1, 1, 0.10, 0.10, 0.10),
                    (2, 2, 0.40, 0.40, 0.40),
                    (3, 2, 0.70, 0.70, 0.70),
                ],
            )
        )
    write_dump(chunk / "dump.elastic.lammpstrj", frames, cell=1.0)

    out = tmp_path / "elastic_candidates"
    main(
        [
            "--config",
            str(config),
            "--project-root",
            str(tmp_path),
            "--poscar",
            str(reference),
            "--output-root",
            str(out),
            "--elasticity-correction",
            "--elastic-frames-per-run",
            "2",
            "--elastic-tail-fraction",
            "1.0",
            "--elastic-min-separation-ps",
            "2.0",
        ]
    )

    runlist = [line for line in (out / "runlist.txt").read_text(encoding="utf-8").splitlines() if line]
    rows = read_csv(out / "candidate_index.csv")
    assert len(runlist) == 2
    assert len(rows) == 2
    assert "T300K/uniaxial_x_p005" in runlist[0]
    case_info = json.loads((out / runlist[0] / "case_info.json").read_text(encoding="utf-8"))
    assert case_info["intended_training_role"] == "elasticity_correction"
    assert case_info["expected_labels"] == ["energy", "forces", "stress"]
    assert case_info["dft_stress_required"] is True
    assert case_info["md_stress_used_as_training_label"] is False
    assert case_info["strain_family"] == "uniaxial_tetragonal"
    assert case_info["strain_amplitude"] == 0.005


def test_elasticity_correction_reduces_large_frame_to_representative_subcells(tmp_path: Path) -> None:
    reference = tmp_path / "POSCAR_2x2x2"
    write_small_reference(reference)
    template = tmp_path / "VASP_TEMPLATE"
    write_template(template)
    stage = {
        "name": "nvt_stress_900K_shear_xy_p005",
        "temperature": 900,
        "chunk_name": "chunk_01",
        "elastic_run": True,
        "deformation": {
            "mode": "xy",
            "strain": 0.005,
            "voigt_strain": [0.0, 0.0, 0.0, 0.0, 0.0, 0.005],
        },
    }
    config = write_elastic_config(tmp_path, stage)
    chunk = tmp_path / "stages" / stage["name"] / "chunk_01"
    chunk.mkdir(parents=True)
    (chunk.parent / "PASS").write_text("ok\n", encoding="utf-8")

    atoms = []
    atom_id = 1
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                ox = 0.002 * (ix + iy + iz)
                atoms.append((atom_id, 1, (ix + 0.10) / 2.0, (iy + 0.10) / 2.0, (iz + 0.10) / 2.0))
                atom_id += 1
                atoms.append(
                    (
                        atom_id,
                        2,
                        (ix + 0.60 + ox) / 2.0,
                        (iy + 0.60) / 2.0,
                        (iz + 0.60) / 2.0,
                    )
                )
                atom_id += 1
    write_dump(chunk / "dump.elastic.lammpstrj", [(1000, atoms)], cell=2.0)

    out = tmp_path / "elastic_reduced"
    main(
        [
            "--config",
            str(config),
            "--project-root",
            str(tmp_path),
            "--output-root",
            str(out),
            "--vasp-template",
            str(template),
            "--elasticity-correction",
            "--elastic-frames-per-run",
            "1",
            "--reduce-large-md-to-2x2x2",
            "--reference-poscar-2x2x2",
            str(reference),
            "--large-to-small-replicate",
            "2",
            "2",
            "2",
            "--keep-all-subcells",
        ]
    )

    runlist = [line for line in (out / "runlist.txt").read_text(encoding="utf-8").splitlines() if line]
    assert len(runlist) == 8
    assert all("T900K/shear_xy_p005" in line for line in runlist)
    for entry in runlist:
        run_dir = out / entry
        atoms_out = read(run_dir / "POSCAR")
        assert atoms_out.get_chemical_symbols() == ["U", "O"]
        assert (run_dir / "INCAR").read_text(encoding="utf-8").splitlines()[-1] == "MAGMOM = 2 -2"
        info = json.loads((run_dir / "case_info.json").read_text(encoding="utf-8"))
        assert info["subcell_offset"] in (
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        )
        assert info["expected_subcell_count"] == 8
        assert info["valid_subcell_count"] == 8
        assert info["kept_subcell_count"] == 8
        assert info["large_to_small_replicate"] == [2, 2, 2]
        assert info["magmom_status"] == "ready"
        assert info["magmom_ready_for_reference_order"] is True
        assert info["magmom_count"] == 2
        assert info["md_stress_used_as_training_label"] is False
        assert info["dft_stress_required"] is True
        assert "Long-wavelength correlations" in info["reduction_note"]
