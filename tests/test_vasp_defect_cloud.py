from __future__ import annotations

import csv
import json
from pathlib import Path

from ase.io import read

from atomi.vasp.defect_cloud import main


def write_poscar(path: Path, title: str = "defect motif") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                title,
                "1.0",
                "5.0 0.0 0.0",
                "0.0 5.0 0.0",
                "0.0 0.0 5.0",
                "Gd U O",
                "1 2 4",
                "Direct",
                "0.00 0.00 0.00",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
                "0.50 0.50 0.00",
                "0.50 0.00 0.50",
                "0.00 0.50 0.50",
                "0.50 0.50 0.50",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_template(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("INCAR", "KPOINTS", "POTCAR"):
        (path / name).write_text(f"{name}\n", encoding="utf-8")
    write_poscar(path / "POSCAR", title="template")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_defect_cloud_generates_compact_vasp_runs(tmp_path: Path) -> None:
    seed_root = tmp_path / "seeds"
    write_poscar(seed_root / "motif_A" / "POSCAR", title="motif A")
    write_poscar(seed_root / "motif_B" / "POSCAR", title="motif B")
    template = tmp_path / "VASP_TEMPLATE"
    write_template(template)
    output = tmp_path / "DEFECT_CLOUD"

    main(
        [
            "--seed-root",
            str(seed_root),
            "--output-root",
            str(output),
            "--vasp-template",
            str(template),
            "--seed",
            "123",
        ]
    )

    runlist = output / "runlist.txt"
    index = output / "defect_cloud_index.csv"
    summary = json.loads((output / "defect_cloud_summary.json").read_text(encoding="utf-8"))
    run_dirs = [line.strip() for line in runlist.read_text(encoding="utf-8").splitlines() if line.strip()]
    index_rows = rows(index)

    assert len(run_dirs) == 16
    assert len(index_rows) == 16
    assert summary["n_seed_motifs"] == 2
    assert summary["n_candidate_runs"] == 16
    assert (output / "motif_A" / "base" / "INCAR").exists()
    assert (output / "motif_A" / "base" / "KPOINTS").exists()
    assert (output / "motif_A" / "base" / "POTCAR").exists()
    assert (output / "motif_A" / "bias_O_001" / "POSCAR").exists()
    assert index_rows[0]["motif_id"] == "motif_A"
    assert {row["family"] for row in index_rows} >= {
        "base",
        "random_displacement",
        "isotropic_strain",
        "species_biased_displacement",
        "mixed_displacement",
    }
    base_atoms = read(output / "motif_A" / "base" / "POSCAR")
    assert base_atoms.get_chemical_symbols() == ["Gd", "U", "U", "O", "O", "O", "O"]


def test_defect_cloud_per_motif_ten_fills_structured_variants(tmp_path: Path) -> None:
    seed = tmp_path / "seed_POSCAR"
    write_poscar(seed)
    output = tmp_path / "cloud"

    main(
        [
            "--seed-poscar",
            str(seed),
            "--output-root",
            str(output),
            "--per-motif",
            "10",
            "--seed",
            "7",
        ]
    )

    index_rows = rows(output / "defect_cloud_index.csv")
    assert len(index_rows) == 10
    assert sum(1 for row in index_rows if row["family"] == "structured_displacement") == 2
    assert (output / "seed_POSCAR" / "disp_small_extra_001" / "POSCAR").exists()
