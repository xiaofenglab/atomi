from pathlib import Path

import numpy as np

from atomi.cp2k.rotate_seed import main, read_xyz, rotation_matrix


def test_rotation_matrix_maps_vector_to_axis() -> None:
    matrix = rotation_matrix(np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    rotated = matrix @ np.array([1.0, 1.0, 0.0])

    assert np.allclose(rotated[:2], [0.0, 0.0], atol=1.0e-12)
    assert rotated[2] > 0


def test_rotate_seed_writes_xyz_matrix_and_pointcharges(tmp_path: Path) -> None:
    xyz = tmp_path / "seed.xyz"
    xyz.write_text(
        "2\n"
        "seed\n"
        "Ga 1.0 1.0 1.0\n"
        "Cl 2.0 1.0 1.0\n",
        encoding="utf-8",
    )
    pc = tmp_path / "pointcharges.dat"
    pc.write_text("# q\n2.0 1.0 1.0 -1.0 Cl tag 2 source\n", encoding="utf-8")
    out = tmp_path / "rot.xyz"
    pc_out = tmp_path / "pc_rot.dat"
    mat = tmp_path / "matrix.txt"

    main(
        [
            str(xyz),
            "--atom1",
            "1",
            "--atom2",
            "2",
            "--axis",
            "z",
            "--output",
            str(out),
            "--pointcharges",
            str(pc),
            "--pointcharges-out",
            str(pc_out),
            "--matrix-out",
            str(mat),
        ]
    )

    _, coords, _ = read_xyz(out)
    assert np.allclose(coords[0], [0.0, 0.0, 0.0], atol=1.0e-10)
    assert np.allclose(coords[1], [0.0, 0.0, 1.0], atol=1.0e-10)
    assert mat.is_file()
    assert pc_out.read_text(encoding="utf-8").splitlines()[1].split()[4:] == [
        "Cl",
        "tag",
        "2",
        "source",
    ]
