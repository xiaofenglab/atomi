import argparse
import sys
from pathlib import Path

import numpy as np


AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-14:
        raise ValueError("zero-length vector cannot be normalized")
    return vector / norm


def rotation_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix R such that R @ source points along target."""
    a = normalize(source)
    b = normalize(target)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    if np.linalg.norm(cross) < 1.0e-12:
        if dot > 0:
            return np.eye(3)
        trial = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, trial))
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (np.linalg.norm(cross) ** 2))


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str, list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"not enough lines for XYZ file: {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"first XYZ line must be atom count: {path}") from exc
    comment = lines[1]
    symbols = []
    coords = []
    extras = []
    for line in lines[2 : 2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"bad XYZ atom line in {path}: {line!r}")
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        extras.append(parts[4:])
    if len(symbols) != natoms:
        raise ValueError(f"XYZ atom count mismatch in {path}")
    return symbols, np.array(coords, dtype=float), comment, extras


def write_xyz(
    path: Path,
    symbols: list[str],
    coords: np.ndarray,
    comment: str,
    extras: list[list[str]] | None = None,
) -> None:
    extras = extras or [[] for _ in symbols]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, coord, extra in zip(symbols, coords, extras):
            line = f"{symbol:2s}  {coord[0]: .10f}  {coord[1]: .10f}  {coord[2]: .10f}"
            if extra:
                line += "  " + " ".join(extra)
            handle.write(line + "\n")


def rotate_coords(coords: np.ndarray, matrix: np.ndarray, origin: np.ndarray) -> np.ndarray:
    return (matrix @ (coords - origin).T).T


def resolve_origin(coords: np.ndarray, atom_index: int, mode: str) -> np.ndarray:
    if mode == "atom1":
        return coords[atom_index].copy()
    if mode == "geometric-center":
        return coords.mean(axis=0)
    raise ValueError(f"unknown origin mode: {mode}")


def rotate_pointcharges(infile: Path, outfile: Path, matrix: np.ndarray, origin: np.ndarray) -> int:
    """Rotate point charges with format: x y z charge [extra columns...]."""
    count = 0
    with infile.open(encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("!"):
                fout.write(line)
                continue
            parts = stripped.split()
            if len(parts) < 4:
                fout.write(line)
                continue
            try:
                coord = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
                charge = float(parts[3])
            except ValueError:
                fout.write(line)
                continue
            new_coord = matrix @ (coord - origin)
            rest = parts[4:]
            out = (
                f"{new_coord[0]:16.8f}  {new_coord[1]:16.8f}  "
                f"{new_coord[2]:16.8f}  {charge:12.6f}"
            )
            if rest:
                out += "  " + "  ".join(rest)
            fout.write(out + "\n")
            count += 1
    return count


def write_rotation_matrix(path: Path, matrix: np.ndarray, origin: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Coordinate transform used:\n")
        handle.write("#   r_new = R @ (r_old - origin)\n")
        handle.write("# Origin:\n")
        handle.write(f"{origin[0]: .12f}  {origin[1]: .12f}  {origin[2]: .12f}\n")
        handle.write("# Rotation matrix R:\n")
        for row in matrix:
            handle.write(f"{row[0]: .12f}  {row[1]: .12f}  {row[2]: .12f}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp2k-rotate-seed",
        description="Rotate a metal-ligand XYZ seed before CP2K box building.",
    )
    parser.add_argument("xyz", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--atom1", type=int, default=1, help="1-based origin atom, usually metal.")
    parser.add_argument(
        "--atom2",
        type=int,
        default=2,
        help="1-based atom defining the alignment vector.",
    )
    parser.add_argument(
        "--axis",
        choices=sorted(AXES),
        default="z",
        help="Target axis for atom1 -> atom2.",
    )
    parser.add_argument("--pointcharges", type=Path, default=None)
    parser.add_argument("--pointcharges-out", type=Path, default=None)
    parser.add_argument("--matrix-out", type=Path, default=None)
    parser.add_argument(
        "--origin",
        choices=("atom1", "geometric-center"),
        default="atom1",
        help="Coordinate origin to remove before rotation.",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Only center/translate coordinates; do not align atom1 -> atom2.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.xyz.is_file():
        parser.error(f"XYZ file not found: {args.xyz}")
    if args.pointcharges is not None and not args.pointcharges.is_file():
        parser.error(f"point charge file not found: {args.pointcharges}")

    symbols, coords, old_comment, extras = read_xyz(args.xyz)
    i = args.atom1 - 1
    j = args.atom2 - 1
    if i < 0 or i >= len(coords) or j < 0 or j >= len(coords):
        parser.error("--atom1/--atom2 are 1-based and must be inside the XYZ atom range")
    if i == j and not args.no_rotate:
        parser.error("--atom1 and --atom2 must be different")

    origin = resolve_origin(coords, i, args.origin)
    vector = coords[j] - coords[i]
    matrix = np.eye(3) if args.no_rotate else rotation_matrix(vector, AXES[args.axis])
    rotated = rotate_coords(coords, matrix, origin)
    suffix = "centered" if args.no_rotate else "rot"
    output = args.output or args.xyz.with_name(f"{args.xyz.stem}_{suffix}.xyz")
    matrix_out = args.matrix_out or args.xyz.with_name(f"{args.xyz.stem}_rotation_matrix.txt")
    action = "Centered" if args.no_rotate else "Rotated"
    alignment = (
        "no rotation" if args.no_rotate else f"atom{args.atom1}->{args.atom2} to {args.axis}"
    )
    comment = (
        f"{action} from {args.xyz.name}; origin={args.origin}; "
        f"{alignment}; old_comment={old_comment}"
    )
    write_xyz(output, symbols, rotated, comment, extras)
    write_rotation_matrix(matrix_out, matrix, origin)

    pc_count = None
    pc_out = args.pointcharges_out
    if args.pointcharges is not None:
        pc_out = pc_out or args.pointcharges.with_name(f"{args.pointcharges.stem}_rot.dat")
        pc_count = rotate_pointcharges(args.pointcharges, pc_out, matrix, origin)

    print(f"Wrote XYZ: {output}")
    print(f"Wrote rotation matrix: {matrix_out}")
    if pc_count is not None:
        print(f"Wrote rotated point charges: {pc_out} ({pc_count} rows)")
    print(f"Origin mode: {args.origin}")
    print(f"Origin removed: {origin[0]:.8f} {origin[1]:.8f} {origin[2]:.8f}")
    if not args.no_rotate:
        rotated_vector = matrix @ vector
        print(
            f"Aligned atom {args.atom1} ({symbols[i]}) -> "
            f"atom {args.atom2} ({symbols[j]}) to {args.axis}"
        )
        print(
            "Rotated vector: "
            f"{rotated_vector[0]:.8f} {rotated_vector[1]:.8f} {rotated_vector[2]:.8f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
