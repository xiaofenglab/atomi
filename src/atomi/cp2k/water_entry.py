"""Find water-entry CP2K seed frames and write two-CV restart inputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


METALS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
}

STEP_PATTERNS = (
    re.compile(r"(?:STEP|Step|step)\s*[=:]?\s*(\d+)"),
    re.compile(r"\bi\s*=\s*(\d+)\b"),
    re.compile(r"(?:MD|md)\s*(?:step)?\s*[=:]?\s*(\d+)"),
)


@dataclass(frozen=True)
class XyzFrame:
    index: int
    comment: str
    symbols: list[str]
    coords: np.ndarray


@dataclass(frozen=True)
class Candidate:
    frame: XyzFrame
    step: int
    time_ps: float
    oxygen_index: int
    hydrogen_indices: tuple[int, int]
    metal_ligand_distance: float
    metal_water_distance: float
    angle_deg: float
    score: float


def read_xyz_trajectory(path: Path) -> list[XyzFrame]:
    frames: list[XyzFrame] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            first = first.strip()
            if not first:
                continue
            try:
                natoms = int(first)
            except ValueError as exc:
                raise ValueError(f"Malformed XYZ atom-count line in {path}: {first!r}") from exc
            comment = handle.readline().rstrip("\n")
            symbols: list[str] = []
            coords: list[list[float]] = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ atom line in {path}")
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append(
                XyzFrame(
                    index=len(frames),
                    comment=comment,
                    symbols=symbols,
                    coords=np.array(coords, dtype=float),
                )
            )
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}")
    return frames


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, coord in zip(symbols, coords):
            handle.write(f"{symbol:2s}  {coord[0]: .8f}  {coord[1]: .8f}  {coord[2]: .8f}\n")


def parse_step_from_comment(comment: str) -> int | None:
    for pattern in STEP_PATTERNS:
        match = pattern.search(comment)
        if match:
            return int(match.group(1))
    return None


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_angle = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def find_first_metal(symbols: list[str], requested_index: int | None = None) -> int:
    if requested_index is not None:
        index = requested_index - 1
        if index < 0 or index >= len(symbols):
            raise ValueError("Requested metal index is out of range")
        return index
    for index, symbol in enumerate(symbols):
        if symbol in METALS:
            return index
    raise ValueError("No metal atom was found automatically; pass --metal-index")


def build_water_groups(
    symbols: list[str],
    coords: np.ndarray,
    oh_cutoff: float = 1.25,
) -> list[tuple[int, int, int]]:
    oxygen_indices = [index for index, symbol in enumerate(symbols) if symbol == "O"]
    hydrogen_indices = [index for index, symbol in enumerate(symbols) if symbol == "H"]
    assigned_hydrogens: set[int] = set()
    waters: list[tuple[int, int, int]] = []
    for oxygen_index in oxygen_indices:
        candidates = sorted(
            (
                (distance(coords[oxygen_index], coords[hydrogen_index]), hydrogen_index)
                for hydrogen_index in hydrogen_indices
                if distance(coords[oxygen_index], coords[hydrogen_index]) <= oh_cutoff
            ),
            key=lambda item: item[0],
        )
        chosen: list[int] = []
        for _, hydrogen_index in candidates:
            if hydrogen_index not in assigned_hydrogens:
                chosen.append(hydrogen_index)
            if len(chosen) == 2:
                break
        if len(chosen) == 2:
            assigned_hydrogens.update(chosen)
            waters.append((oxygen_index, chosen[0], chosen[1]))
    return waters


def parse_cp2k_input(inp_path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "project": None,
        "timestep_fs": None,
        "md_steps": None,
        "temperature": None,
        "colvar_atoms": [],
        "target_angstrom": None,
        "k_kcalmol": None,
        "traj_filename": None,
    }
    section_stack: list[str] = []
    for raw in inp_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("&END"):
            if section_stack:
                section_stack.pop()
            continue
        if upper.startswith("&"):
            section_stack.append(upper[1:].split()[0])
            continue
        if section_stack and section_stack[-1] == "GLOBAL":
            match = re.match(r"PROJECT\s+(.+)", line, re.IGNORECASE)
            if match:
                info["project"] = match.group(1).strip()
        if "MD" in section_stack:
            match = re.match(r"TIMESTEP\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["timestep_fs"] = float(match.group(1))
            match = re.match(r"STEPS\s+([0-9]+)", line, re.IGNORECASE)
            if match:
                info["md_steps"] = int(match.group(1))
            match = re.match(r"TEMPERATURE\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["temperature"] = float(match.group(1))
        if "DISTANCE" in section_stack:
            match = re.match(r"ATOMS\s+([0-9]+)\s+([0-9]+)", line, re.IGNORECASE)
            if match:
                colvar_atoms = info["colvar_atoms"]
                assert isinstance(colvar_atoms, list)
                colvar_atoms.append((int(match.group(1)), int(match.group(2))))
        if "COLLECTIVE" in section_stack:
            match = re.match(r"TARGET(?:\s+\[.*?\])?\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["target_angstrom"] = float(match.group(1))
        if "RESTRAINT" in section_stack:
            match = re.match(r"K(?:\s+\[.*?\])?\s+([0-9Ee+\-.]+)", line, re.IGNORECASE)
            if match:
                info["k_kcalmol"] = float(match.group(1))
        if "TRAJECTORY" in section_stack:
            match = re.match(r"FILENAME\s*=?\s*(.+)", line, re.IGNORECASE)
            if match:
                info["traj_filename"] = match.group(1).strip()
    return info


def infer_leaving_ligand_index(
    colvar_atoms: Iterable[tuple[int, int]],
    metal_index: int,
) -> int | None:
    for first, second in colvar_atoms:
        first_index = first - 1
        second_index = second - 1
        if first_index == metal_index:
            return second_index
        if second_index == metal_index:
            return first_index
    for _, second in colvar_atoms:
        return second - 1
    return None


def frame_time_metadata(
    frames: list[XyzFrame],
    timestep_fs: float,
) -> list[tuple[XyzFrame, int, float]]:
    parsed_steps: list[int] = []
    for fallback_index, frame in enumerate(frames, start=1):
        parsed_steps.append(parse_step_from_comment(frame.comment) or fallback_index)
    return [(frame, step, step * timestep_fs / 1000.0) for frame, step in zip(frames, parsed_steps)]


def candidate_for_frame(
    frame: XyzFrame,
    step: int,
    time_ps: float,
    metal_index: int,
    ligand_index: int,
    angle_max_deg: float,
    d_mo_min: float,
    d_mo_max: float,
    preferred_mo: float,
) -> Candidate | None:
    coords = frame.coords
    metal_pos = coords[metal_index]
    ligand_pos = coords[ligand_index]
    best: Candidate | None = None
    for oxygen_index, hydrogen_1, hydrogen_2 in build_water_groups(frame.symbols, coords):
        metal_water_distance = distance(metal_pos, coords[oxygen_index])
        angle = angle_degrees(ligand_pos - metal_pos, coords[oxygen_index] - metal_pos)
        if not d_mo_min <= metal_water_distance <= d_mo_max:
            continue
        if angle > angle_max_deg:
            continue
        score = abs(metal_water_distance - preferred_mo) + 0.015 * angle
        candidate = Candidate(
            frame=frame,
            step=step,
            time_ps=time_ps,
            oxygen_index=oxygen_index,
            hydrogen_indices=(hydrogen_1, hydrogen_2),
            metal_ligand_distance=distance(metal_pos, ligand_pos),
            metal_water_distance=metal_water_distance,
            angle_deg=angle,
            score=score,
        )
        if best is None or candidate.score < best.score:
            best = candidate
    return best


def find_water_entry_candidates(
    frames: list[XyzFrame],
    timestep_fs: float,
    metal_index: int,
    ligand_index: int,
    last_ps: float,
    max_candidates: int,
    min_frame_gap: int,
    angle_max_deg: float,
    d_mo_min: float,
    d_mo_max: float,
    preferred_mo: float,
) -> list[Candidate]:
    timed_frames = frame_time_metadata(frames, timestep_fs)
    total_ps = timed_frames[-1][2]
    start_ps = max(0.0, total_ps - last_ps)
    candidates: list[Candidate] = []
    for frame, step, time_ps in timed_frames:
        if time_ps < start_ps:
            continue
        candidate = candidate_for_frame(
            frame=frame,
            step=step,
            time_ps=time_ps,
            metal_index=metal_index,
            ligand_index=ligand_index,
            angle_max_deg=angle_max_deg,
            d_mo_min=d_mo_min,
            d_mo_max=d_mo_max,
            preferred_mo=preferred_mo,
        )
        if candidate is not None:
            candidates.append(candidate)

    selected: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.score, -item.time_ps)):
        if all(abs(candidate.frame.index - kept.frame.index) >= min_frame_gap for kept in selected):
            selected.append(candidate)
        if len(selected) >= max_candidates:
            break
    return sorted(selected, key=lambda item: item.time_ps)


def remove_block(text: str, block_name: str) -> str:
    return re.sub(
        rf"(?ms)^\s*&{re.escape(block_name)}\b.*?^\s*&END\s+{re.escape(block_name)}\s*\n?",
        "",
        text,
    )


def insert_colvars(text: str, colvar_block: str) -> str:
    if re.search(r"(?m)^\s*&KIND\s+", text):
        return re.sub(r"(?m)^(\s*&KIND\s+)", colvar_block + r"\1", text, count=1)
    if re.search(r"(?m)^\s*&END\s+SUBSYS\s*$", text):
        return re.sub(r"(?m)^(\s*&END\s+SUBSYS\s*$)", colvar_block + r"\1", text, count=1)
    return text.rstrip() + "\n\n" + colvar_block


def replace_or_insert_constraint(text: str, constraint_block: str) -> str:
    if re.search(r"(?m)^\s*&CONSTRAINT\b", text):
        return re.sub(
            r"(?ms)^\s*&CONSTRAINT\b.*?^\s*&END\s+CONSTRAINT\s*\n?",
            constraint_block,
            text,
            count=1,
        )
    if re.search(r"(?m)^\s*&MOTION\b", text):
        return re.sub(r"(?m)^(\s*&MOTION\b.*\n)", r"\1" + constraint_block, text, count=1)
    return text.rstrip() + "\n\n" + constraint_block


def patch_template_input(
    template_text: str,
    project_name: str,
    coord_file: str,
    metal_index_1based: int,
    ligand_index_1based: int,
    water_oxygen_index_1based: int,
    ligand_target: float,
    water_target: float,
    ligand_k: float,
    water_k: float,
    fresh_start: bool = True,
) -> str:
    text = re.sub(
        r"(^\s*PROJECT\s+)(\S+)",
        rf"\1{project_name}",
        template_text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"(^\s*COORD_FILE_NAME\s+)(\S+)",
        rf"\1{coord_file}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(r"(^\s*SCF_GUESS\s+)RESTART", r"\1ATOMIC", text, flags=re.MULTILINE)
    if fresh_start:
        text = remove_block(text, "EXT_RESTART")

    lagrange_match = re.search(
        r"(?ms)^\s*&LAGRANGE_MULTIPLIERS\b.*?^\s*&END\s+LAGRANGE_MULTIPLIERS\s*",
        text,
    )
    lagrange_block = f"{lagrange_match.group(0)}\n" if lagrange_match else ""
    text = remove_block(text, "LAGRANGE_MULTIPLIERS")
    text = remove_block(text, "COLVAR")

    colvar_block = (
        "    &COLVAR\n"
        "      &DISTANCE\n"
        f"        ATOMS {metal_index_1based} {ligand_index_1based}\n"
        "      &END DISTANCE\n"
        "    &END COLVAR\n\n"
        "    &COLVAR\n"
        "      &DISTANCE\n"
        f"        ATOMS {metal_index_1based} {water_oxygen_index_1based}\n"
        "      &END DISTANCE\n"
        "    &END COLVAR\n"
    )
    text = insert_colvars(text, colvar_block)

    constraint_block = (
        "  &CONSTRAINT\n"
        "    CONSTRAINT_INIT T\n"
        "    &COLLECTIVE\n"
        "      COLVAR 1\n"
        "      INTERMOLECULAR T\n"
        f"      TARGET [angstrom] {ligand_target:.3f}\n"
        "      &RESTRAINT\n"
        f"        K [kcalmol] {ligand_k:.1f}\n"
        "      &END RESTRAINT\n"
        "    &END COLLECTIVE\n\n"
        "    &COLLECTIVE\n"
        "      COLVAR 2\n"
        "      INTERMOLECULAR T\n"
        f"      TARGET [angstrom] {water_target:.3f}\n"
        "      &RESTRAINT\n"
        f"        K [kcalmol] {water_k:.1f}\n"
        "      &END RESTRAINT\n"
        "    &END COLLECTIVE\n\n"
        f"{lagrange_block}"
        "  &END CONSTRAINT\n"
    )
    return replace_or_insert_constraint(text, constraint_block)


def write_candidate_outputs(
    candidates: list[Candidate],
    outdir: Path,
    template_text: str,
    metal_index: int,
    ligand_index: int,
    ligand_target: float,
    water_target: float,
    ligand_k: float,
    water_k: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "water_entry_candidates.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tag",
                "frame",
                "step",
                "time_ps",
                "metal_index",
                "ligand_index",
                "water_oxygen_index",
                "water_hydrogen_indices",
                "metal_ligand_distance",
                "metal_water_distance",
                "angle_deg",
                "score",
                "xyz",
                "labeled_xyz",
                "inp",
            ]
        )
        for rank, candidate in enumerate(candidates, start=1):
            frame = candidate.frame
            tag = f"cand_{rank:02d}_f{frame.index + 1}_step{candidate.step}"
            frame_xyz = outdir / f"{tag}.xyz"
            labeled_xyz = outdir / f"{tag}_labeled.xyz"
            report_txt = outdir / f"{tag}.txt"
            input_path = outdir / f"{tag}_water_entry.inp"
            write_xyz(
                frame_xyz,
                frame.symbols,
                frame.coords,
                (
                    f"frame={frame.index + 1} step={candidate.step} "
                    f"t_ps={candidate.time_ps:.4f} | {frame.comment}"
                ),
            )
            h1, h2 = candidate.hydrogen_indices
            write_xyz(
                labeled_xyz,
                frame.symbols,
                frame.coords,
                (
                    f"METAL={metal_index + 1} LIGAND={ligand_index + 1} "
                    f"Owater={candidate.oxygen_index + 1} Hs={h1 + 1},{h2 + 1} "
                    f"M-O={candidate.metal_water_distance:.4f} angle={candidate.angle_deg:.2f}"
                ),
            )
            report_txt.write_text(
                "\n".join(
                    [
                        f"frame_index_1based = {frame.index + 1}",
                        f"md_step = {candidate.step}",
                        f"time_ps = {candidate.time_ps:.6f}",
                        f"metal_index_1based = {metal_index + 1}",
                        f"ligand_index_1based = {ligand_index + 1}",
                        f"candidate_water_O_index_1based = {candidate.oxygen_index + 1}",
                        f"candidate_water_H_indices_1based = {h1 + 1},{h2 + 1}",
                        f"metal_ligand_distance = {candidate.metal_ligand_distance:.6f}",
                        f"metal_water_distance = {candidate.metal_water_distance:.6f}",
                        f"angle_metal_ligand_metal_water_deg = {candidate.angle_deg:.6f}",
                        f"selection_score = {candidate.score:.6f}",
                        f"comment = {frame.comment}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            patched = patch_template_input(
                template_text=template_text,
                project_name=f"{tag}_water_entry",
                coord_file=frame_xyz.name,
                metal_index_1based=metal_index + 1,
                ligand_index_1based=ligand_index + 1,
                water_oxygen_index_1based=candidate.oxygen_index + 1,
                ligand_target=ligand_target,
                water_target=water_target,
                ligand_k=ligand_k,
                water_k=water_k,
                fresh_start=True,
            )
            input_path.write_text(patched, encoding="utf-8")
            writer.writerow(
                [
                    tag,
                    frame.index + 1,
                    candidate.step,
                    f"{candidate.time_ps:.6f}",
                    metal_index + 1,
                    ligand_index + 1,
                    candidate.oxygen_index + 1,
                    f"{h1 + 1},{h2 + 1}",
                    f"{candidate.metal_ligand_distance:.6f}",
                    f"{candidate.metal_water_distance:.6f}",
                    f"{candidate.angle_deg:.6f}",
                    f"{candidate.score:.6f}",
                    frame_xyz.name,
                    labeled_xyz.name,
                    input_path.name,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find water-entry frames near a dissociating ligand and write CP2K inputs "
            "with metal-ligand and metal-water collective variables."
        )
    )
    parser.add_argument(
        "xyz",
        type=Path,
        help="Multi-frame CP2K trajectory XYZ, usually *-pos.xyz.",
    )
    parser.add_argument("--inp", type=Path, required=True, help="Template CP2K input file.")
    parser.add_argument("--outdir", type=Path, default=Path("water_entry_candidates"))
    parser.add_argument("--metal-index", type=int, help="1-based metal atom index.")
    parser.add_argument("--ligand-index", type=int, help="1-based leaving ligand index.")
    parser.add_argument(
        "--clstar-index",
        type=int,
        help="Compatibility alias for --ligand-index for chloride dissociation workflows.",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        help="MD timestep in fs. Defaults to TIMESTEP read from the CP2K input.",
    )
    parser.add_argument("--last-ps", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--min-frame-gap", type=int, default=30)
    parser.add_argument("--angle-max-deg", type=float, default=60.0)
    parser.add_argument("--d-mo-min", type=float, default=2.4)
    parser.add_argument("--d-mo-max", type=float, default=4.2)
    parser.add_argument("--preferred-mo", type=float, default=2.8)
    parser.add_argument("--cl-target", type=float, default=2.80)
    parser.add_argument("--water-target", type=float, default=2.80)
    parser.add_argument("--cl-k", type=float, default=50.0)
    parser.add_argument("--water-k", type=float, default=15.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ligand_arg = args.ligand_index if args.ligand_index is not None else args.clstar_index
    frames = read_xyz_trajectory(args.xyz)
    info = parse_cp2k_input(args.inp)
    timestep_fs = args.timestep_fs or info["timestep_fs"]
    if timestep_fs is None:
        raise ValueError("Could not read TIMESTEP from input; pass --timestep-fs")
    first_frame = frames[0]
    metal_index = find_first_metal(first_frame.symbols, args.metal_index)
    colvar_atoms = info["colvar_atoms"]
    assert isinstance(colvar_atoms, list)
    ligand_index = (
        ligand_arg - 1
        if ligand_arg is not None
        else infer_leaving_ligand_index(colvar_atoms, metal_index)
    )
    if ligand_index is None:
        raise ValueError("Could not infer leaving ligand from input; pass --ligand-index")
    if ligand_index < 0 or ligand_index >= len(first_frame.symbols):
        raise ValueError("Leaving ligand index is out of range")

    candidates = find_water_entry_candidates(
        frames=frames,
        timestep_fs=float(timestep_fs),
        metal_index=metal_index,
        ligand_index=ligand_index,
        last_ps=args.last_ps,
        max_candidates=args.max_candidates,
        min_frame_gap=args.min_frame_gap,
        angle_max_deg=args.angle_max_deg,
        d_mo_min=args.d_mo_min,
        d_mo_max=args.d_mo_max,
        preferred_mo=args.preferred_mo,
    )
    write_candidate_outputs(
        candidates=candidates,
        outdir=args.outdir,
        template_text=args.inp.read_text(encoding="utf-8"),
        metal_index=metal_index,
        ligand_index=ligand_index,
        ligand_target=args.cl_target,
        water_target=args.water_target,
        ligand_k=args.cl_k,
        water_k=args.water_k,
    )
    print(f"Scanned {len(frames)} frame(s); selected {len(candidates)} water-entry candidate(s).")
    print(f"Wrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
