"""Summarize metal-ligand exchange coordinates from CP2K AIMD trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from atomi.cp2k.aimd_common import (
    distance,
    iter_xyz_frames,
    mean,
    parse_cp2k_input,
    parse_step_from_comment,
    tail_values,
)


def _parse_cutoffs(values: list[str]) -> dict[str, float]:
    cutoffs = {"Cl": 2.8, "O": 2.5}
    for value in values:
        for item in value.split(","):
            if not item.strip():
                continue
            key, raw = item.split("=", 1)
            cutoffs[key.strip()] = float(raw)
    return cutoffs


def _tail_stats(values: list[float], fraction: float) -> dict[str, float | None]:
    tail = tail_values(values, fraction)
    return {
        "mean_all": mean(values),
        "mean_tail": mean(tail),
        "min_tail": min(tail) if tail else None,
        "max_tail": max(tail) if tail else None,
    }


def _nearest_h_distances(
    symbols: list[str],
    coords: list[tuple[float, float, float]],
    atom_index: int,
    cell_abc_A: list[float] | None,
    n: int = 2,
) -> list[float]:
    atom = coords[atom_index - 1]
    distances = [
        distance(atom, xyz, cell_abc_A)
        for idx, (sym, xyz) in enumerate(zip(symbols, coords), start=1)
        if idx != atom_index and sym == "H"
    ]
    return sorted(distances)[:n]


def summarize_ligand_exchange(
    trajectory: Path,
    *,
    input_file: Path | None = None,
    metal_index: int = 1,
    leaving_index: int | None = None,
    entering_index: int | None = None,
    ligand_elements: set[str] | None = None,
    coordination_cutoffs: dict[str, float] | None = None,
    tail_fraction: float = 0.2,
    cell_abc_A: list[float] | None = None,
) -> dict[str, object]:
    input_info = parse_cp2k_input(input_file) if input_file else {}
    if cell_abc_A is None:
        cell = input_info.get("cell_abc_A")
        cell_abc_A = list(cell) if isinstance(cell, list) else None
    if leaving_index is None or entering_index is None:
        colvars = input_info.get("colvars")
        if isinstance(colvars, list):
            atoms = [item.get("atoms") for item in colvars if isinstance(item, dict)]
            pairs = [item for item in atoms if isinstance(item, list) and len(item) >= 2]
            if leaving_index is None and pairs:
                leaving_index = int(pairs[0][1])
            if entering_index is None and len(pairs) > 1:
                entering_index = int(pairs[1][1])
    coordination_cutoffs = coordination_cutoffs or {"Cl": 2.8, "O": 2.5}
    ligand_elements = ligand_elements or {"Cl", "O"}

    leaving_series: list[float] = []
    entering_series: list[float] = []
    steps: list[int] = []
    last_coordination: dict[str, int] = {}
    last_nearest: dict[str, float] = {}
    entering_oh_tail: list[list[float]] = []
    leaving_h_tail: list[float] = []
    frame_count = 0
    atom_count = 0
    last_symbols: list[str] = []

    for frame_idx, (comment, symbols, coords) in enumerate(iter_xyz_frames(trajectory), start=1):
        frame_count += 1
        atom_count = len(symbols)
        last_symbols = symbols
        steps.append(parse_step_from_comment(comment, frame_idx))
        metal = coords[metal_index - 1]
        if leaving_index is not None:
            leaving_series.append(distance(metal, coords[leaving_index - 1], cell_abc_A))
        if entering_index is not None:
            entering_series.append(distance(metal, coords[entering_index - 1], cell_abc_A))

        counts = {element: 0 for element in coordination_cutoffs}
        nearest = {element: math.inf for element in coordination_cutoffs}
        for idx, (sym, xyz) in enumerate(zip(symbols, coords), start=1):
            if idx == metal_index or sym not in ligand_elements:
                continue
            d = distance(metal, xyz, cell_abc_A)
            if sym in coordination_cutoffs and d <= coordination_cutoffs[sym]:
                counts[sym] += 1
            if sym in nearest:
                nearest[sym] = min(nearest[sym], d)
        last_coordination = counts
        last_nearest = {key: value for key, value in nearest.items() if math.isfinite(value)}

        if entering_index is not None:
            entering_oh_tail.append(_nearest_h_distances(symbols, coords, entering_index, cell_abc_A))
        if leaving_index is not None:
            cl_h = _nearest_h_distances(symbols, coords, leaving_index, cell_abc_A, n=1)
            if cl_h:
                leaving_h_tail.append(cl_h[0])

    leaving_stats = _tail_stats(leaving_series, tail_fraction)
    entering_stats = _tail_stats(entering_series, tail_fraction)
    tail_oh = tail_values(entering_oh_tail, tail_fraction)
    tail_clh = tail_values(leaving_h_tail, tail_fraction)
    oh1 = mean([item[0] for item in tail_oh if len(item) >= 1])
    oh2 = mean([item[1] for item in tail_oh if len(item) >= 2])
    clh = mean(tail_clh)

    product_label = "unknown"
    leaving_tail = leaving_stats.get("mean_tail")
    entering_tail = entering_stats.get("mean_tail")
    if isinstance(leaving_tail, float) and isinstance(entering_tail, float):
        if leaving_tail >= 3.0 and entering_tail <= 2.3:
            product_label = "water-bound/chloride-dissociated"
        elif leaving_tail <= 2.6 and entering_tail >= 2.8:
            product_label = "chloride-bound/water-separated"
        else:
            product_label = "intermediate ligand-exchange window"
    entering_identity = "unknown"
    if isinstance(oh1, float) and isinstance(oh2, float):
        entering_identity = "water-like" if oh1 < 1.25 and oh2 < 1.25 else "partly deprotonated/ambiguous"
    elif isinstance(oh1, float):
        entering_identity = "hydroxide-like" if oh1 < 1.25 else "unbound/ambiguous"

    return {
        "trajectory": str(trajectory),
        "input_file": str(input_file) if input_file else None,
        "frame_count": frame_count,
        "atom_count": atom_count,
        "composition": {sym: last_symbols.count(sym) for sym in sorted(set(last_symbols))},
        "metal_index": metal_index,
        "leaving_index": leaving_index,
        "entering_index": entering_index,
        "tail_fraction": tail_fraction,
        "leaving_distance_A": leaving_stats,
        "entering_distance_A": entering_stats,
        "last_coordination_counts": last_coordination,
        "last_nearest_distances_A": last_nearest,
        "entering_OH_tail_mean_A": {"nearest_H1": oh1, "nearest_H2": oh2},
        "leaving_ClH_tail_mean_A": clh,
        "entering_ligand_identity": entering_identity,
        "product_state_label": product_label,
        "steps": {"first": steps[0] if steps else None, "last": steps[-1] if steps else None},
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "trajectory",
        "frame_count",
        "metal_index",
        "leaving_index",
        "entering_index",
        "leaving_tail_mean_A",
        "entering_tail_mean_A",
        "product_state_label",
        "entering_ligand_identity",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "trajectory": row["trajectory"],
                    "frame_count": row["frame_count"],
                    "metal_index": row["metal_index"],
                    "leaving_index": row["leaving_index"],
                    "entering_index": row["entering_index"],
                    "leaving_tail_mean_A": (row["leaving_distance_A"] or {}).get("mean_tail"),  # type: ignore[union-attr]
                    "entering_tail_mean_A": (row["entering_distance_A"] or {}).get("mean_tail"),  # type: ignore[union-attr]
                    "product_state_label": row["product_state_label"],
                    "entering_ligand_identity": row["entering_ligand_identity"],
                }
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="cp2k-ligand-exchange-summary",
        description="Summarize CP2K AIMD metal-ligand exchange and product-state checks.",
    )
    parser.add_argument("trajectory", type=Path, nargs="+", help="CP2K multi-frame *-pos.xyz files.")
    parser.add_argument("--inp", type=Path, help="CP2K input file for COLVAR/cell metadata.")
    parser.add_argument("--metal-index", type=int, default=1)
    parser.add_argument("--leaving-index", type=int, help="1-based leaving ligand atom index.")
    parser.add_argument("--entering-index", type=int, help="1-based entering ligand atom index.")
    parser.add_argument("--ligand-elements", default="Cl,O", help="Comma-separated elements for coordination counts.")
    parser.add_argument("--coordination-cutoff", action="append", default=[], help="Element=cutoff_A, e.g. Cl=2.8,O=2.5")
    parser.add_argument("--tail-fraction", type=float, default=0.2)
    parser.add_argument("--box", type=float, help="Use cubic minimum-image box length in Angstrom.")
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    ligand_elements = {item.strip() for item in args.ligand_elements.split(",") if item.strip()}
    cell = [args.box, args.box, args.box] if args.box else None
    rows = [
        summarize_ligand_exchange(
            trajectory,
            input_file=args.inp,
            metal_index=args.metal_index,
            leaving_index=args.leaving_index,
            entering_index=args.entering_index,
            ligand_elements=ligand_elements,
            coordination_cutoffs=_parse_cutoffs(args.coordination_cutoff),
            tail_fraction=args.tail_fraction,
            cell_abc_A=cell,
        )
        for trajectory in args.trajectory
    ]

    for row in rows:
        print("CP2K ligand-exchange summary")
        print("----------------------------")
        print(f"trajectory   : {row['trajectory']}")
        print(f"frames       : {row['frame_count']}")
        print(f"leaving idx  : {row['leaving_index']}  stats={row['leaving_distance_A']}")
        print(f"entering idx : {row['entering_index']}  stats={row['entering_distance_A']}")
        print(f"coordination : {row['last_coordination_counts']}")
        print(f"product      : {row['product_state_label']}")
        print(f"entering     : {row['entering_ligand_identity']}")
    if args.summary_csv:
        write_summary_csv(args.summary_csv, rows)
        print(f"Wrote summary CSV: {args.summary_csv}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
