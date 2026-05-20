"""Convert LAMMPS dump trajectories into extxyz files for MD/PDF workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomi.lammps.rdf_pdf import read_frames_from_dump, write_selected_frames


def parse_type_map(items: list[str] | None) -> dict[int, str]:
    type_map: dict[int, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected type map item like 1=U, got {item!r}")
        key, value = item.split("=", 1)
        type_map[int(key.strip())] = value.strip()
    return type_map


def load_type_map_json(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("lammps_type_map") or payload.get("type_map") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a lammps_type_map object.")
    out: dict[int, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and key.strip().lstrip("+-").isdigit():
            out[int(key)] = str(value)
        else:
            out[int(value)] = str(key)
    return out


def merge_type_maps(json_map: dict[int, str], explicit_map: dict[int, str]) -> dict[int, str]:
    merged = dict(json_map)
    merged.update(explicit_map)
    return merged


def convert_lammps_dump_to_extxyz(
    dump: Path,
    type_map: dict[int, str],
    dt: float,
    dump_every: int,
    window_ps: float | None,
    outprefix: Path,
    dump_format: str = "lammps-dump-text",
) -> dict[str, object]:
    if not type_map:
        raise ValueError("A LAMMPS type map is required. Use --type-map 1=U 2=O or --type-map-json.")
    frames, summary = read_frames_from_dump(
        dump=dump,
        dump_format=dump_format,
        type_map=type_map,
        dt_ps=dt,
        dump_every=dump_every,
        window_ps=window_ps,
    )
    outdir = outprefix.parent if str(outprefix.parent) else Path(".")
    prefix = outprefix.name
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = write_selected_frames(outdir, prefix, frames)
    summary.update(
        {
            "schema": "atomi.lammps.dump_extxyz.v1",
            "outputs": outputs,
            "notes": [
                "Use *_lastwindow.extxyz for RDF/PDF/S(Q) windowed workflows.",
                "Use *_lastframe.extxyz for a single final snapshot.",
                "Use *_avgframe.extxyz mainly for visualization/reference.",
                "Full pdf_lammps/pdf_lammps_series analysis already writes these selected extxyz files by default.",
            ],
        }
    )
    summary_path = outdir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lammps2extxyz",
        description="Convert the last window of a LAMMPS dump trajectory to extxyz for MD/PDF/ASE analysis.",
    )
    parser.add_argument("--dump", required=True, type=Path, help="LAMMPS dump/trajectory file.")
    parser.add_argument("--dump-format", default="lammps-dump-text", help="ASE dump format. Default: lammps-dump-text.")
    parser.add_argument("--type-map", nargs="+", help="LAMMPS type to element map, e.g. 1=U 2=O.")
    parser.add_argument(
        "--type-map-json",
        type=Path,
        help="JSON sidecar from poscar2lammps containing lammps_type_map.",
    )
    parser.add_argument("--dt", type=float, required=True, help="MD timestep in ps, e.g. 0.0001.")
    parser.add_argument("--dump-every", type=int, required=True, help="MD steps between dump frames.")
    parser.add_argument("--window-ps", type=float, default=5.0, help="Select the last this many ps. Default: 5.")
    parser.add_argument("--all-frames", action="store_true", help="Ignore --window-ps and write the whole trajectory.")
    parser.add_argument("--outprefix", required=True, type=Path, help="Output prefix, e.g. uo2_1500K.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    json_map = load_type_map_json(args.type_map_json)
    explicit_map = parse_type_map(args.type_map)
    type_map = merge_type_maps(json_map, explicit_map)
    summary = convert_lammps_dump_to_extxyz(
        dump=args.dump,
        type_map=type_map,
        dt=args.dt,
        dump_every=args.dump_every,
        window_ps=None if args.all_frames else args.window_ps,
        outprefix=args.outprefix,
        dump_format=args.dump_format,
    )
    print(f"Total frames read         : {summary['n_total_frames']}")
    print(f"Frame spacing (ps)        : {summary['dt_frame_ps']:.6f}")
    print(f"Total time available (ps) : {summary['total_time_ps_available']:.6f}")
    print(f"Selected frames           : {summary['n_selected_frames']}")
    print(f"Selected window used (ps) : {summary['window_ps_used']:.6f}")
    for label, path in summary["outputs"].items():
        print(f"{label:<24}: {path}")
    print(f"summary_json             : {summary['summary_json']}")


if __name__ == "__main__":  # pragma: no cover
    main()
