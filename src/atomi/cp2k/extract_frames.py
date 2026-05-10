"""Extract CP2K AIMD frames into QM clusters, embedding atoms, and point charges."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from atomi.cp2k.acid_box import METALS

STEP_PATTERNS = [
    re.compile(r"(?:STEP|Step|step)\s*[=:]?\s*(\d+)"),
    re.compile(r"i\s*=\s*(\d+)"),
    re.compile(r"(?:MD|md)\s*(?:step)?\s*[=:]?\s*(\d+)"),
]


def read_xyz_trajectory(path):
    frames = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            n = int(line)
            comment = f.readline().rstrip("\n")
            symbols = []
            coords = []
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ line in {path}")
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            frames.append((comment, symbols, np.array(coords, dtype=float)))
    return frames


def write_xyz(path, symbols, coords, comment="cluster"):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{comment}\n")
        for s, r in zip(symbols, coords):
            f.write(f"{s:2s}  {r[0]: .8f}  {r[1]: .8f}  {r[2]: .8f}\n")


def dist(a, b):
    return float(np.linalg.norm(a - b))


def find_first_metal(symbols, metal_symbol=None):
    if metal_symbol is not None:
        for i, s in enumerate(symbols):
            if s == metal_symbol:
                return i, s
        raise RuntimeError(f"Requested metal '{metal_symbol}' not found.")
    for i, s in enumerate(symbols):
        if s in METALS:
            return i, s
    raise RuntimeError("No metal atom found.")


def build_water_like_groups(symbols, coords, oh_cut=1.25):
    oxygen_indices = [i for i, s in enumerate(symbols) if s == "O"]
    hydrogen_indices = [i for i, s in enumerate(symbols) if s == "H"]

    atom_to_group = {}
    groups = {}
    gid = 0
    assigned_h = set()

    for io in oxygen_indices:
        nearby = []
        for ih in hydrogen_indices:
            d = dist(coords[io], coords[ih])
            if d < oh_cut:
                nearby.append((d, ih))
        nearby.sort()

        hlist = []
        for _, ih in nearby:
            if ih not in assigned_h:
                hlist.append(ih)
            if len(hlist) == 3:
                break

        if len(hlist) in (2, 3):
            for ih in hlist:
                assigned_h.add(ih)
            gtype = "H2O" if len(hlist) == 2 else "H3O"
            atoms = [io] + hlist
            groups[gid] = {
                "type": gtype,
                "atom_indices": atoms,
                "oxygen_index": io,
            }
            for ia in atoms:
                atom_to_group[ia] = gid
            gid += 1

    return atom_to_group, groups


def classify_singletons(symbols, atom_to_group):
    groups = {}
    gid_base = max(atom_to_group.values(), default=-1) + 1
    gid = gid_base
    for i, s in enumerate(symbols):
        if i not in atom_to_group:
            groups[gid] = {
                "type": s,
                "atom_indices": [i],
                "oxygen_index": None,
            }
            atom_to_group[i] = gid
            gid += 1
    return atom_to_group, groups


def merge_groups(groups1, groups2):
    out = {}
    out.update(groups1)
    out.update(groups2)
    return out


def apply_shell_presets(args):
    if args.system == "aqua":
        if args.shell_mode == "auto":
            args.shell_mode = "tight"
        if args.hb_cut is None:
            args.hb_cut = 3.05
        if args.ion_cut is None:
            args.ion_cut = 3.20
        if args.embed_cut is None:
            args.embed_cut = 8.0
    elif args.system == "chloro":
        if args.shell_mode == "auto":
            args.shell_mode = "soft"
        if args.hb_cut is None:
            args.hb_cut = 3.10
        if args.ion_cut is None:
            args.ion_cut = 3.30
        if args.embed_cut is None:
            args.embed_cut = 8.0

    if args.shell_mode == "tight":
        if args.ga_o_cut is None:
            args.ga_o_cut = 2.30
        if args.ga_cl_cut is None:
            args.ga_cl_cut = 2.55
        if args.generic_cut is None:
            args.generic_cut = 2.55
    elif args.shell_mode in ("soft", "custom"):
        if args.ga_o_cut is None:
            args.ga_o_cut = 2.45
        if args.ga_cl_cut is None:
            args.ga_cl_cut = 2.80
        if args.generic_cut is None:
            args.generic_cut = 2.80

    if args.hb_cut is None:
        args.hb_cut = 3.20
    if args.ion_cut is None:
        args.ion_cut = 3.50
    if args.embed_cut is None:
        args.embed_cut = 10.0

    return args


def detect_first_shell(symbols, coords, metal_idx, ga_o_cut, ga_cl_cut, generic_cut):
    shell_atoms = []
    shell_dists = []
    for i, s in enumerate(symbols):
        if i == metal_idx:
            continue
        d = dist(coords[metal_idx], coords[i])
        include = False
        if s == "O" and d <= ga_o_cut:
            include = True
        elif s == "Cl" and d <= ga_cl_cut:
            include = True
        elif s not in ("H",) and d <= generic_cut:
            include = True
        if include:
            shell_atoms.append(i)
            shell_dists.append(d)
    return shell_atoms, shell_dists


def collect_full_first_shell_groups(shell_atoms, atom_to_group, groups):
    selected_group_ids = set()
    for ia in shell_atoms:
        gid = atom_to_group[ia]
        selected_group_ids.add(gid)
    selected_atoms = set()
    for gid in selected_group_ids:
        selected_atoms.update(groups[gid]["atom_indices"])
    return selected_group_ids, selected_atoms


def add_second_shell(
    groups,
    coords,
    first_shell_group_ids,
    selected_atoms,
    hb_to_firstshell_cut,
    ion_to_firstshell_cut,
):
    firstshell_anchors = []
    for gid in first_shell_group_ids:
        for ia in groups[gid]["atom_indices"]:
            firstshell_anchors.append(ia)

    second_group_ids = set()
    for gid, g in groups.items():
        if gid in first_shell_group_ids:
            continue
        atoms = g["atom_indices"]
        gtype = g["type"]
        if gtype in ("H2O", "H3O") and g["oxygen_index"] is not None:
            rep_candidates = [g["oxygen_index"]]
            cutoff = hb_to_firstshell_cut
        else:
            rep_candidates = atoms
            cutoff = ion_to_firstshell_cut
        found = False
        for ia in rep_candidates:
            for ja in firstshell_anchors:
                d = dist(coords[ia], coords[ja])
                if d <= cutoff:
                    second_group_ids.add(gid)
                    found = True
                    break
            if found:
                break
    for gid in second_group_ids:
        selected_atoms.update(groups[gid]["atom_indices"])
    return second_group_ids, selected_atoms


def build_embedding_atoms(coords, selected_atoms, embed_cut_from_metal, metal_idx):
    embed_atoms = []
    for i in range(len(coords)):
        if i in selected_atoms:
            continue
        if dist(coords[metal_idx], coords[i]) <= embed_cut_from_metal:
            embed_atoms.append(i)
    return embed_atoms


def assign_simple_point_charge(symbol, group_type=None):
    if symbol == "Cl":
        return -1.0
    if symbol == "Na":
        return 1.0
    if symbol == "O":
        return -0.2 if group_type == "H3O" else -0.834
    if symbol == "H":
        return 0.4 if group_type == "H3O" else 0.417
    return 0.0


def summarize_first_shell(symbols, shell_atoms):
    counts = {}
    for ia in shell_atoms:
        s = symbols[ia]
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "none"
    return "".join(f"{key}{counts[key]}" for key in sorted(counts))


def process_single_frame(frame_index, comment, symbols, coords, args, root_outdir):
    metal_idx, metal_symbol = find_first_metal(symbols, args.metal)
    atom_to_group_1, water_groups = build_water_like_groups(symbols, coords, oh_cut=1.25)
    atom_to_group, singleton_groups = classify_singletons(symbols, atom_to_group_1.copy())
    groups = merge_groups(water_groups, singleton_groups)

    shell_atoms, shell_dists = detect_first_shell(
        symbols=symbols,
        coords=coords,
        metal_idx=metal_idx,
        ga_o_cut=args.ga_o_cut,
        ga_cl_cut=args.ga_cl_cut,
        generic_cut=args.generic_cut,
    )

    first_shell_group_ids, selected_atoms = collect_full_first_shell_groups(
        shell_atoms=shell_atoms,
        atom_to_group=atom_to_group,
        groups=groups,
    )
    selected_atoms.add(metal_idx)
    second_shell_group_ids, selected_atoms = add_second_shell(
        groups=groups,
        coords=coords,
        first_shell_group_ids=first_shell_group_ids,
        selected_atoms=selected_atoms,
        hb_to_firstshell_cut=args.hb_cut,
        ion_to_firstshell_cut=args.ion_cut,
    )
    embed_indices = build_embedding_atoms(
        coords=coords,
        selected_atoms=selected_atoms,
        embed_cut_from_metal=args.embed_cut,
        metal_idx=metal_idx,
    )

    qm_indices = sorted(selected_atoms)
    qm_symbols = [symbols[i] for i in qm_indices]
    qm_coords = np.array([coords[i] for i in qm_indices], dtype=float)
    metal_pos = coords[metal_idx].copy()
    qm_coords_centered = qm_coords - metal_pos

    emb_symbols = [symbols[i] for i in embed_indices]
    emb_coords = np.array([coords[i] - metal_pos for i in embed_indices], dtype=float)

    frame_dir = root_outdir / f"f{frame_index}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    qm_xyz = frame_dir / "qm.xyz"
    emb_xyz = frame_dir / "embed.xyz"
    pc_file = frame_dir / "pointcharges.dat"
    report = frame_dir / "report.txt"
    ref_xyz = frame_dir / f"frame_{frame_index}.xyz"

    write_xyz(ref_xyz, symbols, coords, comment=f"full frame={frame_index} | {comment}")
    write_xyz(
        qm_xyz,
        qm_symbols,
        qm_coords_centered,
        comment=f"QM cluster from frame={frame_index}, metal={metal_symbol}",
    )
    write_xyz(
        emb_xyz,
        emb_symbols,
        emb_coords,
        comment=f"Embedding atoms from frame={frame_index}, metal={metal_symbol}",
    )

    with open(pc_file, "w") as f:
        f.write("# x  y  z  charge  symbol  group_type  atom_index\n")
        for i in embed_indices:
            gid = atom_to_group[i]
            gtype = groups[gid]["type"]
            q = assign_simple_point_charge(symbols[i], gtype)
            x, y, z = coords[i] - metal_pos
            f.write(f"{x: .8f}  {y: .8f}  {z: .8f}  {q: .6f}  {symbols[i]:2s}  {gtype:4s}  {i}\n")

    first_shell_formula = summarize_first_shell(symbols, shell_atoms)
    avg_shell_dist = float(np.mean(shell_dists)) if shell_dists else float("nan")
    min_shell_dist = float(np.min(shell_dists)) if shell_dists else float("nan")
    max_shell_dist = float(np.max(shell_dists)) if shell_dists else float("nan")

    with open(report, "w") as f:
        f.write(f"frame = {frame_index}\n")
        f.write(f"comment = {comment}\n")
        f.write(f"metal_index = {metal_idx}\n")
        f.write(f"metal_symbol = {metal_symbol}\n")
        f.write(f"shell_mode = {args.shell_mode}\nsystem = {args.system}\n")
        f.write(f"ga_o_cut = {args.ga_o_cut}\n")
        f.write(f"ga_cl_cut = {args.ga_cl_cut}\n")
        f.write(f"generic_cut = {args.generic_cut}\n")
        f.write(
            f"hb_cut = {args.hb_cut}\nion_cut = {args.ion_cut}\nembed_cut = {args.embed_cut}\n\n"
        )
        f.write("First-shell donor atoms:\n")
        for ia, d in sorted(zip(shell_atoms, shell_dists), key=lambda x: x[1]):
            gid = atom_to_group[ia]
            f.write(f"  atom {ia:4d}  {symbols[ia]:2s}  d={d:.4f} A  group={groups[gid]['type']}\n")
        f.write("\nFirst-shell groups:\n")
        for gid in sorted(first_shell_group_ids):
            g = groups[gid]
            f.write(f"  group {gid:4d}  type={g['type']:4s}  atoms={g['atom_indices']}\n")
        f.write("\nSecond-shell groups:\n")
        for gid in sorted(second_shell_group_ids):
            g = groups[gid]
            f.write(f"  group {gid:4d}  type={g['type']:4s}  atoms={g['atom_indices']}\n")
        f.write("\nCounts:\n")
        f.write(f"  QM atoms = {len(qm_indices)}\n  Embedding atoms = {len(embed_indices)}\n")
        f.write(f"  First-shell formula = {first_shell_formula}\n")
        f.write(f"  First-shell CN = {len(shell_atoms)}\n")
        f.write(
            f"  First-shell avg distance = {avg_shell_dist:.4f}\n"
            f"  First-shell min distance = {min_shell_dist:.4f}\n"
            f"  First-shell max distance = {max_shell_dist:.4f}\n"
        )

    return {
        "frame": frame_index,
        "metal": metal_symbol,
        "first_shell": first_shell_formula,
        "first_shell_cn": len(shell_atoms),
        "avg_shell_dist": avg_shell_dist,
        "min_shell_dist": min_shell_dist,
        "max_shell_dist": max_shell_dist,
        "qm_atoms": len(qm_indices),
        "embed_atoms": len(embed_indices),
        "frame_dir": str(frame_dir),
    }


def evenly_spaced_indices(start, end, n):
    if n <= 1:
        return [start]
    vals = [int(round(x)) for x in np.linspace(start, end, n)]
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def parse_frame_selection(args, nframes):
    if args.auto_window:
        if args.discard_fraction is not None and args.discard_frames is not None:
            raise ValueError(
                "Use only one of --discard-fraction or --discard-frames in auto-window mode."
            )
        if args.discard_fraction is not None:
            if not (0.0 <= args.discard_fraction < 1.0):
                raise ValueError("--discard-fraction must be in [0,1).")
            start = int(round(nframes * args.discard_fraction))
        elif args.discard_frames is not None:
            start = int(args.discard_frames)
        else:
            start = int(round(nframes * 0.4))
        start = max(0, min(start, nframes - 1))
        end = nframes - 1
        nsnap = args.n_snapshots if args.n_snapshots is not None else 10
        frames = evenly_spaced_indices(start, end, nsnap)
        return frames, {"mode": "auto-window", "start": start, "end": end, "n_snapshots": nsnap}

    if args.frame_start is not None or args.frame_end is not None or args.frame_step is not None:
        start = 0 if args.frame_start is None else args.frame_start
        end = (nframes - 1) if args.frame_end is None else args.frame_end
        step = 1 if args.frame_step is None else args.frame_step
        if start < 0:
            start = nframes + start
        if end < 0:
            end = nframes + end
        start = max(0, start)
        end = min(nframes - 1, end)
        if step <= 0:
            raise ValueError("--frame-step must be positive.")
        frames = list(range(start, end + 1, step))
        if args.max_clusters is not None:
            frames = frames[: args.max_clusters]
        return frames, {"mode": "manual-window", "start": start, "end": end, "step": step}

    frame = args.frame
    if frame < 0:
        frame = nframes + frame
    if frame < 0 or frame >= nframes:
        raise IndexError(f"Frame {args.frame} out of range for {nframes} frames.")
    return [frame], {"mode": "single-frame", "frame": frame}


def parse_md_step(comment, fallback_frame, traj_every):
    for pat in STEP_PATTERNS:
        m = pat.search(comment)
        if m:
            return int(m.group(1)), True
    return fallback_frame * traj_every, False


def find_good_frames(frames, args):
    candidates = []
    nframes = len(frames)
    start_frame = 0
    if args.last_fraction is not None:
        if not (0.0 < args.last_fraction <= 1.0):
            raise ValueError("--last-fraction must be in (0,1].")
        start_frame = max(0, int(np.floor(nframes * (1.0 - args.last_fraction))))
    if args.last_frames is not None:
        start_frame = max(start_frame, max(0, nframes - args.last_frames))

    for iframe, (comment, symbols, coords) in enumerate(frames):
        if iframe < start_frame:
            continue
        metal_idx, _ = find_first_metal(symbols, args.metal)
        shell_atoms, shell_dists = detect_first_shell(
            symbols=symbols,
            coords=coords,
            metal_idx=metal_idx,
            ga_o_cut=args.ga_o_cut,
            ga_cl_cut=args.ga_cl_cut,
            generic_cut=args.generic_cut,
        )
        shell_formula = summarize_first_shell(symbols, shell_atoms)
        cl_dists = sorted(
            dist(coords[metal_idx], coords[i]) for i, s in enumerate(symbols) if s == "Cl"
        )
        if len(cl_dists) < 4:
            continue
        first4 = cl_dists[:4]
        d5 = cl_dists[4] if len(cl_dists) > 4 else np.nan
        spread4 = max(first4) - min(first4)
        avg4 = float(np.mean(first4))
        gap45 = float(d5 - first4[3]) if np.isfinite(d5) else 999.0

        good_formula = shell_formula == args.target_shell
        good_spread = spread4 <= args.max_spread
        good_gap = gap45 >= args.min_gap45
        good_avg = args.avg_min <= avg4 <= args.avg_max

        if good_formula and good_spread and good_gap and good_avg:
            md_step, parsed = parse_md_step(comment, iframe, args.traj_every)
            exact_restart_step = md_step % args.restart_every == 0
            if args.require_exact_restart and not exact_restart_step:
                continue
            restart_floor = (md_step // args.restart_every) * args.restart_every
            restart_ceil = (
                (md_step + args.restart_every - 1) // args.restart_every
            ) * args.restart_every
            score = (
                abs(avg4 - args.target_avg) + args.spread_weight * spread4 - args.gap_weight * gap45
            )
            candidates.append(
                {
                    "frame": iframe,
                    "comment": comment,
                    "md_step": md_step,
                    "step_from_comment": parsed,
                    "restart_floor": restart_floor,
                    "restart_ceil": restart_ceil,
                    "exact_restart_step": exact_restart_step,
                    "shell": shell_formula,
                    "avg4": avg4,
                    "spread4": spread4,
                    "gap45": gap45,
                    "d1": first4[0],
                    "d2": first4[1],
                    "d3": first4[2],
                    "d4": first4[3],
                    "d5": d5,
                    "score": score,
                }
            )

    candidates.sort(key=lambda x: x["score"])
    picked = []
    for row in candidates:
        if all(abs(row["frame"] - p["frame"]) >= args.min_frame_separation for p in picked):
            picked.append(row)
        if len(picked) >= args.top_good_frames:
            break
    return candidates, picked


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cp2k-extract-frames",
        description=(
            "Extract CP2K AIMD frames into full-frame, QM-cluster, embedding, "
            "point-charge, and report files."
        ),
    )
    ap.add_argument("trajectory")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--frame-start", type=int, default=None)
    ap.add_argument("--frame-end", type=int, default=None)
    ap.add_argument("--frame-step", type=int, default=None)
    ap.add_argument("--max-clusters", type=int, default=None)
    ap.add_argument("--auto-window", action="store_true")
    ap.add_argument("--discard-fraction", type=float, default=None)
    ap.add_argument("--discard-frames", type=int, default=None)
    ap.add_argument("--n-snapshots", type=int, default=None)
    ap.add_argument("--metal", default=None)
    ap.add_argument("--system", choices=["aqua", "chloro"], default="chloro")
    ap.add_argument("--shell-mode", choices=["auto", "tight", "soft", "custom"], default="auto")
    ap.add_argument("--ga-o-cut", type=float, default=None)
    ap.add_argument("--ga-cl-cut", type=float, default=None)
    ap.add_argument("--generic-cut", type=float, default=None)
    ap.add_argument("--hb-cut", type=float, default=None)
    ap.add_argument("--ion-cut", type=float, default=None)
    ap.add_argument("--embed-cut", type=float, default=None)
    ap.add_argument("--prefix", default="cluster")
    ap.add_argument("--outdir", type=Path, default=Path("extracted"))

    ap.add_argument(
        "--find-good-frames",
        action="store_true",
        help="Scan the trajectory and rank several stable GaCl4-like frames.",
    )
    ap.add_argument(
        "--target-shell", default="Cl4", help="Desired first-shell formula, default Cl4"
    )
    ap.add_argument("--top-good-frames", type=int, default=5, help="How many good frames to keep")
    ap.add_argument(
        "--min-frame-separation",
        type=int,
        default=100,
        help="Minimum gap in frame index between selected good frames",
    )
    ap.add_argument(
        "--target-avg",
        type=float,
        default=2.28,
        help="Target average of 4 shortest Ga-Cl distances",
    )
    ap.add_argument("--avg-min", type=float, default=2.15)
    ap.add_argument("--avg-max", type=float, default=2.45)
    ap.add_argument(
        "--max-spread", type=float, default=0.25, help="Max spread among 4 shortest Ga-Cl distances"
    )
    ap.add_argument(
        "--min-gap45",
        type=float,
        default=0.35,
        help="Minimum gap between 4th and 5th shortest Ga-Cl distances",
    )
    ap.add_argument("--spread-weight", type=float, default=2.0)
    ap.add_argument("--gap-weight", type=float, default=0.5)
    ap.add_argument(
        "--traj-every", type=int, default=10, help="Trajectory print interval in MD steps"
    )
    ap.add_argument(
        "--restart-every", type=int, default=100, help="Restart history interval in MD steps"
    )
    ap.add_argument(
        "--last-fraction",
        type=float,
        default=0.20,
        help="Only search the last fraction of the trajectory for good frames; default 0.20",
    )
    ap.add_argument(
        "--last-frames",
        type=int,
        default=None,
        help="Optional absolute override: only search the last N frames",
    )
    ap.add_argument(
        "--require-exact-restart",
        action="store_true",
        default=True,
        help="Keep only frames whose MD step exactly matches a restart-history step",
    )
    ap.add_argument(
        "--allow-nearest-restart",
        dest="require_exact_restart",
        action="store_false",
        help="Allow frames that only map to nearest lower/upper restart steps",
    )

    args = ap.parse_args(argv)
    args = apply_shell_presets(args)

    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    frames = read_xyz_trajectory(traj_path)
    if not frames:
        raise RuntimeError("No frames found in trajectory.")

    prefix_name = Path(args.prefix).name
    root_outdir = args.outdir / prefix_name
    root_outdir.mkdir(parents=True, exist_ok=True)

    if args.find_good_frames:
        all_candidates, picked = find_good_frames(frames, args)
        cand_csv = root_outdir / "good_frame_candidates.csv"
        pick_csv = root_outdir / "good_frames_selected.csv"
        with open(cand_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=list(all_candidates[0].keys()) if all_candidates else ["frame"]
            )
            w.writeheader()
            for row in all_candidates:
                w.writerow(row)
        with open(pick_csv, "w", newline="") as f:
            fields = list(picked[0].keys()) if picked else ["frame"]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in picked:
                w.writerow(row)

        selected_rows = []
        for row in picked:
            iframe = row["frame"]
            comment, symbols, coords = frames[iframe]
            selected_rows.append(
                process_single_frame(iframe, comment, symbols, coords, args, root_outdir)
            )

        txt = root_outdir / "restart_download_steps.txt"
        with open(txt, "w") as f:
            f.write("Chosen good frames and restart-history steps to fetch from HPC\n")
            f.write("Only frames in the requested late-trajectory window are considered.\n")
            f.write(
                "If require_exact_restart is on, md_step is guaranteed to match "
                "an available restart-history step.\n\n"
            )
            f.write(
                "frame,md_step,exact_restart_step,restart_floor,restart_ceil,"
                "step_from_comment,shell,avg4,spread4,gap45\n"
            )
            for row in picked:
                f.write(
                    f"{row['frame']},{row['md_step']},{row['exact_restart_step']},"
                    f"{row['restart_floor']},{row['restart_ceil']},"
                    f"{row['step_from_comment']},{row['shell']},{row['avg4']:.4f},"
                    f"{row['spread4']:.4f},{row['gap45']:.4f}\n"
                )
        print(f"Wrote candidates: {cand_csv}")
        print(f"Wrote selected frames: {pick_csv}")
        print(f"Wrote restart guidance: {txt}")
        if args.last_fraction is not None:
            print(f"Good-frame search limited to last {args.last_fraction:.2%} of frames.")
        if args.require_exact_restart:
            print("Only exact restart-history-matching frames were kept.")
        print(f"Output root: {root_outdir}")
        return

    selected_frames, selection_info = parse_frame_selection(args, len(frames))
    summary_rows = []
    for iframe in selected_frames:
        comment, symbols, coords = frames[iframe]
        row = process_single_frame(iframe, comment, symbols, coords, args, root_outdir)
        summary_rows.append(row)
        print(f"Extracted frame {iframe} -> {row['frame_dir']}")

    summary_csv = root_outdir / "summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "metal",
                "first_shell",
                "first_shell_cn",
                "avg_shell_dist",
                "min_shell_dist",
                "max_shell_dist",
                "qm_atoms",
                "embed_atoms",
                "frame_dir",
            ]
        )
        for row in summary_rows:
            writer.writerow(
                [
                    row["frame"],
                    row["metal"],
                    row["first_shell"],
                    row["first_shell_cn"],
                    f"{row['avg_shell_dist']:.6f}",
                    f"{row['min_shell_dist']:.6f}",
                    f"{row['max_shell_dist']:.6f}",
                    row["qm_atoms"],
                    row["embed_atoms"],
                    row["frame_dir"],
                ]
            )
    print(f"Wrote summary: {summary_csv}")
    print(f"Selection mode: {selection_info}")
    print(f"Output root: {root_outdir}")


if __name__ == "__main__":
    main()
