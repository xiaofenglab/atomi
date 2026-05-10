"""Post-run CP2K AIMD metal-ligand bond analysis."""

import argparse
import csv
import math
import re
from pathlib import Path

from atomi.cp2k.acid_box import METALS


def read_xyz_trajectory(path: Path):
    frames = []
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            try:
                nat = int(line)
            except ValueError:
                raise ValueError(f"{path}: expected atom count, got: {line}")

            comment = f.readline().rstrip("\n")
            symbols = []
            coords = []
            for _ in range(nat):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"{path}: malformed XYZ line: {' '.join(parts)}")
                symbols.append(parts[0])
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

            frames.append((comment, symbols, coords))
    return frames


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def moving_average(y, window):
    if window <= 1:
        return y[:]
    out = []
    half = window // 2
    n = len(y)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def find_metal_index(symbols, user_index=None):
    if user_index is not None:
        idx = user_index - 1
        if idx < 0 or idx >= len(symbols):
            raise ValueError(f"metal index {user_index} out of range")
        return idx

    for i, s in enumerate(symbols):
        if s in METALS:
            return i

    raise ValueError("No metal atom found automatically; pass --metal-index")


def parse_step_from_comment(comment):
    patterns = [
        r"\bi\s*=\s*([0-9]+)\b",
        r"\bstep\s*=\s*([0-9]+)\b",
        r"\bStep\s*number\s*=?\s*([0-9]+)\b",
        r"\bSTEP\s*=?\s*([0-9]+)\b",
    ]
    for pat in patterns:
        m = re.search(pat, comment)
        if m:
            return int(m.group(1))
    return None


def choose_initial_ligands(symbols, coords, metal_idx, ligand_elements=None, n_nearest=4):
    mpos = coords[metal_idx]
    candidates = []
    for i, s in enumerate(symbols):
        if i == metal_idx:
            continue
        if ligand_elements is not None and s not in ligand_elements:
            continue
        if ligand_elements is None and s == "H":
            continue
        candidates.append((dist(mpos, coords[i]), i, s))

    candidates.sort(key=lambda x: x[0])

    if len(candidates) < n_nearest:
        raise ValueError(
            f"Only found {len(candidates)} ligand candidates, fewer than n_nearest={n_nearest}"
        )

    return candidates[:n_nearest]


def strip_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def parse_cp2k_input(inp_path: Path):
    info = {
        "project": None,
        "timestep_fs": None,
        "md_steps": None,
        "temperature": None,
        "colvar_atoms": [],
        "target_angstrom": None,
        "k_kcalmol": None,
        "traj_filename": None,
    }

    section_stack = []

    with open(inp_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        line = strip_inline_comment(raw).strip()
        if not line:
            i += 1
            continue

        upper = line.upper()

        if upper.startswith("&END"):
            if section_stack:
                section_stack.pop()
            i += 1
            continue

        if upper.startswith("&"):
            sec = upper[1:].split()[0]
            section_stack.append(sec)
            i += 1
            continue

        # PROJECT
        if len(section_stack) >= 1 and section_stack[-1] == "GLOBAL":
            m = re.match(r"PROJECT\s+(.+)", line, re.IGNORECASE)
            if m:
                info["project"] = m.group(1).strip()

        # MD block
        if "MD" in section_stack:
            m = re.match(r"TIMESTEP\s+([0-9Ee+\-\.]+)", line, re.IGNORECASE)
            if m:
                info["timestep_fs"] = float(m.group(1))

            m = re.match(r"STEPS\s+([0-9]+)", line, re.IGNORECASE)
            if m:
                info["md_steps"] = int(m.group(1))

            m = re.match(r"TEMPERATURE\s+([0-9Ee+\-\.]+)", line, re.IGNORECASE)
            if m:
                info["temperature"] = float(m.group(1))

        # COLVAR / DISTANCE atoms
        if "DISTANCE" in section_stack:
            m = re.match(r"ATOMS\s+([0-9]+)\s+([0-9]+)", line, re.IGNORECASE)
            if m:
                info["colvar_atoms"].append((int(m.group(1)), int(m.group(2))))

        # RESTRAINT target / K
        if "COLLECTIVE" in section_stack:
            m = re.match(r"TARGET(?:\s+\[.*?\])?\s+([0-9Ee+\-\.]+)", line, re.IGNORECASE)
            if m:
                info["target_angstrom"] = float(m.group(1))

        if "RESTRAINT" in section_stack:
            m = re.match(r"K(?:\s+\[.*?\])?\s+([0-9Ee+\-\.]+)", line, re.IGNORECASE)
            if m:
                info["k_kcalmol"] = float(m.group(1))

        # trajectory filename
        if "TRAJECTORY" in section_stack:
            m = re.match(r"FILENAME\s*=?\s*(.+)", line, re.IGNORECASE)
            if m:
                info["traj_filename"] = m.group(1).strip()

        i += 1

    return info


def analyze_one_file(
    path, metal_index=None, ligand_elements=None, n_nearest=4, tail_fraction=0.2, timestep_fs=None
):
    frames = read_xyz_trajectory(path)
    if len(frames) < 2:
        raise ValueError(f"{path}: need a multi-frame trajectory")

    first_comment, symbols0, coords0 = frames[0]
    metal_idx = find_metal_index(symbols0, metal_index)

    chosen = choose_initial_ligands(
        symbols0, coords0, metal_idx, ligand_elements=ligand_elements, n_nearest=n_nearest
    )

    tracked_indices = [i for _, i, _ in chosen]
    tracked_symbols = [s for _, _, s in chosen]

    raw_xvals = []
    tracked_series = [[] for _ in tracked_indices]
    shell_min = []
    shell_max = []
    shell_mean = []

    parsed_steps = True
    for iframe, (comment, symbols, coords) in enumerate(frames, start=1):
        mpos = coords[metal_idx]

        step = parse_step_from_comment(comment)
        if step is None:
            parsed_steps = False
            raw_xvals.append(iframe)
        else:
            raw_xvals.append(step)

        dvals_fixed = []
        for j, atom_idx in enumerate(tracked_indices):
            d = dist(mpos, coords[atom_idx])
            tracked_series[j].append(d)
            dvals_fixed.append(d)

        shell_min.append(min(dvals_fixed))
        shell_max.append(max(dvals_fixed))
        shell_mean.append(mean(dvals_fixed))

    nframes = len(frames)
    tail_n = max(1, int(math.ceil(nframes * tail_fraction)))
    tail_start = nframes - tail_n

    x_label = "Step (if parsed) or frame"
    xvals_plot = raw_xvals
    total_time_ps = None
    tail_start_ps = None
    tail_end_ps = None

    if timestep_fs is not None:
        xvals_plot = [x * timestep_fs / 1000.0 for x in raw_xvals]
        x_label = "Time (ps)"
        total_time_ps = xvals_plot[-1]
        tail_start_ps = xvals_plot[tail_start]
        tail_end_ps = xvals_plot[-1]

    summary = {
        "nframes_total": nframes,
        "tail_nframes": tail_n,
        "tail_fraction": tail_fraction,
        "metal_index_1based": metal_idx + 1,
        "metal_symbol": symbols0[metal_idx],
        "tracked_indices_1based": [i + 1 for i in tracked_indices],
        "tracked_symbols": tracked_symbols,
        "tracked_mean_tail": [mean(s[tail_start:]) for s in tracked_series],
        "tracked_min_tail": [min(s[tail_start:]) for s in tracked_series],
        "tracked_max_tail": [max(s[tail_start:]) for s in tracked_series],
        "tracked_mean_all": [mean(s) for s in tracked_series],
        "shell_mean_tail": mean(shell_mean[tail_start:]),
        "shell_min_tail": mean(shell_min[tail_start:]),
        "shell_max_tail": mean(shell_max[tail_start:]),
        "shell_mean_all": mean(shell_mean),
        "shell_min_all": mean(shell_min),
        "shell_max_all": mean(shell_max),
        "raw_xvals": raw_xvals,
        "xvals_plot": xvals_plot,
        "x_label": x_label,
        "parsed_steps": parsed_steps,
        "timestep_fs": timestep_fs,
        "total_time_ps": total_time_ps,
        "tail_start_ps": tail_start_ps,
        "tail_end_ps": tail_end_ps,
        "tracked_series": tracked_series,
        "shell_min_series": shell_min,
        "shell_max_series": shell_max,
        "shell_mean_series": shell_mean,
    }
    return summary


def make_plot(summary, infile, outfile, smooth_window=11, show_tail_avg=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it or rerun with --no-plot."
        ) from exc

    xvals = summary["xvals_plot"]
    tracked_series = summary["tracked_series"]
    tracked_idx = summary["tracked_indices_1based"]
    tracked_sym = summary["tracked_symbols"]

    nframes = summary["nframes_total"]
    tail_n = summary["tail_nframes"]
    tail_start = nframes - tail_n

    plt.figure(figsize=(10, 6))

    for i, series in enumerate(tracked_series):
        global_avg = mean(series)
        smooth = moving_average(series, smooth_window)
        tail_avg = mean(series[tail_start:])

        base_label = f"{tracked_sym[i]} idx {tracked_idx[i]}"

        plt.plot(xvals, series, linewidth=0.8, alpha=0.18)
        plt.plot(xvals, smooth, linewidth=2.0, label=base_label)
        plt.axhline(global_avg, linestyle="--", linewidth=1.0, alpha=0.8)
        if show_tail_avg:
            plt.axhline(tail_avg, linestyle=":", linewidth=1.0, alpha=0.8)

    plt.xlabel(summary["x_label"])
    plt.ylabel("Metal–ligand distance (Å)")
    plt.title(f"Bond evolution: {Path(infile).name}")
    plt.tight_layout()
    plt.legend(fontsize=8)
    plt.savefig(outfile, dpi=200)
    plt.close()


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "file",
        "metal",
        "metal_index",
        "tracked_index",
        "tracked_symbol",
        "tracked_mean_all",
        "tracked_mean_tail",
        "tracked_min_tail",
        "tracked_max_tail",
        "shell_mean_all",
        "shell_mean_tail",
        "shell_min_tail",
        "shell_max_tail",
        "nframes_total",
        "tail_nframes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cp2k-bond-analysis",
        description=(
            "Analyze metal-ligand distances from CP2K XYZ trajectories, "
            "optionally reading metadata from a CP2K input file."
        ),
    )
    parser.add_argument("xyz_files", nargs="+", help="Multi-frame XYZ trajectory file(s).")
    parser.add_argument(
        "--inp",
        type=str,
        default=None,
        help="Optional CP2K input file to extract timestep, project, restraints, and K",
    )
    parser.add_argument(
        "--metal-index",
        type=int,
        default=None,
        help="1-based metal atom index; if omitted, first metal atom is used",
    )
    parser.add_argument(
        "--ligand-elements",
        type=str,
        default=None,
        help="Comma-separated ligand elements to consider, e.g. Cl or O,Cl",
    )
    parser.add_argument(
        "--n-nearest",
        type=int,
        default=4,
        help="Number of nearest ligands in the first frame to track",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of final frames to average, default 0.20",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Moving-average window for smoothed plot curves, default 11",
    )
    parser.add_argument(
        "--timestep-fs",
        type=float,
        default=None,
        help="Override timestep in fs. If omitted, script tries to read it from --inp.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--summary-csv", type=Path, default=None, help="Write ligand summary CSV.")
    args = parser.parse_args(argv)

    inp_info = None
    if args.inp is not None:
        inp_path = Path(args.inp)
        if not inp_path.exists():
            raise FileNotFoundError(f"Input file not found: {inp_path}")
        inp_info = parse_cp2k_input(inp_path)

    timestep_fs = args.timestep_fs
    if timestep_fs is None and inp_info is not None:
        timestep_fs = inp_info["timestep_fs"]

    ligand_elements = None
    if args.ligand_elements:
        ligand_elements = {x.strip() for x in args.ligand_elements.split(",") if x.strip()}

    csv_rows = []
    for xyz in args.xyz_files:
        path = Path(xyz)
        summary = analyze_one_file(
            path,
            metal_index=args.metal_index,
            ligand_elements=ligand_elements,
            n_nearest=args.n_nearest,
            tail_fraction=args.tail_fraction,
            timestep_fs=timestep_fs,
        )

        print("=" * 80)
        print(f"File: {path.name}")

        if inp_info is not None:
            print("Input metadata:")
            if inp_info["project"] is not None:
                print(f"  project            = {inp_info['project']}")
            if inp_info["timestep_fs"] is not None:
                print(f"  timestep           = {inp_info['timestep_fs']:.4f} fs")
            if inp_info["md_steps"] is not None:
                print(f"  md steps (input)   = {inp_info['md_steps']}")
            if inp_info["temperature"] is not None:
                print(f"  temperature        = {inp_info['temperature']:.2f} K")
            if len(inp_info["colvar_atoms"]) == 1:
                a1, a2 = inp_info["colvar_atoms"][0]
                print(f"  restrained pair    = {a1}-{a2}")
            elif len(inp_info["colvar_atoms"]) > 1:
                pairs = ", ".join([f"{a}-{b}" for a, b in inp_info["colvar_atoms"]])
                print(f"  restrained pairs   = {pairs}")
            if inp_info["target_angstrom"] is not None:
                print(f"  target distance    = {inp_info['target_angstrom']:.4f} Å")
            if inp_info["k_kcalmol"] is not None:
                print(f"  restraint K        = {inp_info['k_kcalmol']:.4f} kcal/mol/Å²")
            if inp_info["traj_filename"] is not None:
                print(f"  traj filename      = {inp_info['traj_filename']}")

        print(f"Metal: {summary['metal_symbol']}  index={summary['metal_index_1based']}")
        print(f"Total frames: {summary['nframes_total']}")
        print(
            f"Averaging last {summary['tail_nframes']} frames "
            f"({100.0 * summary['tail_fraction']:.1f}%)"
        )

        if summary["timestep_fs"] is not None:
            if summary["parsed_steps"]:
                print(
                    "Time mapping: using parsed MD step from XYZ comment with "
                    f"timestep {summary['timestep_fs']:.4f} fs"
                )
            else:
                print(
                    "Time mapping: MD step not parsed; using frame number with "
                    f"timestep {summary['timestep_fs']:.4f} fs"
                )
            print(f"Total sampled time: {summary['total_time_ps']:.4f} ps")
            print(
                f"Tail averaging window: {summary['tail_start_ps']:.4f} ps -> "
                f"{summary['tail_end_ps']:.4f} ps"
            )

        print("Tracked ligands from first frame:")
        for i, idx in enumerate(summary["tracked_indices_1based"]):
            sym = summary["tracked_symbols"][i]
            avg_all = summary["tracked_mean_all"][i]
            avg_tail = summary["tracked_mean_tail"][i]
            mind = summary["tracked_min_tail"][i]
            maxd = summary["tracked_max_tail"][i]
            print(
                f"  {sym:2s} index={idx:4d}   "
                f"full avg={avg_all:8.4f} Å   "
                f"tail avg={avg_tail:8.4f} Å   "
                f"tail min={mind:8.4f}   "
                f"tail max={maxd:8.4f}"
            )
            csv_rows.append(
                {
                    "file": str(path),
                    "metal": summary["metal_symbol"],
                    "metal_index": summary["metal_index_1based"],
                    "tracked_index": idx,
                    "tracked_symbol": sym,
                    "tracked_mean_all": f"{avg_all:.6f}",
                    "tracked_mean_tail": f"{avg_tail:.6f}",
                    "tracked_min_tail": f"{mind:.6f}",
                    "tracked_max_tail": f"{maxd:.6f}",
                    "shell_mean_all": f"{summary['shell_mean_all']:.6f}",
                    "shell_mean_tail": f"{summary['shell_mean_tail']:.6f}",
                    "shell_min_tail": f"{summary['shell_min_tail']:.6f}",
                    "shell_max_tail": f"{summary['shell_max_tail']:.6f}",
                    "nframes_total": summary["nframes_total"],
                    "tail_nframes": summary["tail_nframes"],
                }
            )

        print("Shell summary (tracked ligands only):")
        print(f"  full-run mean distance   = {summary['shell_mean_all']:.4f} Å")
        print(f"  last-20% mean distance   = {summary['shell_mean_tail']:.4f} Å")
        print(f"  last-20% mean of minima  = {summary['shell_min_tail']:.4f} Å")
        print(f"  last-20% mean of maxima  = {summary['shell_max_tail']:.4f} Å")

        if not args.no_plot:
            png = path.with_name(path.stem + "_bond_evolution.png")
            make_plot(summary, path, png, smooth_window=args.smooth_window, show_tail_avg=True)
            print(f"Saved plot: {png}")

    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, csv_rows)
        print(f"Wrote summary CSV: {args.summary_csv}")


if __name__ == "__main__":
    main()
