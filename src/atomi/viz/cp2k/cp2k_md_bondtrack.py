#!/usr/bin/env python3
import sys
import math
from pathlib import Path

METALS = {
    "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Cs","Ba",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Th","Pa","U","Np","Pu"
}

COMMON_EXCLUDE = {"H"}

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
                n = int(line)
            except ValueError:
                break

            comment = f.readline().rstrip("\n")
            syms = []
            pos = []
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed XYZ line in {path}")
                syms.append(parts[0])
                pos.append((float(parts[1]), float(parts[2]), float(parts[3])))
            frames.append((comment, syms, pos))
    return frames

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def find_metal_index(symbols):
    for i, s in enumerate(symbols):
        if s in METALS:
            return i, s
    raise RuntimeError("No metal atom detected in first frame.")

def detect_shell(symbols, pos, metal_idx, max_candidates=12, max_display=5):
    mpos = pos[metal_idx]
    pairs = []
    for i, s in enumerate(symbols):
        if i == metal_idx:
            continue
        if s in COMMON_EXCLUDE:
            continue
        pairs.append((dist(mpos, pos[i]), i, s))
    pairs.sort(key=lambda x: x[0])

    if not pairs:
        raise RuntimeError("No candidate ligand atoms found.")

    ncheck = min(max_candidates, len(pairs))
    gaps = []
    for k in range(ncheck - 1):
        gaps.append((pairs[k+1][0] - pairs[k][0], k+1))

    if gaps:
        biggest_gap, split = max(gaps, key=lambda x: x[0])
        if biggest_gap > 0.35:
            cn = split
        else:
            cn = min(6, len(pairs))
    else:
        cn = 1

    display_count = max(cn, min(max_display, len(pairs)))
    shell = pairs[:cn]
    display_shell = pairs[:display_count]
    ligand_types = sorted(set(s for _, _, s in shell))
    cutoff = shell[-1][0] + 0.25
    display_ligand_types = sorted(set(s for _, _, s in display_shell))
    return cn, ligand_types, cutoff, shell, display_shell, display_ligand_types

def label_shell(shell):
    counts = {}
    labels = []
    for _, _, symbol in shell:
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels

def auto_find_xyz(logfile: Path, xyz_hint: str | None = None):
    if xyz_hint:
        p = Path(xyz_hint)
        if p.exists():
            return p
        raise RuntimeError(f"Explicit xyz file not found: {xyz_hint}")

    stem = logfile.with_suffix("")
    parent = logfile.parent

    # strongest preference: same basename + common CP2K pos-style names
    preferred = [
        Path(str(stem) + "-pos.xyz"),
        Path(str(stem) + "-pos-1.xyz"),
        Path(str(stem) + ".pos.xyz"),
        Path(str(stem) + ".pos-1.xyz"),
        Path(str(stem) + "_pos.xyz"),
        Path(str(stem) + "_pos-1.xyz"),
    ]
    for p in preferred:
        if p.exists():
            return p

    # next: any *pos*.xyz in same directory, prefer largest/newest
    pos_candidates = sorted(parent.glob("*pos*.xyz"))
    if pos_candidates:
        pos_candidates.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
        return pos_candidates[0]

    # then: same basename plain xyz
    plain = Path(str(stem) + ".xyz")
    if plain.exists():
        return plain

    # last fallback: exactly one xyz in folder
    xyzs = sorted(parent.glob("*.xyz"))
    if len(xyzs) == 1:
        return xyzs[0]

    raise RuntimeError(
        "Could not auto-find a suitable trajectory xyz. "
        "Pass it explicitly or ensure a *pos*.xyz file exists."
    )

def main():
    if len(sys.argv) < 2:
        print(
            "Usage: cp2k_md_bondtrack.py cp2k_md.log [trajectory.xyz] [output.dat]",
            file=sys.stderr,
        )
        sys.exit(1)

    logfile = Path(sys.argv[1])
    xyzfile = auto_find_xyz(logfile, sys.argv[2] if len(sys.argv) >= 3 else None)
    outfile = Path(sys.argv[3]) if len(sys.argv) >= 4 else Path("cp2k_md_bonds.dat")
    metaout = outfile.with_suffix(".meta")

    if not logfile.exists():
        raise RuntimeError(f"Log file not found: {logfile}")
    if not xyzfile.exists():
        raise RuntimeError(f"XYZ file not found: {xyzfile}")

    frames = read_xyz_trajectory(xyzfile)
    if not frames:
        raise RuntimeError(f"No frames found in {xyzfile}")

    _, syms0, pos0 = frames[0]
    metal_idx, metal_sym = find_metal_index(syms0)
    cn, ligand_types, cutoff, shell0, display_shell0, display_ligand_types = detect_shell(
        syms0,
        pos0,
        metal_idx,
    )
    distance_labels = label_shell(display_shell0)
    display_count = len(distance_labels)

    lines = []
    header_cols = ["frame", "min_d", "max_d", "mean_d"] + distance_labels

    for iframe, (_, syms, pos) in enumerate(frames, start=1):
        mpos = pos[metal_idx]
        cand = []
        for i, s in enumerate(syms):
            if i == metal_idx:
                continue
            if s in COMMON_EXCLUDE:
                continue
            if s not in display_ligand_types:
                continue
            cand.append((dist(mpos, pos[i]), i, s))
        cand.sort(key=lambda x: x[0])

        dvals = [x[0] for x in cand[:display_count]]
        if len(dvals) < display_count:
            dvals += [float("nan")] * (display_count - len(dvals))

        finite = [x for x in dvals if x == x]
        dmin = min(finite) if finite else float("nan")
        dmax = max(finite) if finite else float("nan")
        dmean = sum(finite) / len(finite) if finite else float("nan")

        lines.append([iframe, dmin, dmax, dmean] + dvals)

    with open(outfile, "w") as f:
        f.write("# " + " ".join(header_cols) + "\n")
        for row in lines:
            out = []
            for x in row:
                if isinstance(x, float):
                    out.append(f"{x:.8f}" if x == x else "nan")
                else:
                    out.append(str(x))
            f.write(" ".join(out) + "\n")

    with open(metaout, "w") as f:
        f.write(f"xyzfile={xyzfile}\n")
        f.write(f"metal_index={metal_idx + 1}\n")
        f.write(f"metal_symbol={metal_sym}\n")
        f.write(f"coordination_number={cn}\n")
        f.write(f"ligand_types={','.join(ligand_types)}\n")
        f.write(f"display_count={display_count}\n")
        f.write(f"display_ligand_types={','.join(display_ligand_types)}\n")
        f.write(f"distance_labels={','.join(distance_labels)}\n")
        f.write(f"initial_cutoff={cutoff:.6f}\n")
        f.write("initial_shell=" + ",".join(f"{d:.4f}:{i+1}:{s}" for d, i, s in shell0) + "\n")
        f.write(
            "display_shell="
            + ",".join(
                f"{label}:{d:.4f}:{i+1}:{s}"
                for label, (d, i, s) in zip(distance_labels, display_shell0)
            )
            + "\n"
        )

    print(f"Wrote {outfile}")
    print(f"Wrote {metaout}")
    print(f"Using xyzfile: {xyzfile}")
    print(f"Detected metal: {metal_sym} (index {metal_idx + 1})")
    print(f"Detected CN: {cn}")
    print(f"Detected ligand types: {', '.join(ligand_types)}")
    print(f"Displayed distance labels: {', '.join(distance_labels)}")

if __name__ == "__main__":
    main()
