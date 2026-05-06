#!/usr/bin/env python3
import sys
import time
from pathlib import Path
from collections import deque

def parse_latest_step(logfile: Path):
    latest = None
    with logfile.open("r", errors="ignore") as f:
        for line in f:
            if "MD| Step number" in line:
                parts = line.split()
                try:
                    latest = int(parts[-1])
                except Exception:
                    pass
    return latest

def parse_total_steps_from_input(inpfile: Path):
    if not inpfile.exists():
        return None
    total = None
    with inpfile.open("r", errors="ignore") as f:
        for line in f:
            s = line.strip().upper()
            if s.startswith("STEPS"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        total = int(parts[1])
                    except Exception:
                        pass
    return total

def choose_block_rate(rows, preferred_step_block=50, min_step_block=20):
    """
    rows: list of (time_sec, md_step)
    returns:
      min_per_step, steps_per_min, used_dstep
    """
    if len(rows) < 2:
        return None, None, None

    t_last, s_last = rows[-1]

    # Search backwards for a point about preferred_step_block earlier.
    best = None
    for i in range(len(rows) - 2, -1, -1):
        t0, s0 = rows[i]
        dstep = s_last - s0
        dt = t_last - t0
        if dstep <= 0 or dt <= 0:
            continue

        # first choice: at least preferred block
        if dstep >= preferred_step_block:
            best = (dt, dstep)
            break

        # fallback candidate: largest available >= min block
        if dstep >= min_step_block:
            best = (dt, dstep)

    # if still nothing, use full recent span
    if best is None:
        t0, s0 = rows[0]
        dstep = s_last - s0
        dt = t_last - t0
        if dstep > 0 and dt > 0:
            best = (dt, dstep)

    if best is None:
        return None, None, None

    dt, dstep = best
    min_per_step = (dt / 60.0) / dstep
    steps_per_min = dstep / (dt / 60.0)
    return min_per_step, steps_per_min, dstep

def main():
    if len(sys.argv) < 4:
        print("Usage: cp2k_md_eta.py <logfile> <inpfile> <historyfile> [history_points] [preferred_step_block] [min_step_block]", file=sys.stderr)
        sys.exit(1)

    logfile = Path(sys.argv[1])
    inpfile = Path(sys.argv[2])
    histfile = Path(sys.argv[3])

    history_points = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    preferred_step_block = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    min_step_block = int(sys.argv[6]) if len(sys.argv) > 6 else 20

    latest_step = parse_latest_step(logfile)
    total_steps = parse_total_steps_from_input(inpfile)
    now = time.time()

    rows = deque(maxlen=history_points)

    if histfile.exists():
        with histfile.open("r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    t = float(parts[0])
                    s = int(parts[1])
                    rows.append((t, s))
                except Exception:
                    pass

    if latest_step is None:
        print("latest_step=NA")
        print("min_per_step=NA")
        print("steps_per_min=NA")
        print("remaining_steps=NA")
        print("eta_min=NA")
        print("eta_hms=NA")
        print("rate_block_steps=NA")
        return

    # append only if step advanced
    if len(rows) == 0 or rows[-1][1] != latest_step:
        rows.append((now, latest_step))

    with histfile.open("w") as f:
        for t, s in rows:
            f.write(f"{t:.3f} {s}\n")

    min_per_step, steps_per_min, used_dstep = choose_block_rate(
        list(rows),
        preferred_step_block=preferred_step_block,
        min_step_block=min_step_block
    )

    remaining_steps = None
    eta_min = None
    eta_hms = "NA"

    if total_steps is not None:
        remaining_steps = max(total_steps - latest_step, 0)
        if min_per_step is not None:
            eta_min = remaining_steps * min_per_step
            eta_sec = int(round(eta_min * 60.0))
            h = eta_sec // 3600
            m = (eta_sec % 3600) // 60
            sec = eta_sec % 60
            eta_hms = f"{h:02d}:{m:02d}:{sec:02d}"

    def fmt(x, nd=4):
        return "NA" if x is None else f"{x:.{nd}f}"

    print(f"latest_step={latest_step}")
    print(f"min_per_step={fmt(min_per_step, 4)}")
    print(f"steps_per_min={fmt(steps_per_min, 3)}")
    print(f"remaining_steps={remaining_steps if remaining_steps is not None else 'NA'}")
    print(f"eta_min={fmt(eta_min, 2)}")
    print(f"eta_hms={eta_hms}")
    print(f"rate_block_steps={used_dstep if used_dstep is not None else 'NA'}")

if __name__ == "__main__":
    main()