"""Prepare and optionally run PyMOL rendering for CP2K AIMD trajectories."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import tarfile
from pathlib import Path


def count_xyz_frames(path: Path) -> int:
    frames = 0
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
            comment = handle.readline()
            if not comment:
                raise ValueError(f"XYZ frame {frames + 1} in {path} is missing a comment line")
            for _ in range(natoms):
                atom_line = handle.readline()
                if not atom_line:
                    raise ValueError(f"XYZ frame {frames + 1} in {path} is truncated")
            frames += 1
    if frames == 0:
        raise ValueError(f"No XYZ frames found in {path}")
    return frames


def resolve_frame_plan(
    start: int,
    stop: int | None,
    step: int,
    available_frames: int,
) -> tuple[int, int, int]:
    if step <= 0:
        raise ValueError("--step must be a positive integer")
    if start < 1:
        raise ValueError("--start must be >= 1")
    if start > available_frames:
        raise ValueError(f"--start {start} is beyond the available {available_frames} frame(s)")
    requested_stop = available_frames if stop is None else stop
    if requested_stop < start:
        raise ValueError("--stop must be >= --start")
    effective_stop = min(requested_stop, available_frames)
    n_render = len(range(start, effective_stop + 1, step))
    return requested_stop, effective_stop, n_render


PYMOL_HELPER_TEMPLATE = r'''from pymol import cmd
import math
import os

TRAJ_OBJECT = "@@TRAJ_OBJECT@@"

GA_O_CUTOFF = @@GA_O_CUTOFF@@
GA_CL_CUTOFF = @@GA_CL_CUTOFF@@
O_H_CUTOFF = @@O_H_CUTOFF@@

EMBED_SCALE = @@EMBED_SCALE@@
EMBED_TRANS = @@EMBED_TRANS@@

CL_SCALE = @@CL_SCALE@@
CL_TRANS = @@CL_TRANS@@

CORE_SCALE = @@CORE_SCALE@@
GA_SCALE = @@GA_SCALE@@

SHOW_CLOUD = @@SHOW_CLOUD@@
CLOUD_SCALE = @@CLOUD_SCALE@@
CLOUD_O_BOOST = @@CLOUD_O_BOOST@@
CLOUD_CL_BOOST = @@CLOUD_CL_BOOST@@
CLOUD_GA_BOOST = @@CLOUD_GA_BOOST@@
CLOUD_TRANS = @@CLOUD_TRANS@@

STICK_RADIUS = @@STICK_RADIUS@@

RAY_W = @@RAY_W@@
RAY_H = @@RAY_H@@
DPI = @@DPI@@

FIXED_VIEW = None


def _dist(a, b):
    dx = a.coord[0] - b.coord[0]
    dy = a.coord[1] - b.coord[1]
    dz = a.coord[2] - b.coord[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _idx_sel(obj, idx):
    return f"{obj} and index {idx}"


def _make_selection(name, obj, idxs):
    if not idxs:
        cmd.select(name, "none")
        return
    expr = " or ".join(_idx_sel(obj, i) for i in sorted(set(idxs)))
    cmd.select(name, expr)


def _clear_temp():
    for name in [
        "frameobj",
        "cloud",
        "ga_center",
        "firstO",
        "shellH",
        "nearCl",
        "allCl",
        "core",
        "embed",
        "bonded_atoms",
    ]:
        try:
            cmd.delete(name)
        except Exception:
            pass


def style_publication():
    cmd.bg_color("white")
    cmd.set("orthoscopic", 0)
    cmd.set("depth_cue", 0)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_opaque_background", 0)
    cmd.set("sphere_quality", 4)
    cmd.set("stick_quality", 16)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.58)
    cmd.set("direct", 0.22)
    cmd.set("specular", 0.08)
    cmd.set("reflect", 0.0)
    cmd.set("shininess", 20)
    cmd.set("two_sided_lighting", 1)
    cmd.set("stick_ball", 0)
    cmd.set("stick_radius", STICK_RADIUS)
    cmd.set("transparency_mode", 2)
    cmd.set("auto_show_selections", 0)


def build_frame(state=1):
    global FIXED_VIEW
    state = int(state)
    _clear_temp()

    cmd.enable(TRAJ_OBJECT)
    cmd.create("frameobj", TRAJ_OBJECT, state, 1)
    cmd.frame(1)
    cmd.hide("everything", TRAJ_OBJECT)
    cmd.unbond("frameobj", "frameobj")

    if cmd.count_atoms("frameobj") == 0:
        print(f"Warning: frame {state} is empty")
        return

    model_all = cmd.get_model("frameobj")
    ga_atoms = [a for a in model_all.atom if a.symbol == "Ga"]
    o_atoms = [a for a in model_all.atom if a.symbol == "O"]
    h_atoms = [a for a in model_all.atom if a.symbol == "H"]
    cl_atoms = [a for a in model_all.atom if a.symbol == "Cl"]

    ga_idx = [a.index for a in ga_atoms]
    all_cl_idx = [a.index for a in cl_atoms]
    first_o_idx = set()
    near_cl_idx = set()

    for ga in ga_atoms:
        for o in o_atoms:
            if _dist(ga, o) <= GA_O_CUTOFF:
                first_o_idx.add(o.index)
        for cl in cl_atoms:
            if _dist(ga, cl) <= GA_CL_CUTOFF:
                near_cl_idx.add(cl.index)

    shell_h_idx = set()
    for h in h_atoms:
        for o in o_atoms:
            if o.index in first_o_idx and _dist(h, o) <= O_H_CUTOFF:
                shell_h_idx.add(h.index)
                break

    core_idx = set(ga_idx) | set(first_o_idx) | set(shell_h_idx) | set(all_cl_idx)
    all_idx = {a.index for a in model_all.atom}
    embed_idx = all_idx - core_idx

    _make_selection("ga_center", "frameobj", ga_idx)
    _make_selection("firstO", "frameobj", first_o_idx)
    _make_selection("shellH", "frameobj", shell_h_idx)
    _make_selection("nearCl", "frameobj", near_cl_idx)
    _make_selection("allCl", "frameobj", all_cl_idx)
    _make_selection("core", "frameobj", core_idx)
    _make_selection("embed", "frameobj", embed_idx)

    cmd.hide("everything", "all")

    if embed_idx:
        cmd.show("spheres", "embed")
        cmd.set("sphere_scale", EMBED_SCALE, "embed")
        cmd.set("sphere_transparency", EMBED_TRANS, "embed")
        cmd.color("gray", "embed and elem H")
        cmd.color("pink", "embed and elem O")
        cmd.color("gray", "embed and not elem O+H+Cl+Ga")

    if all_cl_idx:
        cmd.show("spheres", "allCl")
        cmd.set("sphere_scale", CL_SCALE, "allCl")
        cmd.set("sphere_transparency", CL_TRANS, "allCl")
        cmd.color("green", "allCl")

    if SHOW_CLOUD:
        cloud_idx = set(ga_idx) | set(first_o_idx) | set(near_cl_idx)
        if cloud_idx:
            expr = " or ".join(_idx_sel("frameobj", i) for i in sorted(cloud_idx))
            cmd.create("cloud", expr, 1, 1)
            cmd.hide("everything", "cloud")
            cmd.show("spheres", "cloud")
            cmd.alter("cloud", f"vdw=vdw*{CLOUD_SCALE}")
            cmd.alter("cloud and elem O", f"vdw=vdw*{CLOUD_O_BOOST}")
            cmd.alter("cloud and elem Cl", f"vdw=vdw*{CLOUD_CL_BOOST}")
            cmd.alter("cloud and elem Ga", f"vdw=vdw*{CLOUD_GA_BOOST}")
            cmd.rebuild("cloud")
            cmd.set("sphere_scale", 1.0, "cloud")
            cmd.set("sphere_transparency", CLOUD_TRANS, "cloud")
            cmd.color("yellow", "cloud and elem Ga")
            cmd.color("gray", "cloud and elem Cl")
            cmd.color("pink", "cloud and elem O")

    if core_idx:
        cmd.show("spheres", "core")
        cmd.set("sphere_scale", CORE_SCALE, "core")
        cmd.set("sphere_transparency", 0.0, "core")
        cmd.color("yellow", "ga_center")
        cmd.color("green", "core and elem Cl")
        cmd.color("red", "core and elem O")
        cmd.color("white", "core and elem H")
        cmd.set("sphere_scale", GA_SCALE, "ga_center")

    bonded_idx = set()
    for ga in ga_atoms:
        ga_sel = _idx_sel("frameobj", ga.index)
        for o in o_atoms:
            if _dist(ga, o) <= GA_O_CUTOFF:
                o_sel = _idx_sel("frameobj", o.index)
                try:
                    cmd.bond(ga_sel, o_sel)
                except Exception:
                    pass
                bonded_idx.add(ga.index)
                bonded_idx.add(o.index)
        for cl in cl_atoms:
            if _dist(ga, cl) <= GA_CL_CUTOFF:
                cl_sel = _idx_sel("frameobj", cl.index)
                try:
                    cmd.bond(ga_sel, cl_sel)
                except Exception:
                    pass
                bonded_idx.add(ga.index)
                bonded_idx.add(cl.index)

    if bonded_idx:
        _make_selection("bonded_atoms", "frameobj", bonded_idx)
        cmd.show("sticks", "bonded_atoms")
        cmd.set("stick_radius", STICK_RADIUS, "bonded_atoms")
        try:
            cmd.set("stick_color", "black", "bonded_atoms")
        except Exception:
            cmd.set_bond("stick_color", "black", "bonded_atoms", "bonded_atoms")

    if cmd.count_atoms("frameobj") > 0:
        if FIXED_VIEW is not None:
            cmd.set_view(FIXED_VIEW)
        else:
            cmd.orient("frameobj")
            cmd.zoom("frameobj", buffer=2.0)

    for selection in [
        "ga_center",
        "firstO",
        "shellH",
        "nearCl",
        "allCl",
        "core",
        "embed",
        "bonded_atoms",
    ]:
        try:
            cmd.disable(selection)
        except Exception:
            pass

    cmd.rebuild()
    cmd.refresh()


def set_reference_view(state=1):
    global FIXED_VIEW
    build_frame(state)
    cmd.orient("frameobj")
    cmd.zoom("frameobj", buffer=2.0)
    FIXED_VIEW = cmd.get_view()
    print(f"Stored reference view from frame {state}")


def snapshot(state=1, out_png="snapshot.png"):
    build_frame(state)
    if cmd.count_atoms("frameobj") == 0:
        print(f"Skipping snapshot for empty frame {state}")
        return
    cmd.ray(RAY_W, RAY_H)
    cmd.png(out_png, width=RAY_W, height=RAY_H, dpi=DPI)
    print(f"Wrote {out_png}")


def render_movie(start=1, stop=None, prefix="frames/frame", step=1, do_ray=1):
    n_states = cmd.count_states(TRAJ_OBJECT)
    if stop is None or stop == "None":
        stop = n_states

    folder = os.path.dirname(prefix)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    for state in range(int(start), int(stop) + 1, int(step)):
        build_frame(state)
        if cmd.count_atoms("frameobj") == 0:
            print(f"Skipping empty frame {state}")
            continue
        out_png = f"{prefix}{state:04d}.png"
        if int(do_ray):
            cmd.ray(RAY_W, RAY_H)
            cmd.png(out_png, width=RAY_W, height=RAY_H, dpi=DPI)
        else:
            cmd.viewport(RAY_W, RAY_H)
            cmd.png(out_png, width=RAY_W, height=RAY_H, dpi=DPI)
        print(f"Wrote {out_png}")


cmd.extend("build_frame", build_frame)
cmd.extend("set_reference_view", set_reference_view)
cmd.extend("snapshot", snapshot)
cmd.extend("render_movie", render_movie)

style_publication()
print("Loaded helper functions:")
print("  build_frame(state)")
print("  set_reference_view(state)")
print("  snapshot(state, png)")
print("  render_movie(start, stop, prefix, step, do_ray)")
'''


def render_helper_text(args: argparse.Namespace) -> str:
    replacements = {
        "@@TRAJ_OBJECT@@": args.object_name,
        "@@GA_O_CUTOFF@@": f"{args.ga_o_cutoff:.6g}",
        "@@GA_CL_CUTOFF@@": f"{args.ga_cl_cutoff:.6g}",
        "@@O_H_CUTOFF@@": f"{args.o_h_cutoff:.6g}",
        "@@EMBED_SCALE@@": f"{args.embed_scale:.6g}",
        "@@EMBED_TRANS@@": f"{args.embed_transparency:.6g}",
        "@@CL_SCALE@@": f"{args.cl_scale:.6g}",
        "@@CL_TRANS@@": f"{args.cl_transparency:.6g}",
        "@@CORE_SCALE@@": f"{args.core_scale:.6g}",
        "@@GA_SCALE@@": f"{args.ga_scale:.6g}",
        "@@SHOW_CLOUD@@": "1" if args.show_cloud else "0",
        "@@CLOUD_SCALE@@": f"{args.cloud_scale:.6g}",
        "@@CLOUD_O_BOOST@@": f"{args.cloud_o_boost:.6g}",
        "@@CLOUD_CL_BOOST@@": f"{args.cloud_cl_boost:.6g}",
        "@@CLOUD_GA_BOOST@@": f"{args.cloud_ga_boost:.6g}",
        "@@CLOUD_TRANS@@": f"{args.cloud_transparency:.6g}",
        "@@STICK_RADIUS@@": f"{args.stick_radius:.6g}",
        "@@RAY_W@@": str(args.width),
        "@@RAY_H@@": str(args.height),
        "@@DPI@@": str(args.dpi),
    }
    text = PYMOL_HELPER_TEMPLATE
    for marker, value in replacements.items():
        text = text.replace(marker, value)
    return text


def quote_pymol_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def write_pml(args: argparse.Namespace, helper_name: str, effective_stop: int) -> str:
    prefix = Path("frames") / args.frame_prefix
    lines = [
        f"load {quote_pymol_path(args.xyz.resolve())}, {args.object_name}",
        f"run {helper_name}",
        f"set_reference_view {args.reference_state}",
    ]
    for state in args.snapshot:
        lines.append(f"snapshot {state}, snapshots/snapshot_{int(state):04d}.png")
    do_ray = 1 if args.ray else 0
    lines.append(
        f"render_movie {args.start}, {effective_stop}, {prefix.as_posix()}, {args.step}, {do_ray}"
    )
    lines.append("quit")
    return "\n".join(lines) + "\n"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def write_shell_scripts(args: argparse.Namespace, outdir: Path, archive_path: Path) -> None:
    pymol_cmd = [args.pymol_exe, "-cq", "render_movie.pml"]
    if args.xvfb:
        pymol_cmd = ["xvfb-run", "-a", *pymol_cmd]
    (outdir / "run_pymol_render.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        f"{shell_join(pymol_cmd)}\n",
        encoding="utf-8",
    )
    ffmpeg_cmd = [
        args.ffmpeg_exe,
        "-framerate",
        str(args.framerate),
        "-pattern_type",
        "glob",
        "-i",
        f"frames/{args.frame_prefix}*.png",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        args.movie_name,
    ]
    (outdir / "make_movie.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        f"{shell_join(ffmpeg_cmd)}\n",
        encoding="utf-8",
    )
    (outdir / "pack_for_download.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/..\"\n"
        f"tar -czf {shlex.quote(archive_path.name)} {shlex.quote(outdir.name)}\n",
        encoding="utf-8",
    )
    for script in ("run_pymol_render.sh", "make_movie.sh", "pack_for_download.sh"):
        (outdir / script).chmod(0o755)


def make_archive(outdir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(outdir, arcname=outdir.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run PyMOL rendering scripts for CP2K AIMD XYZ files."
    )
    parser.add_argument("xyz", type=Path, help="CP2K multi-frame XYZ trajectory.")
    parser.add_argument("--outdir", type=Path, default=Path("pymol_render"))
    parser.add_argument("--object-name", default="traj")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--reference-state", type=int, default=1)
    parser.add_argument("--snapshot", type=int, action="append", default=[])
    parser.add_argument("--frame-prefix", default="frame")
    parser.add_argument("--movie-name", default="movie.mp4")
    parser.add_argument("--framerate", type=int, default=12)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--width", type=int, default=3600)
    parser.add_argument("--height", type=int, default=2700)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ray", dest="ray", action="store_true", default=True)
    parser.add_argument("--no-ray", dest="ray", action="store_false")
    parser.add_argument("--ga-o-cutoff", type=float, default=2.55)
    parser.add_argument("--ga-cl-cutoff", type=float, default=2.95)
    parser.add_argument("--o-h-cutoff", type=float, default=1.25)
    parser.add_argument("--embed-scale", type=float, default=0.14)
    parser.add_argument("--embed-transparency", type=float, default=0.78)
    parser.add_argument("--cl-scale", type=float, default=0.30)
    parser.add_argument("--cl-transparency", type=float, default=0.00)
    parser.add_argument("--core-scale", type=float, default=0.38)
    parser.add_argument("--ga-scale", type=float, default=0.50)
    parser.add_argument("--stick-radius", type=float, default=0.12)
    parser.add_argument("--show-cloud", action="store_true")
    parser.add_argument("--cloud-scale", type=float, default=1.30)
    parser.add_argument("--cloud-o-boost", type=float, default=1.15)
    parser.add_argument("--cloud-cl-boost", type=float, default=1.35)
    parser.add_argument("--cloud-ga-boost", type=float, default=1.05)
    parser.add_argument("--cloud-transparency", type=float, default=0.78)
    parser.add_argument("--pymol-exe", default="pymol")
    parser.add_argument("--ffmpeg-exe", default="ffmpeg")
    parser.add_argument("--xvfb", action="store_true", help="Run PyMOL through xvfb-run -a.")
    parser.add_argument("--run", action="store_true", help="Run PyMOL after writing scripts.")
    parser.add_argument(
        "--make-movie",
        action="store_true",
        help="Run ffmpeg after rendering frames.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create a tar.gz archive for download.",
    )
    parser.add_argument("--archive-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.xyz.is_file():
        raise FileNotFoundError(f"XYZ trajectory not found: {args.xyz}")
    available_frames = count_xyz_frames(args.xyz)
    requested_stop, effective_stop, n_render = resolve_frame_plan(
        start=args.start,
        stop=args.stop,
        step=args.step,
        available_frames=available_frames,
    )
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frames").mkdir(exist_ok=True)
    (outdir / "snapshots").mkdir(exist_ok=True)

    helper_name = "aimd_render_dynamic.py"
    (outdir / helper_name).write_text(render_helper_text(args), encoding="utf-8")
    (outdir / "render_movie.pml").write_text(
        write_pml(args, helper_name, effective_stop=effective_stop),
        encoding="utf-8",
    )

    archive_path = args.archive_path or outdir.with_suffix(".tar.gz")
    write_shell_scripts(args, outdir=outdir, archive_path=archive_path)

    summary_lines = [
        f"xyz = {args.xyz.resolve()}",
        f"available_frames = {available_frames}",
        f"requested_start = {args.start}",
        f"requested_stop = {requested_stop}",
        f"effective_stop = {effective_stop}",
        f"step = {args.step}",
        f"frames_to_render = {n_render}",
        f"snapshots_requested = {','.join(str(item) for item in args.snapshot) or 'none'}",
    ]
    (outdir / "render_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Wrote PyMOL render workspace: {outdir}")
    print(f"  available XYZ frames: {available_frames}")
    print(
        f"  frames selected for rendering: {n_render} "
        f"({args.start}:{effective_stop}:{args.step})"
    )
    if requested_stop != effective_stop:
        print(f"  requested stop {requested_stop} was clamped to available frame {effective_stop}")
    print(f"  helper: {outdir / helper_name}")
    print(f"  driver: {outdir / 'render_movie.pml'}")
    print(f"  summary: {outdir / 'render_summary.txt'}")
    print(f"  run script: {outdir / 'run_pymol_render.sh'}")
    print(f"  movie script: {outdir / 'make_movie.sh'}")
    print(f"  download pack script: {outdir / 'pack_for_download.sh'}")

    if args.run:
        subprocess.run([str(outdir / "run_pymol_render.sh")], check=True)
    if args.make_movie:
        subprocess.run([str(outdir / "make_movie.sh")], check=True)
    if args.archive:
        make_archive(outdir, archive_path)
        print(f"Wrote download archive: {archive_path}")


if __name__ == "__main__":
    main()
