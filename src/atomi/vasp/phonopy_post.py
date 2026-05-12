from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from atomi.vasp.phonopy_band import write_standalone_script


@dataclass
class PoscarInfo:
    path: Path
    comment: str
    lattice: list[list[float]]
    species: list[str]
    counts: list[int]
    natoms: int
    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    volume: float
    dup: tuple[int, int, int]

    @property
    def reference_lengths(self) -> tuple[float, float, float]:
        return tuple(length / dup for length, dup in zip(self.lengths, self.dup))

    @property
    def reference_volume(self) -> float:
        dx, dy, dz = self.dup
        return self.volume / (dx * dy * dz)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(dot(a, a))


def angle_degrees(a: list[float], b: list[float]) -> float:
    denom = norm(a) * norm(b)
    if denom == 0:
        return math.nan
    value = max(-1.0, min(1.0, dot(a, b) / denom))
    return math.degrees(math.acos(value))


def determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def read_poscar(path: Path, dup: tuple[int, int, int]) -> PoscarInfo:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 8:
        raise ValueError(f"POSCAR seems too short: {path}")

    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[index].split()[:3]] for index in range(2, 5)]

    species_line = lines[5].split()
    counts_line = lines[6].split()
    try:
        counts = [int(x) for x in counts_line]
        species = species_line
    except ValueError:
        # VASP 4 style POSCAR may omit species names.
        counts = [int(x) for x in species_line]
        species = [f"X{i + 1}" for i in range(len(counts))]

    if len(species) != len(counts):
        raise ValueError("POSCAR species/count lines have different lengths")

    lengths = tuple(norm(vector) for vector in lattice)
    a_vec, b_vec, c_vec = lattice
    angles = (
        angle_degrees(b_vec, c_vec),
        angle_degrees(a_vec, c_vec),
        angle_degrees(a_vec, b_vec),
    )
    volume = abs(determinant3(lattice))
    return PoscarInfo(
        path=path.resolve(),
        comment=comment,
        lattice=lattice,
        species=species,
        counts=counts,
        natoms=sum(counts),
        lengths=lengths,
        angles=angles,
        volume=volume,
        dup=dup,
    )


def band_path_for_dup(dup: tuple[int, int, int], override: str | None = None) -> str:
    if override:
        return " ".join(override.split())
    if dup == (2, 1, 1):
        return "0 0 0  0 0.5 0  0 0 0  0 0 0.5  0 0 0  0 0.5 0.5"
    return "0 0 0  0.5 0 0  0 0 0  0 0.5 0  0 0 0  0 0 0.5"


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def write_env(path: Path, info: PoscarInfo, mesh: tuple[int, int, int], band_path: str) -> None:
    dx, dy, dz = info.dup
    a, b, c = info.lengths
    alpha, beta, gamma = info.angles
    ref_a, ref_b, ref_c = info.reference_lengths
    mesh_text = " ".join(str(x) for x in mesh)
    content = [
        f'export POSCAR_PATH="{info.path}"',
        f"export DUP_X={dx}",
        f"export DUP_Y={dy}",
        f"export DUP_Z={dz}",
        f"export NATOMS={info.natoms}",
        f"export CELL_A={a:.10f}",
        f"export CELL_B={b:.10f}",
        f"export CELL_C={c:.10f}",
        f"export CELL_ALPHA={alpha:.10f}",
        f"export CELL_BETA={beta:.10f}",
        f"export CELL_GAMMA={gamma:.10f}",
        f"export CELL_VOLUME={info.volume:.10f}",
        f"export REF_A={ref_a:.10f}",
        f"export REF_B={ref_b:.10f}",
        f"export REF_C={ref_c:.10f}",
        f"export REF_VOLUME={info.reference_volume:.10f}",
        f'export MESH="{mesh_text}"',
        f'export BAND_PATH="{band_path}"',
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_run_script(
    path: Path,
    env_file: Path,
    phonopy_module: str | None,
    phonopy: str,
    phonopy_load: str,
    run_thermal: bool,
    run_dos: bool,
    run_band: bool,
    run_band_plot: bool,
    band_plot_script: str,
    band_plot_png: str,
    band_plot_title: str,
    module_purge: bool,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"',
        'cd "${WORKDIR}"',
        "mkdir -p logs",
        "",
        f"if [[ -f {shell_quote(env_file.name)} ]]; then",
        f"  source {shell_quote(env_file.name)}",
        "else",
        f"  echo \"ERROR: {env_file.name} not found in ${{WORKDIR}}\"",
        "  exit 1",
        "fi",
        "",
        'echo "POSCAR path: ${POSCAR_PATH}"',
        'echo "Duplication: ${DUP_X} ${DUP_Y} ${DUP_Z}"',
        'echo "natoms: ${NATOMS}"',
        'echo "Actual cell: a=${CELL_A}, b=${CELL_B}, c=${CELL_C}"',
        'echo "Reference cell: a=${REF_A}, b=${REF_B}, c=${REF_C}"',
        'echo "Actual volume: ${CELL_VOLUME}"',
        'echo "Reference volume: ${REF_VOLUME}"',
        'echo "Mesh: ${MESH}"',
        'echo "Band path: ${BAND_PATH}"',
        "",
    ]
    if module_purge:
        lines.append("module purge")
    if phonopy_module:
        lines.append(f"module load {shell_quote(phonopy_module)}")
    lines.extend(
        [
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            "",
            'echo "=== Step 1: FORCE_SETS ==="',
            "if [[ -f FORCE_SETS ]]; then",
            '  echo "FORCE_SETS already exists. Skipping rebuild."',
            "else",
            "  if compgen -G 'disp-*/vasprun.xml' >/dev/null; then",
            '    echo "Building FORCE_SETS from disp-*/vasprun.xml"',
            f"    {shell_quote(phonopy)} -f disp-*/vasprun.xml",
            "  elif compgen -G 'disp-*/vasprun.xml.gz' >/dev/null; then",
            '    echo "Building FORCE_SETS from disp-*/vasprun.xml.gz"',
            f"    {shell_quote(phonopy)} -f disp-*/vasprun.xml.gz",
            "  else",
            '    echo "ERROR: FORCE_SETS missing and no vasprun.xml(.gz) found in disp-*"',
            "    exit 1",
            "  fi",
            "fi",
            "",
        ]
    )
    if run_thermal:
        lines.extend(
            [
                'echo "=== Step 2: Thermal properties ==="',
                f"{shell_quote(phonopy_load)} --mesh ${{MESH}} -t",
                'cp -f thermal_properties.yaml "thermal_properties_mesh_${MESH// /x}.yaml"',
                "",
            ]
        )
    if run_dos:
        lines.extend(
            [
                'echo "=== Step 3: DOS ==="',
                f"{shell_quote(phonopy_load)} --mesh ${{MESH}} --dos",
                'cp -f total_dos.dat "total_dos_mesh_${MESH// /x}.dat"',
                "",
            ]
        )
    if run_band:
        lines.extend(
            [
                'echo "=== Step 4: Band structure ==="',
                f'{shell_quote(phonopy_load)} --band "${{BAND_PATH}}"',
                "if [[ -f band.yaml ]]; then",
                '  cp -f band.yaml "band_${DUP_X}x${DUP_Y}x${DUP_Z}.yaml"',
                "fi",
                "",
            ]
        )
    if run_band and run_band_plot:
        lines.extend(
            [
                'echo "=== Step 5: Band plot ==="',
                'mkdir -p "${WORKDIR}/.matplotlib"',
                'export MPLCONFIGDIR="${WORKDIR}/.matplotlib"',
                f"if [[ -f band.yaml && -f {shell_quote(band_plot_script)} ]]; then",
                f"  python3 {shell_quote(band_plot_script)} "
                f"--band-yaml band.yaml --outpng {shell_quote(band_plot_png)} "
                f"--title {shell_quote(band_plot_title)} "
                '|| echo "WARNING: band plot failed; band.yaml is still available."',
                "else",
                '  echo "WARNING: band.yaml or band plot script missing; skipping band plot."',
                "fi",
                "",
            ]
        )
    lines.extend(
        [
            'echo "=== Finished ==="',
            "ls -lh FORCE_SETS 2>/dev/null || true",
            "ls -lh thermal_properties*.yaml total_dos*.dat band*.yaml 2>/dev/null || true",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_sbatch_script(
    path: Path,
    run_script: Path,
    job_name: str,
    time_limit: str,
    cpus: int,
    mem: str,
) -> None:
    content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}

set -euo pipefail
mkdir -p logs
bash {shell_quote(run_script.name)}
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def write_summary(path: Path, info: PoscarInfo, mesh: tuple[int, int, int], band_path: str) -> None:
    a, b, c = info.lengths
    alpha, beta, gamma = info.angles
    ref_a, ref_b, ref_c = info.reference_lengths
    lines = [
        "POSCAR summary",
        "=" * 40,
        f"POSCAR       : {info.path}",
        f"Comment      : {info.comment}",
        f"Species      : {' '.join(info.species)}",
        f"Counts       : {info.counts}",
        f"natoms       : {info.natoms}",
        f"duplication  : {' '.join(str(x) for x in info.dup)}",
        "",
        "Actual cell",
        f"  a b c      : {a:.6f}  {b:.6f}  {c:.6f}",
        f"  alpha beta gamma : {alpha:.4f}  {beta:.4f}  {gamma:.4f}",
        f"  volume     : {info.volume:.6f}",
        "",
        "Inferred reference cell",
        f"  a b c      : {ref_a:.6f}  {ref_b:.6f}  {ref_c:.6f}",
        f"  volume     : {info.reference_volume:.6f}",
        "",
        f"Mesh         : {' '.join(str(x) for x in mesh)}",
        f"Band path    : {band_path}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vasp-phonopy-post",
        description="Generate and optionally run phonopy post-analysis for thermal, DOS, and band outputs.",
    )
    parser.add_argument("--poscar", type=Path, default=Path("POSCAR"))
    parser.add_argument("--dup", nargs=3, type=int, default=(1, 1, 1), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--mesh", nargs=3, type=int, default=(20, 20, 20), metavar=("MX", "MY", "MZ"))
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument("--env-file", default="poscar_info.env")
    parser.add_argument("--run-script", default="run_phonopy_post.sh")
    parser.add_argument("--sbatch-script", default="submit_phonopy_post.sbatch")
    parser.add_argument("--summary", default="phonopy_post_summary.txt")
    parser.add_argument("--phonopy-module", default="phys/phonopy/2.38.1")
    parser.add_argument("--no-module", action="store_true", help="Do not write a module load line.")
    parser.add_argument("--module-purge", action="store_true", help="Write module purge before module load.")
    parser.add_argument("--phonopy", default="phonopy")
    parser.add_argument("--phonopy-load", default="phonopy-load")
    parser.add_argument("--band-path", default=None, help="Override fractional band path string.")
    parser.add_argument("--no-thermal", action="store_true")
    parser.add_argument("--no-dos", action="store_true")
    parser.add_argument("--no-band", action="store_true")
    parser.add_argument("--no-band-plot", action="store_true", help="Do not write/run the band plotting helper.")
    parser.add_argument("--band-plot-script", default="plot_phonopy_band.py")
    parser.add_argument("--band-plot-png", default="phonon_band.png")
    parser.add_argument("--band-plot-title", default="Phonon band structure")
    parser.add_argument("--job-name", default="phonopy_post")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--submit", action="store_true", help="Run sbatch after writing files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    info = read_poscar(args.poscar.resolve(), tuple(args.dup))
    mesh = tuple(args.mesh)
    band_path = band_path_for_dup(info.dup, args.band_path)

    env_file = outdir / args.env_file
    run_script = outdir / args.run_script
    sbatch_script = outdir / args.sbatch_script
    summary = outdir / args.summary
    band_plot_script = outdir / args.band_plot_script

    write_env(env_file, info, mesh, band_path)
    if not args.no_band and not args.no_band_plot:
        write_standalone_script(band_plot_script)
    write_run_script(
        run_script,
        env_file,
        None if args.no_module else args.phonopy_module,
        args.phonopy,
        args.phonopy_load,
        run_thermal=not args.no_thermal,
        run_dos=not args.no_dos,
        run_band=not args.no_band,
        run_band_plot=not args.no_band_plot,
        band_plot_script=args.band_plot_script,
        band_plot_png=args.band_plot_png,
        band_plot_title=args.band_plot_title,
        module_purge=args.module_purge,
    )
    write_sbatch_script(sbatch_script, run_script, args.job_name, args.time, args.cpus, args.mem)
    write_summary(summary, info, mesh, band_path)

    print(summary.read_text(encoding="utf-8"))
    print(f"Wrote env file      : {env_file}")
    print(f"Wrote run script    : {run_script}")
    print(f"Wrote sbatch script : {sbatch_script}")
    if band_plot_script.exists():
        print(f"Wrote band plotter  : {band_plot_script}")
    expected = []
    if not args.no_thermal:
        expected.append("thermal_properties.yaml")
    if not args.no_dos:
        expected.append("total_dos.dat")
    if not args.no_band:
        expected.append("band.yaml")
    if not args.no_band and not args.no_band_plot:
        expected.append(args.band_plot_png)
    print(f"Expected outputs    : {', '.join(expected)}")
    if args.submit:
        subprocess.run(["sbatch", str(sbatch_script)], cwd=outdir, check=True)


if __name__ == "__main__":
    main()
