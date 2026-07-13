"""Optional FDMNES bridge for quick periodic/cluster XANES screens.

FDMNES is kept outside Atomi's dependency set.  This bridge prepares a
reviewable FDMNES workspace from a VASP-relaxed structure, records VASP
provenance, writes a Slurm wrapper, and collects lightweight spectrum
summaries.  It is a route-C scaffold for fast XANES screening, not a substitute
for project-specific FDMNES convergence checks or for Molcas/OCEAN physics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Na": 11,
    "Cl": 17,
    "K": 19,
    "Ca": 20,
    "Ti": 22,
    "Nb": 41,
    "Ce": 58,
    "Gd": 64,
    "U": 92,
}

SPECTRUM_CANDIDATES = (
    "xanes.dat",
    "spectrum.dat",
    "fdmnes_xanes.dat",
    "fdmnes_spectrum.dat",
    "fdmnes_out.txt",
    "out.txt",
)

VASP_INCAR_KEYS = {
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "GGA",
    "ISMEAR",
    "SIGMA",
    "ISPIN",
    "MAGMOM",
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "LASPH",
    "LMAXMIX",
    "LREAL",
}


def _vec_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _angle_degrees(a: list[float], b: list[float]) -> float:
    denom = _vec_norm(a) * _vec_norm(b)
    if denom <= 0:
        raise ValueError("Cannot calculate angle for zero-length lattice vector")
    cosang = sum(x * y for x, y in zip(a, b)) / denom
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _invert3(m: list[list[float]]) -> list[list[float]]:
    det = _det3(m)
    if abs(det) < 1e-14:
        raise ValueError("POSCAR lattice is singular")
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]


def _row_times_matrix(row: list[float], matrix: list[list[float]]) -> list[float]:
    return [sum(row[i] * matrix[i][j] for i in range(3)) for j in range(3)]


def _parse_poscar(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path} does not look like a POSCAR/CONTCAR")
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    coord_line = 7
    if lines[coord_line].lower().startswith("s"):
        coord_line += 1
    mode = lines[coord_line].lower()
    start = coord_line + 1
    total = sum(counts)
    coords = [[float(x) for x in lines[start + i].split()[:3]] for i in range(total)]
    if mode.startswith("c") or mode.startswith("k"):
        inv_lattice = _invert3(lattice)
        frac = [_row_times_matrix(coord, inv_lattice) for coord in coords]
    else:
        frac = coords
    atoms: list[dict[str, Any]] = []
    first_indices: dict[str, int] = {}
    cursor = 0
    for element, count in zip(elements, counts):
        first_indices.setdefault(element, cursor + 1)
        z = ATOMIC_NUMBERS.get(element)
        if z is None:
            raise ValueError(f"Missing atomic number mapping for POSCAR element {element!r}")
        for _ in range(count):
            atoms.append({"index": cursor + 1, "element": element, "z": z, "frac": frac[cursor]})
            cursor += 1
    lengths = [_vec_norm(v) for v in lattice]
    angles = [
        _angle_degrees(lattice[1], lattice[2]),
        _angle_degrees(lattice[0], lattice[2]),
        _angle_degrees(lattice[0], lattice[1]),
    ]
    return {
        "lattice": lattice,
        "lengths": lengths,
        "angles": angles,
        "elements": elements,
        "counts": counts,
        "atoms": atoms,
        "first_indices": first_indices,
        "natom": total,
    }


def _find_structure(structure: Path | None, vasp_dir: Path | None) -> Path:
    if structure is not None:
        return structure.expanduser()
    if vasp_dir is None:
        raise ValueError("Either --structure or --vasp-dir is required")
    root = vasp_dir.expanduser()
    for name in ("CONTCAR", "POSCAR"):
        candidate = root / name
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FileNotFoundError(f"No non-empty CONTCAR/POSCAR found in {root}")


def _parse_incar(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    tags: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        if key in VASP_INCAR_KEYS:
            tags[key] = value.strip()
    return tags


def _read_head(path: Path, n_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[:n_lines]


def _extract_final_energy(outcar: Path) -> float | None:
    if not outcar.exists():
        return None
    final: float | None = None
    for line in outcar.read_text(encoding="utf-8", errors="replace").splitlines():
        if "free  energy   TOTEN" not in line:
            continue
        parts = line.split()
        try:
            final = float(parts[4])
        except (IndexError, ValueError):
            continue
    return final


def build_vasp_context(vasp_dir: Path | None) -> dict[str, Any]:
    if vasp_dir is None:
        return {}
    root = vasp_dir.expanduser()
    return {
        "path": str(root),
        "files_present": {
            name: (root / name).exists()
            for name in ("INCAR", "KPOINTS", "POSCAR", "CONTCAR", "OUTCAR", "vasprun.xml")
        },
        "incar_tags": _parse_incar(root / "INCAR"),
        "kpoints_head": _read_head(root / "KPOINTS", 10),
        "final_toten_ev": _extract_final_energy(root / "OUTCAR"),
    }


def default_fdmnes_executable() -> str:
    for key in ("ATOMI_FDMNES_EXE", "FDMNES_EXE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    bin_dir = os.environ.get("ATOMI_FDMNES_BIN", "").strip()
    if bin_dir:
        for name in ("fdmnes", "fdmnes_mpi"):
            candidate = Path(bin_dir).expanduser() / name
            if candidate.exists():
                return str(candidate)
    return shutil.which("fdmnes") or shutil.which("fdmnes_mpi") or "fdmnes"


def probe_fdmnes(executable: str | None = None, root: str | None = None, bin_dir: str | None = None) -> dict[str, Any]:
    exe = executable or default_fdmnes_executable()
    resolved = shutil.which(exe) if not Path(exe).is_absolute() else exe
    exists = bool(resolved and Path(resolved).expanduser().exists())
    payload: dict[str, Any] = {
        "available": exists,
        "executable": exe,
        "resolved_executable": str(Path(resolved).expanduser()) if resolved else "",
        "root": root or os.environ.get("ATOMI_FDMNES_ROOT", ""),
        "bin": bin_dir or os.environ.get("ATOMI_FDMNES_BIN", ""),
        "module": os.environ.get("ATOMI_FDMNES_MODULE", ""),
    }
    if exists:
        try:
            proc = subprocess.run(
                [str(Path(resolved).expanduser()), "-h"],
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            payload["help_returncode"] = proc.returncode
            payload["help_head"] = "\n".join((proc.stdout or proc.stderr).splitlines()[:8])
        except Exception as exc:  # pragma: no cover - executable-specific behavior
            payload["help_error"] = str(exc)
    return payload


def _json_dump(data: Any, path: Path | None = None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _fdmnes_output_prefix(args: argparse.Namespace) -> str:
    raw = args.output_prefix or f"fdmnes_{args.absorber}_{args.edge}"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)


def write_fdmnes_input(args: argparse.Namespace, outdir: Path) -> tuple[Path, dict[str, Any]]:
    structure = _find_structure(args.structure, args.vasp_dir)
    poscar = _parse_poscar(structure)
    absorber = args.absorber.strip()
    if absorber not in poscar["first_indices"]:
        raise ValueError(f"Absorber {absorber!r} is not present in {structure}")
    absorber_index = int(args.absorber_index or poscar["first_indices"][absorber])
    if absorber_index < 1 or absorber_index > poscar["natom"]:
        raise ValueError(f"--absorber-index {absorber_index} is outside 1..{poscar['natom']}")
    output_prefix = _fdmnes_output_prefix(args)
    range_tokens = str(args.energy_range).split()
    if len(range_tokens) < 2:
        raise ValueError("--energy-range should contain at least start and stop values, e.g. '-20 80 0.5'")
    a, b, c = poscar["lengths"]
    alpha, beta, gamma = poscar["angles"]
    lines = [
        "! FDMNES input generated by Atomi route C from POSCAR/CONTCAR.",
        "! Review edge, absorber site, radius, broadening, spin/orbit, and convolution before production use.",
        "! Atomic coordinates below are fractional and use FDMNES Crystal mode.",
        "Filout",
        output_prefix,
        "",
        "Range",
        " ".join(range_tokens),
        "",
        "Radius",
        f"{float(args.radius):.6f}",
        "",
        "Edge",
        args.edge,
        "",
        "Absorber",
        str(absorber_index),
        "",
    ]
    if args.green:
        lines.extend(["Green", ""])
    if args.scf:
        lines.extend(["SCF", ""])
    if args.quadrupole:
        lines.extend(["Quadrupole", ""])
    if args.spinorbit:
        lines.extend(["Spinorbit", ""])
    if args.convolution:
        lines.extend(["Convolution", ""])
    if args.extra:
        lines.extend([str(item) for item in args.extra])
        lines.append("")
    lines.extend(
        [
            "Crystal",
            f"{a:.10f} {b:.10f} {c:.10f} {alpha:.8f} {beta:.8f} {gamma:.8f}",
        ]
    )
    for atom in poscar["atoms"]:
        x, y, z = atom["frac"]
        marker = " ! absorber" if atom["index"] == absorber_index else ""
        lines.append(f"{atom['z']:3d} {x:.12f} {y:.12f} {z:.12f}{marker}")
    lines.extend(["End", ""])
    path = outdir / "fdmnes.inp"
    path.write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "structure": str(structure.resolve()),
        "absorber": absorber,
        "absorber_index": absorber_index,
        "absorber_element_at_index": poscar["atoms"][absorber_index - 1]["element"],
        "edge": args.edge,
        "output_prefix": output_prefix,
        "natom": poscar["natom"],
        "elements": poscar["elements"],
        "counts": poscar["counts"],
        "cell": {"lengths_angstrom": poscar["lengths"], "angles_degrees": poscar["angles"]},
        "fdmnes_input_notes": [
            "FDMNES input is a conservative scaffold from VASP structure only.",
            "VASP CHGCAR/WAVECAR are not consumed directly by this bridge.",
            "Validate absorber site, edge, radius, relativistic/spin-orbit choices, and broadening before interpreting spectra.",
        ],
    }
    return path, metadata


def write_run_scripts(args: argparse.Namespace, outdir: Path, input_file: Path) -> dict[str, str]:
    exe = args.executable or default_fdmnes_executable()
    module_name = args.module or os.environ.get("ATOMI_FDMNES_MODULE", "")
    run_script = outdir / "run_fdmnes_xanes.sh"
    module_block = ""
    if module_name:
        module_block = (
            f"export ATOMI_FDMNES_MODULE={shlex.quote(str(module_name))}\n"
            'if ! type module >/dev/null 2>&1; then\n'
            '  source /etc/profile.d/modules.sh 2>/dev/null || true\n'
            'fi\n'
            'if type module >/dev/null 2>&1; then\n'
            '  module load "${ATOMI_FDMNES_MODULE}"\n'
            'fi\n'
        )
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${SCRIPT_DIR}"\n'
        f"{module_block}"
        f"export ATOMI_FDMNES_EXE={shlex.quote(str(exe))}\n"
        f"export ATOMI_FDMNES_ROOT={shlex.quote(str(args.root or os.environ.get('ATOMI_FDMNES_ROOT', '')))}\n"
        f"export ATOMI_FDMNES_BIN={shlex.quote(str(args.bin or os.environ.get('ATOMI_FDMNES_BIN', '')))}\n"
        'echo "Running FDMNES route-C XANES bridge workspace"\n'
        'echo "Started $(date -Is)"\n'
        "set +e\n"
        f"{shlex.quote(str(exe))} {shlex.quote(input_file.name)} > fdmnes.stdout.log 2> fdmnes.stderr.log\n"
        "status=$?\n"
        "if [[ ${status} -ne 0 ]]; then\n"
        '  echo "Direct argument run failed with status ${status}; retrying stdin filename mode." >&2\n'
        f"  printf '%s\\n' {shlex.quote(input_file.name)} | {shlex.quote(str(exe))} >> fdmnes.stdout.log 2>> fdmnes.stderr.log\n"
        "  status=$?\n"
        "fi\n"
        "set -e\n"
        'echo "Finished $(date -Is) status=${status}" | tee fdmnes.status.txt\n'
        "exit ${status}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    sbatch = outdir / "submit_fdmnes_xanes.sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={args.job_name}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={args.ntasks}\n"
        f"#SBATCH --cpus-per-task={args.cpus_per_task}\n"
        f"#SBATCH --mem={args.mem}\n"
        f"#SBATCH --time={args.time}\n"
        "#SBATCH --output=fdmnes_xanes.%j.out\n"
        "#SBATCH --error=fdmnes_xanes.%j.err\n"
        "\n"
        "set -euo pipefail\n"
        "bash run_fdmnes_xanes.sh\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    return {"run_script": str(run_script.resolve()), "sbatch_script": str(sbatch.resolve())}


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "schema": "atomi.fdmnes_xanes_status.v1",
        "fdmnes": probe_fdmnes(args.executable, args.root, args.bin),
        "environment": {
            "ATOMI_FDMNES_ROOT": os.environ.get("ATOMI_FDMNES_ROOT", ""),
            "ATOMI_FDMNES_BIN": os.environ.get("ATOMI_FDMNES_BIN", ""),
            "ATOMI_FDMNES_MODULE": os.environ.get("ATOMI_FDMNES_MODULE", ""),
            "ATOMI_FDMNES_EXE": os.environ.get("ATOMI_FDMNES_EXE", ""),
        },
    }
    if args.json:
        _json_dump(report)
    else:
        fdmnes = report["fdmnes"]
        print("Atomi FDMNES/XANES route-C bridge status")
        print(f"  executable : {fdmnes.get('resolved_executable') or fdmnes.get('executable')}")
        print(f"  available  : {'yes' if fdmnes.get('available') else 'no'}")
        print(f"  root       : {fdmnes.get('root') or '(not set)'}")
        print(f"  module     : {fdmnes.get('module') or '(not set)'}")
    return report


def install_plan_main(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "atomi.fdmnes_xanes_install_plan.v1",
        "recommendation": "Keep FDMNES as a separate external runtime and call it from Atomi route C.",
        "why": [
            "FDMNES is a compiled XANES/XAS code with its own executable/runtime assumptions.",
            "Keeping it outside m_lammps_env avoids disturbing Atomi, VASP, LAMMPS, and SLUSCHI Python dependencies.",
            "The Atomi bridge prepares input, provenance, Slurm wrappers, and spectrum summaries; FDMNES physics settings remain reviewable.",
        ],
        "bridge_roles": {
            "Route A FEFF/Larch": "Local absorber-cluster and MD/ensemble XAFS processing.",
            "Route B OCEAN": "Periodic-solid BSE/core-hole route, slower but closer to band-structure screening.",
            "Route C FDMNES": "Quick XANES screen from VASP-relaxed periodic structures or clusters.",
            "Molcas/OpenMolcas": "Multireference cluster multiplet/core-level state analysis.",
        },
        "hpc_pattern": [
            "Build or load FDMNES outside m_lammps_env, e.g. $HOME/atomi_hpc/fdmnes.",
            "Record FDMNES paths in private KIT JSON under profiles.fdmnes.",
            "Use VASP relaxation workflow output as --vasp-dir or pass --structure CONTCAR explicitly.",
            "Check with: fdmnes-xanes-status",
            "Prepare with: fdmnes-xanes-bridge prepare --vasp-dir vasp_relax --absorber Ce --edge L3 --outdir fdmnes_Ce_L3",
        ],
        "example_profile": {
            "profiles": {
                "fdmnes": {
                    "root": "$HOME/atomi_hpc/fdmnes",
                    "bin": "$HOME/atomi_hpc/fdmnes/bin",
                    "module": "",
                    "executable": "$HOME/atomi_hpc/fdmnes/bin/fdmnes",
                    "environment": {
                        "ATOMI_FDMNES_ROOT": "$HOME/atomi_hpc/fdmnes",
                        "ATOMI_FDMNES_BIN": "$HOME/atomi_hpc/fdmnes/bin",
                        "ATOMI_FDMNES_MODULE": "",
                        "ATOMI_FDMNES_EXE": "$HOME/atomi_hpc/fdmnes/bin/fdmnes",
                    },
                }
            }
        },
    }
    if args.json:
        _json_dump(payload)
    else:
        print("FDMNES / Atomi route-C install plan")
        for item in payload["why"]:
            print(f"  - {item}")
        print("  Recommended: separate FDMNES runtime, configured through private KIT JSON.")
    return payload


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    input_file, input_metadata = write_fdmnes_input(args, outdir)
    scripts = write_run_scripts(args, outdir, input_file)
    metadata = {
        "schema": "atomi.fdmnes_xanes_project.v1",
        "mode": "prepare",
        "role": "route C quick FDMNES XANES bridge",
        "fdmnes_input": str(input_file.resolve()),
        "scripts": scripts,
        "vasp": build_vasp_context(args.vasp_dir),
        "fdmnes": {
            "executable": args.executable or default_fdmnes_executable(),
            "module": args.module or os.environ.get("ATOMI_FDMNES_MODULE", ""),
            "root": args.root or os.environ.get("ATOMI_FDMNES_ROOT", ""),
            "bin": args.bin or os.environ.get("ATOMI_FDMNES_BIN", ""),
        },
        **input_metadata,
        "recommendations": [
            "Use this as a quick route-C XANES screen after VASP relaxation guards pass.",
            "For Ce/U L or M edges, compare against Molcas/OCEAN where multiplet or BSE physics matters.",
            "Record FDMNES edge, absorber index, radius, convolution, and spin-orbit choices in the project report.",
        ],
    }
    project_path = outdir / "fdmnes_xanes_project.json"
    project_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote FDMNES route-C XANES workspace: {outdir}")
    print(f"Wrote FDMNES input: {input_file}")
    return metadata


def read_numeric_curve(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "Title", "End")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return rows


def _find_spectrum(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    for name in SPECTRUM_CANDIDATES:
        candidate = root / name
        if candidate.exists() and read_numeric_curve(candidate):
            return candidate
    for candidate in sorted(root.glob("*.dat")) + sorted(root.glob("*.txt")):
        if candidate.name in {"fdmnes.inp", "fdmnes.stdin", "fdmnes.status.txt"}:
            continue
        if read_numeric_curve(candidate):
            return candidate
    raise FileNotFoundError(f"No numeric FDMNES spectrum found in {root}")


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.fdmnes_dir.expanduser().resolve()
    spectrum = _find_spectrum(root, args.spectrum)
    rows = read_numeric_curve(spectrum)
    if not rows:
        raise ValueError(f"No numeric two-column spectrum data found in {spectrum}")
    energies = [row[0] for row in rows]
    intensities = [row[1] for row in rows]
    summary = {
        "schema": "atomi.fdmnes_xanes_summary.v1",
        "spectrum": str(spectrum.resolve()),
        "n_points": len(rows),
        "energy_min": min(energies),
        "energy_max": max(energies),
        "intensity_min": min(intensities),
        "intensity_max": max(intensities),
        "peak_energy": rows[intensities.index(max(intensities))][0],
    }
    if args.write:
        _json_dump(summary, args.write)
    else:
        _json_dump(summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "doctor"):
        p = sub.add_parser(name, help="Check configured FDMNES executable/runtime.")
        p.add_argument("--executable")
        p.add_argument("--root")
        p.add_argument("--bin")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=status_main)
    p = sub.add_parser("install-plan", help="Print recommended FDMNES HPC installation/configuration pattern.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=install_plan_main)
    p = sub.add_parser("prepare", help="Prepare a reviewable FDMNES XANES workspace from VASP/POSCAR context.")
    p.add_argument("--structure", type=Path, help="POSCAR/CONTCAR structure. Defaults to --vasp-dir CONTCAR/POSCAR.")
    p.add_argument("--vasp-dir", type=Path, help="Converged VASP relaxation/SCF directory for structure and provenance.")
    p.add_argument("--absorber", required=True, help="Absorber element, e.g. Ce, U, Nb.")
    p.add_argument("--edge", default="L3", help="Absorption edge label passed to FDMNES, e.g. K, L2, L3, M4, M5.")
    p.add_argument("--absorber-index", type=int, default=0, help="1-based absorber atom index; defaults to first absorber element.")
    p.add_argument("--outdir", type=Path, default=Path("fdmnes_xanes"))
    p.add_argument("--output-prefix", default="", help="FDMNES Filout prefix. Defaults to fdmnes_<absorber>_<edge>.")
    p.add_argument("--radius", type=float, default=6.0, help="FDMNES cluster radius in Angstrom.")
    p.add_argument("--energy-range", default="-20 80 0.5", help="FDMNES Range line, e.g. '-20 80 0.5'.")
    p.add_argument("--green", action="store_true", default=True, help="Include Green keyword.")
    p.add_argument("--no-green", dest="green", action="store_false", help="Do not include Green keyword.")
    p.add_argument("--scf", action="store_true", default=True, help="Include SCF keyword.")
    p.add_argument("--no-scf", dest="scf", action="store_false", help="Do not include SCF keyword.")
    p.add_argument("--quadrupole", action="store_true", help="Include Quadrupole keyword.")
    p.add_argument("--spinorbit", action="store_true", help="Include Spinorbit keyword.")
    p.add_argument("--convolution", action="store_true", default=True, help="Include Convolution keyword.")
    p.add_argument("--no-convolution", dest="convolution", action="store_false", help="Do not include Convolution keyword.")
    p.add_argument("--extra", action="append", default=[], help="Extra line to append before Crystal block.")
    p.add_argument("--executable", default="")
    p.add_argument("--root", default="")
    p.add_argument("--bin", default="")
    p.add_argument("--module", default=os.environ.get("ATOMI_FDMNES_MODULE", ""))
    p.add_argument("--job-name", default="fdmnes-xanes")
    p.add_argument("--ntasks", type=int, default=1)
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--mem", default="8G")
    p.add_argument("--time", default="04:00:00")
    p.set_defaults(func=prepare_main)
    p = sub.add_parser("collect", help="Collect a compact summary from a FDMNES spectrum file.")
    p.add_argument("--fdmnes-dir", type=Path, default=Path("."))
    p.add_argument("--spectrum", type=Path)
    p.add_argument("--write", type=Path)
    p.set_defaults(func=collect_main)
    return parser


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


def status_cli() -> Any:
    return main(["status", *sys.argv[1:]])


def install_plan_cli() -> Any:
    return main(["install-plan", *sys.argv[1:]])


if __name__ == "__main__":
    main()
