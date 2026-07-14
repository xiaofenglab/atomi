"""Optional FDMNES bridge for quick periodic/cluster XANES screens.

FDMNES is kept outside Atomi's dependency set.  This bridge prepares a
reviewable FDMNES workspace from a VASP-relaxed structure, records VASP
provenance, writes a Slurm wrapper, and collects lightweight spectrum
summaries.  It is a route-C scaffold for fast XANES screening, not a substitute
for project-specific FDMNES convergence checks or for Molcas/OCEAN physics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
ATOMIC_SYMBOLS = {z: symbol for symbol, z in ATOMIC_NUMBERS.items()}

EDGE_CORE_ORBITALS = {
    "K": "1s",
    "L1": "2s",
    "L2": "2p1/2",
    "L3": "2p3/2",
    "M1": "3s",
    "M2": "3p1/2",
    "M3": "3p3/2",
    "M4": "3d3/2",
    "M5": "3d5/2",
    "N4": "4d3/2",
    "N5": "4d5/2",
}

D_TARGET_ORBITALS = {
    "Ti": "3d",
    "Nb": "4d",
    "Ce": "5d",
    "Gd": "5d",
    "U": "6d",
}

F_TARGET_ORBITALS = {
    "Ce": "4f",
    "Gd": "4f",
    "U": "5f",
}

ELEMENT_NAMES = {
    "cerium": "Ce",
    "gadolinium": "Gd",
    "uranium": "U",
    "niobium": "Nb",
    "titanium": "Ti",
    "calcium": "Ca",
    "oxygen": "O",
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


def _normalize_poscar_element(label: str) -> str:
    """Map VASP split-species labels such as U_probe or Ce4plus to elements."""
    cleaned = label.strip()
    if cleaned in ATOMIC_NUMBERS:
        return cleaned
    title = cleaned[:1].upper() + cleaned[1:]
    if title in ATOMIC_NUMBERS:
        return title
    for symbol in sorted(ATOMIC_NUMBERS, key=len, reverse=True):
        if cleaned.startswith(symbol):
            return symbol
        if title.startswith(symbol):
            return symbol
    raise ValueError(f"Missing atomic number mapping for POSCAR element {label!r}")


def _parse_poscar(path: Path) -> dict[str, Any]:
    text = path.expanduser().read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path} does not look like a POSCAR/CONTCAR")
    scale = float(lines[1].split()[0])
    lattice = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    raw_elements = lines[5].split()
    elements = [_normalize_poscar_element(element) for element in raw_elements]
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
        z = ATOMIC_NUMBERS[element]
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
        "raw_elements": raw_elements,
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
        raise ValueError(
            "--energy-range should follow FDMNES Range grammar, e.g. '-20 0.5 80' "
            "for first energy, step, last energy."
        )
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
        "raw_elements": poscar["raw_elements"],
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
    fdmfile = outdir / "fdmfile.txt"
    fdmfile.write_text(f"1\n{input_file.name}\n", encoding="utf-8")
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
        f"cat > fdmfile.txt <<'FDMNES_FDMFILE'\n1\n{input_file.name}\nFDMNES_FDMFILE\n"
        'echo "Running FDMNES route-C XANES bridge workspace"\n'
        'echo "Started $(date -Is)"\n'
        'echo "Started $(date -Is) status=RUNNING" > fdmnes.status.txt\n'
        "set +e\n"
        f"{shlex.quote(str(exe))} > fdmnes.stdout.log 2> fdmnes.stderr.log\n"
        "status=$?\n"
        "if grep -Eqi 'Error opening the file|fdmfile.txt' fdmnes.stdout.log fdmnes.stderr.log; then\n"
        '  echo "FDMNES did not consume the generated fdmfile.txt/input deck." >&2\n'
        "  status=2\n"
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
        f"#SBATCH --chdir={outdir.resolve()}\n"
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
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index: int | None = None
    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "energy" in lowered and ("xanes" in lowered or "mu" in lowered):
            header_index = idx
            break
    if header_index is not None:
        lines = lines[header_index + 1 :]
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "Title", "End")):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        if header_index is not None and len(parts) != 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return rows


def _find_raw_spectrum(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    skip_fragments = ("_conv", "_bav", "stdout", "stderr", "status", "project", "fdmnes.inp", "fdmfile")
    for candidate in sorted(root.glob("*.txt")) + sorted(root.glob("*.dat")):
        lowered = candidate.name.lower()
        if any(fragment in lowered for fragment in skip_fragments):
            continue
        if read_numeric_curve(candidate):
            return candidate
    raise FileNotFoundError(f"No raw/unbroadened numeric FDMNES spectrum found in {root}")


def _next_keyword_value(lines: list[str], keyword: str) -> str:
    lowered = keyword.lower()
    for idx, line in enumerate(lines[:-1]):
        if line.strip().lower() == lowered:
            return lines[idx + 1].strip()
    return ""


def _parse_z_from_spectrum_header(path: Path) -> int | None:
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except IndexError:
        return None
    if "E_edge" not in first or "Z" not in first:
        return None
    parts = first.split("=", 1)[0].split()
    if len(parts) < 2:
        return None
    try:
        return int(float(parts[1]))
    except ValueError:
        return None


def _parse_edge_from_spectrum_header(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except IndexError:
        return ""
    if "E_edge" not in first or "n_edge" not in first:
        return ""
    parts = first.split("=", 1)[0].split()
    if len(parts) < 4:
        return ""
    try:
        n_edge = int(float(parts[2]))
        j_edge = int(float(parts[3]))
    except ValueError:
        return ""
    if n_edge == 1:
        return "K"
    if n_edge == 2 and j_edge == 1:
        return "L1"
    if n_edge == 2 and j_edge == 2:
        return "L2"
    if n_edge == 2 and j_edge == 3:
        return "L3"
    if n_edge == 3 and j_edge in {4, 5}:
        return f"M{j_edge}"
    return ""


def _parse_fdmnes_context(root: Path, spectrum: Path | None = None) -> dict[str, Any]:
    """Extract conservative edge/channel labels from a FDMNES workspace.

    FDMNES is a multiple-scattering continuum code.  Unless a projected
    transition table is supplied by a separate analysis, local maxima are
    feature markers rather than state-resolved oscillator strengths.
    """

    context: dict[str, Any] = {
        "absorber": "",
        "absorber_z": None,
        "edge": "",
        "core_orbital": "",
        "transition_operator": "dipole",
        "target_orbital": "unoccupied continuum",
        "ligand_hint": "",
        "state_resolved": False,
        "source": "fdmnes workspace metadata",
    }

    input_path = root / "fdmnes.inp"
    if input_path.exists():
        lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
        edge = _next_keyword_value(lines, "Edge").split()[0] if _next_keyword_value(lines, "Edge") else ""
        if edge:
            context["edge"] = edge.upper()
        z_values: list[int] = []
        in_crystal = False
        for raw in lines:
            stripped = raw.strip()
            if not stripped or stripped.startswith("!"):
                continue
            if stripped.lower() == "crystal":
                in_crystal = True
                continue
            if not in_crystal:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                z_values.append(int(float(parts[0])))
            except ValueError:
                continue
        if z_values:
            context["absorber_z"] = z_values[0]
            context["absorber"] = ATOMIC_SYMBOLS.get(z_values[0], f"Z{z_values[0]}")
            if 8 in z_values:
                context["ligand_hint"] = "O 2p hybridization possible"

    if spectrum is not None:
        z = _parse_z_from_spectrum_header(spectrum)
        if z is not None:
            context["absorber_z"] = z
            context["absorber"] = ATOMIC_SYMBOLS.get(z, f"Z{z}")
        edge = _parse_edge_from_spectrum_header(spectrum)
        if edge:
            context["edge"] = edge

    for candidate in sorted(root.glob("*_bav.txt")) + [root / "fdmnes.stdout.log"]:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        threshold = re.search(r"Threshold:\s+([A-Za-z]+)\s+([A-Za-z0-9]+)\s+edge", text)
        if threshold:
            absorber_text = threshold.group(1)
            absorber_symbol = ELEMENT_NAMES.get(absorber_text.lower(), "")
            if not absorber_symbol:
                try:
                    absorber_symbol = _normalize_poscar_element(absorber_text)
                except ValueError:
                    absorber_symbol = absorber_text
            context["absorber"] = absorber_symbol
            context["absorber_z"] = ATOMIC_NUMBERS.get(absorber_symbol, context["absorber_z"])
            context["edge"] = threshold.group(2).upper()
        if "Quadrupole component" in text:
            context["transition_operator"] = "dipole/quadrupole"
        elif "Dipole component" in text:
            context["transition_operator"] = "dipole"
        if re.search(r"\bO\b|\bZ\s*=\s*8\b|^\s*\d+\s+8\s+", text, re.MULTILINE):
            context["ligand_hint"] = "O 2p hybridization possible"
        break

    edge = str(context.get("edge") or "").upper()
    absorber = str(context.get("absorber") or "absorber")
    core = EDGE_CORE_ORBITALS.get(edge, f"{edge} core" if edge else "core")
    context["core_orbital"] = core
    if edge in {"K", "L1", "M1"}:
        target = "p-like unoccupied states"
    elif edge in {"L2", "L3", "M2", "M3"}:
        target = D_TARGET_ORBITALS.get(absorber, "d-like unoccupied states")
    elif edge in {"M4", "M5", "N4", "N5"}:
        target = F_TARGET_ORBITALS.get(absorber, "f-like unoccupied states")
    else:
        target = "unoccupied continuum states"
    if str(context.get("transition_operator")) == "dipole/quadrupole":
        target = f"{target}; quadrupole channel may add higher-l character"
    context["target_orbital"] = target
    return context


def _feature_region(energy: float, peak_energy: float) -> str:
    if energy < 0:
        return "pre-edge"
    if energy <= peak_energy - 3:
        return "rising-edge"
    if abs(energy - peak_energy) <= 3:
        return "white-line"
    if energy <= peak_energy + 25:
        return "near-edge shoulder"
    return "post-edge/MS"


def _feature_assignment(context: dict[str, Any], *, energy: float, peak_energy: float, index: int) -> tuple[str, str]:
    absorber = str(context.get("absorber") or "Abs")
    edge = str(context.get("edge") or "edge")
    core = str(context.get("core_orbital") or "core")
    target = str(context.get("target_orbital") or "unoccupied continuum")
    ligand = str(context.get("ligand_hint") or "ligand hybridization not resolved")
    operator = str(context.get("transition_operator") or "dipole")
    region = _feature_region(energy, peak_energy)
    short_target = target.replace(" unoccupied states", "").replace(" unoccupied continuum states", " continuum")
    short_target = short_target.replace("states; quadrupole channel may add higher-l character", "+higher-l")
    if "O 2p" in ligand and "O2p" not in short_target and ("d" in short_target or "continuum" in short_target):
        short_target = f"{short_target}/O2p"
    state_label = f"{absorber} {edge} {core}->{short_target} #{index}"
    assignment = (
        f"FDMNES {region} continuum feature; {operator}-allowed {absorber} {edge} "
        f"{core} -> {absorber} {target}; {ligand}; not a state-resolved transition."
    )
    return state_label, assignment


def extract_feature_sticks(
    rows: list[tuple[float, float]],
    *,
    min_relative: float = 0.08,
    min_separation_ev: float = 1.0,
    max_peaks: int = 24,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    finite = [(energy, intensity) for energy, intensity in rows if math.isfinite(energy) and math.isfinite(intensity)]
    if not finite:
        return []
    ymin = min(intensity for _, intensity in finite)
    ymax = max(intensity for _, intensity in finite)
    span = max(ymax - ymin, ymax, 1.0e-30)
    threshold = ymin + max(0.0, min(1.0, min_relative)) * (ymax - ymin)
    candidates: list[tuple[float, float, float]] = []
    for idx in range(1, len(finite) - 1):
        energy, intensity = finite[idx]
        if intensity < threshold:
            continue
        if intensity >= finite[idx - 1][1] and intensity >= finite[idx + 1][1]:
            candidates.append((energy, intensity, (intensity - ymin) / span))
    if not candidates:
        energy, intensity = max(finite, key=lambda row: row[1])
        candidates.append((energy, intensity, (intensity - ymin) / span))

    selected: list[tuple[float, float, float]] = []
    for energy, intensity, relative in sorted(candidates, key=lambda row: row[1], reverse=True):
        if any(abs(energy - kept_energy) < min_separation_ev for kept_energy, _, _ in selected):
            continue
        selected.append((energy, intensity, relative))
        if len(selected) >= max_peaks:
            break

    peak_energy = max(finite, key=lambda row: row[1])[0]
    context = context or {}
    sticks: list[dict[str, Any]] = []
    for idx, (energy, intensity, relative) in enumerate(sorted(selected, key=lambda row: row[0]), start=1):
        state_label, assignment = _feature_assignment(context, energy=energy, peak_energy=peak_energy, index=idx)
        sticks.append(
            {
                "energy_rel_eV": energy,
                "intensity": max(intensity - ymin, 0.0),
                "relative_intensity": relative,
                "state_label": state_label,
                "assignment": assignment,
                "absorber": context.get("absorber", ""),
                "edge": context.get("edge", ""),
                "core_orbital": context.get("core_orbital", ""),
                "transition_operator": context.get("transition_operator", ""),
                "target_orbital": context.get("target_orbital", ""),
                "feature_region": _feature_region(energy, peak_energy),
                "state_resolved": "false",
            }
        )
    return sticks


def write_feature_sticks_csv(path: Path, sticks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "energy_rel_eV",
        "intensity",
        "relative_intensity",
        "state_label",
        "assignment",
        "absorber",
        "edge",
        "core_orbital",
        "transition_operator",
        "target_orbital",
        "feature_region",
        "state_resolved",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sticks:
            writer.writerow(
                {
                    "energy_rel_eV": f"{float(row['energy_rel_eV']):.10g}",
                    "intensity": f"{float(row['intensity']):.10g}",
                    "relative_intensity": f"{float(row['relative_intensity']):.10g}",
                    "state_label": row["state_label"],
                    "assignment": row["assignment"],
                    "absorber": row.get("absorber", ""),
                    "edge": row.get("edge", ""),
                    "core_orbital": row.get("core_orbital", ""),
                    "transition_operator": row.get("transition_operator", ""),
                    "target_orbital": row.get("target_orbital", ""),
                    "feature_region": row.get("feature_region", ""),
                    "state_resolved": row.get("state_resolved", "false"),
                }
            )


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
    write_feature_sticks = getattr(args, "write_feature_sticks", None)
    if write_feature_sticks:
        feature_source = _find_raw_spectrum(root, getattr(args, "feature_source", None))
        feature_rows = read_numeric_curve(feature_source)
        feature_context = _parse_fdmnes_context(root, spectrum=feature_source)
        feature_sticks = extract_feature_sticks(
            feature_rows,
            min_relative=getattr(args, "feature_min_relative", 0.08),
            min_separation_ev=getattr(args, "feature_min_separation_ev", 1.0),
            max_peaks=getattr(args, "feature_max_peaks", 24),
            context=feature_context,
        )
        write_feature_sticks_csv(write_feature_sticks, feature_sticks)
        summary["feature_sticks"] = {
            "source": str(feature_source.resolve()),
            "csv": str(write_feature_sticks),
            "n_features": len(feature_sticks),
            "context": feature_context,
            "min_relative": getattr(args, "feature_min_relative", 0.08),
            "min_separation_ev": getattr(args, "feature_min_separation_ev", 1.0),
            "max_peaks": getattr(args, "feature_max_peaks", 24),
            "note": "FDMNES feature sticks are raw-spectrum local maxima annotated by edge/channel metadata, not state-resolved oscillator-strength transitions.",
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
    p.add_argument(
        "--energy-range",
        default="-20 0.5 80",
        help="FDMNES Range line: first energy, step, intermediate energy, step, last energy; e.g. '-20 0.5 80'.",
    )
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
    p.add_argument("--write-feature-sticks", type=Path, help="Write raw FDMNES local maxima as feature-stick CSV for xanes-overlay --sticks.")
    p.add_argument("--feature-source", type=Path, help="Raw/unbroadened FDMNES spectrum used for feature-stick extraction.")
    p.add_argument("--feature-min-relative", type=float, default=0.08)
    p.add_argument("--feature-min-separation-ev", type=float, default=1.0)
    p.add_argument("--feature-max-peaks", type=int, default=24)
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
