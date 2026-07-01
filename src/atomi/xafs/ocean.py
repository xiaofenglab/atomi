"""Optional OCEAN bridge for periodic-solid XANES/XAS workflows.

OCEAN is kept outside Atomi's dependency set.  This bridge records the DFT+U
ground-state handoff, writes a conservative reviewable OCEAN workspace, and
collects lightweight spectrum summaries.  It is intentionally a bridge, not a
replacement for OCEAN's own convergence and input checks.
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



BOHR_PER_ANGSTROM = 1.8897261246257702

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

EDGE_QUANTUM_NUMBERS = {
    "K": (1, 0),
    "L1": (2, 0),
    "L2": (2, 1),
    "L3": (2, 1),
    "L": (2, 1),
    "M1": (3, 0),
    "M2": (3, 1),
    "M3": (3, 1),
    "M4": (3, 2),
    "M5": (3, 2),
    "M": (3, 2),
}


def _vec_norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


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
    lines = [line.strip() for line in path.expanduser().read_text(encoding="utf-8").splitlines() if line.strip()]
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
    typat: list[int] = []
    first_indices: dict[str, int] = {}
    atom_index = 1
    for type_index, (element, count) in enumerate(zip(elements, counts), start=1):
        first_indices.setdefault(element, atom_index)
        typat.extend([type_index] * count)
        atom_index += count
    return {
        "lattice": lattice,
        "elements": elements,
        "counts": counts,
        "frac": frac,
        "typat": typat,
        "first_indices": first_indices,
        "natom": total,
    }


def _edge_quantum(edge: str) -> tuple[int, int]:
    key = edge.strip().upper()
    if key not in EDGE_QUANTUM_NUMBERS:
        raise ValueError(f"Unsupported OCEAN edge label {edge!r}; expected one of {', '.join(sorted(EDGE_QUANTUM_NUMBERS))}")
    return EDGE_QUANTUM_NUMBERS[key]


def _format_block(values: list[Any], per_line: int = 12) -> str:
    parts = [str(value) for value in values]
    return "\n  ".join(" ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line))


def _pseudo_list(elements: list[str], pseudo_dir: str, pp_list: list[str] | None) -> list[str]:
    if pp_list:
        if len(pp_list) != len(elements):
            raise ValueError("--pp-list must provide one pseudopotential filename per POSCAR element, in POSCAR order")
        return pp_list
    root = Path(pseudo_dir).expanduser() if pseudo_dir else None
    names: list[str] = []
    for element in elements:
        match = None
        if root and root.exists():
            candidates = sorted(root.glob(f"{element}*.UPF")) + sorted(root.glob(f"{element}*.upf"))
            if candidates:
                match = candidates[0].name
        names.append(match or f"{element}.UPF")
    return names


def _write_native_ocean_input(args: argparse.Namespace, outdir: Path) -> tuple[Path, dict[str, Any]]:
    poscar = _parse_poscar(args.structure)
    absorber = args.absorber.strip()
    if absorber not in poscar["first_indices"]:
        raise ValueError(f"Absorber {absorber!r} is not present in {args.structure}")
    edge_n, edge_l = _edge_quantum(args.edge)
    edge_atom_index = int(getattr(args, "edge_atom_index", 0) or poscar["first_indices"][absorber])
    lengths_bohr = [_vec_norm(vec) * BOHR_PER_ANGSTROM for vec in poscar["lattice"]]
    rprim = [[component * BOHR_PER_ANGSTROM / length for component in vec] for vec, length in zip(poscar["lattice"], lengths_bohr)]
    pp_names = _pseudo_list(poscar["elements"], getattr(args, "pseudo_dir", ""), getattr(args, "pp_list", []))
    znucl = [ATOMIC_NUMBERS.get(element) for element in poscar["elements"]]
    missing = [element for element, z in zip(poscar["elements"], znucl) if z is None]
    if missing:
        raise ValueError(f"Missing atomic number mapping for: {', '.join(missing)}")
    nkpt = getattr(args, "nkpt", "") or "10 10 6"
    screen_nkpt = getattr(args, "screen_nkpt", "") or "2 2 2"
    xmesh = getattr(args, "xmesh", "") or nkpt
    lines = [
        "# OCEAN input generated by Atomi from POSCAR/CONTCAR.",
        "# Review against your local OCEAN version/tutorial before production.",
        "# NOTE: OCEAN edge n/l does not distinguish spin-orbit split L2/L3 or M4/M5 by itself.",
        "para_prefix { srun }",
        f"dft{{ {args.dft_engine} }}",
        f"nkpt {{ {nkpt} }}",
        f"screen.nkpt {{ {screen_nkpt} }}",
        f"screen.nbands {getattr(args, 'screen_nbands', 180)}",
        f"nbands {getattr(args, 'nbands', 240)}",
        "acell { " + " ".join(f"{x:.10f}" for x in lengths_bohr) + " }",
        "rprim {",
        *["  " + " ".join(f"{x:.16f}" for x in row) for row in rprim],
        "}",
        f"ntypat {{ {len(poscar['elements'])} }}",
        f"natom {{ {poscar['natom']} }}",
        "znucl { " + " ".join(str(z) for z in znucl) + " }",
        "ppdir { ../ }",
        "pp_list{ " + " ".join(pp_names) + " }",
        "typat {",
        "  " + _format_block(poscar["typat"]),
        "}",
        "xred {",
        *["  " + " ".join(f"{x:.16f}" for x in row) for row in poscar["frac"]],
        "}",
        f"ecut {getattr(args, 'ecut', '90')}",
        f"diemac {getattr(args, 'diemac', '10.0')}",
        f"CNBSE.xmesh {{ {xmesh} }}",
        f"edges{{ {edge_atom_index} {edge_n} {edge_l} }}",
        f"cnbse.broaden{{ {getattr(args, 'broaden', '0.4')} }}",
        "screen.shells{ 4.0 }",
        "cnbse.rad{ 4.0 }",
        "scfac 0.80",
    ]
    if args.energy_window:
        lines.append(f"# energy_window_note {args.energy_window}")
    if args.dft_plus_u:
        lines.append(f"# dft_plus_u_note {args.dft_plus_u}")
    if args.vasp_dir:
        lines.append(f"# vasp_dir {args.vasp_dir}")
    if args.extra:
        lines.extend(args.extra)
    path = outdir / "ocean.in"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    native = {
        "native_ocean_input": True,
        "edge_atom_index": edge_atom_index,
        "edge_quantum": {"n": edge_n, "l": edge_l},
        "elements": poscar["elements"],
        "counts": poscar["counts"],
        "pseudopotentials": pp_names,
        "ppdir_policy": "workspace-relative ../ is used because OCEAN OPF runs inside OPF/",
        "nkpt": nkpt,
        "screen_nkpt": screen_nkpt,
        "xmesh": xmesh,
    }
    return path, native


SPECTRUM_CANDIDATES = (
    "absspct",
    "absspct.dat",
    "ocean_absspct.dat",
    "xanes.dat",
    "xas.dat",
    "spectrum.dat",
)


def _json_dump(data: Any, path: Path | None = None) -> None:
    text = json.dumps(data, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _nonempty(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def default_ocean_executable() -> str:
    for key in ("ATOMI_OCEAN_EXE", "ATOMI_OCEAN_RUN", "OCEAN_EXE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    bin_dir = os.environ.get("ATOMI_OCEAN_BIN", "").strip()
    if bin_dir:
        for name in ("ocean.pl", "ocean", "ocean.pl"):
            candidate = Path(bin_dir).expanduser() / name
            if candidate.exists():
                return str(candidate)
    return shutil.which("ocean.pl") or shutil.which("ocean") or "ocean.pl"


def probe_ocean(executable: str | None = None, root: str | None = None, bin_dir: str | None = None) -> dict[str, Any]:
    exe = executable or default_ocean_executable()
    resolved = shutil.which(exe) if not Path(exe).is_absolute() else exe
    exists = bool(resolved and Path(resolved).expanduser().exists())
    payload: dict[str, Any] = {
        "available": exists,
        "executable": exe,
        "resolved_executable": str(Path(resolved).expanduser()) if resolved else "",
        "root": root or os.environ.get("ATOMI_OCEAN_ROOT", ""),
        "bin": bin_dir or os.environ.get("ATOMI_OCEAN_BIN", ""),
        "module": os.environ.get("ATOMI_OCEAN_MODULE", ""),
        "pseudo_dir": os.environ.get("ATOMI_OCEAN_PSEUDO_DIR", ""),
        "dft_engine": os.environ.get("ATOMI_OCEAN_DFT_ENGINE", "vasp"),
    }
    if exists:
        try:
            proc = subprocess.run(
                [str(Path(resolved).expanduser()), "--help"],
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


def status_main(args: argparse.Namespace) -> dict[str, Any]:
    report = {
        "schema": "atomi.ocean_xanes_status.v1",
        "ocean": probe_ocean(args.executable, args.root, args.bin),
        "environment": {
            "ATOMI_OCEAN_ROOT": os.environ.get("ATOMI_OCEAN_ROOT", ""),
            "ATOMI_OCEAN_BIN": os.environ.get("ATOMI_OCEAN_BIN", ""),
            "ATOMI_OCEAN_MODULE": os.environ.get("ATOMI_OCEAN_MODULE", ""),
            "ATOMI_OCEAN_EXE": os.environ.get("ATOMI_OCEAN_EXE", ""),
            "ATOMI_OCEAN_PSEUDO_DIR": os.environ.get("ATOMI_OCEAN_PSEUDO_DIR", ""),
            "ATOMI_OCEAN_DFT_ENGINE": os.environ.get("ATOMI_OCEAN_DFT_ENGINE", ""),
        },
    }
    if args.json:
        _json_dump(report)
    else:
        ocean = report["ocean"]
        print("Atomi OCEAN/XANES bridge status")
        print(f"  executable : {ocean.get('resolved_executable') or ocean.get('executable')}")
        print(f"  available  : {'yes' if ocean.get('available') else 'no'}")
        print(f"  root       : {ocean.get('root') or '(not set)'}")
        print(f"  module     : {ocean.get('module') or '(not set)'}")
        print(f"  pseudo dir : {ocean.get('pseudo_dir') or '(not set)'}")
    return report


def install_plan_main(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema": "atomi.ocean_xanes_install_plan.v1",
        "recommendation": "Keep OCEAN as a separate compiled MPI/runtime stack and call it from Atomi.",
        "why": [
            "OCEAN is a periodic-solid XAS/BSE code with compiled dependencies, MPI, pseudopotentials, and a DFT ground-state handoff.",
            "Installing OCEAN inside m_lammps_env would risk disturbing the stable Atomi/LAMMPS/VASP Python environment.",
            "OCEAN itself is normally a CPU/MPI workflow; GPU time is not needed for installation unless the upstream DFT/VASP step uses GPUs.",
        ],
        "bridge_roles": {
            "Molcas/OpenMolcas": "Small molecules or embedded clusters with explicit excited-state/core-hole treatment.",
            "FEFF/Larch": "MD/cluster XAFS and EXAFS-style ensemble comparison already supported by Atomi.",
            "OCEAN": "Extended periodic solids, XANES/XAS, BSE screening/core-hole physics, and DFT+U ground-state handoff.",
        },
        "hpc_pattern": [
            "Build or load OCEAN in a separate module/runtime, e.g. $HOME/atomi_hpc/ocean or a project software stack.",
            "Keep VASP/DFT+U ground-state preparation in the VASP module/workflow; record the VASP run directory in the OCEAN bridge metadata.",
            "Record OCEAN paths in private KIT JSON under profiles.ocean.",
            "Apply with: eval \"$(confighpc --config ~/atomi_hpc/atomi_hpc_config.kit.local.json --shell)\"",
            "Check with: ocean-xanes-status",
        ],
        "example_profile": {
            "profiles": {
                "ocean": {
                    "root": "$HOME/atomi_hpc/ocean",
                    "bin": "$HOME/atomi_hpc/ocean/bin",
                    "module": "",
                    "executable": "$HOME/atomi_hpc/ocean/bin/ocean.pl",
                    "pseudo_dir": "$HOME/atomi_hpc/ocean/pseudos",
                    "dft_engine": "vasp",
                    "environment": {
                        "ATOMI_OCEAN_ROOT": "$HOME/atomi_hpc/ocean",
                        "ATOMI_OCEAN_BIN": "$HOME/atomi_hpc/ocean/bin",
                        "ATOMI_OCEAN_MODULE": "",
                        "ATOMI_OCEAN_EXE": "$HOME/atomi_hpc/ocean/bin/ocean.pl",
                        "ATOMI_OCEAN_PSEUDO_DIR": "$HOME/atomi_hpc/ocean/pseudos",
                        "ATOMI_OCEAN_DFT_ENGINE": "vasp",
                    },
                }
            }
        },
    }
    if args.json:
        _json_dump(payload)
    else:
        print("OCEAN / Atomi HPC install plan")
        for item in payload["why"]:
            print(f"  - {item}")
        print("  Recommended: separate OCEAN runtime, configured through private KIT JSON.")
    return payload


def write_ocean_stub(args: argparse.Namespace, outdir: Path) -> Path:
    """Backward-compatible name for writing a native OCEAN input."""
    path, _ = _write_native_ocean_input(args, outdir)
    return path


def write_run_scripts(args: argparse.Namespace, outdir: Path, ocean_input: Path) -> dict[str, str]:
    exe = args.executable or default_ocean_executable()
    module_name = args.module or os.environ.get("ATOMI_OCEAN_MODULE", "")
    run_script = outdir / "run_ocean_xanes.sh"
    module_block = ""
    if module_name:
        module_block = (
            f"export ATOMI_OCEAN_MODULE={shlex.quote(str(module_name))}\n"
            'if ! type module >/dev/null 2>&1; then\n'
            '  source /etc/profile.d/modules.sh 2>/dev/null || true\n'
            'fi\n'
            'if type module >/dev/null 2>&1; then\n'
            '  module load "${ATOMI_OCEAN_MODULE}"\n'
            'else\n'
            '  echo "WARNING: module command is unavailable; expecting OCEAN runtime already on PATH." >&2\n'
            'fi\n'
        )
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${SCRIPT_DIR}"\n'
        f"{module_block}"
        f"export ATOMI_OCEAN_EXE={shlex.quote(str(exe))}\n"
        f"export ATOMI_OCEAN_ROOT={shlex.quote(str(args.root or os.environ.get('ATOMI_OCEAN_ROOT', '')))}\n"
        f"export ATOMI_OCEAN_BIN={shlex.quote(str(args.bin or os.environ.get('ATOMI_OCEAN_BIN', '')))}\n"
        f"export ATOMI_OCEAN_PSEUDO_DIR={shlex.quote(str(args.pseudo_dir or os.environ.get('ATOMI_OCEAN_PSEUDO_DIR', '')))}\n"
        'echo "Running OCEAN XANES bridge workspace"\n'
        f"{shlex.quote(str(exe))} {shlex.quote(ocean_input.name)}\n",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    sbatch = outdir / "submit_ocean_xanes.sbatch"
    sbatch.write_text(
        "#!/bin/bash\n"
        f"#SBATCH --job-name={args.job_name}\n"
        "#SBATCH --nodes=1\n"
        f"#SBATCH --ntasks={args.ntasks}\n"
        f"#SBATCH --cpus-per-task={args.cpus_per_task}\n"
        f"#SBATCH --mem={args.mem}\n"
        f"#SBATCH --time={args.time}\n"
        "#SBATCH --output=ocean_xanes.%j.out\n"
        "#SBATCH --error=ocean_xanes.%j.err\n"
        "\n"
        "set -euo pipefail\n"
        "bash run_ocean_xanes.sh\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    return {"run_script": str(run_script.resolve()), "sbatch_script": str(sbatch.resolve())}


def prepare_main(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    ocean_input, native_metadata = _write_native_ocean_input(args, outdir)
    scripts = write_run_scripts(args, outdir, ocean_input)
    metadata = {
        "schema": "atomi.ocean_xanes_project.v1",
        "mode": "prepare",
        "role": "periodic-solid OCEAN XANES/BSE bridge",
        "absorber": args.absorber,
        "edge": args.edge,
        "structure": str(args.structure.expanduser().resolve()),
        "vasp_dir": str(args.vasp_dir.expanduser().resolve()) if args.vasp_dir else "",
        "dft_engine": args.dft_engine,
        "dft_plus_u": args.dft_plus_u,
        "module": args.module or os.environ.get("ATOMI_OCEAN_MODULE", ""),
        "ocean_input": str(ocean_input.resolve()),
        "native_ocean": native_metadata,
        "scripts": scripts,
        "recommendations": [
            "Use VASP/DFT+U to converge the periodic ground state before OCEAN.",
            "Check absorber species, edge, k-grid, band count, screening, and broadening in native OCEAN inputs.",
            "Benchmark against a simple oxide/carbide reference before production actinide or mixed-valence spectra.",
        ],
    }
    (outdir / "ocean_xanes_project.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote OCEAN XANES workspace: {outdir}")
    print(f"Wrote native OCEAN input: {ocean_input}")
    return metadata


def read_numeric_curve(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return rows


def collect_main(args: argparse.Namespace) -> dict[str, Any]:
    root = args.ocean_dir.expanduser().resolve()
    candidates = [args.spectrum] if args.spectrum else [root / name for name in SPECTRUM_CANDIDATES]
    spectrum = next((path for path in candidates if path and path.exists()), None)
    if spectrum is None:
        raise FileNotFoundError(f"No OCEAN spectrum found in {root}; checked {', '.join(SPECTRUM_CANDIDATES)}")
    rows = read_numeric_curve(spectrum)
    if not rows:
        raise ValueError(f"No numeric two-column spectrum data found in {spectrum}")
    energies = [row[0] for row in rows]
    intensities = [row[1] for row in rows]
    summary = {
        "schema": "atomi.ocean_xanes_summary.v1",
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
        p = sub.add_parser(name, help="Check configured OCEAN executable/runtime.")
        p.add_argument("--executable")
        p.add_argument("--root")
        p.add_argument("--bin")
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=status_main)
    p = sub.add_parser("install-plan", help="Print recommended OCEAN HPC installation/configuration pattern.")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=install_plan_main)
    p = sub.add_parser("prepare", help="Prepare a reviewable OCEAN XANES workspace from VASP/DFT+U context.")
    p.add_argument("--structure", type=Path, required=True, help="POSCAR/CIF/structure used for the periodic solid.")
    p.add_argument("--vasp-dir", type=Path, help="Converged VASP/DFT+U ground-state directory to hand off.")
    p.add_argument("--absorber", required=True, help="Absorber element, e.g. U, O, Gd.")
    p.add_argument("--edge", default="K", help="Absorption edge label, e.g. K, L3, M4.")
    p.add_argument("--outdir", type=Path, default=Path("ocean_xanes"))
    p.add_argument("--dft-engine", default=os.environ.get("ATOMI_OCEAN_DFT_ENGINE", "vasp"))
    p.add_argument("--dft-plus-u", default="", help="Short DFT+U note, e.g. 'VASP LDAU U=4.5 eV on U 5f'.")
    p.add_argument("--executable", default="")
    p.add_argument("--root", default="")
    p.add_argument("--bin", default="")
    p.add_argument("--module", default=os.environ.get("ATOMI_OCEAN_MODULE", ""), help="Environment module to load before running OCEAN, e.g. chem/ocean/2.9.7.")
    p.add_argument("--pseudo-dir", default=os.environ.get("ATOMI_OCEAN_PSEUDO_DIR", ""))
    p.add_argument("--energy-window", default="", help="Human-readable energy window note, e.g. '-10 60 eV'.")
    p.add_argument("--edge-atom-index", type=int, default=0, help="1-based absorber atom index for OCEAN edges; defaults to first absorber atom.")
    p.add_argument("--nkpt", default="10 10 6", help="OCEAN ground-state k mesh, e.g. '10 10 6'.")
    p.add_argument("--screen-nkpt", default="2 2 2", help="OCEAN screening k mesh, e.g. '2 2 2'.")
    p.add_argument("--xmesh", default="", help="CNBSE.xmesh; defaults to --nkpt.")
    p.add_argument("--nbands", type=int, default=240)
    p.add_argument("--screen-nbands", type=int, default=180)
    p.add_argument("--ecut", default="90")
    p.add_argument("--diemac", default="10.0")
    p.add_argument("--broaden", default="0.4")
    p.add_argument("--pp-list", nargs="*", default=[], help="Pseudopotential filenames in POSCAR element order.")
    p.add_argument("--extra", action="append", default=[], help="Extra line to append to ocean.in scaffold.")
    p.add_argument("--job-name", default="ocean-xanes")
    p.add_argument("--ntasks", type=int, default=8)
    p.add_argument("--cpus-per-task", type=int, default=1)
    p.add_argument("--mem", default="16G")
    p.add_argument("--time", default="08:00:00")
    p.set_defaults(func=prepare_main)
    p = sub.add_parser("collect", help="Collect a compact summary from an OCEAN spectrum file.")
    p.add_argument("--ocean-dir", type=Path, default=Path("."))
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
