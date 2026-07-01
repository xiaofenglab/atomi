"""Optional OCEAN bridge for periodic-solid XANES/XAS workflows.

OCEAN is kept outside Atomi's dependency set.  This bridge records the DFT+U
ground-state handoff, writes a conservative reviewable OCEAN workspace, and
collects lightweight spectrum summaries.  It is intentionally a bridge, not a
replacement for OCEAN's own convergence and input checks.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


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
                    "executable": "$HOME/atomi_hpc/ocean/bin/ocean.pl",
                    "pseudo_dir": "$HOME/atomi_hpc/ocean/pseudos",
                    "dft_engine": "vasp",
                    "environment": {
                        "ATOMI_OCEAN_ROOT": "$HOME/atomi_hpc/ocean",
                        "ATOMI_OCEAN_BIN": "$HOME/atomi_hpc/ocean/bin",
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
    lines = [
        "# OCEAN input scaffold generated by Atomi.",
        "# Review against your local OCEAN version/tutorial before production.",
        f"absorber {args.absorber}",
        f"edge {args.edge}",
        f"dft_engine {args.dft_engine}",
        f"structure {args.structure}",
    ]
    if args.vasp_dir:
        lines.append(f"vasp_dir {args.vasp_dir}")
    if args.pseudo_dir:
        lines.append(f"pseudo_dir {args.pseudo_dir}")
    if args.energy_window:
        lines.append(f"energy_window {args.energy_window}")
    if args.extra:
        lines.extend(args.extra)
    path = outdir / "ocean.in"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_run_scripts(args: argparse.Namespace, outdir: Path, ocean_input: Path) -> dict[str, str]:
    exe = args.executable or default_ocean_executable()
    run_script = outdir / "run_ocean_xanes.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'cd "${SCRIPT_DIR}"\n'
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
    ocean_input = write_ocean_stub(args, outdir)
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
        "ocean_input": str(ocean_input.resolve()),
        "scripts": scripts,
        "recommendations": [
            "Use VASP/DFT+U to converge the periodic ground state before OCEAN.",
            "Check absorber species, edge, k-grid, band count, screening, and broadening in native OCEAN inputs.",
            "Benchmark against a simple oxide/carbide reference before production actinide or mixed-valence spectra.",
        ],
    }
    (outdir / "ocean_xanes_project.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote OCEAN XANES workspace: {outdir}")
    print(f"Wrote OCEAN input scaffold: {ocean_input}")
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
    p.add_argument("--pseudo-dir", default=os.environ.get("ATOMI_OCEAN_PSEUDO_DIR", ""))
    p.add_argument("--energy-window", default="", help="Human-readable energy window note, e.g. '-10 60 eV'.")
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
    return args.func(args)


def status_cli() -> Any:
    return main(["status", *sys.argv[1:]])


def install_plan_cli() -> Any:
    return main(["install-plan", *sys.argv[1:]])


if __name__ == "__main__":
    main()
