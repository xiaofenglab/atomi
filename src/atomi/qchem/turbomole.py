"""Turbomole setup helpers for relaxation-stage quantum chemistry workflows.

The key output is a blank-line-preserving stdin file for ``define``.  This is
intentionally plain text because Turbomole's interactive setup is sensitive to
empty answers and menu exits.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


CLEAN_RUN_FILES = (
    "energy",
    "gradient",
    "mos",
    "alpha",
    "beta",
    "statistics",
    "damping",
    "restartfile",
    "job.last",
    "GEO_OPT_RUNNING",
    "converged",
    "not.converged",
    "basis",
    "auxbasis",
    "dens*",
    "diff_dens*",
    "fock*",
    "errvec",
    "oldfock",
    "lmo",
    "trajectory*",
    "out.ccf",
)


@dataclass(frozen=True)
class DefineOptions:
    coord: str = "coord"
    symmetry: str = "ci"
    use_desy: bool = False
    desy_tolerance: str = "1d-1"
    use_ired: bool = True
    use_ecp: bool = True
    charge: int = 0
    dft: bool = True
    grid: str = "m4"
    ri: bool = True
    scf_iter: int = 240
    shift: str = "0.25"
    damp_start: str = "5.0"
    damp_end: str = "0.5"


def build_define_lines(options: DefineOptions) -> list[str]:
    """Return the exact stdin lines for Turbomole ``define``.

    Empty strings are deliberate blank answers to the interactive prompts.
    """

    lines = [f"a {options.coord}"]
    if options.use_desy:
        lines.append(f"desy {options.desy_tolerance}")
    elif options.symmetry:
        lines.append(f"sy {options.symmetry}")
    if options.use_ired:
        lines.append("ired")
    lines.extend(["*", "bl"])
    if options.use_ecp:
        lines.append("ecpl")
    lines.extend(["*", "eht", "", str(options.charge), ""])
    if options.dft:
        lines.extend(["dft", "on", f"grid {options.grid}", "*"])
    lines.extend(["ri", "on" if options.ri else "off", "*"])
    lines.extend(["scf", "iter", str(options.scf_iter), "shift", "", str(options.shift), ""])
    lines.extend(["damp", str(options.damp_start), str(options.damp_end), "", "", "*", "*", "*"])
    return lines


def render_define_stdin(options: DefineOptions) -> str:
    return "\n".join(build_define_lines(options)) + "\n"


def render_clean_script() -> str:
    quoted = " \\\n      ".join(CLEAN_RUN_FILES)
    return f"""#!/usr/bin/env bash
set -euo pipefail
rm -f {quoted}
"""


def render_run_define_script(xyz: str, define_stdin: str = "define.stdin", coord: str = "coord") -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail
: "${{TURBODIR:?Load Turbomole first, e.g. mlt/module load turbomole.}}"
x2t {shlex.quote(xyz)} > {shlex.quote(coord)}
define < {shlex.quote(define_stdin)} | tee define.log
"""


def render_relax_sbatch(job_name: str = "Turbomole", tasks: int = 4, hours: int = 12, module: str = "chem/turbomole/7.5") -> str:
    """Render the JUSTUS2-style Turbomole relaxation sbatch used in U-BPDC.

    The calculation sequence follows the existing U-BPDC runs:
    ``ridft`` -> ``jobex -ri -c 200`` -> ``aoforce -ri -central``.
    """

    safe_job = job_name[:30] or "Turbomole"
    return f"""#!/bin/bash
#SBATCH --job-name {safe_job}
#SBATCH --ntasks={tasks} --nodes=1
#SBATCH --mem-per-cpu=2000
#SBATCH --time={hours}:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ka_hq7637

set -euo pipefail
unset LANG; unset LC_CTYPE
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export QS_USER=${{SLURM_JOB_USER:=$(logname)}}
export QS_JOBID=${{SLURM_JOB_ID:=$(date +%s)}}
export QS_SUBMITDIR=${{SLURM_SUBMIT_DIR:=$(pwd)}}
export QS_JOBNAME=${{SLURM_JOB_NAME:=$(basename "$0")}}
export QS_NPROCS=${{SLURM_NTASKS:=1}}
export QS_NNODES=${{SLURM_NNODES:=1}}

runDIR=${{TMPDIR:-/tmp/${{USER}}_job_${{QS_JOBID}}}}
if [[ "${{runDIR}}" == "/scratch" ]] ; then runDIR="/scratch/$USER/Job_${{QS_JOBID}}" ; fi
if [[ "${{runDIR}}" == "/tmp"     ]] ; then runDIR="/tmp/$USER/Job_${{QS_JOBID}}" ; fi
mkdir -vp "${{runDIR}}"
cd "${{runDIR}}"

module unload chem/turbomole || true
export TURBOMOLE_MODE="compute"
if [ "$QS_NPROCS" -gt 1 ] ; then
  export PARA_ARCH="SMP"
  export PARNODES="$QS_NPROCS"
else
  unset PARNODES || true
  unset PARA_ARCH || true
fi
module load {module}
export TURBOTMPDIR="${{runDIR}}"

cp -v "$QS_SUBMITDIR"/{{coord,*basis,control,mos,alpha,beta}} "$runDIR"/ 2>/dev/null || \
cp -v "$QS_SUBMITDIR"/{{coord,*basis,control}} "$runDIR"/

time ridft > "$QS_SUBMITDIR/ridft.out.$QS_JOBID"
time jobex -ri -c 200
cp job.last "$QS_SUBMITDIR/job.last.$QS_JOBID" || true
time aoforce -ri -central > "$QS_SUBMITDIR/aoforce.out.$QS_JOBID"

rm -vf dens errvec fock oldfock slave* || true
cp -v coord control energy gradient mos alpha beta "$QS_SUBMITDIR"/ 2>/dev/null || true
if command -v t2x >/dev/null 2>&1; then
  t2x coord > "$QS_SUBMITDIR/final.xyz" || true
fi
cd "$(dirname "$runDIR")"
tar -zcvf "$QS_SUBMITDIR/${{QS_JOBID}}.tgz" "$(basename "$runDIR")"
"""


def render_basis_policy_readme() -> str:
    return """# Turbomole to OpenMolcas Basis Policy

This workspace is intended for Turbomole geometry preconditioning/relaxation.
For actinide spectroscopy and covalency interpretation, especially U4O9 U M-edge
HERFD-XANES work, keep the electronic-structure stage in OpenMolcas with U
ANO-RCC basis sets. The JUSTUS2 Turbomole 7.5 installation used in the U-BPDC
examples provides U ECP/def-style basis choices, not the OpenMolcas ANO-RCC U
basis used for the final ground-state and excited-state calculations.

Consistent handoff rule:

- Turbomole output may be used as a structural preconditioner when its U ECP
  setup is documented.
- OpenMolcas/ANO-RCC output is the source for U electronic configuration,
  covalency, M-edge excitation, and HERFD-XANES interpretation.
- Do not compare Turbomole and OpenMolcas total energies as if they came from
  the same Hamiltonian. Compare geometries, then run the spectroscopy model in
  OpenMolcas.
- If a custom Turbomole ANO-RCC-like all-electron basis is later imported, record
  the exact basis provenance in this folder before submitting calculations.
"""


def write_define_workspace(
    outdir: Path,
    xyz: Path,
    options: DefineOptions,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    targets = {
        "define_stdin": outdir / "define.stdin",
        "clean_script": outdir / "clean_turbomole_run.sh",
        "run_script": outdir / "run_define.sh",
        "metadata": outdir / "turbomole_define_metadata.json",
        "relax_sbatch": outdir / "runturbo_relax.sh",
        "basis_policy": outdir / "README_basis_policy.md",
    }
    for path in targets.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    targets["define_stdin"].write_text(render_define_stdin(options), encoding="utf-8")
    targets["clean_script"].write_text(render_clean_script(), encoding="utf-8")
    targets["run_script"].write_text(render_run_define_script(str(xyz), "define.stdin", options.coord), encoding="utf-8")
    targets["relax_sbatch"].write_text(render_relax_sbatch(job_name=outdir.name[:30]), encoding="utf-8")
    targets["basis_policy"].write_text(render_basis_policy_readme(), encoding="utf-8")
    targets["clean_script"].chmod(0o755)
    targets["run_script"].chmod(0o755)
    targets["relax_sbatch"].chmod(0o755)
    payload = {
        "schema": "atomi.qchem.turbomole_define.v1",
        "xyz": str(xyz),
        "options": asdict(options),
        "outputs": {key: str(path) for key, path in targets.items()},
        "note": "define.stdin intentionally preserves blank lines for Turbomole interactive prompts.",
        "basis_policy": "Turbomole is treated as a geometry-preconditioning stage for U4O9 unless a custom U ANO-RCC-equivalent basis is explicitly imported. OpenMolcas/ANO-RCC remains the spectroscopy/covalency stage.",
    }
    targets["metadata"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {key: str(path) for key, path in targets.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Turbomole coord/define setup files.")
    parser.add_argument("--xyz", type=Path, required=True, help="Input XYZ geometry passed to x2t.")
    parser.add_argument("--outdir", type=Path, default=Path("turbomole_define_setup"))
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--symmetry", default="ci", help="Symmetry command after coord load, e.g. ci or c1.")
    parser.add_argument("--desy", action="store_true", help="Use desy instead of sy.")
    parser.add_argument("--desy-tolerance", default="1d-1")
    parser.add_argument("--no-ired", action="store_true")
    parser.add_argument("--no-ecp", action="store_true")
    parser.add_argument("--no-dft", action="store_true")
    parser.add_argument("--grid", default="m4")
    parser.add_argument("--ri", choices=("on", "off"), default="on")
    parser.add_argument("--scf-iter", type=int, default=240)
    parser.add_argument("--shift", default="0.25")
    parser.add_argument("--damp-start", default="5.0")
    parser.add_argument("--damp-end", default="0.5")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = DefineOptions(
        symmetry=args.symmetry,
        use_desy=args.desy,
        desy_tolerance=args.desy_tolerance,
        use_ired=not args.no_ired,
        use_ecp=not args.no_ecp,
        charge=args.charge,
        dft=not args.no_dft,
        grid=args.grid,
        ri=args.ri == "on",
        scf_iter=args.scf_iter,
        shift=args.shift,
        damp_start=args.damp_start,
        damp_end=args.damp_end,
    )
    outputs = write_define_workspace(args.outdir, args.xyz, options, overwrite=args.overwrite)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
