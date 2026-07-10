"""External-runtime bridge for Quantum ESPRESSO and Wannier90 workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


QE_EXECUTABLES = {
    "pw": ("ATOMI_QE_PW", "pw.x"),
    "hp": ("ATOMI_QE_HP", "hp.x"),
    "pw2wannier90": ("ATOMI_QE_PW2WANNIER90", "pw2wannier90.x"),
    "wannier2pw": ("ATOMI_QE_WANNIER2PW", "wannier2pw.x"),
}
W90_EXECUTABLES = {
    "wannier90": ("ATOMI_WANNIER90_EXE", "wannier90.x"),
    "postw90": ("ATOMI_POSTW90_EXE", "postw90.x"),
}


def _resolve_executable(env_key: str, name: str, bin_env: str) -> str:
    explicit = os.environ.get(env_key, "").strip()
    if explicit:
        return str(Path(explicit).expanduser())
    bin_dir = os.environ.get(bin_env, "").strip()
    if bin_dir:
        candidate = Path(bin_dir).expanduser() / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name) or ""


def _version_tuple(text: str) -> tuple[int, ...]:
    match = re.search(r"(?<!\d)(\d+)\.(\d+)(?:\.(\d+))?", text)
    if not match:
        return ()
    return tuple(int(value) for value in match.groups(default="0"))


def _version_text(executable: str) -> str:
    if not executable:
        return ""
    for option in ("--version", "-version", "-v"):
        try:
            result = subprocess.run(
                [executable, option],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        if text:
            return "\n".join(text.splitlines()[:6])
    return ""


def probe_runtime() -> dict[str, Any]:
    executables: dict[str, dict[str, Any]] = {}
    for key, (env_key, name) in QE_EXECUTABLES.items():
        path = _resolve_executable(env_key, name, "ATOMI_QE_BIN")
        executables[key] = {"path": path, "available": bool(path and Path(path).is_file())}
    for key, (env_key, name) in W90_EXECUTABLES.items():
        path = _resolve_executable(env_key, name, "ATOMI_WANNIER90_BIN")
        executables[key] = {"path": path, "available": bool(path and Path(path).is_file())}

    pw_version_text = _version_text(executables["pw"]["path"])
    w90_version_text = _version_text(executables["wannier90"]["path"])
    qe_version = _version_tuple(pw_version_text or os.environ.get("ESPRESSO_VERSION", ""))
    w90_version = _version_tuple(w90_version_text or os.environ.get("WANNIER90_VERSION", ""))
    modern_hubbard = qe_version >= (7, 3, 1)
    mlwf_projectors = modern_hubbard and all(
        executables[name]["available"]
        for name in ("pw", "pw2wannier90", "wannier90", "wannier2pw")
    )
    stock_hp = modern_hubbard and executables["hp"]["available"]

    piotr_root = os.environ.get("ATOMI_PIOTR_QE_ROOT", "").strip()
    piotr_commit = os.environ.get("ATOMI_PIOTR_QE_COMMIT", "").strip()
    piotr_response = os.environ.get("ATOMI_PIOTR_QE_RESPONSE_EXE", "").strip()
    piotr_ready = bool(
        piotr_root
        and piotr_commit
        and piotr_response
        and Path(piotr_root).expanduser().is_dir()
        and Path(piotr_response).expanduser().is_file()
    )

    warnings: list[str] = []
    if executables["pw"]["available"] and not modern_hubbard:
        warnings.append("QE is older than 7.3.1; do not use the modern HUBBARD-card templates.")
    if executables["pw2wannier90"]["available"] and not executables["wannier90"]["available"]:
        warnings.append("pw2wannier90.x is present, but the external Wannier90 executable is missing.")
    if mlwf_projectors and not piotr_ready:
        warnings.append(
            "Stock QE can build MLWF Hubbard projectors, but Piotr's matched response branch is not pinned."
        )

    return {
        "schema": "atomi.qe_wannier_status.v1",
        "qe_root": os.environ.get("ATOMI_QE_ROOT", ""),
        "qe_bin": os.environ.get("ATOMI_QE_BIN", ""),
        "qe_module": os.environ.get("ATOMI_QE_MODULE", ""),
        "wannier90_root": os.environ.get("ATOMI_WANNIER90_ROOT", ""),
        "wannier90_bin": os.environ.get("ATOMI_WANNIER90_BIN", ""),
        "qe_version": ".".join(map(str, qe_version)) if qe_version else "unknown",
        "wannier90_version": ".".join(map(str, w90_version)) if w90_version else "unknown",
        "version_text": {"qe": pw_version_text, "wannier90": w90_version_text},
        "executables": executables,
        "capabilities": {
            "modern_hubbard_card": modern_hubbard,
            "stock_hp_atomic_response": stock_hp,
            "mlwf_hubbard_projectors": mlwf_projectors,
            "piotr_matched_response": piotr_ready,
            "uo2_piotr_production_ready": mlwf_projectors and piotr_ready,
        },
        "piotr": {
            "root": piotr_root,
            "commit": piotr_commit,
            "response_executable": piotr_response,
        },
        "warnings": warnings,
    }


def install_plan() -> dict[str, Any]:
    return {
        "schema": "atomi.qe_wannier_install_plan.v1",
        "recommendation": "Keep QE/Wannier90 as a compiled sidecar runtime outside m_lammps_env.",
        "target": {
            "quantum_espresso": "7.5",
            "wannier90": "3.1.0",
            "root": "$HOME/atomi_hpc/qe-wannier",
        },
        "why_not_ocean_qe": [
            "A legacy OCEAN/QE 7.0 runtime predates the modern Hubbard-card workflow.",
            "A partial runtime with pw.x and pw2wannier90.x is insufficient without hp.x, Wannier90, and wannier2pw.x.",
        ],
        "route_gates": [
            "Stock hp.x is an atomic/ortho-atomic projector baseline unless proven otherwise.",
            "MLWF projectors require pw.x -> pw2wannier90.x -> wannier90.x -> wannier2pw.x.",
            "Piotr replication additionally requires his exact response branch, commit, and equations.",
            "A UO2 run also requires validated U/O UPFs, magnetic/occupation guards, and window QA.",
        ],
        "environment": {
            "ATOMI_QE_ROOT": "$HOME/atomi_hpc/qe-wannier/qe-7.5",
            "ATOMI_QE_BIN": "$HOME/atomi_hpc/qe-wannier/qe-7.5/bin",
            "ATOMI_WANNIER90_ROOT": "$HOME/atomi_hpc/qe-wannier/wannier90-3.1.0",
            "ATOMI_WANNIER90_BIN": "$HOME/atomi_hpc/qe-wannier/wannier90-3.1.0/bin",
            "ATOMI_PIOTR_QE_ROOT": "set only after collaborator branch is obtained",
            "ATOMI_PIOTR_QE_COMMIT": "required before claiming Piotr replication",
            "ATOMI_PIOTR_QE_RESPONSE_EXE": "required before calculating matched Wannier U",
        },
    }


def write_install_script(
    outdir: Path,
    *,
    root: str,
    qe_version: str,
    wannier_version: str,
    cpus: int,
    time_limit: str,
    module_loads: Sequence[str],
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    script = outdir / "install_qe_wannier.sbatch"
    root_shell = root.replace("$HOME", "${HOME}").replace('"', '\\"')
    module_lines = ["module purge"]
    module_lines.extend(f"module load {module}" for module in module_loads)
    module_block = "\n".join(module_lines)
    script.write_text(
        f"""#!/usr/bin/env bash
#SBATCH --job-name=qe-w90-build
#SBATCH --output=qe-w90-build.%j.out
#SBATCH --error=qe-w90-build.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem=64G
#SBATCH --time={time_limit}

set -euo pipefail
{module_block}
export OMP_NUM_THREADS=1

ROOT="{root_shell}"
QE_VERSION={qe_version!r}
W90_VERSION={wannier_version!r}
SRC="$ROOT/src"
BUILD="$ROOT/build"
QE_PREFIX="$ROOT/qe-$QE_VERSION"
W90_PREFIX="$ROOT/wannier90-$W90_VERSION"
JOBS="${{SLURM_CPUS_PER_TASK:-{cpus}}}"
mkdir -p "$SRC" "$BUILD" "$QE_PREFIX" "$W90_PREFIX/bin"

cd "$SRC"
if [[ ! -f "q-e-$QE_VERSION.tar.gz" ]]; then
  curl -L --fail --retry 3 -o "q-e-$QE_VERSION.tar.gz" \
    "https://gitlab.com/QEF/q-e/-/archive/qe-$QE_VERSION/q-e-qe-$QE_VERSION.tar.gz"
fi
if [[ ! -d "q-e-qe-$QE_VERSION" ]]; then
  tar -xzf "q-e-$QE_VERSION.tar.gz"
fi
rm -rf "$BUILD/qe-$QE_VERSION"
cp -a "$SRC/q-e-qe-$QE_VERSION" "$BUILD/qe-$QE_VERSION"
cd "$BUILD/qe-$QE_VERSION"
./configure --prefix="$QE_PREFIX" CC=mpicc FC=mpif90 F77=mpif77 MPIF90=mpif90
make -j "$JOBS" pw hp pp
make install

cd "$SRC"
if [[ ! -f "wannier90-$W90_VERSION.tar.gz" ]]; then
  curl -L --fail --retry 3 -o "wannier90-$W90_VERSION.tar.gz" \
    "https://github.com/wannier-developers/wannier90/archive/refs/tags/v$W90_VERSION.tar.gz"
fi
if [[ ! -d "wannier90-$W90_VERSION" ]]; then
  tar -xzf "wannier90-$W90_VERSION.tar.gz"
fi
rm -rf "$BUILD/wannier90-$W90_VERSION"
cp -a "$SRC/wannier90-$W90_VERSION" "$BUILD/wannier90-$W90_VERSION"
cd "$BUILD/wannier90-$W90_VERSION"
cp config/make.inc.gfort make.inc
make -j "$JOBS"
cp -f wannier90.x postw90.x w90chk2chk.x w90spn2spn.x "$W90_PREFIX/bin/" 2>/dev/null || true

test -x "$QE_PREFIX/bin/pw.x"
test -x "$QE_PREFIX/bin/hp.x"
test -x "$QE_PREFIX/bin/pw2wannier90.x"
test -x "$QE_PREFIX/bin/wannier2pw.x"
test -x "$W90_PREFIX/bin/wannier90.x"

printf '%s\n' \
  '# Source this file before QE/Wannier work.' \
  "export ATOMI_QE_ROOT=$QE_PREFIX" \
  "export ATOMI_QE_BIN=$QE_PREFIX/bin" \
  'export ATOMI_QE_PW=$ATOMI_QE_BIN/pw.x' \
  'export ATOMI_QE_HP=$ATOMI_QE_BIN/hp.x' \
  'export ATOMI_QE_PW2WANNIER90=$ATOMI_QE_BIN/pw2wannier90.x' \
  'export ATOMI_QE_WANNIER2PW=$ATOMI_QE_BIN/wannier2pw.x' \
  "export ATOMI_WANNIER90_ROOT=$W90_PREFIX" \
  "export ATOMI_WANNIER90_BIN=$W90_PREFIX/bin" \
  'export ATOMI_WANNIER90_EXE=$ATOMI_WANNIER90_BIN/wannier90.x' \
  'export PATH=$ATOMI_QE_BIN:$ATOMI_WANNIER90_BIN:$PATH' \
  > "$ROOT/activate_qe_wannier.sh"

"$QE_PREFIX/bin/pw.x" --version || true
"$W90_PREFIX/bin/wannier90.x" --version || true
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _dump(payload: Any, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qe-wannier-bridge")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Probe QE, Wannier90, and Piotr-route readiness")
    status.add_argument("--json", action="store_true")
    plan = sub.add_parser("install-plan", help="Print the separate-runtime installation plan")
    plan.add_argument("--json", action="store_true")
    write = sub.add_parser("write-install", help="Write a compute-node Slurm build script")
    write.add_argument("--outdir", type=Path, required=True)
    write.add_argument("--root", default="$HOME/atomi_hpc/qe-wannier")
    write.add_argument("--qe-version", default="7.5")
    write.add_argument("--wannier-version", default="3.1.0")
    write.add_argument("--cpus", type=int, default=24)
    write.add_argument("--time", default="06:00:00")
    write.add_argument(
        "--module-load",
        action="append",
        default=[],
        help="Environment module to load in the compute job; repeat as needed",
    )
    write.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        payload = probe_runtime()
    elif args.command == "install-plan":
        payload = install_plan()
    else:
        path = write_install_script(
            args.outdir.resolve(),
            root=args.root,
            qe_version=args.qe_version,
            wannier_version=args.wannier_version,
            cpus=args.cpus,
            time_limit=args.time,
            module_loads=args.module_load,
        )
        payload = {"schema": "atomi.qe_wannier_install_script.v1", "script": str(path)}
    _dump(payload, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
