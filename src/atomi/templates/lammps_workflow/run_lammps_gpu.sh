#!/bin/bash
#SBATCH --job-name=md-engine
#SBATCH --output=lammps_gpu.%x.%j.out
#SBATCH --error=lammps_gpu.%x.%j.err
# Set cluster-specific partition/resource directives in your private copy.
##SBATCH --partition=your_gpu_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3500M
#SBATCH --time=2:00:00

##### E-MAIL SPECIFICATION:
#
# No emails are sent by default unless specified.
#
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your_email_or_username

unset LANG
export LC_ALL="C"

ATOMI_HPC_DIR="${ATOMI_HPC_DIR:-$HOME/atomi_hpc}"
ATOMI_HPC_ENV="${ATOMI_HPC_ENV:-$ATOMI_HPC_DIR/atomi_hpc_env.sh}"

atomi_local_hpc_json_exists() {
    [ -d "$ATOMI_HPC_DIR" ] && find "$ATOMI_HPC_DIR" -maxdepth 1 -type f -name "*.local.json" -print -quit | grep -q .
}

apply_atomi_hpc_environment() {
    if command -v confighpc >/dev/null 2>&1; then
        if [ -n "${ATOMI_HPC_CONFIG:-}" ] && [ -f "$ATOMI_HPC_CONFIG" ]; then
            eval "$(confighpc --config "$ATOMI_HPC_CONFIG" --shell)"
            return
        fi
        if atomi_local_hpc_json_exists; then
            eval "$(confighpc --dir "$ATOMI_HPC_DIR" --no-env-var --shell)"
            return
        fi
    elif atomi_local_hpc_json_exists; then
        echo "WARNING: local Atomi HPC JSON exists in $ATOMI_HPC_DIR, but confighpc is not on PATH; not sourcing stale env file."
        return
    fi

    if [ -z "${ATOMI_HPC_CONFIG:-}" ] && [ -f "$ATOMI_HPC_ENV" ]; then
        source "$ATOMI_HPC_ENV"
    fi
}

apply_atomi_hpc_environment

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PSM2_CUDA=1
export PYTHONUNBUFFERED=1
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"

export USER="${USER:=`logname`}"
export SLURM_JOB_ID="${SLURM_JOB_ID:=`date +%s`}"
export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:=`pwd`}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME:=`basename "$0"`}"
export SLURM_JOB_NAME="${SLURM_JOB_NAME//[^a-zA-Z0-9._-]/_}"

# ---- modules ----
if command -v module >/dev/null 2>&1; then
    module purge
    if [ -n "${ATOMI_LAMMPS_MODULES:-}" ]; then
        for mod in ${ATOMI_LAMMPS_MODULES}; do
            module load "$mod"
        done
    else
        echo "ATOMI_LAMMPS_MODULES is not set; using the current environment after module purge."
    fi
fi

if [ -n "${ATOMI_LAMMPS_ENV:-}" ] && [ -f "$ATOMI_LAMMPS_ENV/bin/activate" ]; then
    source "$ATOMI_LAMMPS_ENV/bin/activate"
fi

# ---- optional env ----
unset PYTHONPATH

INPUT="$1"
INPUT_BASE="$(basename "${INPUT:-}")"
GK_REQUESTED=0
if [ "${ATOMI_LAMMPS_USE_GK_EXE:-0}" = "1" ] || [[ "${INPUT_BASE}" == in.gk_* ]]; then
    GK_REQUESTED=1
fi

if [ -d "$ATOMI_HPC_DIR" ]; then
    if [ -z "${ATOMI_LMP_EXE:-}" ] || { [ "${GK_REQUESTED}" = "1" ] && [ -z "${ATOMI_LMP_GK_EXE:-}" ]; }; then
        apply_atomi_hpc_environment
    fi
fi

# ---- runtime libraries ----
LAMMPS_PROFILE="production"
if [ "${GK_REQUESTED}" = "1" ] && [ -z "${ATOMI_LMP_GK_EXE:-}" ]; then
    echo "ERROR: GK/ML-IAP LAMMPS was requested, but ATOMI_LMP_GK_EXE is not set."
    echo "Run confighpc or update the local JSON in $ATOMI_HPC_DIR so the Slurm job can see profiles.lammps_gk_mliap."
    exit 2
fi
if [ "${GK_REQUESTED}" = "1" ] && [ -n "${ATOMI_LMP_GK_EXE:-}" ]; then
    ATOMI_LMP_EXE="${ATOMI_LMP_GK_EXE}"
    if [ -n "${ATOMI_LAMMPS_GK_PREFIX:-}" ]; then
        ATOMI_LAMMPS_PREFIX="${ATOMI_LAMMPS_GK_PREFIX}"
    fi
    LAMMPS_PROFILE="gk_mliap"
fi

if [ -z "${ATOMI_LMP_EXE:-}" ]; then
    echo "ERROR: set ATOMI_LMP_EXE to the private path of your LAMMPS executable."
    exit 2
fi

ATOMI_LMP_BIN_DIR="$(cd "$(dirname "${ATOMI_LMP_EXE}")" && pwd)"
ATOMI_LMP_INSTALL_DIR="$(cd "${ATOMI_LMP_BIN_DIR}/.." && pwd)"
ATOMI_LIBTORCH_LIB="${ATOMI_LIBTORCH_LIB:-}"
if [ -z "${ATOMI_LIBTORCH_LIB}" ] && [ -n "${ATOMI_LAMMPS_PREFIX:-}" ]; then
    ATOMI_LIBTORCH_LIB="$ATOMI_LAMMPS_PREFIX/src/libtorch-gpu/lib"
fi

atomi_add_ld_library_path() {
    if [ -n "${1:-}" ] && [ -d "$1" ]; then
        if [ -n "${LD_LIBRARY_PATH:-}" ]; then
            export LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH}"
        else
            export LD_LIBRARY_PATH="$1"
        fi
    fi
}

if [ -n "${ATOMI_LIBTORCH_LIB}" ]; then
    export LD_LIBRARY_PATH=$ATOMI_LMP_INSTALL_DIR/lib64:$ATOMI_LMP_INSTALL_DIR/lib:$ATOMI_LIBTORCH_LIB:${LD_LIBRARY_PATH:-}
else
    export LD_LIBRARY_PATH=$ATOMI_LMP_INSTALL_DIR/lib64:$ATOMI_LMP_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}
fi

if [ "${LAMMPS_PROFILE}" = "gk_mliap" ]; then
    if [ -n "${ATOMI_PYTHON_LIBDIRS:-}" ]; then
        for atomi_python_libdir in ${ATOMI_PYTHON_LIBDIRS}; do
            atomi_add_ld_library_path "$atomi_python_libdir"
        done
    fi
    ATOMI_DETECTED_PYTHON_LIBDIRS="$(python - <<'PY' 2>/dev/null || true
import pathlib
import sys
import sysconfig

names = {
    sysconfig.get_config_var("LDLIBRARY"),
    sysconfig.get_config_var("LIBRARY"),
    f"libpython{sys.version_info.major}.{sys.version_info.minor}.so",
}
roots = {
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("LIBPL"),
    str(pathlib.Path(sys.executable).resolve().parents[1] / "lib"),
    str(pathlib.Path(sys.base_prefix) / "lib"),
    str(pathlib.Path(sys.exec_prefix) / "lib"),
}
out = []
for root in roots:
    if not root:
        continue
    path = pathlib.Path(root)
    if not path.is_dir():
        continue
    if any((path / name).exists() for name in names if name):
        out.append(str(path))
for item in dict.fromkeys(out):
    print(item)
PY
)"
    for atomi_python_libdir in ${ATOMI_DETECTED_PYTHON_LIBDIRS}; do
        atomi_add_ld_library_path "$atomi_python_libdir"
    done
fi

atomi_add_pythonpath() {
    if [ -n "${1:-}" ] && [ -d "$1" ]; then
        if [ -n "${PYTHONPATH:-}" ]; then
            export PYTHONPATH="${PYTHONPATH}:$1"
        else
            export PYTHONPATH="$1"
        fi
    fi
}

if [ -n "${ATOMI_LAMMPS_PYTHONPATH:-}" ]; then
    export PYTHONPATH="${ATOMI_LAMMPS_PYTHONPATH}"
fi

for atomi_py_path in \
    "$ATOMI_LMP_INSTALL_DIR"/lib/python*/site-packages \
    "$ATOMI_LMP_INSTALL_DIR"/lib64/python*/site-packages \
    "${ATOMI_LAMMPS_PREFIX:-}"/src/lammps/python \
    "${ATOMI_LAMMPS_PREFIX:-}"/build_mliap/python \
    "${ATOMI_LAMMPS_PREFIX:-}"/build_mliap/cython
do
    atomi_add_pythonpath "$atomi_py_path"
done


cd "${SLURM_SUBMIT_DIR}" || exit 1

if [ -z "$INPUT" ]; then
    echo "Usage: sbatch run_lammps.sh input_file"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "ERROR: input file '$INPUT' not found"
    exit 2
fi

echo "========================================"
echo "LAMMPS JOB INFORMATION"
echo "========================================"

echo "START_TIME        = $(date +'%y-%m-%d %H:%M:%S %s')"
echo "HOSTNAME          = ${HOSTNAME}"
echo "USER              = ${USER}"
echo "SLURM_JOB_NAME    = ${SLURM_JOB_NAME}"
echo "SLURM_JOB_ID      = ${SLURM_JOB_ID}"
echo "SLURM_SUBMIT_DIR  = ${SLURM_SUBMIT_DIR}"
echo "SLURM_NTASKS      = ${SLURM_NTASKS}"
echo "OMP_NUM_THREADS   = ${OMP_NUM_THREADS}"
echo "INPUT_FILE        = ${INPUT}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "LMP_EXE           = ${ATOMI_LMP_EXE}"
echo "LMP_INSTALL_DIR   = ${ATOMI_LMP_INSTALL_DIR}"
echo "LAMMPS_PROFILE    = ${LAMMPS_PROFILE}"
echo "PYTHON_EXE        = $(command -v python || true)"
echo "PYTHON_LIBDIRS    = ${ATOMI_DETECTED_PYTHON_LIBDIRS:-}"
echo "LAMMPS_PYTHONPATH = ${PYTHONPATH:-}"
echo "========================================"

echo "----- TOOLCHAIN -----"
which mpicxx || true
which nvcc || true
nvidia-smi || true
if [ "${LAMMPS_PROFILE}" = "gk_mliap" ]; then
    python - <<'PY' || true
import importlib.util
for name in ("lammps", "mliap_unified_couple", "torch", "mace"):
    spec = importlib.util.find_spec(name)
    origin = spec.origin if spec is not None else "missing"
    print(f"python module {name}: {origin}")
PY
fi
echo "---------------------"

atomi_fail_preflight() {
    echo "ERROR: Atomi LAMMPS preflight failed: $1"
    echo "This job stopped before requesting the full GK run because the array would fail with the same setup."
    exit 2
}

if [ "${LAMMPS_PROFILE}" = "gk_mliap" ]; then
    echo "----- ATOMI GK/ML-IAP PREFLIGHT -----"
    LMP_HELP_LOG="atomi_lammps_help.${SLURM_JOB_ID}.txt"
    if ! "${ATOMI_LMP_EXE}" -h >"${LMP_HELP_LOG}" 2>&1; then
        cat "${LMP_HELP_LOG}" || true
        atomi_fail_preflight "LAMMPS executable could not start. Check CUDA, libtorch, and LD_LIBRARY_PATH."
    fi
    if ! grep -qi "ML-IAP" "${LMP_HELP_LOG}" || ! grep -Eq '(^|[[:space:]])mliap(/kk)?([[:space:]]|$)' "${LMP_HELP_LOG}"; then
        grep -iE "mliap|mace|python|kokkos|ml-" "${LMP_HELP_LOG}" || true
        atomi_fail_preflight "selected GK executable does not expose the ML-IAP mliap pair style."
    fi

    MLIP_MODEL_PATH="$(awk '$1=="pair_style" && $2=="mliap" && $3=="unified" {print $4; exit}' "${INPUT}")"
    if [ -z "${MLIP_MODEL_PATH}" ]; then
        awk '$1=="pair_style" || $1=="pair_coeff" {print "INPUT_PAIR:", $0}' "${INPUT}" || true
        atomi_fail_preflight "GK/ML-IAP run requested, but input does not contain 'pair_style mliap unified <model> 0'."
    fi
    if [ ! -f "${MLIP_MODEL_PATH}" ]; then
        atomi_fail_preflight "ML-IAP model file not found: ${MLIP_MODEL_PATH}"
    fi
    export ATOMI_MLIP_MODEL_PATH="${MLIP_MODEL_PATH}"

    python - <<'PY'
import importlib
import os
import sys

required = ("lammps", "lammps.mliap", "torch")
# Some CMake ML-IAP builds compile the Cython mliap_unified_couple module into
# liblammps instead of installing it as a standalone Python extension. Treat it
# as diagnostic here; the LAMMPS run-0 probe below is the authoritative check.
optional = ("mliap_unified_couple", "mace")
failed = False
for name in required + optional:
    try:
        module = importlib.import_module(name)
        origin = getattr(module, "__file__", "built-in")
        print(f"Atomi preflight python import {name}: OK {origin}")
    except Exception as exc:
        level = "ERROR" if name in required else "WARNING"
        print(f"Atomi preflight python import {name}: {level} {exc.__class__.__name__}: {exc}")
        if name in required:
            failed = True
if failed:
    sys.exit(1)

try:
    import torch

    model_path = os.environ.get("ATOMI_MLIP_MODEL_PATH", "")
    print(f"Atomi preflight torch cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Atomi preflight torch cuda device: {torch.cuda.get_device_name(0)}")
    if model_path:
        model = torch.jit.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Atomi preflight torch.jit.load model: OK {model_path}")
        del model
except Exception as exc:
    print(f"Atomi preflight torch.jit.load model: WARNING {exc.__class__.__name__}: {exc}")
PY
    if [ "$?" -ne 0 ]; then
        atomi_fail_preflight "required ML-IAP Python modules could not be imported."
    fi
    echo "Atomi GK/ML-IAP preflight: PASS"
    echo "-------------------------------------"
fi

LMP_EXE="${ATOMI_LMP_EXE}"

${LMP_EXE} \
    -nonbuf \
    -k on g 1 \
    -sf kk \
    -pk kokkos newton on neigh half \
    -in "${INPUT}" \
    -log "log.${INPUT}"

LMP_STATUS=$?

echo "========================================"
echo "LAMMPS RUN FINISHED"
echo "END_TIME = $(date +'%y-%m-%d %H:%M:%S %s')"
echo "EXIT_STATUS       = ${LMP_STATUS}"
echo "========================================"

exit ${LMP_STATUS}
