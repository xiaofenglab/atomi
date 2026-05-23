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

if [ -z "${ATOMI_HPC_CONFIG:-}" ] && [ -f "$HOME/atomi_hpc/atomi_hpc_env.sh" ]; then
    source "$HOME/atomi_hpc/atomi_hpc_env.sh"
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PSM2_CUDA=1

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

# ---- optional env ----
unset PYTHONPATH

INPUT="$1"
INPUT_BASE="$(basename "${INPUT:-}")"
GK_REQUESTED=0
if [ "${ATOMI_LAMMPS_USE_GK_EXE:-0}" = "1" ] || [[ "${INPUT_BASE}" == in.gk_* ]]; then
    GK_REQUESTED=1
fi

if [ -f "$HOME/atomi_hpc/atomi_hpc_env.sh" ]; then
    if [ -z "${ATOMI_LMP_EXE:-}" ] || { [ "${GK_REQUESTED}" = "1" ] && [ -z "${ATOMI_LMP_GK_EXE:-}" ]; }; then
        source "$HOME/atomi_hpc/atomi_hpc_env.sh"
    fi
fi

# ---- runtime libraries ----
LAMMPS_PROFILE="production"
if [ "${GK_REQUESTED}" = "1" ] && [ -z "${ATOMI_LMP_GK_EXE:-}" ]; then
    echo "ERROR: GK/ML-IAP LAMMPS was requested, but ATOMI_LMP_GK_EXE is not set."
    echo "Run confighpc or update $HOME/atomi_hpc/atomi_hpc_env.sh so the Slurm job can see profiles.lammps_gk_mliap."
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
if [ -n "${ATOMI_LAMMPS_PREFIX:-}" ]; then
    ATOMI_LIBTORCH_LIB="${ATOMI_LIBTORCH_LIB:-$ATOMI_LAMMPS_PREFIX/src/libtorch-gpu/lib}"
    export LD_LIBRARY_PATH=$ATOMI_LAMMPS_PREFIX/install/lib64:$ATOMI_LAMMPS_PREFIX/install/lib:$ATOMI_LIBTORCH_LIB:$LD_LIBRARY_PATH
fi


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
echo "LAMMPS_PROFILE    = ${LAMMPS_PROFILE}"
echo "========================================"

echo "----- TOOLCHAIN -----"
which mpicxx || true
which nvcc || true
nvidia-smi || true
echo "---------------------"

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
