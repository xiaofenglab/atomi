#!/bin/bash
#SBATCH --job-name=md-engine
#SBATCH --output=lammps_gpu.%x.%j.out
#SBATCH --error=lammps_gpu.%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
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
    for mod in ${ATOMI_LAMMPS_MODULES:-compiler/gnu mpi/openmpi numlib/mkl/2020.2 devel/cuda/12.3}; do
        module load "$mod"
    done
fi

# ---- optional env ----
unset PYTHONPATH

# ---- runtime libraries ----
ATOMI_LAMMPS_PREFIX="${ATOMI_LAMMPS_PREFIX:-$HOME/projects/lammps/gup_run}"
ATOMI_LIBTORCH_LIB="${ATOMI_LIBTORCH_LIB:-$ATOMI_LAMMPS_PREFIX/src/libtorch-gpu/lib}"
export LD_LIBRARY_PATH=$ATOMI_LAMMPS_PREFIX/install/lib64:$ATOMI_LAMMPS_PREFIX/install/lib:$ATOMI_LIBTORCH_LIB:$LD_LIBRARY_PATH


cd "${SLURM_SUBMIT_DIR}" || exit 1

INPUT="$1"

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
echo "LMP_EXE           = ${ATOMI_LMP_EXE:-$ATOMI_LAMMPS_PREFIX/install/bin/lmp}"
echo "========================================"

echo "----- TOOLCHAIN -----"
which mpicxx || true
which nvcc || true
nvidia-smi || true
echo "---------------------"

LMP_EXE="${ATOMI_LMP_EXE:-$ATOMI_LAMMPS_PREFIX/install/bin/lmp}"

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
