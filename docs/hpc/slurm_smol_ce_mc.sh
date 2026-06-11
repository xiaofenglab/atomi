#!/bin/bash
#SBATCH --job-name=atomi-smol-ce-mc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load miniconda || module load anaconda || true
source activate atomi-smol

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

atomi-defects backend doctor --backend smol_ce_mc

# Future production shape:
# atomi-defects large-state-space run \
#   --backend smol_ce_mc \
#   --config configs/gd_uo2.defect_engine.yaml \
#   --training-set build/gd_uo2.ce_training.jsonl \
#   --output build/smol_ce_mc_gd_uo2
