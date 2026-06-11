#!/bin/bash
#SBATCH --job-name=atomi-casm-ce-mc-ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load miniconda || module load anaconda || true
source activate atomi-casm

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

atomi-defects backend doctor --backend casm_ce_mc_ti

# Future production shape:
# atomi-defects large-state-space run \
#   --backend casm_ce_mc_ti \
#   --config configs/gd_uo2.defect_engine.yaml \
#   --training-set build/gd_uo2.ce_training.jsonl \
#   --output build/casm_ce_mc_ti_gd_uo2
