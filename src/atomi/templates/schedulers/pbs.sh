#!/usr/bin/env bash
#PBS -N $job_name
#PBS -A $account
#PBS -q $partition
#PBS -l select=$nodes:ncpus=$ntasks
#PBS -l walltime=$time
#PBS -j oe

set -euo pipefail

cd "$PBS_O_WORKDIR"

$modules

$command

