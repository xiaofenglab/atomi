#!/usr/bin/env bash
#SBATCH --job-name=$job_name
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --nodes=$nodes
#SBATCH --ntasks=$ntasks
#SBATCH --time=$time
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

$modules

$command

