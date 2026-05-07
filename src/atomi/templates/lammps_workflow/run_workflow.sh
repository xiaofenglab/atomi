#!/bin/bash
#SBATCH --job-name=md_engine
#SBATCH --output=logs/md_engine_%j.out
#SBATCH --error=logs/md_engine_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000M
#SBATCH --time=48:00:00

# ============================================================
# MD engine launcher for md-engine
#
# USAGE:
#
#   1) Resume from latest finished stage/chunk using default config.json
#      sbatch run_workflow.sh resume
#
#   2) Fresh start from the beginning using default config.json
#      sbatch run_workflow.sh fresh
#
#   3) Resume and start from a specific stage
#      sbatch run_workflow.sh resume npt_200K
#      -> calls:
#         md-engine --resume --start-from npt_200K --config config.json
#
#   4) Fresh start beginning at a specific stage
#      sbatch run_workflow.sh fresh nvt_300K
#      -> calls:
#         md-engine --start-from nvt_300K --config config.json
#
#   5) Resume using a different config file
#      sbatch run_workflow.sh resume "" config_600_1200K.json
#
#   6) Resume from a specific stage using a different config file
#      sbatch run_workflow.sh resume nvt_ramp_700K config_600_1200K.json
#
#   7) If no argument is given, default is:
#      sbatch run_workflow.sh
#      -> equivalent to:
#         sbatch run_workflow.sh resume
#
# ARGUMENTS:
#
#   $1 = MODE
#        allowed values:
#          resume
#          fresh
#
#   $2 = START_STAGE   (optional)
#        examples:
#          npt_200K
#          nvt_large_relax_350K
#          nvt_ramp_700K
#
#   $3 = CONFIG_FILE   (optional)
#        default:
#          config.json
#
# NOTES:
#
#   - "resume" adds:
#         --resume
#
#   - "fresh" does not add --resume
#
#   - If START_STAGE is provided, it adds:
#         --start-from <stage_name>
#
#   - If CONFIG_FILE is provided, it adds:
#         --config <config_file>
#
#   - A true clean "fresh" run does NOT automatically delete old stages/.
#     If you want a completely clean run, move or remove old stage folders first.
#
# EXAMPLE CLEANUP BEFORE FRESH:
#      mv stages stages_backup_$(date +%Y%m%d_%H%M%S)
#      mkdir -p stages
#
# ============================================================

ATOMI_LAMMPS_ENV="${ATOMI_LAMMPS_ENV:-$HOME/m_lammps_env}"
source "$ATOMI_LAMMPS_ENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

MODE=${1:-resume}
START_STAGE=${2:-}
CONFIG_FILE=${3:-config.json}

echo "=============================="
echo "Workflow launcher"
echo "Mode        : $MODE"
echo "Start stage : ${START_STAGE:-<none>}"
echo "Config      : $CONFIG_FILE"
echo "Directory   : $(pwd)"
echo "Job ID      : $SLURM_JOB_ID"
echo "=============================="

CMD=(md-engine --config "$CONFIG_FILE")

if [ "$MODE" = "resume" ]; then
    CMD+=(--resume)
elif [ "$MODE" = "fresh" ]; then
    :
else
    echo "ERROR: unknown mode '$MODE'"
    echo
    echo "Usage:"
    echo "  sbatch run_workflow.sh [fresh|resume] [optional_stage_name] [optional_config]"
    echo
    echo "Examples:"
    echo "  sbatch run_workflow.sh resume"
    echo "  sbatch run_workflow.sh fresh"
    echo "  sbatch run_workflow.sh resume npt_200K"
    echo "  sbatch run_workflow.sh fresh nvt_300K"
    echo "  sbatch run_workflow.sh resume \"\" config_600_1200K.json"
    echo "  sbatch run_workflow.sh resume nvt_ramp_700K config_600_1200K.json"
    exit 1
fi

if [ -n "$START_STAGE" ]; then
    CMD+=(--start-from "$START_STAGE")
fi

echo "Running command:"
printf '  %q' "${CMD[@]}"
echo
echo "=============================="

"${CMD[@]}"
