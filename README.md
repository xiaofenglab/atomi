# Atomi

Reusable automation tools for atomistic modeling on HPC systems.

This repository is meant to collect scripts for:

- preparing calculations
- writing scheduler scripts
- submitting and tracking jobs
- post-processing outputs
- moving data between project folders and shared storage
- standardizing workflows across VASP, CP2K, LAMMPS, Turbomole, and OpenMolcas

## Install

For active development:

```bash
git clone <your-github-repo-url>
cd atomi
python -m pip install -e ".[dev,materials]"
```

For use on an HPC system:

```bash
python -m pip install git+https://github.com/xiaofenglab/atomi.git
```

To update an existing Atomi install on an HPC system:

```bash
python -m pip install --upgrade --force-reinstall git+https://github.com/xiaofenglab/atomi.git
```

If Atomi was installed in editable mode from a cloned repository:

```bash
cd atomi
git pull
python -m pip install -e ".[materials]"
```

If the compute environment has no internet access, install on a login node or build a wheel:

```bash
python -m pip wheel . -w dist
python -m pip install dist/atomi-*.whl
```

For an offline update, rebuild the wheel from the latest repository checkout, copy the new wheel to the HPC environment, and reinstall it:

```bash
python -m pip install --upgrade --force-reinstall dist/atomi-*.whl
```

## Command Line

After installation, the main command is:

```bash
atomi --help
```

Examples:

```bash
atomi init-project my_vasp_run --code vasp
atomi write-submit --scheduler slurm --profile generic_cpu
atomi inspect .
atomi doctor
atomi lammps-md-init my_lammps_md_project
```

## HPC Environment Check

Before using Atomi on a new cluster, run:

```bash
atomi doctor
atomi doctor --write atomi_hpc_config.json
```

The doctor report checks common executables, scheduler commands, plotting tools, atomistic engine names, and Python packages used by the currently packaged workflows. It also records cluster-specific assumptions that should be reviewed before a command is treated as portable.

Atomi looks for configuration in this order:

```text
--hpc-config PATH
ATOMI_HPC_CONFIG
./atomi_hpc_config.json
~/.config/atomi/hpc.json
```

For example, edit the `profiles.mace_lammps` block in `atomi_hpc_config.json` to match a cluster's GPU partition, gres string, wall time, and MACE environment path. Then:

```bash
convertmace modelname.model --hpc-config atomi_hpc_config.json
```

For LAMMPS MD workflows, also review the `profiles.lammps_md_workflow` block. The project launcher and GPU wrapper use these environment variables when needed:

```text
ATOMI_LAMMPS_ENV
ATOMI_LAMMPS_MODULES
ATOMI_LAMMPS_PREFIX
ATOMI_LMP_EXE
ATOMI_LIBTORCH_LIB
```

## Familiar Plot Commands

Atomi keeps short compatibility commands for day-to-day monitoring:

```bash
plotvasp vasp.out 2000
plotvasp4 vasp.outA vasp.outB vasp.outC vasp.outD 200
plotlammps log.lammps
plotcp2k cp2k.log
plotcp2k cp2k.log trajectory.xyz
plotmace mace_train.log
plotmace mace_train.log 200 5
convertmace modelname.model
extv OUTCAR
lammps-md-init my_lammps_md_project
lammps-md-workflow --config config.json
lammps-md-workflow --resume --config config.json
lammps-postprocess --log stages/npt_prod_1400K/chunk_production/log.in.npt_prod_1400K_production --temperature 1400 --outdir analysis/npt_prod_1400K
lammps-thermo-series --config config_production.json --outdir analysis/thermo_0_300K_uq
mace-build-dataset --neareq neareq_train.extxyz --phonopy phonopy.extxyz --force-spread forces.extxyz --prefail-group prefail=prefail.extxyz
mace-energy-outliers --extxyz training.extxyz --model model.model --outdir energy_outliers --device cuda --dtype float32 --top-n 30 --write-poscars
mace-update-outliers --report energy_outliers/report.txt --train-in training.extxyz --valid-in validation.extxyz --train-out training_clean.extxyz --valid-out validation_clean.extxyz
mace-check-extxyz --train training.extxyz --valid validation.extxyz --show-tags --write-bad-csv
mace-vasp2extxyz --runlist runlist.txt --out train.extxyz --index index.csv --failed failed.txt
mace-convert-lammps modelname.model
```

For the same MACE dataset builder through the grouped command:

```bash
atomi mace-build-dataset --neareq neareq_train.extxyz --phonopy phonopy.extxyz --force-spread forces.extxyz --prefail-group prefail=prefail.extxyz
```

For MACE energy outlier detection on a GPU allocation:

```bash
atomi mace-energy-outliers --extxyz training.extxyz --model model.model --outdir energy_outliers --device cuda --dtype float32 --top-n 30 --write-poscars
```

To remove outlier frames and optionally append rerun results:

```bash
mace-update-outliers --report energy_outliers/report.txt --train-in training.extxyz --valid-in validation.extxyz --train-out training_clean.extxyz --valid-out validation_clean.extxyz
mace-update-outliers --report energy_outliers/report.txt --train-in training.extxyz --valid-in validation.extxyz --train-out training_updated.extxyz --valid-out validation_updated.extxyz --add-extxyz rerun_bad_energy.extxyz
```

To QA extxyz labels, plots, and optional MACE-safe key rewriting:

```bash
mace-check-extxyz --train training.extxyz --valid validation.extxyz --show-tags --write-bad-csv
mace-check-extxyz --train training.extxyz --valid validation.extxyz --rewrite-refkeys --train-out training_ref.extxyz --valid-out validation_ref.extxyz
```

To collect completed VASP DFT run folders into an extxyz training file:

```bash
mace-vasp2extxyz --runlist runlist.txt --out mlacs_550K_train.extxyz --index index_mlacs_550K_train.csv --failed failed_mlacs_550K_train.txt
```

To convert a trained MACE model for LAMMPS:

```bash
convertmace modelname.model
convertmace modelname.model --env ~/m_lammps_env --partition gpu --gres gpu:1
convertmace modelname.model --hpc-config atomi_hpc_config.json
convertmace modelname.model --local
```

`plotcp2ck` is also accepted as an alias for `plotcp2k`.

## LAMMPS MD Workflow

The packaged LAMMPS MD engine runs staged equilibration or production workflows from a JSON config. Create a project skeleton with:

```bash
lammps-md-init my_lammps_md_project
cd my_lammps_md_project
```

Then edit `config.json` for equilibration or `config_production.json` for production runs. Put structures under `structures/`, MACE-LAMMPS `.pt` models under `models/`, and keep generated run state under `stages/`.

Run directly on a login/interactive workflow allocation:

```bash
lammps-md-workflow --config config.json
lammps-md-workflow --resume --config config.json
lammps-md-workflow --resume --start-from npt_200K --config config.json
lammps-md-workflow --resume --only npt_prod_1400K --config config_production.json
```

When a regular equilibration workflow finishes, Atomi automatically writes `config_production.json` from completed NPT equilibrium stages. A completed stage must have `stages/<stage>/PASS` plus `<stage>.restart` or `<stage>.data`. The generated production config is flagged with `generated_by`, `source_config`, and `source_equilibration_stage` fields so it is clear it came from the finished `config.json` run.

To change the production length during generation:

```bash
lammps-md-workflow --config config.json --production-time-ps 100
lammps-md-workflow --config config.json --production-steps 1000000
lammps-md-workflow --config config.json --production-config-out config_production_100ps.json
```

To skip production config generation:

```bash
lammps-md-workflow --config config.json --no-write-production-config
```

Or submit the orchestrator itself to Slurm:

```bash
sbatch run_workflow.sh resume
sbatch run_workflow.sh fresh
sbatch run_workflow.sh resume npt_200K config.json
sbatch run_workflow.sh resume "" config_production.json
```

The workflow submits one LAMMPS chunk at a time using `run_lammps_gpu.sh`, waits for `squeue` to clear, checks wrapper exit status, parses thermo output, writes `summary.txt` and optional `thermo.png`, and stores stage `PASS` markers for resume.

For one temperature or one log file, use the single-run postprocessor:

```bash
lammps-postprocess --log stages/npt_prod_1400K/chunk_production/log.in.npt_prod_1400K_production --temperature 1400 --timestep-ps 0.0001 --natoms 96 --discard-ps 20 --window-ps 5 --window-stride-ps 1 --plot-bin-ps 0.5 --outdir analysis/npt_prod_1400K
```

This writes `thermo_summary.json`, `window_summaries.csv/json`, `selected_timeseries.csv`, and diagnostic plots for the selected window and subwindows.

After production runs finish, analyze the temperature series from `config_production.json`:

```bash
lammps-thermo-series --config config_production.json --outdir analysis/thermo_0_300K_uq --min-window-ps 18 --window-stride-ps 2 --plot-bin-ps 0.5 --raw-decimate 5 --natoms 96 --plot-T-min 0 --plot-T-max 300 --plot-T-step 10 --anchor-zero --n-bootstrap 300
```

Use `--cp-source dH` to use dH/dT for Cp:

```bash
lammps-thermo-series --config config_production.json --outdir analysis/thermo_0_300K_uq_dH --min-window-ps 18 --window-stride-ps 2 --plot-bin-ps 0.5 --natoms 96 --plot-T-min 0 --plot-T-max 300 --plot-T-step 10 --anchor-zero --cp-source dH --n-bootstrap 300
```

For high-temperature integration anchored at 300 K:

```bash
lammps-thermo-series --config config_production.json --outdir analysis/thermo_anchor_300K --min-window-ps 20 --window-stride-ps 2 --plot-bin-ps 0.5 --natoms 96 --plot-T-min 300 --plot-T-max 1500 --plot-T-step 10 --cp-source dH --thermo-anchor-T 300 --thermo-anchor-S 78.0 --thermo-anchor-Cp 64.0 --use-anchor-for-integration --use-anchor-Cp-in-fit --n-bootstrap 100
```

This command packages the v4 anchor-capable analyzer, which also supports the earlier v3-style fluctuation and dH workflows.

## VASP Live Plotting

The first visualization tools wrap your gnuplot terminal monitors for VASP SCF progress.

For one output file:

```bash
atomi vasp-live vasp.out
atomi vasp-live vasp.out --window 200
```

For one to four output files:

```bash
atomi vasp-live4 run1/vasp.out run2/vasp.out run3/vasp.out run4/vasp.out --window 100
```

These commands require `gnuplot` on `PATH`. On an HPC system, that usually means:

```bash
module load gnuplot
```

## LAMMPS Live Plotting

LAMMPS thermo logs can be monitored in the terminal:

```bash
atomi lammps-live log.lammps
atomi lammps-live log.lammps --window 80 --interval 5
```

For a text summary of thermo data:

```bash
atomi lammps-summary log.lammps
atomi lammps-summary log.lammps --last-fraction 0.25
```

The live plot assumes thermo columns similar to:

```text
Step Temp PotEng TotEng Press Volume
```

More flexible column-name parsing can be added as the LAMMPS toolkit grows.

## CP2K Live Plotting

The CP2K command auto-detects common MD and GEO optimization logs:

```bash
atomi cp2k-live cp2k.log
atomi cp2k-live cp2k.log trajectory.xyz
atomi cp2k-all cp2k_geoopt.log
```

For MD logs, `plotcp2k` calls the packaged bond-tracking and ETA helpers when useful.
If no trajectory is found, the MD monitor still plots energy/temperature/SCF data and skips bond panels.

## Recommended Migration Pattern

1. Put reusable Python logic under `src/atomi/`.
2. Put command-line interfaces under `src/atomi/cli/`.
3. Put code-specific logic under `src/atomi/codes/`.
4. Keep bash scripts only when they truly need shell behavior.
5. Put cluster-specific settings in YAML profiles instead of hard-coding paths.
6. Add one small test whenever you migrate an important script.

See `docs/migration.md` for a fuller plan.
