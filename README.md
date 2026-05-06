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
python -m pip install git+https://github.com/<user>/<repo>.git
```

If the compute environment has no internet access, install on a login node or build a wheel:

```bash
python -m pip wheel . -w dist
python -m pip install dist/atomi-*.whl
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
