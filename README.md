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

Check the installed version after updating:

```bash
python -m pip show atomi
atomi --version
which md-engine
```

Atomi uses simple patch version bumps for package updates, for example `0.1.0` to `0.1.1`, so an HPC environment can confirm it is using the expected install.

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
atomi md-engine-init my_lammps_md_project
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

For LAMMPS MD engines, also review the `profiles.lammps_md_engine` block. The project launcher and GPU wrapper use these environment variables when needed:

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
extv OUTCAR --mag-lines 80
checkvasp runlist.txt
checkscf runlist.txt 1e-6
checkscf runlist.txt 5e-6 --out bad_runs.txt --clean --dry-run
magit Gd U --dry-run
magit Gd U
vasp-phonopy-neareq --output-root DFT_PHONOPY_NEAREQ --vasp-template VASP_TEMPLATE --mode both --dim 1 1 1
vasp-prefail-candidates --config dump_npt_900K.json --output-root DFT_PREFail_CANDIDATES --vasp-template VASP_TEMPLATE
vasp-stress-force-candidates --output-root DFT_STRESS_FORCE --vasp-template VASP_TEMPLATE --mode both --target-size 80 --bias-species O
vasp-defect-candidates --output-root DFT_DEFECTS --vasp-template VASP_TEMPLATE --species O U
vasp-md-snapshot-candidates --config config_production.json --output-root DFT_MD_SNAPSHOTS --vasp-template VASP_TEMPLATE --last-frames 5
md-engine-init my_lammps_md_project
md-engine --config config.json
md-engine --resume --config config.json
lammps-postprocess --log stages/npt_prod_1400K/chunk_production/log.in.npt_prod_1400K_production --temperature 1400 --outdir analysis/npt_prod_1400K
lammps-thermo-series --config config_production.json --outdir analysis/thermo_0_300K_uq
mace-build-dataset --neareq neareq_train.extxyz --phonopy phonopy.extxyz --force-spread forces.extxyz --prefail-group prefail=prefail.extxyz
mace-energy-outliers --extxyz training.extxyz --model model.model --outdir energy_outliers --device cuda --dtype float32 --top-n 30 --write-poscars
mace-update-outliers --report energy_outliers/report.txt --train-in training.extxyz --valid-in validation.extxyz --train-out training_clean.extxyz --valid-out validation_clean.extxyz
mace-check-extxyz --train training.extxyz --valid validation.extxyz --show-tags --write-bad-csv
mace-vasp2extxyz --runlist runlist.txt --out train.extxyz --index index.csv --failed failed.txt
mace-convert-lammps modelname.model
```

For VASP array DFT checks, `checkvasp` reads a `runlist.txt` whose lines are run directories and reports `DONE`, `RUNNING`, `NOTSTART`, or `MISSING` using `OUTCAR`, `OUTCAR.gz`, and `vasp.out*` files:

```bash
checkvasp runlist.txt
atomi vasp-check runlist.txt
```

For SCF convergence, `checkscf` keeps your original convention where run `N` in `runlist.txt` is checked against `vasp.out*.N` in the current directory:

```bash
checkscf runlist.txt 1e-6
checkscf runlist.txt 5e-6 --out bad_runs.txt --clean --dry-run
checkscf runlist.txt 5e-6 --out bad_runs.txt --clean
```

Use `--dry-run` first when cleaning. Without `--dry-run`, `--clean` removes `OUTCAR*`, `CONTCAR`, `vasprun.xml`, and `OSZICAR` from runs that fail the threshold or have stale VASP outputs without a matching log.

To update `INCAR` magnetic moments from the final `OUTCAR` magnetization table:

```bash
magit Gd U --dry-run
magit Gd U
```

The command reads `OUTCAR`, `POSCAR`, and `INCAR`. It uses the POSCAR species order and counts, copies the final OUTCAR `magnetization (x)` total moments for only the listed elements, and sets unlisted species to zero. For a POSCAR ordered `Gd U O`, the output is written as Gd atom moments, then U atom moments, then a compact oxygen block such as `96*0`. By default it writes `INCAR.bak` before replacing or appending the `MAGMOM` line. Use `--preserve-unselected` to keep existing unlisted-element values instead of zeroing them.

To prepare near-equilibrium phonopy and MLIP VASP datasets from one reference POSCAR:

```bash
vasp-phonopy-neareq \
  --output-root DFT_PHONOPY_NEAREQ \
  --vasp-template VASP_TEMPLATE \
  --mode both \
  --dim 1 1 1
```

For each volume, Atomi writes a base `POSCAR` and can produce two related datasets:

```text
V1.000/phonopy_001/POSCAR       # from phonopy -d
V1.000/rd_001/POSCAR            # from phonopy --rd, useful for MLIP
V1.000/strain_001_hydro_p0p0050/POSCAR
V1.000/strainrd_001_hydro_p0p0050_001/POSCAR
```

`--mode phonopy` generates only finite-displacement phonopy folders. `--mode mlip` generates only the compact MLIP near-equilibrium folders. `--mode both` does both and writes one combined:

```text
DFT_PHONOPY_NEAREQ/runlist.txt
DFT_PHONOPY_NEAREQ/runlist_phonopy.txt
DFT_PHONOPY_NEAREQ/runlist_mlip_neareq.txt
DFT_PHONOPY_NEAREQ/candidate_index.csv
DFT_PHONOPY_NEAREQ/manifest.csv
```

For a 2x2x2 UO2 POSCAR, using `--dim 1 1 1` means the `phonopy -d` step can generate the full symmetry-expanded finite-displacement set appropriate for that already-expanded cell. If this creates hundreds of structures, keep them as the phonopy-specific pool and use the random-displacement/strain folders as the MLIP near-equilibrium pool.

For all VASP preparation commands, `VASP_TEMPLATE` can contain:

```text
VASP_TEMPLATE/POSCAR
VASP_TEMPLATE/INCAR
VASP_TEMPLATE/KPOINTS
VASP_TEMPLATE/POTCAR
```

`POSCAR` is the template/reference structure used for generating modified structures when `--poscar` or `reference_poscar` is not provided. `INCAR`, `KPOINTS`, and `POTCAR` are copied into each generated VASP run folder. The commands print the POSCAR composition and parsed POTCAR symbols, and warn if POSCAR species order and POTCAR order do not match.

To prepare prefail MD frames for VASP array DFT, use one merged command:

```bash
vasp-prefail-candidates \
  --config dump_npt_900K.json \
  --output-root UO2_V4_r4_ACTIVE/DFT_V4_CANDIDATES \
  --vasp-template UO2_V4_r4_ACTIVE/VASP_TEMPLATE
```

The JSON gives the LAMMPS dump, selected timesteps, and atom type map. It can also give `reference_poscar`; if omitted, Atomi uses `VASP_TEMPLATE/POSCAR`. Paths inside the JSON are resolved relative to the JSON file location, so this is portable when the workflow folder moves. The command first writes selected parent structures as:

```text
MD_SELECT/<label>/md_<timestep>/base/POSCAR
```

Then it writes 16 VASP candidate folders per parent:

```text
DFT_V4_CANDIDATES/<label>/md_<timestep>/base/POSCAR
DFT_V4_CANDIDATES/<label>/md_<timestep>/rd_small_001/POSCAR
...
DFT_V4_CANDIDATES/<label>/md_<timestep>/strain_disp_002/POSCAR
```

Each candidate folder receives `INCAR`, `KPOINTS`, and `POTCAR` from `--vasp-template`. Add `--copy-template-all` to copy every template file except `POSCAR`. The command also writes:

```text
DFT_V4_CANDIDATES/runlist.txt
DFT_V4_CANDIDATES/candidate_index.csv
```

Use the runlist for array DFT, then feed the finished runs into `mace-vasp2extxyz`.

To prepare stress/force backbone structures from one equilibrium POSCAR:

```bash
vasp-stress-force-candidates \
  --output-root DFT_STRESS_FORCE \
  --vasp-template VASP_TEMPLATE \
  --mode both \
  --target-size 80 \
  --bias-species O
```

This generates a larger stress/force pool, selects a family-balanced set, writes DFT-ready folders, and creates:

```text
DFT_STRESS_FORCE/runlist.txt
DFT_STRESS_FORCE/candidate_index.csv
DFT_STRESS_FORCE/dataset_summary.json
DFT_STRESS_FORCE/selection_report.json
```

Each selected run folder contains `POSCAR`, `case_info.json`, and the VASP template files. Use `--mode stress` or `--mode force` to build only one family, and add `--copy-template-all` when the template folder has submit scripts or extra VASP helper files that should be copied into every run.

To add point-defect coverage for MLIP training:

```bash
vasp-defect-candidates \
  --output-root DFT_DEFECTS \
  --vasp-template VASP_TEMPLATE \
  --species O U \
  --n-vacancy 6 \
  --n-interstitial 6 \
  --n-frenkel 6 \
  --n-schottky 4
```

The command reads the template POSCAR species order and writes vacancy, interstitial, Frenkel, Schottky, and antisite candidate folders. Schottky candidates remove one reduced formula unit, so a UO2 POSCAR removes one U and two O per Schottky sample. Use `--no-antisite` for ionic systems where antisites are not physically useful, or `--interstitial-species O` if you only want oxygen interstitials.

To harvest successful thermal MD snapshots from an `md-engine` project:

```bash
vasp-md-snapshot-candidates \
  --config config_production.json \
  --output-root DFT_MD_SNAPSHOTS \
  --vasp-template VASP_TEMPLATE \
  --last-frames 5
```

This command reads the md-engine JSON, searches `stages/<stage>/PASS`, enters the latest chunk for each passed stage, finds `dump.*.lammpstrj`, and converts selected frames into DFT-ready VASP folders. By default it uses the last five frames from each successful latest chunk. Useful filters are:

```bash
vasp-md-snapshot-candidates --config config_production.json --output-root DFT_MD_900K --vasp-template VASP_TEMPLATE --stage npt_prod_900K --last-frames 10
vasp-md-snapshot-candidates --config config_production.json --output-root DFT_MD_900K --vasp-template VASP_TEMPLATE -T 900 --last-frames 10
vasp-md-snapshot-candidates --config config_production.json --output-root DFT_MD_300_1500K --vasp-template VASP_TEMPLATE --temperature-range 300 1500 --last-frames 3
vasp-md-snapshot-candidates --config config.json --output-root DFT_MD_EQM --vasp-template VASP_TEMPLATE --stage-regex 'npt.*eqm' --last-frames 3
vasp-md-snapshot-candidates --config config.json --output-root DFT_MD_PICKED --vasp-template VASP_TEMPLATE --timesteps 50000,100000 --atom-type-map 1=O 2=U
```

Temperature filters read `temperature`, `temperature_end`, or a stage name like `npt_prod_900K`. Use repeated or comma-separated `-T` values for several single points, for example `-T 300 -T 900` or `-T 300,900`. If the md-engine JSON contains `atom_type_map`, Atomi uses it. Otherwise the current U/O md-engine default is inferred from `mass_O` and `mass_U`; on another material, pass `--atom-type-map 1=ElementA 2=ElementB` or add that map to the config. The command writes `runlist.txt`, `candidate_index.csv`, and `dataset_summary.json`.

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

## LAMMPS MD Engine

The packaged LAMMPS MD engine runs staged equilibration or production workflows from a JSON config. Create a project skeleton with:

```bash
md-engine-init my_lammps_md_project
cd my_lammps_md_project
```

Then edit `config.json` for equilibration or `config_production.json` for production runs. Put structures under `structures/`, MACE-LAMMPS `.pt` models under `models/`, and keep generated run state under `stages/`.

Run directly on a login/interactive workflow allocation:

```bash
md-engine --config config.json
md-engine --resume --config config.json
md-engine --resume --start-from npt_200K --config config.json
md-engine --resume --only npt_prod_1400K --config config_production.json
```

`--start-from` and `--only` must match a stage name in the config exactly. If the stage is misspelled, `md-engine` stops before running anything or generating `config_production.json`, and prints close stage-name suggestions.

When a regular equilibration workflow finishes, Atomi automatically writes `config_production.json` from completed NPT equilibrium stages. A completed stage must have `stages/<stage>/PASS` plus `<stage>.restart` or `<stage>.data`. The generated production config is flagged with `generated_by`, `source_config`, and `source_equilibration_stage` fields so it is clear it came from the finished `config.json` run.

To change the production length during generation:

```bash
md-engine --config config.json --production-time-ps 100
md-engine --config config.json --production-steps 1000000
md-engine --config config.json --production-config-out config_production_100ps.json
```

To skip production config generation:

```bash
md-engine --config config.json --no-write-production-config
```

Or submit the orchestrator itself to Slurm:

```bash
sbatch run_workflow.sh resume
sbatch run_workflow.sh fresh
sbatch run_workflow.sh resume npt_200K config.json
sbatch run_workflow.sh resume "" config_production.json
```

The MD engine submits one LAMMPS chunk at a time using `run_lammps_gpu.sh`, waits for `squeue` to clear, checks wrapper exit status, parses thermo output, writes `summary.txt` and optional `thermo.png`, and stores stage `PASS` markers for resume.

For a stage with a fixed length, put `fixed_steps` in the stage block:

```json
{
  "name": "lc_nvt_ramp_400K",
  "type": "nvt",
  "temperature_start": 300,
  "temperature_end": 400,
  "fixed_steps": 100000
}
```

`md-engine` writes `run 100000` into the generated LAMMPS input and estimates the Slurm wall time from the same step count. Fixed-step stages run one chunk by default; add `max_chunks` only if you intentionally want repeated fixed-size chunks. You can also specify a duration with `time_ps`, `run_time_ps`, or `duration_ps`, which is converted to steps using the config `timestep`.

NPT stages whose names contain `_eqm` keep a constant chunk size from `adaptive_steps.initial_small` or `adaptive_steps.initial_large`, so an equilibrium block can retry convergence with repeated 50,000-step chunks without growing to longer chunks. Set `"adaptive_growth": true` on a stage if you want the old increasing chunk size behavior.

For NVT ramp stages with both `temperature_start` and `temperature_end`, continuation chunks use the previous chunk's tail temperature instead of restarting the ramp schedule from the original start temperature. By default, `md-engine` averages the last `2 ps` of the previous chunk, keeps restart velocities, and ramps from that average toward the target. If the tail average reaches the target within the temperature tolerance, the ramp stage is accepted and marked `PASS`.

Optional ramp controls can be placed globally or in a stage `ramp_override` block:

```json
"ramp_rules": {
  "tail_average_ps": 2.0,
  "continue_from_tail_temperature": true,
  "accept_temperature_tol_min": 20.0,
  "accept_temperature_tol_fraction": 0.03
}
```

The Slurm `#SBATCH --time` value is computational wall time, not MD physical time. By default, it is estimated from:

```text
reference_hours * atoms/reference_atoms * steps/reference_steps * safety_factor
```

If `reference_hours` already includes your safety margin, use `reference_walltime_hours` instead. You can also override the request globally or per stage:

```json
"performance": {
  "reference_atoms": 768,
  "reference_steps": 50000,
  "reference_walltime_hours": 6.0,
  "atoms_small": 768,
  "atoms_large": 768
}
```

or:

```json
{
  "name": "lc_npt_eqm_300K",
  "type": "npt",
  "temperature": 300,
  "walltime_hours": 6.0
}
```

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
