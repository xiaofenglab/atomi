# Atomi

Reusable automation tools for atomistic modeling on HPC systems.

Atomi collects scripts for preparing calculations, writing scheduler inputs, tracking jobs, post-processing outputs, and standardizing workflows across VASP, CP2K, LAMMPS, Turbomole, OpenMolcas, and MLIP/MACE workflows.

## Install

For active development:

```bash
git clone https://github.com/xiaofenglab/atomi.git
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

Atomi uses patch version bumps for package updates, for example `0.2.7` to `0.2.8`, so an HPC environment can confirm it is using the expected install.

## Command Discovery

After installation, use the command help rather than copying project-specific recipes from public documentation:

```bash
atomi --help
```

Most workflows also install short console commands for compatibility with older scripts. Each command has its own help text, so the preferred pattern is:

```bash
command-name --help
```

Public documentation intentionally avoids detailed project recipes, molecule names, dataset paths, scheduler partitions, and cluster-specific resource settings.

## Command Groups

Atomi currently includes command families for:

- Visualization and live monitoring for VASP, LAMMPS, CP2K, and MACE logs.
- VASP result checks, energy/convergence summaries, magnetic-moment utilities, and DFT input preparation.
- VASP structural-variance generation for phonopy, near-equilibrium, prefail-MD, stress/force, defect, and MD snapshot datasets.
- MACE/MLIP dataset construction, validation, outlier detection, extxyz updating, and model conversion.
- LAMMPS MD engine setup, staged equilibration/production workflows, post-processing, thermodynamic analysis, and RDF/PDF/S(Q)/F(Q) total-scattering outputs.
- CP2K input preparation, trajectory extraction, molecular box construction, bond analysis, and run cleanup.
- MOOSE and CALPHAD environment discovery modules for project-specific app executables and pycalphad database checks.
- HPC environment diagnostics and local configuration discovery.

The command names and options may evolve as workflows are cleaned up, so use `--help` from the installed version you are actually running.

For LAMMPS total-scattering analysis, `pdf_lammps` accepts a single dump or trajectory, while `pdf_lammps_series` can scan a config file or MD root and analyze only NPT stages. Series outputs include per-temperature RDF/PDF/S(Q)/F(Q) files, explicit PDFgui/RMC-style fitting exports, transition-colored overlay plots, a `series_index.csv`, `series_summary.json`, and a default `.tar.gz` archive.

Single-temperature `pdf_lammps` averages RDF/PDF/S(Q)/F(Q) over the selected time window. Optional `--frame-overlays` writes per-frame overlay curves from that same window, with the averaged structure shown as a black solid curve, and `--adp` writes per-atom and per-species Uiso/Biso displacement summaries in Angstrom-squared units.

For `pdf_lammps_series`, `--frame-overlays` writes those dynamic per-frame G(r)/S(Q) overlay plots inside each temperature folder. `--adp` also aggregates selected-window volume, lattice parameters, and per-element Uiso versus temperature, with uncertainty plots from frame-window/statistical spreads and one combined all-element Uiso plot. Long series analyses can be prepared for Slurm with `--write-sbatch` or submitted directly with `--submit`.

Experimental PDF matching starts with `pdf_md_compare` for ranking MD-derived curves against PDFgetX/PDFgui/RMC-style two-column data, followed by `pdf_md_reweight` for conservative maximum-entropy-style reweighting of MD temperature/window candidates.

## HPC Environment Check

Before using Atomi on a new cluster, run the environment doctor and review the generated report. The report checks common executables, scheduler commands, plotting tools, atomistic engine names, and Python packages used by the packaged workflows.

Atomi looks for HPC configuration in this order:

```text
--hpc-config PATH
ATOMI_HPC_CONFIG
./atomi_hpc_config.json
~/.config/atomi/hpc.json
```

Cluster paths, module names, scheduler partitions, GPU resources, Python environments, and executable names should be treated as local configuration. Do not assume settings from one HPC system are portable to another without checking the doctor output.

For routine use on an HPC where a private config already exists, place the local-only config in `~/atomi_hpc/` with a `*.local.json` name and apply it with:

```bash
confighpc
source ~/atomi_hpc/atomi_hpc_env.sh
```

`confighpc` automatically prefers `atomi_hpc_config*.local.json` files in `~/atomi_hpc/` and writes a sourceable env file for Atomi workflows. To apply the exports directly in the current shell:

```bash
eval "$(confighpc --shell)"
```

Use auto-setup to prefer an existing private config and otherwise start the discovery workflow:

```bash
atomi-doctor --auto-setup --site my_hpc
```

If a config is found, Atomi writes a local `atomi_hpc_env.sh` file with sourceable exports for generated shell/sbatch scripts. Source it in the shell that launches workflows:

```bash
source atomi_hpc_env.sh
```

If no config is found, auto-setup writes a private config template and `atomi_hpc_discover.sh`.

For a new HPC, generate local-only helper files first:

```bash
atomi-doctor --write-config-template atomi_hpc_config.my_hpc.local.json --site my_hpc
atomi-doctor --write-discovery-script atomi_hpc_discover.sh
```

Run the discovery script on the HPC login node, and run it again inside a GPU allocation when GPU workflows are needed. If you already know exact private module stacks, pass them through `ATOMI_PROBE_*_MODULES` environment variables before running the script. Copy confirmed local values into the private config file and keep it ignored by Git.

When multiple module choices are available, run the discovery script interactively so you can choose the stack after reading the module candidates:

```bash
ATOMI_DISCOVERY_INTERACTIVE=1 bash atomi_hpc_discover.sh
```

## Shared Google Drive Development

If this repository is edited from a Google Drive shared folder by more than one session or computer, use the guard script before changing files:

```bash
python tools/shared_sync_guard.py status --probe-wait 2
python tools/shared_sync_guard.py acquire --note "short description of planned edit"
```

After committing and pushing, release the edit lock:

```bash
git status
git push origin main
python tools/shared_sync_guard.py release --note "pushed latest changes"
```

The guard writes root-level `ATOMI_EDIT_LOCK.json`, `ATOMI_SYNC_STATUS.json`, and `ATOMI_SYNC_PROBE.txt` files. These files are synced by Google Drive but ignored by Git. A clean status with no lock means another editing session can proceed. This is a practical coordination flag, not a perfect proof that every Google Drive client has finished downloading.
