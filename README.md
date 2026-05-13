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
- LAMMPS MD engine setup, staged equilibration/production workflows, post-processing, and thermodynamic analysis.
- CP2K input preparation, trajectory extraction, molecular box construction, bond analysis, and run cleanup.
- MOOSE and CALPHAD environment discovery modules for project-specific app executables and pycalphad database checks.
- HPC environment diagnostics and local configuration discovery.

The command names and options may evolve as workflows are cleaned up, so use `--help` from the installed version you are actually running.

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

## Shared Google Drive Development

If this repository is edited from a Google Drive shared folder by more than one Codex session or computer, use the guard script before changing files:

```bash
python tools/codex_sync_guard.py status --probe-wait 2
python tools/codex_sync_guard.py acquire --note "short description of planned edit"
```

After committing and pushing, release the edit lock:

```bash
git status
git push origin main
python tools/codex_sync_guard.py release --note "pushed latest changes"
```

The guard writes root-level `CODEX_EDIT_LOCK.json`, `CODEX_SYNC_STATUS.json`, and `CODEX_SYNC_PROBE.txt` files. These files are synced by Google Drive but ignored by Git. A clean status with no lock means another Codex can proceed. This is a practical coordination flag, not a perfect proof that every Google Drive client has finished downloading.
