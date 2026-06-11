# smol Backend Installation For Atomi Defect Thermodynamics

Use `smol` only as an optional Mode 4 backend:

```text
large_state_space.backend = smol_ce_mc
```

Do not install it into every Atomi environment unless CE-MC work is needed.

## Workstation Or Simple HPC

```bash
python -m venv .venv-atomi-smol
source .venv-atomi-smol/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install smol
python -m pip install -e ".[defects,smol]"

atomi-defects backend doctor --backend smol_ce_mc
```

## Conda/Mamba HPC Profile

```bash
module load miniconda || module load anaconda || true
mamba create -n atomi-smol -c conda-forge \
  python=3.11 pip numpy scipy pandas pymatgen spglib ase h5py netcdf4 xarray scikit-learn -y
conda activate atomi-smol

python -m pip install smol
cd /path/to/atomi
python -m pip install -e ".[defects,smol]"

atomi-defects backend doctor --backend smol_ce_mc
```

## Role In The Engine

Atomi writes a backend-neutral `CETrainingSet`, then the smol adapter will map it to a Python-native cluster expansion and MC workflow. Absolute free energies still require thermodynamic integration or a validated estimator; raw MC averages alone are not enough for pycalphad export.
