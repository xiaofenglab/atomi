# CASM Backend Installation For Atomi Defect Thermodynamics

Use CASM only as an optional production Mode 4 backend:

```text
large_state_space.backend = casm_ce_mc_ti
```

Keep CASM isolated from the base Atomi environment. Do not add `casm-cpp` as a hard dependency in `pyproject.toml`.

## Conda Profile

```bash
module load miniconda || module load anaconda || true
conda create -n atomi-casm \
  --override-channels -c prisms-center -c conda-forge \
  casm-cpp=1.2.0 python=3 -y
conda activate atomi-casm

python -m pip install -U pip wheel setuptools
python -m pip install casm-python
cd /path/to/atomi
python -m pip install -e ".[defects]"

atomi-defects backend doctor --backend casm_ce_mc_ti
```

## Modular Python Profile

```bash
module load miniconda || module load anaconda || true
mamba create -n atomi-libcasm -c conda-forge python=3.11 pip numpy scipy pandas -y
conda activate atomi-libcasm

python -m pip install \
  libcasm-xtal libcasm-composition libcasm-mapping libcasm-configuration \
  libcasm-clexulator casm-bset libcasm-monte libcasm-clexmonte casm-project casm-tools

cd /path/to/atomi
python -m pip install -e ".[defects]"
atomi-defects backend doctor --backend casm_ce_mc_ti
```

## Container Profile

```bash
module load apptainer || module load singularity || true
apptainer pull casm.sif docker://casmcode/casm
```

Run CASM through an explicit execution profile in the defect-engine config. Atomi should treat CASM results as file-based inputs/outputs so that site-specific modules and containers remain outside the base package.
