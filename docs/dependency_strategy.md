# Dependency Strategy

This page lists Atomi's public dependency groups and the recommended install
strategy for shared HPC environments.

## Base Install

The normal Atomi install is intended to support the common VASP, CP2K, LAMMPS,
MOOSE, total-scattering, and environment-discovery workflows without forcing
large scientific-stack upgrades.

Base dependencies from `pyproject.toml`:

- `ase>=3.23`
- `mp-api`
- `periodictable`
- `xraydb`

`xraydb` supplies X-ray form-factor data for X-ray PDF/XAFS workflows.
`periodictable` supplies neutron coherent scattering lengths for neutron PDF
weighting.

Recommended update command:

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed \
  git+https://github.com/xiaofenglab/atomi.git@main
```

Use `--upgrade-strategy only-if-needed` on production HPC environments so pip
keeps already-working packages unless Atomi truly requires a newer version.

## Optional Extras

Atomi keeps heavier or less-universal workflows behind extras:

- `dev`: testing and linting helpers such as `pytest` and `ruff`.
- `materials`: materials-data helpers such as `pymatgen`.
- `calphad`: CALPHAD helpers such as `pycalphad`.
- `scattering`: explicit scattering-analysis stack with `ase`,
  `periodictable`, and `xraydb`.
- `xafs`: XAFS/Larch workflow support with `xraydb` and `xraylarch`.

Examples:

```bash
python -m pip install -e ".[dev,materials]"
python -m pip install --upgrade --upgrade-strategy only-if-needed \
  "atomi[xafs] @ git+https://github.com/xiaofenglab/atomi.git@main"
```

## Larch/XAFS Guidance

`xraylarch` is intentionally not part of the base install. Larch is powerful,
but it can require a newer and broader scientific Python stack than an existing
LAMMPS or VASP environment wants to change.

Recommended options:

- Keep the main MD/DFT environment on the base Atomi install.
- Install `atomi[xafs]` only in an environment meant to run Larch directly.
- For maximum stability, put Larch in its own virtual environment and point
  Atomi to that environment or to FEFF/Larch outputs through local HPC config.
- Keep FEFF executables such as `feff8l` or `feff6l` outside of package
  dependencies and configure them with local paths or environment variables.

After a `--no-deps` Atomi update, use:

```bash
xafs_status
```

If Larch lives in a separate environment, configure one of:

```bash
export ATOMI_XAFS_LARCH_PYTHON="$HOME/atomi_hpc/larch_env/bin/python"
export ATOMI_XAFS_LARCH_ENV="$HOME/atomi_hpc/larch_env"
```

`xafs_larch_run` will use active-env Larch when available, otherwise it can
fall back to the configured external Larch Python for the Larch `xftf`
transform.

## When To Keep Existing Packages

If the cluster environment already has working versions of NumPy, SciPy,
Matplotlib, ASE, PyTorch, or CUDA-related packages, prefer keeping them unless a
specific Atomi workflow fails and clearly requires a newer version.

For stable updates:

```bash
python -m pip install --upgrade --upgrade-strategy only-if-needed \
  git+https://github.com/xiaofenglab/atomi.git@main
```

For a clean rebuild in a throwaway environment:

```bash
python -m pip install --upgrade --force-reinstall \
  git+https://github.com/xiaofenglab/atomi.git@main
```

Use `--force-reinstall` sparingly on shared or long-lived HPC environments
because it can rebuild the full dependency tree and disturb unrelated packages.
