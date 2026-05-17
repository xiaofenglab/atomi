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
- `zentropy`: optional materials zentropy runtime bridge with `pyzentropy`.
- `elastic-viz`: dependency-light elastic visualization command group. ELATE is
  detected at runtime if the user installs it separately.

Examples:

```bash
python -m pip install -e ".[dev,materials]"
python -m pip install --upgrade --upgrade-strategy only-if-needed \
  "atomi[xafs] @ git+https://github.com/xiaofenglab/atomi.git@main"
```

## Elastic Visualization Guidance

The `elastic_viz` command is a post-analysis layer for elastic tensors already
produced by VASP or MD workflows. It does not need ELATE for its summary tables
or native directional formulas. Use `elate_status` to check whether the active
Python environment can import ELATE. If ELATE is missing, `elastic_viz
--backend auto` falls back to native directional tensor formulas.

ELATE is not kept as a hard dependency of an Atomi extra because it may not be
available from the normal package index on all HPC systems. Install it
explicitly when needed:

```bash
python -m pip install "ELATE @ git+https://github.com/coudertlab/elate.git@master"
```

ElasTool is useful as an external reference workflow for broader postprocessing
such as Christoffel surfaces, hardness estimates, and minimum thermal
conductivity models. Because it is GPLv3 and dependency-heavy, Atomi should not
vendor or copy its code into the base package. Prefer independent formulas in
Atomi plus export/call support for users who install ElasTool separately.

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

## Zentropy Guidance

Atomi's zentropy tools are organized as a staged workflow:

- Stage 1 indexes DFT defect motifs and exports MLIP-ready structures.
- Later stages attach microstate free energies, solve ensemble statistics,
  export CALPHAD/MOOSE-ready tables, and drive active-learning loops.

The materials zentropy runtime is treated as optional because it is less
universal than the base MD/DFT workflow stack. Prefer the same strategy used
for Larch:

- Use base Atomi for motif databases, manifests, and workflow scaffolding.
- Install `atomi[zentropy]` only in Python 3.10+ environments meant to run
  `pyzentropy` directly.
- Or keep `pyzentropy` in a separate environment and point Atomi to it through
  local HPC config or environment variables.

Useful status check:

```bash
zentropy_status
```

External runtime configuration:

```bash
export ATOMI_ZENTROPY_PYTHON="$HOME/atomi_hpc/zentropy_env/bin/python"
export ATOMI_ZENTROPY_ENV="$HOME/atomi_hpc/zentropy_env"
export ATOMI_ZENTROPY_EXE="$HOME/atomi_hpc/zentropy_env/bin/pyzentropy"
```

The package named `zentropy` is not assumed to be the materials workflow
runtime; Atomi checks for `pyzentropy` and warns when only a plain `zentropy`
package is visible.

## CALPHAD And MOOSE Guidance

MOOSE applications and pycalphad databases are best kept outside the Atomi base
environment. This avoids coupling a project-specific MOOSE build, CALPHAD
database set, or MOOSE-managed Python environment to routine Atomi updates.

Use status checks to confirm what Atomi can see:

```bash
moose_status
calphad_status
```

If pycalphad is installed in a MOOSE/CALPHAD virtual environment, point Atomi
to that Python instead of installing pycalphad into every MD/DFT environment:

```bash
export ATOMI_CALPHAD_PYTHON="$HOME/moose_env/bin/python"
export ATOMI_CALPHAD_DATABASES="$HOME/tdb/database.tdb"
calphad_status
```

For MOOSE, point Atomi to the project-specific app executable:

```bash
export ATOMI_MOOSE_APP="$HOME/moose_projects/app/app-opt"
moose_status
```

The same values can live in the local-only HPC config under
`profiles.calphad` and `profiles.moose`. Applying that config with `confighpc`
exports the relevant `ATOMI_CALPHAD_*` and `ATOMI_MOOSE_*` variables. Keep
those local JSON files out of Git when they contain private paths, databases,
or cluster details.

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
