# Migration Plan

## Target Shape

Keep old project folders focused on science results. Move reusable workflow logic into this package.

Recommended classification:

- One-off commands stay in project notes or notebooks.
- Reused shell snippets become `shell/*.sh` helpers.
- Reused Python scripts become package functions.
- User-facing workflows become `atomi` subcommands.
- Cluster-specific settings become YAML profiles.

## Suggested Inventory Spreadsheet

Create a small table for your existing scripts:

| Script | Language | Used for | Code | Inputs | Outputs | HPC-specific? | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prepare_vasp.py | Python | setup | VASP | POSCAR | run dirs | no | migrate to `codes/vasp.py` |
| submit_all.sh | Bash | submit | all | folders | job ids | yes | wrap as CLI or shell helper |

## Migration Order

1. Start with scripts you use every week.
2. Move parsing and file generation into Python functions.
3. Add CLI wrappers only after the functions are reusable.
4. Replace hard-coded paths, accounts, queues, and modules with profile YAML.
5. Add tests for parsers and input generators.
6. Tag releases in GitHub once the package is stable enough to install elsewhere.

## HPC Installation Advice

Avoid relying on editable installs on production systems. Prefer:

- a GitHub release tag
- a built wheel
- a conda/mamba environment file
- cluster profiles committed as examples, with private values overridden locally

## Naming Advice

Choose a short name that is not already a major package. Good examples:

- `atomi`
- `molflow-hpc`
- `<groupname>-atomkit`
- `<initials>-hpc-workflows`
