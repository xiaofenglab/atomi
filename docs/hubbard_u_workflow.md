# First-principles Hubbard-U workflows

Atomi keeps every U value attached to its correlated projector and response
definition. A VASP PAW linear-response U, a VASP MLWF-cRPA U, and a QE
Wannier-projector U are related benchmarks, not interchangeable numbers.

## VASP PAW linear response

Prepare a reference and symmetric perturbation series:

```bash
hubbard-u-workflow vasp-lr-prepare \
  --seed /path/to/healthy/static \
  --outdir uo2_vasp_lr \
  --probe-atom 1 \
  --nodes 2 --ntasks-per-node 48 \
  --mem-per-cpu 3500M \
  --module-load chem/vasp/6.2.1 \
  --vasp-command "vasp -s std"
```

The command splits the probe atom into its own VASP species, duplicates the
matching POTCAR dataset, preserves/reorders MAGMOM, writes a U=0 reference,
and prepares fixed-density and self-consistent `LDAUTYPE=3` calculations.
After the jobs complete:

```bash
hubbard-u-workflow vasp-lr-analyze --root uo2_vasp_lr
```

The analyzer extracts the final on-site occupation, fits `chi0` and `chi`,
reports linearity, and computes `U = 1/chi - 1/chi0`.

For f shells use `LMAXMIX=6`. A UO2 reference must be guarded for AFM moment,
occupation-matrix branch, insulating/metallic character, and linear response
without hysteresis.

Site-specific launch details are explicit inputs rather than source-code
defaults: use `--nodes`, `--ntasks-per-node`, `--cpus-per-task`, repeated
`--mem-per-cpu`, `--module-load`, and `--vasp-command`. This keeps Atomi portable while the
installed HPC profile or local JSON supplies the scheduler policy.

## VASP Wannier-cRPA

```bash
hubbard-u-workflow vasp-crpa-prepare \
  --outdir uo2_crpa --vasp-version 6.6.0 \
  --num-wann 14 --nbands 512 \
  --target-states "..." --crpa-bands "..."
```

This writes staged INCAR additions for the DFT, Wannier, virtual-band, and cRPA
steps. Spectral cRPA is blocked when the declared VASP version is older than
6.6.0. Band indices and Wannier windows are scientific inputs and must be set
after comparing VASP and Wannier-interpolated bands.

## Quantum ESPRESSO

```bash
hubbard-u-workflow qe-prepare --outdir uo2_qe --element U --manifold 5f
```

The baseline route uses stock QE `hp.x` with `ortho-atomic` projectors. The
legacy `HUBBARD (wf)` route uses `pmw.x` projectors and is not an MLWF route.
This is the application layer used in Tesch and Kowalski (2022), but their U
was calculated separately by linear response; the paper explicitly states that
the `pmw.x` projectors were not used to calculate U. Treat this as a labeled
historical reproduction route, not a matched-projector response.
The modern research route requires QE 7.5 or newer, `pw2wannier90.x`,
Wannier90, and `wannier2pw.x`, followed by a matched response implementation
such as a pinned collaborator branch. Stock HP values must not be silently
applied to a different Wannier projector.
