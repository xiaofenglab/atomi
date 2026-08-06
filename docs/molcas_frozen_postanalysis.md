# Frozen OpenMolcas Postanalysis

Atomi separates scientific parsing from figure styling so a large OpenMolcas
output is parsed once and later publication figures can be revised without
silently changing state selection, transition scope, or orbital identity.

## 1. Freeze the data layer

```bash
mocparse RUN.log --inp RUN.inp --outdir RUN_mocparse_r1 \
  --project PROJECT --science-owner "MOLCAS project lead" \
  --scientific-status provisional --quality diagnostic
```

`mocparse` writes an immutable, hash-verified `plot_dataset.json` bundle with
input-scoped module status and reusable CSV tables for spin-free states,
spin-orbit states, printed RASSI parent weights, dipole-transition tables,
one-electron orbitals, and printed AO coefficients. Create a new revision
directory instead of overwriting a bundle. The default status is
`provisional`; parsing success is not scientific acceptance.

## 2. Render edge XANES

```bash
mocxanes RUN_mocparse_r1/plot_dataset.json \
  --element U \
  --edge M5:3545:3560 --edge M4:3745:3760 \
  --initial-states 1,2 --aggregation mean \
  --rassi-section last --gauge length \
  --broadening-profile u-m45-polly-high-resolution \
  --broadening voigt --normalize max \
  --outdir RUN_xanes_r1
```

Every science-changing selection is exposed in the command: RASSI section,
state basis/gauge, initial states, aggregation, edge windows, energy shift,
broadening, and normalization. The element and edge must agree with the
central-atom core-excitation manifold prepared in the OpenMolcas input; RASSI
transition rows do not independently establish that identity.

The default profile is `u-m45-polly-high-resolution`, the established Atomi
Polly comparison profile: Gaussian FWHM `1.2894947 eV` and Lorentzian FWHM
`0.3621724 eV`. It preserves continuity with recent U M4/M5 figures, but it is
a theoretical U M-edge comparison profile rather than a universal lifetime
model. Use `--broadening-profile xraydb-core-hole` for an element/edge-specific
ordinary-XAS width, or `--broadening-profile custom` with explicit
`--gaussian-fwhm` and `--lorentzian-fwhm` values.

The renderer writes the broadened curve and stick table as CSV, render
provenance as JSON, and publication outputs as 300 dpi PNG, PDF, and SVG. Its
default science-clean style uses no grid, a centered explicit energy window,
and dipole sticks in a separate band below the spectral baseline. The y-axis
states the selected policy explicitly: maximum-normalized, area-normalized in
`eV^-1`, or unnormalized arbitrary intensity.

## 3. Render MO and state-splitting figures

```bash
mocmo RUN_mocparse_r1/plot_dataset.json \
  --orbital-section -1 --rassi-section last \
  --mo-window=-20:20 --state-window 0:5 \
  --parent-weight-threshold 0.05 \
  --outdir RUN_mo_r1
```

`mocmo` produces two distinct figures:

- A one-electron MO-level diagram separated by printed Abelian symmetry
  species, referenced to the HOMO of each selected orbital section.
- A many-electron spin-free to RASSI spin-orbit state-correlation diagram.
  Link widths encode printed RASSI parent weights, with the unprinted parent
  remainder reported rather than silently assigned. Spin-free and SO energies
  are each referenced to their own manifold minimum; the plot does not imply a
  shared absolute energy zero.

The second figure describes state interaction, spin-orbit splitting, and
redistribution of many-electron parent character. Its links must not be called
AO percentages or literal MO mixing. Orbital occupations outside 0-2,
RASSI-parent weights outside 0-1, oversummed printed weights, incomplete
initial-state averaging, duplicate transition rows, and missing excitation
energies are hard failures.

## Ownership and provenance

`mocparse` owns scientific extraction. `mocxanes` and `mocmo` read only the
verified frozen bundle and never reopen the raw output. Plot revisions therefore
remain reproducible while the MOLCAS project lead retains authority over active
space, roots, edge assignment, accepted/provisional scope, energy calibration,
and physical interpretation.
