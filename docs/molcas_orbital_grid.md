# OpenMolcas Orbital Grid Visualization

Atomi uses OpenMolcas `GRID_IT` values as the source of truth for
reproducible orbital figures. The report path is headless and does not require
the Qt/VTK graphical stack:

## Ground and spin-free excited natural orbitals

Start from the preserved RASSCF `JobIph` containing the desired root. Atomi
writes a matched state replay and grid plan:

```bash
molcas-postanalysis natural-orbital-plan \
  --jobiph RUN.JobIph_9 \
  --root 166 \
  --coord cluster.xyz \
  --basis ANO-RCC-VDZP \
  --relativity x2c \
  --state-role so-parent \
  --spin-free-state 1125 \
  --so-state 1395 \
  --parent-weight-percent 10.97 \
  --orbital 3:19-21 \
  --orbital 4:25-28 \
  --label m5_so1395_sf1125 \
  --outdir natural_orbitals
```

Run the generated inputs with OpenMolcas in the order recorded in the JSON
manifest. The first uses `RASSI EJOB + NATORB + TRD1` to obtain a
state-specific `SiOrb`; the second uses `GRID_IT` to produce the ASCII grid.
The same command supports `--state-role ground` and
`--state-role spin-free-excited`.

For f-block production, keep the relativistic model explicit. The default
`--relativity x2c` writes `RX2C + AMFI`. The alternative
`--relativity dkh2` writes DKH2 with `R02O02` property picture-change
correction plus AMFI. Match the replay to the parent calculation.

Do not use an `ONEL`-only, zero-density RASSI replay as a natural-orbital
source. Require successful `EJOB`, `NATORB`, and `TRD1`, nonzero natural
occupations, and a nonzero one-particle density.

## Headless rendering

```bash
molcas-postanalysis orbital-grid-render \
  --grid RUN.grid \
  --orbital 3:19-21 \
  --orbital 4:25-28 \
  --isovalue 0.05 \
  --center-atom-index 1 \
  --center-radius-bohr 3.0 \
  --min-center-fraction 0.50 \
  --outdir orbital_grids
```

The command writes:

- one phase-colored PNG for every selected orbital;
- a common-view montage;
- a CSV with extrema, isovalue, phase voxel counts, and center weight;
- a JSON provenance and QA record.

Use a common absolute `--isovalue` when comparing orbitals in one manifold.
The default per-orbital relative value is useful for inspecting shape, but it
can visually hide differences in localization.

The center fraction is the ratio of the grid integral of `|orbital|^2` inside
the selected center sphere to the integral over the complete grid. It is a
diagnostic, not an atomic population analysis. A failed localization guard
should trigger AO-composition and active-space review rather than automatic
rejection of the electronic state.

Pegamoid remains available through `pegamoid-bridge` for interactive
inspection. `pegamoid-status` only reports success when the entry point and
its Qt, VTK, and HDF5 imports all work. Keep this GUI environment separate
from the main Atomi environment on HPC systems.

State labels matter:

- a final generic RasOrb/Molden file is not automatically a state-specific
  spin-orbit orbital;
- the correct hierarchy is AO basis functions -> active MOs -> spin-free
  many-electron CI states -> RASSI SO mixtures;
- RASSI parent percentages are squared SO-mixing coefficients, not AO
  percentages;
- a spin-free parent natural-orbital plot may explain an SO state, but it is
  not a unique SO orbital;
- use RASSI `SONORB` for selected spin-orbit states;
- use RASSI `SONT` for spin-orbit natural transition orbitals;
- run `GRID_IT` on those state-specific orbital files before making the final
  ground/excited or M4/M5 transition comparison.

For figures, show the active orbital basis and the SO parentage as separate
layers. Include natural occupations, AO composition, the five largest printed
spin-free parent weights, and an explicit unprinted remainder. Use a common
absolute isovalue for comparisons. Keep `provenance.json`, the orbital QA CSV,
and the natural-orbital plan beside the figures.
