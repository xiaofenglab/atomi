# OpenMolcas Orbital Grid Visualization

Atomi uses OpenMolcas `GRID_IT` values as the source of truth for
reproducible orbital figures. The report path is headless and does not require
the Qt/VTK graphical stack:

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
- use RASSI `SONORB` for selected spin-orbit states;
- use RASSI `SONT` for spin-orbit natural transition orbitals;
- run `GRID_IT` on those state-specific orbital files before making the final
  ground/excited or M4/M5 transition comparison.
