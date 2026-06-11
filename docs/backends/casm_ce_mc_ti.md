# Backend: casm_ce_mc_ti

`casm_ce_mc_ti` is the production-grade CE-MC/TI route for the Atomi defect thermodynamic engine.

Use it after the Atomi schema is stable and the low/moderate `(Gd,U)O2` static campaign has identified which defect motifs and compensation mechanisms matter.

Inputs from Atomi:

- Parent fluorite lattice and occupant model.
- Training structures and energies in `CETrainingSet`.
- Sublattice model: `(U4, U5, Gd3)_1 (O, VaO)_2`.
- Charge-neutrality or constrained composition axes.

Expected outputs:

- CE fit metadata and validation diagnostics.
- Canonical or semigrand MC observables.
- Thermodynamic integration surfaces.
- `ThermoSurface` rows for pycalphad fitting.

CASM should remain optional and external-environment friendly. Atomi should detect whichever CASM layer is visible: `casm`, `ccasm`, `casm-python`, or modular `libcasm` packages.
