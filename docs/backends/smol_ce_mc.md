# Backend: smol_ce_mc

`smol_ce_mc` is the first Python-native CE-MC target for the Atomi defect thermodynamic engine.

Inputs from Atomi:

- `CETrainingSet` JSONL from symmetry-reduced POCC/defect VASP records.
- Sublattice model such as `(U4, U5, Gd3)_1 (O, VaO)_2`.
- Charge-neutrality constraint, for `(Gd,U)O2`: `N_U5 + 2*N_VaO - N_Gd3 == 0`.
- Static or zentropy-corrected energies.

Expected outputs back to Atomi:

- MC observables versus temperature/composition or chemical potential.
- Motif/SRO averages.
- Thermodynamic-integration-ready energy and composition paths.
- A common `ThermoSurface` once an absolute free-energy route is validated.

Degeneracy rule: use `g_sigma` for finite logsum and reference population counting. Once smol explicitly samples real lattice configurations, do not multiply sampled states by POCC embedding degeneracy again.
