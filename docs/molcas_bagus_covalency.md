# MOLCAS Bagus-Style Covalency Investigation

`molcas-bagus-covalency` is the investigation layer after OpenMolcas
postanalysis. It consumes explicit orbital, configuration, and transition
tables and writes a provenance-preserving, multi-measure covalency packet.
The schema is chemistry-neutral: primary and secondary shells can be U 5f/6d,
Nd 4f/5d, a transition-metal d/s pair, or another declared reference.

This command does not calculate a universal "percent covalent" value. Orbital
delocalization, ligand-hole weight, secondary-shell participation, and
spectral satellite intensity are related but non-equivalent observables.

## Entry Points

```bash
molcas-bagus-covalency --write-template bagus_covalency_spec.json
molcas-bagus-covalency \
  --spec bagus_covalency_spec.json \
  --outdir bagus_covalency_r1

# The same layer inside the formal MOLCAS postanalysis workflow:
molcas-postanalysis bagus-covalency \
  --spec bagus_covalency_spec.json \
  --outdir bagus_covalency_r1
```

An existing nonempty output directory is never overwritten. Use a new
revision directory when the scientific inputs or mapping change.

## Required Scientific Contract

The JSON specification records:

- system label, central element, and central atom;
- primary, secondary, ligand, and optional core shells;
- exact basis, relativistic Hamiltonian, and wavefunction treatment;
- stable state IDs and roles (`ground`, `excited`, `so-parent`, or
  `reference`);
- matched isolated-ion or fragment radial references;
- CSV column maps, exact value maps, row filters, and weight scales;
- accepted/provisional/screening/rejected source quality;
- explicit state-resolved satellite classes;
- project references and caveats.

Ground/excited comparisons require matched state meaning, orbital identity,
basis, Hamiltonian, and projection definitions. A complete XANES calculation
does not rescue a failed orbital-identity comparison.

## Measures

| Family | Output | Interpretation |
|---|---|---|
| Relative orbital energy | Energy span and occupation-weighted centroid | Compare between states only when the orbital identities are matched |
| Natural occupation | Electron count over the declared orbital scope | Scope-dependent, not an oxidation-state assignment by itself |
| Spatial extent | Occupation-weighted radial extent | Measures orbital expansion or contraction |
| Matched radial extent | Ratio to a declared isolated-ion/fragment radius | Requires the same basis, Hamiltonian, and projection convention |
| Isolated-reference projection | Occupation-weighted primary-shell fraction | Identity and localization diagnostic |
| Ligand projection | Electrons projected into the declared ligand subspace | Orbital projection, not a CI ligand-hole weight |
| Ligand-hole weight | Normalized CI weight with one or more ligand holes | Requires configuration-resolved state weights |
| Secondary-shell participation | Expected CI occupation or projected electrons | For example U 6d, Nd 5d, or metal 4s participation |
| Satellite intensity | Fraction of edge oscillator strength | Requires a named, accepted, state-resolved satellite assignment |
| Lz/Sz/Jz expectations | Optional occupation-weighted spinor observables | Unavailable for scalar orbitals unless compatible spinor data are supplied |

Energy-window sidebands are always descriptors. They are not promoted to
charge-transfer satellites by position or visual resemblance alone.

## Input Adapters

Each source accepts:

```json
{
  "id": "ground_ci",
  "path": "configuration_weights.csv",
  "quality": "screening",
  "weight_scale": 0.01,
  "filters": {"root": [1]},
  "defaults": {"state_id": "ground_1"},
  "columns": {
    "weight": "weight_percent",
    "ligand_holes": "n_ligand_holes",
    "primary_electrons": "n_primary",
    "secondary_electrons": "n_secondary"
  }
}
```

`filters` are exact source-column filters. `value_maps` translate source
labels into stable canonical values:

```json
{
  "columns": {"state_id": "state_label"},
  "value_maps": {
    "state_id": {
      "GS SF1": "ground_1",
      "M5 SO1395 SF1125": "m5_parent_1395"
    }
  }
}
```

Configuration weights are fractions after `weight_scale` is applied. The
default norm gate is 0.95-1.05, which catches truncated tables and accidental
percent-versus-fraction mistakes.

For transitions, a satellite is counted only when:

1. `assignment_basis` is `state_resolved_configuration`;
2. the row's class is in the source's explicit `satellite_values`;
3. the row's assignment status is `accepted`.

## Physical Guards

The default guards are intentionally conservative:

- at least 99% of the declared occupied scope has a primary-shell projection;
- at least 80% of the occupied scope remains in the declared primary-shell
  reference subspace;
- the matched radial-extent ratio lies between 0.5 and 2.0;
- configuration weights sum to 0.95-1.05 after scaling;
- all parsed numeric values are finite and physically nonnegative where
  required.

Thresholds are explicit in the spec and copied into the provenance chain.
A rejected state comparison does not automatically reject unrelated
transition energies or spectrum intensities.

## Durable Data Layer

The packet writes canonical plotting tables before aggregation:

- `bagus_covalency_canonical_orbitals.csv`
- `bagus_covalency_canonical_configurations.csv`
- `bagus_covalency_canonical_transitions.csv`

These tables retain source file, source row, source quality, canonical state
IDs, and unit-bearing columns. Plotting can therefore be revised without
reparsing raw MOLCAS outputs.

Derived outputs are:

- `bagus_covalency_orbital_metrics.csv`
- `bagus_covalency_configuration_metrics.csv`
- `bagus_covalency_spectral_metrics.csv`
- `bagus_covalency_measure_status.csv`
- `bagus_covalency_summary.json`
- `bagus_covalency_provenance.json`
- `BAGUS_COVALENCY_README.md`
- optional PNG/PDF audit figures.

The provenance JSON records SHA-256 hashes for every local source table and
the mapping specification.

## Evidence Status

- `accepted`: the declared method, source, and physical guards support use.
- `provisional`: useful but one or more required measures remain incomplete.
- `screening`: a prior or diagnostic calculation, not production evidence.
- `unavailable`: the method or supplied data do not provide the observable.
- `rejected`: a numerical, identity, radial, or configuration gate failed.

The overall status is the strictest status among the measures marked required
for the represented state roles and spectrum.

## Method Boundaries

- A scalar-relativistic natural orbital is not a four-component spinor.
- RASSI SO-parent percentages are many-electron mixing coefficients, not AO
  percentages.
- A projected electron count is not the same observable as an active-space CI
  occupation.
- XANES intensity is not a direct covalency percentage.
- A screening fixed-orbital CI probe must remain labeled screening until its
  root, CSF, orbital-identity, and method guards pass.
