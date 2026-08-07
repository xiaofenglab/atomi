# `moccheck`: pre-ALTER/SUPSYM orbital diagnostics

`moccheck` reads a matching OpenMolcas input and log/output file. It selects
the single last RASSCF block before the first block containing `ALTER` or
`SUPSYM`, verifies the corresponding output module using symmetry, spin, and
active-electron count, and prints frontier-orbital AO character from each
symmetry table in that same module. This matters because the setup inherits
one JobIph/orbital set; it does not independently inherit the last
state-specific calculation for each irrep.

```bash
moccheck RUN.inp RUN.log
atomi moccheck RUN.inp RUN.out
```

The default report contains six occupied and ten virtual orbitals around the
frontier for every symmetry table printed by the final RASSCF reference before
the first `ALTER`/`SUPSYM` setup block. The input and output are parsed once,
even for a large multi-symmetry log. Only orbitals above the default `0.999`
occupation threshold count as occupied; partially occupied orbitals such as
`0.5` are shown on the LUMO/virtual-like side. HOMO/LUMO labels are therefore
occupation-defined by orbital order, not canonical filling. RASSCF
active-space orbital energies need not follow canonical Aufbau ordering, and
`moccheck` warns when the occupation-defined LUMO energy lies below the HOMO
energy.

```bash
atomi moccheck RUN.inp RUN.log \
  --occupied-orbitals 6 \
  --virtual-orbitals 10 \
  --top-aos 6 \
  --json-out moccheck_all_symmetries.json
```

The legacy `--orbitals N` option remains available and overrides both counts.

## RAS3 intent and homing audit

`moccheck` also reads `Inactive`, `RAS1`, `RAS2`, `RAS3`, existing `ALTER`,
and existing `SUPSYM` data from the first setup block. It compares the final
pre-setup orbital identities with the intended RAS3 central-atom shells and
reports:

- the actual final RAS3 slot range in every symmetry;
- the strongest matching source MOs and their occupations;
- existing RAS3 identities after the deck's current `ALTER` sequence;
- minimal missing virtual-homing swaps;
- occupied-source swaps in a separate, conditional partition-change section;
- close-energy same-symmetry neighbors within the configurable
  `--mixing-window-ha` window; and
- a mixing-preserving probe block that keeps only existing `SUPSYM` groups,
  plus an optional identity-lock scaffold that adds the pre-`ALTER` source
  identities intended for RAS3 and reports the expected final RAS3 slots for
  output QA.

Source matching is done component by component (for example `5f0`, `5f2+`,
and `7s`), using the requested atom-shell share normalized over the AO
coefficients printed for each MO. Raw AO coefficients are not ranked across
MOs because their magnitudes are not directly comparable in a nonorthogonal
basis. This remains an identity diagnostic, not a quantitative population
analysis.

Declare intent explicitly near the top of a production input:

```text
* ATOMI RAS3_TARGET U1:5f
* ATOMI RAS3_TARGET U1:5f U1:7s
* ATOMI RAS3_TARGET U1:5f U1:7s U1:7p
```

For compatibility with existing decks, `moccheck` recognizes clear filename
or leading-comment labels such as `5f-only`, `5f+7s`, and `5f+7s+7p`, then
infers a unique central atom from the printed AO character. If shell intent or
the central atom is ambiguous, it stops at diagnosis and asks for the explicit
annotation instead of inventing an `ALTER` recipe.

An occupied source is never labeled as a safe virtual-homing swap. For an
f-electron ion it may still need to enter the active space, but that operation
changes the orbital partition and must be checked against `NACTEL`, inactive
counts, and the intended ground configuration. Likewise, the suggested
`SUPSYM` block is review-required: it can prevent accidental same-symmetry
orbital exchange, but over-constraining RAS3 can suppress physical
metal-ligand mixing. Validate accepted changes in a short RASSCF-only setup
probe before production.

The preferred sequence when metal-ligand mixing is physically meaningful is
therefore `ALTER` homing first, with no new RAS3 `SUPSYM` group. Inspect the
resulting active-orbital character and occupations. Add the optional RAS3
identity lock only if this probe demonstrates destructive identity exchange or
active-space drift. Existing core-orbital `SUPSYM` constraints can be retained
independently.

OpenMolcas applies `SUPSYM` to the starting orbital identities. When `ALTER`
then swaps an identity into a RAS3 slot, its supersymmetry label follows that
identity. Consequently, the generated input scaffold uses pre-`ALTER` source
MO indices; the separately reported final-slot groups are post-`ALTER` QA
expectations and must not be copied back as source indices.

Use `--symmetry N` for a focused report. Repeat the option to preserve a
specific subset and order:

```bash
moccheck RUN.inp RUN.log --symmetry 1
moccheck RUN.inp RUN.log --symmetry 4 --symmetry 2
```

Single-symmetry JSON retains schema `atomi.molcas_moccheck.v1`. An
all-symmetry or multi-symmetry JSON uses
`atomi.molcas_moccheck_collection.v1` and stores the individual diagnostics in
the `diagnostics` list.

Use `--reference-block N` when the deck has no `ALTER`/`SUPSYM` setup or the
desired reference is not the automatically selected block. Its output module
can still contain multiple symmetry tables, and `moccheck` audits those tables
from that one module.

Each AO line reports the signed Molcas coefficient and a normalized display
share, `100*c_i^2/sum_j(c_j^2)`, over the AO coefficients printed for that MO.
This share is useful for rapid identity checks but is not a Mulliken, Loewdin,
or other population analysis because the AO basis is nonorthogonal.

The reference RASSCF may use either `ORBL ALL` with `ORBA FULL` or `ORBA COMP`.
`FULL` is preferred because it provides the complete printed coefficient
matrix. With `COMP`, `moccheck` parses the thresholded per-orbital AO listing
and emits a warning that omitted small coefficients are not included in the
normalized display shares. The compact result is suitable for orbital-identity
QA, but not for quantitative population analysis.
