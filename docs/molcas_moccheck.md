# `moccheck`: pre-ALTER/SUPSYM orbital diagnostics

`moccheck` reads a matching OpenMolcas input and log/output file. It selects
the last requested symmetry block before the first RASSCF block containing
`ALTER` or `SUPSYM`, verifies the corresponding output module using symmetry,
spin, and active-electron count, and prints frontier-orbital AO character.

```bash
moccheck RUN.inp RUN.log
atomi moccheck RUN.inp RUN.out
```

The default report contains six occupied and six virtual orbitals around the
frontier for symmetry 1. Partially occupied active orbitals count as occupied
when their occupation is above the default `1e-3` threshold. HOMO/LUMO labels
are therefore occupation-defined by orbital order. RASSCF active-space orbital
energies need not follow canonical Aufbau ordering, and `moccheck` warns when
the occupation-defined LUMO energy lies below the HOMO energy.

```bash
atomi moccheck RUN.inp RUN.log \
  --symmetry 1 \
  --orbitals 6 \
  --top-aos 6 \
  --json-out moccheck.json
```

Use `--reference-block N` when the deck has no `ALTER`/`SUPSYM` setup or the
desired reference is not the automatically selected block.

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
