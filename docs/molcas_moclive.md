# `moclive`: live OpenMolcas workflow monitoring

`moclive` reads an OpenMolcas input deck together with its growing log or
output file. The input deck is authoritative: only scheduled `RASSCF`,
`CASPT2`, and `RASSI` modules in that file belong to the monitored scope, and
they are matched to output modules in chronological order.

The default terminal view refreshes every five minutes:

```bash
moclive RUN.inp RUN.log
moclive RUN.inp RUN.out --interval 300
```

The first watch refresh parses the output accumulated so far. The monitor then
retains its parser state, byte offset, and any incomplete final line, so each
later refresh reads only bytes appended since the previous refresh. If the log
is truncated or rotated, or if the input deck changes, the state is reset and
rebuilt deliberately. Compressed `.gz` output is treated as a complete archive
and uses a full one-shot audit rather than incremental tailing.

Use a single refresh for scripts, reports, or scheduler checks:

```bash
moclive RUN.inp RUN.log --once
moclive RUN.inp RUN.log --once --json
python -m atomi.qchem.molcas_live RUN.inp RUN.log --once
```

`--ascii` replaces Unicode status marks and the Braille spinner with portable
ASCII symbols. `--no-clear` appends refreshes instead of clearing an
interactive terminal, which is useful when recording a monitoring session.

## Screen contract

Each scheduled block has a stable, one-based order and one status:

| Status | Unicode | ASCII | Meaning |
| --- | --- | --- | --- |
| `finished` | `✓` | `OK` | Module stopped with `_RC_ALL_IS_WELL_` |
| `running` | animated Braille spinner | `\|/-\\` spinner | Module started and has not stopped |
| `pending` | `·` | `.` | Scheduled by the input but not started |
| `failed` | `✗` | `X` | Nonzero return code, nonconvergence, or fatal marker |

For RASSCF, the view shows completed roots against the first `CIROOTS` value.
For CASPT2, live progress counts the per-root `Reference weight` records until
the final numbered `CASPT2 Root N Total energy` table is available. It reports
the larger of those two counts against the `MULTISTATE` or `XMULTISTATE` state
count. If a CASPT2 block omits an explicit count, the preceding RASSCF root
count is the documented fallback.

Some large-root OpenMolcas modules return `_RC_ALL_IS_WELL_` but truncate the
printed root-energy or orbital-file listings near 99-100 entries. In that
case, `moclive` marks the module finished, fills its completion bar, and shows
the partial listing as a warning such as `finished; 99/156 roots printed`.
The warning requires a later JobIph/RASSI state-count audit; it is not a
RASSCF/CASPT2 convergence failure.

For RASSI, `NROFJOBIPHS` or `Nr of JobIph files` defines the selected spin-free
state counts. Their sum is not the final SO-state dimension. `moclive` reports
human-readable stages such as constructing the spin-orbit Hamiltonian, writing
SO states, and transition/property analysis. It does not invent a percentage
when the output has no countable SO-state denominator. A finished RASSI block
uses a full completion bar, but its structured `progress.percent` remains
`null` because that bar describes module completion, not SO-state coverage.

An illustrative live view is:

```text
moclive | RUN.inp + RUN.log | RUNNING
 ✓ 01 RASSCF  UO9 ground A1              3/3  100.0% [████████████████████████]
 ✓ 02 CASPT2  UO9 ground A1 PT2          3/3  100.0% [████████████████████████]
 ⠋ 03 RASSCF  UO9 M5 excited A1          2/4   50.0% [████████████------------]
 · 04 CASPT2  UO9 M5 excited A1 PT2      0/4    0.0% [------------------------]
 · 05 RASSI   [························] pending (7 input spin-free states)
      UO9 M5 spin-orbit

MO orbital guard: pending - final pseudonatural orbitals are not available yet
```

## Snapshot API

`parse_input_plan(input_text)` returns chronological `PlanBlock` records with
`index`, `kind`, `title`, `expected_roots`, and module-specific metadata such
as RASSI `rassi_state_counts`.

`build_snapshot(inp_path, output_path)` returns a JSON-serializable dictionary
with schema `atomi.molcas_live.v1`. Its stable fields are:

- `overall_status`: `waiting`, `running`, `complete`, or `failed` for the full deck.
- `active_block_index`: one-based active/failing block index, or `null`.
- `blocks`: input-scoped block records with status, stage, and progress.
- `guards`: conservative numerical and MO-orbital checks within each block.
- `guard_summary`: counts of pass, warning, failure, pending, and review checks.
- `incremental`: tail offset, bytes read in the latest refresh, cumulative
  bytes read, incomplete buffered bytes, and reset count in watch mode; `null`
  for a one-shot full audit.

Each block progress object contains `completed`, `total`, and `percent`. The
percentage is a display aid for countable roots, not an estimate of remaining
wall time.

`render_snapshot(snapshot, ascii_only=False, spinner_frame=0, bar_width=24)`
returns plain text. Rendering has no file or scheduler side effects, so the
same frozen snapshot can be reformatted without reparsing the run.

`main()` implements both the five-minute watch loop and one-shot modes.
`--json` prints exactly one JSON document and implies `--once`. `--json-out`
atomically updates a reusable machine-readable snapshot at every refresh.

## Physical guards

`moclive` separates numerical completion from physical acceptance.

- A completed RASSCF block must have a successful module return code. Printed
  root coverage is checked separately because large-root listings may be
  truncated even when the complete JobIph state space is written.
- A final pseudonatural-orbital table is checked for printed, finite
  occupations within a tolerant zero-to-two range.
- Supersymmetry warnings, printed-root shortfalls, missing final orbital
  tables, and nonconvergence are surfaced with their source evidence. A
  printed-root shortfall after `_RC_ALL_IS_WELL_` is a warning; a non-success
  return code or explicit fatal marker is a failure.
- While RASSCF is active, MO identity is `pending`; the monitor must not call
  unfinished orbitals physically accepted.
- AO dominance is descriptive. It is not a Mulliken, Loewdin, or other
  quantitative population analysis, and chemistry-specific orbital identity
  still requires MOLCAS-lead review.
- CASPT2 reference-weight and RASSI state-space warnings are shown when those
  quantities are printed. Their absence from a partial log is `pending`, not
  `pass`.

`moclive` is read-only. It does not submit, cancel, restart, copy, truncate, or
modify an OpenMolcas job. Scheduler state can be shown by a separate wrapper,
but the scientific block scope always comes from the supplied input deck.

## Parsing and failure rules

Both `--- Module ... spent` and `--- Stop Module: ... /rc=...` completion forms
should be accepted. `_RC_ALL_IS_WELL_` is success. `_RC_NOT_CONVERGED_`, other
non-success return codes, and explicit fatal/error markers fail the matching
active block. `Happy landing` confirms full-run completion only when all
input-scoped blocks have finished successfully.

Extra output modules not represented by the supplied input are not silently
folded into the plan. They should be reported as unmatched output so a user can
detect a mismatched input/log pair.
