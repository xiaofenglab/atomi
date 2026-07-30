# Plot-Data Bundles

Atomi plot-data bundles separate scientific analysis from figure rendering.
Domain code parses raw computation output, applies physical guards and declared
scientific transformations, then freezes the resulting tables once. Plotting
tools consume that immutable revision without reparsing HPC output.

## Ownership

- The project student or domain lead owns parsing, units, row selection,
  uncertainty, normalization, fitting, broadening, and scientific status.
- Sarah owns portfolio-level acceptance and promotion.
- Dana owns style, layout, rendering, and figure QA.
- Anna owns the reusable bundle contract and domain adapters in Atomi.

Raw HPC outputs remain read-only and are referenced by fingerprints. The bundle
copies only curated rectangular CSV or TSV tables needed for plotting.

## Commands

Write a starter specification:

```bash
atomi plot-data template --write plot_data_spec.json
```

Freeze a new immutable revision:

```bash
atomi plot-data freeze \
  --spec plot_data_spec.json \
  --outdir plot_data_bundles/result_r1
```

Validate and inspect:

```bash
atomi plot-data validate \
  plot_data_bundles/result_r1/plot_dataset.json
atomi plot-data describe \
  plot_data_bundles/result_r1/plot_dataset.json
atomi plot-data resolve \
  plot_data_bundles/result_r1/plot_dataset.json --view main
```

The output directory must not already exist. Scientific changes require a new
revision and directory. Format, typography, color, panel arrangement, legend
position, and other explicitly permitted render changes reuse the same bundle.

## Contract

Schema `atomi.analysis.plot_dataset.v1` records:

- dataset ID, revision, project, question, owner, report, status, and quality;
- source artifacts and SHA-256 fingerprints;
- copied tables, row counts, SHA-256 fingerprints, row scope, and column
  meanings including units and normalization basis;
- approved views with table and column mappings;
- transformations, uncertainty meaning, caveats, and render permissions.

Validation rejects modified tables, non-finite numeric values, missing declared
columns, unsafe identifiers, and table paths escaping the bundle.

## Dana

Dana can render an approved single-table view directly:

```bash
python scripts/dana_plot.py \
  --dataset /project/plot_data_bundles/result_r1/plot_dataset.json \
  --view main \
  --output figure.pdf --output figure.svg --output figure.png
```

Custom multi-panel renderers should call
`atomi.analysis.plot_dataset.load_manifest` or `resolve_view`, then read only
the frozen tables. Each figure provenance sidecar should record the manifest
path, manifest fingerprint, dataset ID/revision, and approved view.
