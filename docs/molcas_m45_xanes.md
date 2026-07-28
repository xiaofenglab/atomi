# OpenMolcas U M4,5 XANES plotting

Use `molcas-postanalysis extract-m45-transitions` to average the selected
initial spin-orbit states and export every positive transition. For a mainline
report, pass explicit M5 and M4 windows so inter-edge satellites are retained
in the all-transition CSV without being mislabeled as M4.

Generate two complementary figures from the same transition CSVs:

```bash
molcas-postanalysis m45-two-panel \
  --m5-transitions-csv run_m5_transitions_for_atomi.csv \
  --m4-transitions-csv run_m4_transitions_for_atomi.csv \
  --profile-preset u-m45-polly-high-resolution \
  --stick-relative-threshold 0 \
  --stick-top 40 \
  --outdir m45_high_resolution

molcas-postanalysis m45-two-panel \
  --m5-transitions-csv run_m5_transitions_for_atomi.csv \
  --m4-transitions-csv run_m4_transitions_for_atomi.csv \
  --profile-preset u-m45-physical-xas \
  --stick-relative-threshold 0 \
  --stick-top 40 \
  --outdir m45_physical_xas
```

The high-resolution preset reproduces the narrow UO8/C2h Polly comparison
profile (`Gaussian FWHM 1.2894947 eV`, `Lorentzian FWHM 0.3621724 eV`). It is
a theoretical multiplet-comparison profile, not an ordinary XAS lifetime
model.

The physical-XAS preset uses edge-specific XrayDB core-hole widths and a
recorded `1.0 eV` Gaussian display resolution. The XrayDB widths are queried
separately for M5 and M4.

In both figures, every positive transition contributes to the envelope.
`--stick-top` limits only the visible sticks and makes detailed plots
reproducible without changing the spectrum. Record the root table, initial
spin-orbit states, energy alignment, profile preset, measurement channel,
width provenance, and visible-stick policy with every published figure.
