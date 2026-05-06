# CP2K Visualization Staging

These are corrected staging copies of the CP2K visualization scripts supplied from Downloads.

Usage:

```bash
./plotcp2k cp2k.log
./plotcp2k cp2k.log trajectory.xyz
./plotcp2kall cp2k_geoopt.log
python3 cp2k_md_bondtrack.py cp2k_md.log trajectory.xyz cp2k_md_bonds.dat
python3 cp2k_md_eta.py cp2k_md.log cp2k.inp cp2k_md_eta.hist
```

Corrections made before packaging:

- Wrappers now locate gnuplot/helper scripts next to themselves instead of using `$HOME/scripts`.
- `plotcp2k` passes helper script paths into gnuplot explicitly.
- `plot_cp2k_md_live.gp` no longer requires `xyzfile`; it skips bond panels if no trajectory is available.
- `plotcp2kall` checks that its gnuplot script exists before launching.
- The `find -printf` dependency in `plotcp2k` was replaced with a small Python path picker, which is friendlier on macOS and HPC systems without GNU find.

Not yet packaged into `src/atomi`; this folder is the cleaned baseline for implementation.
