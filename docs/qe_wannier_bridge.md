# QE and Wannier90 external-runtime bridge

Atomi treats Quantum ESPRESSO and Wannier90 as a compiled sidecar runtime. Do
not install them into `m_lammps_env`.

```bash
qe-wannier-bridge status --json
qe-wannier-bridge install-plan --json
qe-wannier-bridge write-install \
  --outdir qe_wannier_install \
  --root '$HOME/atomi_hpc/qe-wannier' \
  --module-load <compiler-module> \
  --module-load <mpi-module>
```

Submit the generated install script through Slurm. It builds QE 7.5 and
Wannier90 3.1.0 with GCC/OpenMPI and verifies `pw.x`, `hp.x`,
`pw2wannier90.x`, `wannier90.x`, and `wannier2pw.x`.

The bridge reports four distinct readiness levels:

1. QE ground-state capability (`pw.x`).
2. Stock atomic/ortho-atomic response (`hp.x`).
3. MLWF Hubbard-projector construction (`pw2wannier90.x`, `wannier90.x`, and
   `wannier2pw.x`).
4. Piotr matched-response readiness, which additionally requires a recorded
   collaborator root, immutable commit, and response executable.

A legacy OCEAN/QE 7.0 runtime may be useful for OCEAN DFT/BSE work, but it is
not a complete QE/Wannier Hubbard runtime when it lacks `hp.x`, `wannier90.x`,
or `wannier2pw.x`; QE 7.0 also predates the modern Hubbard-card interface.

For the UO2 route, validate U/O UPFs, the AFM and occupation branch, Wannier
windows, interpolated bands, spreads, centers, and U-5f/O-2p character before
calculating or applying U. Never call a stock `hp.x` value a matched MLWF U.
