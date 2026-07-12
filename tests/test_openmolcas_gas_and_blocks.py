from __future__ import annotations

from atomi.qchem.openmolcas_bridge import GasSpace, OpenMolcasPrepareOptions, RasscfStateBlock, render_openmolcas_input


def test_generic_gas_selector_is_terminal_and_group_agnostic() -> None:
    options = OpenMolcasPrepareOptions(
        title="generic_gas",
        xyz_name="cluster.xyz",
        charge=0,
        spin=3,
        symmetry=2,
        nactel="7 0 0",
        inactive="10 10 10",
        gas_spaces=(
            GasSpace("core", "1 1 1", 5, 5),
            GasSpace("acceptor", "2 0 1", 6, 7),
        ),
        gas_selector_dimension=4,
        gas_selector_root=3,
    )
    text = render_openmolcas_input(options)
    assert "GASSCF\n 2\n 1 1 1\n 5 5\n 2 0 1\n 6 7" in text
    assert "CIROOTS\n 1 4\n 3" in text
    assert "&CASPT2" not in text
    assert "&RASSI" not in text
    assert ">>> COPY" not in text


def test_multiblock_ras_preserves_jobiph_jobmix_and_rassi_order() -> None:
    options = OpenMolcasPrepareOptions(
        title="ras_production",
        xyz_name="cluster.xyz",
        charge=0,
        group="X Y Z",
        use_caspt2=True,
        include_orbital_prep=False,
        state_blocks=(
            RasscfStateBlock(
                title="ground",
                symmetry=1,
                spin=1,
                nactel="6 1 1",
                inactive="14 8 8 7 8 7 7 5",
                ras1="0 1 1 0 1 0 0 0",
                ras3="3 0 0 1 0 1 1 0",
                ciroots="1 1 1",
                cionly=True,
            ),
            RasscfStateBlock(
                title="core_b3u",
                symmetry=2,
                spin=1,
                nactel="6 1 1",
                inactive="14 8 8 7 8 7 7 5",
                ras1="0 1 1 0 1 0 0 0",
                ras3="3 0 0 1 0 1 1 0",
                ciroots="4 4 1",
                cionly=True,
            ),
        ),
    )
    text = render_openmolcas_input(options)
    assert text.index("Title\nground") < text.index("$Project.JobIph_1")
    assert text.index("$Project.JobIph_1") < text.index("$Project.JobMix_1")
    assert text.index("$Project.JobMix_1") < text.index("Title\ncore_b3u")
    assert text.index("$Project.JobIph_2") < text.index("$Project.JobMix_2")
    assert text.index(">>> COPY $Project.JobIph_1 JOB001") < text.index(">>> COPY $Project.JobIph_2 JOB002")
    assert text.index(">>> COPY $Project.JobMix_1 JOB001") < text.index(">>> COPY $Project.JobMix_2 JOB002")
    assert text.count("&RASSI") == 2
