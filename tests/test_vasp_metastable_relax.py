from pathlib import Path

from atomi.vasp.metastable_relax import (
    advance_main,
    fingerprint_main,
    main,
    selective_main,
    status_main,
    validate_ldau_species_order,
)


def write_vasp_root(root: Path) -> None:
    root.mkdir()
    (root / "POSCAR").write_text(
        "\n".join(
            [
                "metastable",
                "1.0",
                "5.0 0.0 0.0",
                "0.0 5.0 0.0",
                "0.0 0.0 5.0",
                "Gd U O",
                "1 1 2",
                "Direct",
                "0.0 0.0 0.0",
                "0.5 0.5 0.5",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "INCAR").write_text(
        "\n".join(
            [
                "ENCUT = 650",
                "ISPIN = 2",
                "MAGMOM = 7 -7 2*0",
                "LDAU = .TRUE.",
                "LDAUL = 3 3 -1",
                "LDAUU = 6.0 4.0 0.0",
                "IBRION = 2",
                "ISIF = 3",
                "EDIFFG = -0.03",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "POTCAR").write_text(
        "\n".join(
            [
                "TITEL  = PAW_PBE Gd 23Dec2003",
                "TITEL  = PAW_PBE U 06Sep2000",
                "TITEL  = PAW_PBE O 08Apr2002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "KPOINTS").write_text("Gamma\n1\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")


def test_metastable_prepare_preserves_physics_and_uses_conservative_freeze_sequence(tmp_path: Path) -> None:
    root = tmp_path / "seed"
    out = tmp_path / "staged"
    write_vasp_root(root)

    main([str(root), "--output", str(out)])

    for stage in ("00_static_scf", "01_gentle_relax", "02_continue_relax", "03_final_static"):
        assert (out / stage / "POSCAR").is_file()
        assert (out / stage / "INCAR").is_file()
        assert (out / stage / "POTCAR").is_file()
    static_incar = (out / "00_static_scf" / "INCAR").read_text(encoding="utf-8")
    relax_incar = (out / "01_gentle_relax" / "INCAR").read_text(encoding="utf-8")
    continue_relax_incar = (out / "02_continue_relax" / "INCAR").read_text(encoding="utf-8")
    final_static_incar = (out / "03_final_static" / "INCAR").read_text(encoding="utf-8")
    cation_relax_poscar = (out / "01_gentle_relax" / "POSCAR").read_text(encoding="utf-8")
    oxygen_relax_poscar = (out / "02_continue_relax" / "POSCAR").read_text(encoding="utf-8")

    assert "ENCUT = 650" in static_incar
    assert "MAGMOM = 7 -7 2*0" in static_incar
    assert "IBRION = -1" in static_incar
    assert "ISIF = 2" in static_incar
    assert "NELM = 3000" in static_incar
    assert "LCHARG = .TRUE." in static_incar
    assert "LWAVE = .FALSE." in static_incar
    assert "ISTART = 0" in relax_incar
    assert "ICHARG = 1" in relax_incar
    assert "IBRION = 2" in relax_incar
    assert "POTIM = 0.05" in relax_incar
    assert "NELM = 3000" in relax_incar
    assert "EDIFF = 1E-6" in continue_relax_incar
    assert "EDIFFG = -0.01" in continue_relax_incar
    assert "NSW = 600" in continue_relax_incar
    assert "NELM = 3000" in continue_relax_incar
    assert "NSW = 0" in final_static_incar
    assert "IBRION = -1" in final_static_incar
    assert "LREAL = .FALSE." in final_static_incar
    assert "NELM = 3000" in final_static_incar
    assert "EDIFFG" not in final_static_incar
    assert "Selective dynamics" in cation_relax_poscar
    assert "0.0  0.0  0.0   T T T" in cation_relax_poscar
    assert "0.25  0.25  0.25   F F F" in cation_relax_poscar
    assert "0.0  0.0  0.0   F F F" in oxygen_relax_poscar
    assert "0.25  0.25  0.25   T T T" in oxygen_relax_poscar


def test_selective_dynamics_by_species(tmp_path: Path) -> None:
    root = tmp_path / "seed"
    write_vasp_root(root)
    output = tmp_path / "POSCAR.sd"

    selective_main(
        [
            str(root / "POSCAR"),
            "--output",
            str(output),
            "--freeze-mode",
            "by_species",
            "--freeze-species",
            "U,O",
        ]
    )

    text = output.read_text(encoding="utf-8")
    assert "Selective dynamics" in text
    assert "0.0  0.0  0.0   T T T" in text
    assert "0.5  0.5  0.5   F F F" in text
    assert "0.75  0.75  0.75   F F F" in text


def test_fingerprint_and_status_print_tables(tmp_path: Path, capsys) -> None:
    root = tmp_path / "seed"
    out = tmp_path / "staged"
    write_vasp_root(root)
    main([str(root), "--output", str(out)])

    fingerprint_main([str(root / "POSCAR"), "--reference", str(root / "POSCAR")])
    status_main([str(out), "--reference", str(root / "POSCAR")])

    text = capsys.readouterr().out
    assert "VASP Structure Fingerprint" in text
    assert "Nearest Pair Distances" in text
    assert "VASP Metastable Relaxation Status" in text
    assert "00_static_scf" in text


def test_ldau_species_order_warning_catches_swapped_u_o(tmp_path: Path) -> None:
    root = tmp_path / "seed"
    write_vasp_root(root)
    (root / "INCAR").write_text(
        "\n".join(
            [
                "LDAU = .TRUE.",
                "LDAUL = 3 -1 3",
                "LDAUU = 6.0 4.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    warnings = validate_ldau_species_order(root / "POSCAR", root / "INCAR")

    assert "U usually expects 3, found -1" in "\n".join(warnings)
    assert "O usually expects -1, found 3" in "\n".join(warnings)


def test_advance_copies_chgcar_and_contcar_to_next_stage(tmp_path: Path, capsys) -> None:
    root = tmp_path / "seed"
    out = tmp_path / "staged"
    write_vasp_root(root)
    main([str(root), "--output", str(out)])
    source = out / "00_static_scf"
    target = out / "01_gentle_relax"
    (source / "CHGCAR").write_text("charge\n", encoding="utf-8")
    (source / "CONTCAR").write_text(
        "\n".join(
            [
                "metastable after static",
                "1.0",
                "5.0 0.0 0.0",
                "0.0 5.0 0.0",
                "0.0 0.0 5.0",
                "Gd U O",
                "1 1 2",
                "Direct",
                "0.1 0.0 0.0",
                "0.5 0.6 0.5",
                "0.25 0.25 0.25",
                "0.75 0.75 0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "OUTCAR").write_text(
        " free  energy   TOTEN  =       -10.000000 eV\n"
        " magnetization (x)\n"
        " # of ion       s       p       d       f       tot\n"
        "    1 0.000 0.000 0.000 0.000 7.000000\n"
        " tot\n",
        encoding="utf-8",
    )

    advance_main([str(out), "--from-stage", "00_static_scf", "--reference", str(root / "POSCAR")])

    assert (target / "CHGCAR").read_text(encoding="utf-8") == "charge\n"
    target_poscar = (target / "POSCAR").read_text(encoding="utf-8")
    assert "Selective dynamics" in target_poscar
    assert "0.1  0.0  0.0   T T T" in target_poscar
    assert "0.25  0.25  0.25   F F F" in target_poscar
    text = capsys.readouterr().out
    assert "vasp-spin-report" in text
    assert "vasp-structure-fingerprint" in text
