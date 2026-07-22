import csv

from atomi.xafs import spectrum_overlay


def test_white_line_alignment_shifts_simulated_spectrum_to_experiment(tmp_path):
    exp_path = tmp_path / "exp.csv"
    sim_path = tmp_path / "fdmnes.csv"
    exp_path.write_text(
        "energy_rel_eV,norm\n"
        "0,0.2\n"
        "10,1.5\n"
        "20,1.0\n",
        encoding="utf-8",
    )
    sim_path.write_text(
        "energy_rel_eV,intensity\n"
        "0,0.1\n"
        "15,2.0\n"
        "30,1.2\n",
        encoding="utf-8",
    )

    exp = spectrum_overlay.read_spectrum(
        exp_path,
        label="Exp",
        kind="experiment",
        energy_column="energy_rel_eV",
        intensity_column="norm",
    )
    sim = spectrum_overlay.read_spectrum(
        sim_path,
        label="FDMNES",
        kind="fdmnes",
        energy_column="energy_rel_eV",
        intensity_column="intensity",
    )
    aligned = spectrum_overlay.align_spectra([exp, sim], white_line_window=(0, 25))

    assert aligned[0].white_line_energy == 10.0
    assert aligned[0].energy_shift == 0.0
    assert aligned[1].white_line_energy == 15.0
    assert aligned[1].energy_shift == -5.0
    assert aligned[1].energy_aligned == (-5.0, 10.0, 25.0)


def test_overlay_cli_writes_shifted_csv_and_svg_with_exp_style(tmp_path, capsys):
    exp_path = tmp_path / "exp.csv"
    sim_path = tmp_path / "molcas.csv"
    out_csv = tmp_path / "overlay.csv"
    out_svg = tmp_path / "overlay.svg"
    exp_path.write_text("energy_rel_eV,norm\n0,0.1\n8,1.4\n20,0.9\n", encoding="utf-8")
    sim_path.write_text("energy_rel_eV,intensity\n0,0.2\n12,1.1\n24,0.7\n", encoding="utf-8")

    args = spectrum_overlay.build_parser().parse_args(
        [
            "--exp",
            f"Exp:{exp_path}",
            "--sim",
            f"Molcas:molcas:{sim_path}",
            "--exp-intensity-column",
            "norm",
            "--white-line-window",
            "0 20",
            "--out-csv",
            str(out_csv),
            "--out-svg",
            str(out_svg),
            "--exp-style",
            "hollow-points",
        ]
    )
    summary = spectrum_overlay.overlay_main(args)

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    svg = out_svg.read_text(encoding="utf-8")
    captured = capsys.readouterr()

    assert summary["schema"] == "atomi.xafs.xanes_overlay.v1"
    assert summary["spectra"][1]["energy_shift"] == -4.0
    assert any(row["label"] == "Molcas" and row["energy_raw"] == "12" and row["energy_aligned"] == "8" for row in rows)
    assert "Molcas (molcas, shift -4 eV)" in svg
    assert "<circle" in svg
    assert "atomi.xafs.xanes_overlay.v1" in captured.out


def test_xanes_mode_defaults_to_aligned_energy_window(tmp_path):
    exp_path = tmp_path / "exp.csv"
    sim_path = tmp_path / "fdmnes.csv"
    out_csv = tmp_path / "overlay.csv"
    exp_path.write_text("energy_rel_eV,norm\n-250,0.1\n0,1.0\n250,0.4\n350,0.2\n", encoding="utf-8")
    sim_path.write_text("energy_rel_eV,intensity\n-240,0.1\n10,1.0\n260,0.4\n360,0.2\n", encoding="utf-8")

    args = spectrum_overlay.build_parser().parse_args(
        [
            "--exp",
            f"Exp:{exp_path}",
            "--sim",
            f"FDMNES:fdmnes:{sim_path}",
            "--exp-intensity-column",
            "norm",
            "--white-line-window",
            "-20 30",
            "--out-csv",
            str(out_csv),
        ]
    )
    summary = spectrum_overlay.overlay_main(args)

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert summary["mode"] == "xanes"
    assert summary["energy_window_aligned_eV"] == [-200.0, 300.0]
    assert {row["energy_raw"] for row in rows if row["label"] == "Exp"} == {"0", "250"}
    assert {row["energy_aligned"] for row in rows if row["label"] == "FDMNES"} == {"0", "250"}


def test_exafs_mode_does_not_clip_energy_without_explicit_window(tmp_path):
    exp_path = tmp_path / "exp.csv"
    sim_path = tmp_path / "ocean.csv"
    out_csv = tmp_path / "overlay.csv"
    exp_path.write_text("energy_rel_eV,norm\n-250,0.1\n0,1.0\n350,0.2\n", encoding="utf-8")
    sim_path.write_text("energy_rel_eV,intensity\n-240,0.1\n10,1.0\n360,0.2\n", encoding="utf-8")

    args = spectrum_overlay.build_parser().parse_args(
        [
            "--mode",
            "exafs",
            "--exp",
            f"Exp:{exp_path}",
            "--sim",
            f"OCEAN:ocean:{sim_path}",
            "--exp-intensity-column",
            "norm",
            "--white-line-window",
            "-20 30",
            "--out-csv",
            str(out_csv),
        ]
    )
    summary = spectrum_overlay.overlay_main(args)

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert summary["mode"] == "exafs"
    assert summary["energy_window_aligned_eV"] is None
    assert len([row for row in rows if row["label"] == "Exp"]) == 3
    assert len([row for row in rows if row["label"] == "OCEAN"]) == 3


def test_overlay_cli_aligns_transition_sticks_to_parent_spectrum(tmp_path):
    exp_path = tmp_path / "exp.csv"
    sim_path = tmp_path / "molcas.csv"
    sticks_path = tmp_path / "molcas_sticks.csv"
    out_csv = tmp_path / "overlay.csv"
    out_svg = tmp_path / "overlay.svg"
    out_sticks = tmp_path / "overlay_sticks.csv"
    exp_path.write_text("energy_rel_eV,norm\n0,0.1\n8,1.4\n20,0.9\n", encoding="utf-8")
    sim_path.write_text("energy_rel_eV,intensity\n0,0.2\n12,1.1\n24,0.7\n", encoding="utf-8")
    sticks_path.write_text(
        "energy_rel_eV,oscillator_strength,state_label,assignment\n"
        "12,1.0,SO1->SO8,Ce 2p -> Ce 5d\n"
        "18,0.3,SO1->SO9,shoulder\n",
        encoding="utf-8",
    )

    args = spectrum_overlay.build_parser().parse_args(
        [
            "--exp",
            f"Exp:{exp_path}",
            "--sim",
            f"Molcas:molcas:{sim_path}",
            "--sticks",
            f"Molcas sticks:Molcas:{sticks_path}",
            "--exp-intensity-column",
            "norm",
            "--white-line-window",
            "0 20",
            "--stick-relative-threshold",
            "0",
            "--stick-label-relative-threshold",
            "0.8",
            "--out-csv",
            str(out_csv),
            "--out-svg",
            str(out_svg),
            "--out-sticks-csv",
            str(out_sticks),
        ]
    )
    summary = spectrum_overlay.overlay_main(args)

    rows = list(csv.DictReader(out_sticks.open(encoding="utf-8")))
    svg = out_svg.read_text(encoding="utf-8")

    assert summary["sticks"][0]["energy_shift"] == -4.0
    assert rows[0]["energy_raw"] == "12"
    assert rows[0]["energy_aligned"] == "8"
    assert rows[0]["state_label"] == "SO1->SO8"
    assert "Transition / feature sticks" in svg
    assert "MOLCAS stick key" in svg
    assert "<tspan font-weight=\"700\">1.</tspan>" in svg
    assert "<tspan font-weight=\"700\">2.</tspan>" in svg
    assert "SO1-&gt;SO8" in svg
    assert "shoulder" in svg


def test_overlay_svg_uses_xanes_axis_and_split_route_stick_keys(tmp_path):
    exp_path = tmp_path / "exp.csv"
    fdmnes_path = tmp_path / "fdmnes.csv"
    molcas_path = tmp_path / "molcas.csv"
    fdmnes_sticks_path = tmp_path / "fdmnes_sticks.csv"
    molcas_sticks_path = tmp_path / "molcas_sticks.csv"
    out_csv = tmp_path / "overlay.csv"
    out_svg = tmp_path / "overlay.svg"
    exp_path.write_text("energy_rel_eV,norm\n0,0.1\n8,1.4\n20,0.9\n", encoding="utf-8")
    fdmnes_path.write_text("energy_rel_eV,intensity\n0,0.2\n10,1.1\n24,0.7\n", encoding="utf-8")
    molcas_path.write_text("energy_rel_eV,intensity\n0,0.2\n12,1.1\n24,0.7\n", encoding="utf-8")
    fdmnes_sticks_path.write_text(
        "energy_rel_eV,relative_intensity,state_label,assignment\n"
        "10,1.0,FDMNES feature,Ce L3 continuum/5d-O2p feature\n",
        encoding="utf-8",
    )
    molcas_sticks_path.write_text(
        "energy_rel_eV,oscillator_strength,state_label,assignment\n"
        "12,1.0,SO1->SO8,Ce 2p -> Ce 5d\n",
        encoding="utf-8",
    )

    args = spectrum_overlay.build_parser().parse_args(
        [
            "--exp",
            f"Exp:{exp_path}",
            "--sim",
            f"FDMNES:fdmnes:{fdmnes_path}",
            "--sim",
            f"Molcas:molcas:{molcas_path}",
            "--sticks",
            f"FDMNES features:FDMNES:{fdmnes_sticks_path}",
            "--sticks",
            f"Molcas sticks:Molcas:{molcas_sticks_path}",
            "--exp-intensity-column",
            "norm",
            "--white-line-window",
            "0 20",
            "--stick-relative-threshold",
            "0",
            "--out-csv",
            str(out_csv),
            "--out-svg",
            str(out_svg),
        ]
    )
    spectrum_overlay.overlay_main(args)

    svg = out_svg.read_text(encoding="utf-8")
    assert "Normalized absorption (a.u.)" in svg
    assert "rotate(-90" in svg
    assert "FDMNES stick key" in svg
    assert "MOLCAS stick key" in svg
    assert "opacity=\"0.5\"" in svg
    assert "Ce L3 continuum/5d-O2p feature" in svg
    assert "Ce 2p -&gt; Ce 5d" in svg
