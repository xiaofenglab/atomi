import json

from atomi.cli.registry import command_registry
from atomi.xafs import molcas_orbital_diagram


def test_molcas_orbital_diagram_filters_edge_and_writes_placeholder_svg(tmp_path, capsys):
    levels = tmp_path / "levels.csv"
    arrows = tmp_path / "arrows.csv"
    out_svg = tmp_path / "diagram.svg"
    summary = tmp_path / "summary.json"
    levels.write_text(
        "label,block,energy_ev,occupation,character,color\n"
        "Ce 2p3/2 L3 core,core,0,2,Ce 2p3/2,#355c9a\n"
        "Ce 2p1/2 L2 core,core,420,2,Ce 2p1/2,#8064a2\n"
        "L3 SO34-36,L3,8,0,mixed Ce 5d/6s/4f_L,#b1462f\n"
        "L2 SO263-265,L2,428,0,mixed Ce 5d/6s/4f_L,#6f4aa8\n",
        encoding="utf-8",
    )
    arrows.write_text(
        "label,state_from,state_to,energy_ev,oscillator_strength,source_label,target_label\n"
        "L3 SO34-36,1,34-36,5764.9,0.0032,Ce 2p3/2 L3 core,L3 SO34-36\n"
        "L2 SO263-265,1,263-265,6185.6,0.0021,Ce 2p1/2 L2 core,L2 SO263-265\n",
        encoding="utf-8",
    )

    args = molcas_orbital_diagram.build_parser().parse_args(
        [
            "--levels-csv",
            str(levels),
            "--arrows-csv",
            str(arrows),
            "--edge",
            "L3",
            "--out-svg",
            str(out_svg),
            "--summary",
            str(summary),
        ]
    )
    result = molcas_orbital_diagram.run(args)
    svg = out_svg.read_text(encoding="utf-8")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    captured = capsys.readouterr()

    assert result["schema"] == "atomi.xafs.molcas_orbital_diagram.v1"
    assert payload["n_levels"] == 2
    assert payload["n_arrows"] == 1
    assert "Ce 2p3/2 L3 core" in svg
    assert "L3 SO34-36" in svg
    assert "L2 SO263-265" not in svg
    assert "image slot" in svg
    assert "atomi.xafs.molcas_orbital_diagram.v1" in captured.out


def test_molcas_orbital_diagram_registry_alias_is_available():
    registry = command_registry()

    assert registry["molcas-mo-diagram"].target == "atomi.xafs.molcas_orbital_diagram:main"
