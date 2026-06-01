import json

from atomi.calphad.mivm import main as mivm_main
from atomi.cli.main import main as atomi_main


def test_mivm_default_guide_mentions_ceramic_parameters(capsys):
    mivm_main([])

    out = capsys.readouterr().out
    assert "(Gd,U)O2" in out
    assert "Gd-VO" in out
    assert "MIVM excess Gibbs energy" in out


def test_mivm_ceramic_json_contains_charge_compensation(capsys):
    mivm_main(["guide", "--system", "ceramic", "--format", "json"])

    data = json.loads(capsys.readouterr().out)
    ceramic_text = " ".join(data["ceramic_solid"])
    assert "U5+O2" in ceramic_text
    assert "oxygen vacancies" in ceramic_text


def test_atomi_cli_forwards_calphad_mivm(capsys):
    atomi_main(["calphad-mivm", "guide", "--system", "ceramic"])

    out = capsys.readouterr().out
    assert "Solid/ceramic" in out
    assert "(Gd,U)O2" in out
