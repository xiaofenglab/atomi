import csv
import json
from pathlib import Path

from atomi.aqueous.thermohub_bridge import main


def test_aq_thermo_bridge_writes_request_mode_outputs(tmp_path):
    aimd = tmp_path / "aimd.csv"
    with aimd.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["temperature_c", "log10_K", "sigma", "source"])
        writer.writeheader()
        writer.writerow({"temperature_c": 25, "log10_K": 1.56, "sigma": 0.12, "source": "test"})

    out = tmp_path / "bridge"
    main(["--aimd-k", str(aimd), "--out", str(out), "--temperatures-c", "25,50"])

    status = json.loads((out / "aqueous_bridge_status.json").read_text(encoding="utf-8"))
    assert status["aimd_rows"] == 1
    assert status["reactions"] >= 4
    assert status["thermofun_rows"] == 0
    assert (out / "thermohub_gems_species_request.md").exists()
    assert "GaCl4-" in (out / "reaction_request.csv").read_text(encoding="utf-8")
    assert "AIMD Conditional Constants" in (out / "aqueous_bridge_report.md").read_text(encoding="utf-8")
