from __future__ import annotations

import json
from pathlib import Path

from atomi.thermo_prior.materials_project import (
    SCHEMA,
    main,
    normalize_mp_doc,
    resolve_mp_api_key,
    summarize_mp_cache,
    write_mp_cache,
)


def test_normalize_mp_doc_handles_summary_payload() -> None:
    row = normalize_mp_doc(
        {
            "material_id": "mp-22862",
            "formula_pretty": "NaCl",
            "formula_anonymous": "AB",
            "chemsys": "Cl-Na",
            "energy_per_atom": -3.4,
            "formation_energy_per_atom": -2.0,
            "energy_above_hull": 0.0,
            "volume": 45.0,
            "nsites": 2,
            "is_stable": True,
        }
    )

    assert row["schema"] == SCHEMA
    assert row["material_id"] == "mp-22862"
    assert row["formation_energy_per_atom_eV"] == -2.0
    assert row["volume_per_atom_A3"] == 22.5


def test_write_and_summarize_mp_cache(tmp_path: Path) -> None:
    output = tmp_path / "mp_cache.json"
    records = [
        normalize_mp_doc(
            {
                "material_id": "mp-a",
                "formula_pretty": "NaCl",
                "chemsys": "Cl-Na",
                "energy_above_hull": 0.0,
                "is_stable": True,
            }
        ),
        normalize_mp_doc(
            {
                "material_id": "mp-b",
                "formula_pretty": "Na2Cl2",
                "chemsys": "Cl-Na",
                "energy_above_hull": 0.05,
                "is_stable": False,
            }
        ),
    ]

    summary = write_mp_cache(output, records, query={"formula": "NaCl"})
    reread = summarize_mp_cache(output)

    assert summary.n_records == 2
    assert reread.n_stable == 1
    assert reread.formulas == ["Na2Cl2", "NaCl"]
    assert reread.min_energy_above_hull_eV == 0.0


def test_materials_project_cli_from_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "material_id": "mp-1",
                        "formula_pretty": "UCl3",
                        "chemsys": "Cl-U",
                        "energy_above_hull": 0.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "cache.json"

    result = main(["from-json", "--input", str(raw), "--out", str(output)])

    assert result["n_records"] == 1
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["records"][0]["formula_pretty"] == "UCl3"


def test_resolve_mp_api_key_uses_explicit_private_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MP_API_KEY", raising=False)
    key_json = tmp_path / "atomi_hpc_config.kit.local.json"
    key_json.write_text(
        json.dumps({"materials_project": {"api_key": "secret-test-key"}}),
        encoding="utf-8",
    )

    key, source = resolve_mp_api_key(api_key_json=key_json)

    assert key == "secret-test-key"
    assert source == f"json:{key_json}"
