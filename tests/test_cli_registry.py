from __future__ import annotations

import sys
import types

from atomi.cli.registry import CommandSpec, command_registry, dispatch_registered_command, registered_aliases, specs_by_category


def test_command_registry_exposes_core_bridge_aliases() -> None:
    aliases = registered_aliases()
    registry = command_registry()

    assert "zentropy-mode4-surface" in aliases
    assert "gnn-active-learning" in aliases
    assert "crystal-graph-dataset" in aliases
    assert "thermo-prior" in aliases
    assert "thermo-prior-mp" in aliases
    assert "aq-thermo-bridge" in aliases
    assert registry["mode4-surface"].target == "atomi.zentropy.mode4_surface:main"
    assert registry["crystal-graph-dataset"].target == "atomi.ml.crystal_graph_dataset:main"
    assert "zentropy" in specs_by_category()


def test_command_spec_invokes_target_with_prepended_args(monkeypatch) -> None:
    calls: list[list[str]] = []
    module = types.ModuleType("atomi_fake_cli_target")

    def fake_main(argv: list[str]) -> None:
        calls.append(argv)

    module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atomi_fake_cli_target", module)
    spec = CommandSpec(
        aliases=("fake",),
        target="atomi_fake_cli_target:main",
        category="test",
        help="fake command",
        prepend_args=("subcommand",),
    )

    spec.invoke(["--flag"])

    assert calls == [["subcommand", "--flag"]]


def test_dispatch_registered_command_returns_false_for_unknown() -> None:
    assert dispatch_registered_command(["not-a-real-atomi-command"]) is False
