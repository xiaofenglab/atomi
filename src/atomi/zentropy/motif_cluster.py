"""Cluster and select representative defect motifs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from atomi.zentropy.stage_utils import (
    composition_label,
    energy_per_fu,
    load_motif_records,
    motif_family,
    motif_id,
    volume_per_fu,
    write_csv,
    write_json,
)


SCHEMA = "atomi.zentropy.motif_cluster.v1"

FIELDS = [
    "cluster_id",
    "representative",
    "motif_id",
    "motif_family",
    "composition",
    "spin_order_host",
    "energy_eV_per_fu",
    "delta_E_eV_per_fu",
    "volume_A3_per_fu",
    "run_dir",
    "n_cluster_members",
]


def _spin_label(row: dict[str, Any]) -> str:
    meta = row.get("motif_metadata") if isinstance(row.get("motif_metadata"), dict) else {}
    return str(row.get("spin_order_host") or meta.get("spin_order_host") or row.get("spin_order_all") or meta.get("spin_order_all") or "")


def _base_key(row: dict[str, Any], columns: list[str]) -> str:
    values: list[str] = []
    for column in columns:
        if column == "motif_family":
            values.append(motif_family(row))
        elif column == "composition":
            values.append(composition_label(row))
        elif column == "spin_order_host":
            values.append(_spin_label(row))
        else:
            values.append(str(row.get(column) or ""))
    return "|".join(values) or "all"


def cluster_motifs(
    records: list[dict[str, Any]],
    *,
    group_by: list[str],
    energy_window_eV_per_fu: float,
    volume_window_A3_per_fu: float | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[_base_key(row, group_by)].append(row)

    output: list[dict[str, Any]] = []
    cluster_counter = 0
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (energy_per_fu(row) if energy_per_fu(row) is not None else float("inf"), motif_id(row)))
        clusters: list[list[dict[str, Any]]] = []
        for row in rows:
            e_value = energy_per_fu(row)
            v_value = volume_per_fu(row)
            placed = False
            for cluster in clusters:
                ref = cluster[0]
                ref_e = energy_per_fu(ref)
                ref_v = volume_per_fu(ref)
                energy_ok = e_value is None or ref_e is None or abs(e_value - ref_e) <= energy_window_eV_per_fu
                volume_ok = (
                    volume_window_A3_per_fu is None
                    or v_value is None
                    or ref_v is None
                    or abs(v_value - ref_v) <= volume_window_A3_per_fu
                )
                if energy_ok and volume_ok:
                    cluster.append(row)
                    placed = True
                    break
            if not placed:
                clusters.append([row])
        for cluster in clusters:
            cluster_counter += 1
            ref = cluster[0]
            ref_e = energy_per_fu(ref)
            cluster_id = f"cluster_{cluster_counter:04d}"
            for row in cluster:
                e_value = energy_per_fu(row)
                output.append(
                    {
                        "cluster_id": cluster_id,
                        "representative": "yes" if row is ref else "no",
                        "motif_id": motif_id(row),
                        "motif_family": motif_family(row),
                        "composition": composition_label(row),
                        "spin_order_host": _spin_label(row),
                        "energy_eV_per_fu": e_value,
                        "delta_E_eV_per_fu": e_value - ref_e if e_value is not None and ref_e is not None else None,
                        "volume_A3_per_fu": volume_per_fu(row),
                        "run_dir": row.get("run_dir") or "",
                        "n_cluster_members": len(cluster),
                    }
                )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motif-cluster",
        description="Cluster motif DB records and identify low-energy representatives for zentropy/active learning.",
    )
    parser.add_argument("--motif-db", type=Path, required=True, help="defect_motif_db.json or compatible motif CSV.")
    parser.add_argument("--outdir", type=Path, default=Path("motif_clusters"))
    parser.add_argument("--group-by", action="append", default=[], help="Grouping column. Defaults to family/composition/spin.")
    parser.add_argument("--energy-window-eV-per-fu", type=float, default=0.05)
    parser.add_argument("--volume-window-A3-per-fu", type=float)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    payload, records = load_motif_records(args.motif_db.resolve())
    group_by = args.group_by or ["motif_family", "composition", "spin_order_host"]
    rows = cluster_motifs(
        records,
        group_by=group_by,
        energy_window_eV_per_fu=args.energy_window_eV_per_fu,
        volume_window_A3_per_fu=args.volume_window_A3_per_fu,
    )
    outdir = args.outdir.resolve()
    cluster_csv = outdir / "motif_clusters.csv"
    representative_csv = outdir / "motif_cluster_representatives.csv"
    representatives = [row for row in rows if row.get("representative") == "yes"]
    write_csv(cluster_csv, rows, FIELDS)
    write_csv(representative_csv, representatives, FIELDS)
    metadata = {
        "schema": SCHEMA,
        "motif_db_schema": payload.get("schema", ""),
        "inputs": {"motif_db": str(args.motif_db.resolve())},
        "outputs": {"clusters": str(cluster_csv), "representatives": str(representative_csv)},
        "group_by": group_by,
        "energy_window_eV_per_fu": args.energy_window_eV_per_fu,
        "volume_window_A3_per_fu": args.volume_window_A3_per_fu,
        "n_motifs": len(records),
        "n_clusters": len(representatives),
    }
    write_json(outdir / "motif_cluster_metadata.json", metadata)
    print(f"Motifs        : {len(records)}")
    print(f"Clusters      : {len(representatives)}")
    print(f"Wrote clusters: {cluster_csv}")
    return metadata


if __name__ == "__main__":
    main()
