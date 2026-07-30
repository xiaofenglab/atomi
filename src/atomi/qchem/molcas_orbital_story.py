"""Paper-ready orbital atlases, state pairs, networks, and occupation metrics.

The plotting commands in this module consume explicit CSV manifests instead of
re-parsing large OpenMolcas outputs.  That keeps scientific extraction separate
from style-only rerendering and makes every figure reproducible from a compact
plot-data packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_ATLAS = "atomi.molcas_orbital_atlas.v1"
SCHEMA_PAIR = "atomi.molcas_orbital_pair.v1"
SCHEMA_NETWORK = "atomi.molcas_orbital_network.v1"
SCHEMA_METRICS = "atomi.molcas_orbital_state_metrics.v1"

STATUS_COLORS = {
    "accepted": "#1B6B50",
    "provisional": "#B7791F",
    "diagnostic-only": "#7A5AF8",
    "rejected": "#B42318",
    "pending": "#59636E",
}


def _matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Matplotlib is required; install atomi[molcas-viz].") from exc
    return matplotlib, plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    return rows


def _required_columns(rows: list[dict[str, str]], columns: set[str], path: Path) -> None:
    missing = sorted(columns - set(rows[0]))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value is None or not str(value).strip():
        return default
    return float(value)


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value is None or not str(value).strip():
        return default
    return int(float(value))


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _crop_image(
    image: np.ndarray,
    *,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> np.ndarray:
    if any(value < 0.0 or value >= 0.95 for value in (top, bottom, left, right)):
        raise ValueError("Image crop fractions must be in [0, 0.95)")
    if top + bottom >= 0.95 or left + right >= 0.95:
        raise ValueError("Image crop fractions remove the whole image")
    height, width = image.shape[:2]
    y0 = int(round(top * height))
    y1 = height - int(round(bottom * height))
    x0 = int(round(left * width))
    x1 = width - int(round(right * width))
    return image[y0:y1, x0:x1]


def _load_image(
    path: Path,
    *,
    top: float = 0.0,
    bottom: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
) -> np.ndarray:
    _, plt = _matplotlib()
    if not path.exists():
        raise FileNotFoundError(path)
    image = plt.imread(path)
    return _crop_image(
        image,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )


def _row_crop(row: dict[str, str], args: argparse.Namespace) -> dict[str, float]:
    return {
        "top": _float(row, "crop_top", args.crop_top),
        "bottom": _float(row, "crop_bottom", args.crop_bottom),
        "left": _float(row, "crop_left", args.crop_left),
        "right": _float(row, "crop_right", args.crop_right),
    }


def _status(row: dict[str, str]) -> str:
    status = (row.get("status") or "provisional").strip().lower()
    return status if status in STATUS_COLORS else "provisional"


def _save_figure(figure: Any, outdir: Path, stem: str, dpi: int) -> list[str]:
    outputs: list[str] = []
    for suffix in ("png", "pdf", "svg"):
        path = outdir / f"{stem}.{suffix}"
        kwargs = {"dpi": dpi} if suffix == "png" else {}
        figure.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    return outputs


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _subtitle(row: dict[str, str]) -> str:
    fields: list[str] = []
    if row.get("symmetry"):
        fields.append(row["symmetry"])
    if row.get("occupation"):
        fields.append(f"n={float(row['occupation']):.3f}")
    if row.get("energy_au"):
        fields.append(f"{float(row['energy_au']):.4f} Eh")
    return " | ".join(fields)


def render_orbital_atlas(args: argparse.Namespace) -> int:
    manifest = args.manifest_csv.expanduser().resolve()
    rows = _read_csv(manifest)
    _required_columns(rows, {"label", "image_path", "family"}, manifest)
    base = manifest.parent
    for row in rows:
        row["_image"] = str(_resolve(row["image_path"], base))
        row["_family_order"] = str(_int(row, "family_order", 999))
        row["_order"] = str(_int(row, "order", 999))
    rows.sort(
        key=lambda row: (
            int(row["_family_order"]),
            row["family"],
            int(row["_order"]),
            row["label"],
        )
    )

    groups: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        groups.setdefault(row["family"], []).append(row)
    columns = min(args.columns, max(len(group) for group in groups.values()))
    panel_rows = sum(math.ceil(len(group) / columns) for group in groups.values())
    height_ratios: list[float] = []
    for group in groups.values():
        height_ratios.extend([0.18] + [1.0] * math.ceil(len(group) / columns))

    _, plt = _matplotlib()
    figure = plt.figure(
        figsize=(args.panel_width * columns, 1.15 + args.panel_height * panel_rows),
        facecolor="white",
    )
    grid = figure.add_gridspec(
        len(height_ratios),
        columns,
        height_ratios=height_ratios,
        hspace=0.12,
        wspace=0.10,
    )
    grid_row = 0
    source_images: list[dict[str, str]] = []
    normalized: list[dict[str, Any]] = []
    for family, group in groups.items():
        header = figure.add_subplot(grid[grid_row, :])
        header.set_facecolor("#F1F3F5")
        header.text(
            0.012,
            0.50,
            family,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#252B33",
        )
        header.set_xticks([])
        header.set_yticks([])
        for spine in header.spines.values():
            spine.set_visible(False)
        grid_row += 1
        nrows = math.ceil(len(group) / columns)
        for index, row in enumerate(group):
            local_row, col = divmod(index, columns)
            panel = grid[grid_row + local_row, col].subgridspec(
                2,
                1,
                height_ratios=(0.22, 0.78),
                hspace=0.0,
            )
            label_ax = figure.add_subplot(panel[0, 0])
            ax = figure.add_subplot(panel[1, 0])
            image_path = Path(row["_image"])
            crop = _row_crop(row, args)
            ax.imshow(_load_image(image_path, **crop))
            ax.set_axis_off()
            status = _status(row)
            title = row["label"]
            subtitle = row.get("subtitle", "").strip() or _subtitle(row)
            label_ax.text(
                0.5,
                0.52,
                title + (f"\n{subtitle}" if subtitle else ""),
                ha="center",
                va="center",
                fontsize=8.0,
                color=STATUS_COLORS[status],
            )
            label_ax.set_axis_off()
            source_images.append(_image_record(image_path))
            normalized.append(
                {key: value for key, value in row.items() if not key.startswith("_")}
                | {
                    "resolved_image_path": str(image_path),
                    "status": status,
                    **{f"effective_crop_{key}": value for key, value in crop.items()},
                }
            )
        for index in range(len(group), nrows * columns):
            local_row, col = divmod(index, columns)
            ax = figure.add_subplot(grid[grid_row + local_row, col])
            ax.set_axis_off()
        grid_row += nrows

    figure.suptitle(args.title, fontsize=14, y=0.995)
    figure.text(
        0.5,
        0.006,
        args.footnote,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#59636E",
    )
    figure.subplots_adjust(left=0.025, right=0.985, top=0.955, bottom=0.045)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = _save_figure(figure, outdir, args.figure_stem, args.dpi)
    plt.close(figure)

    rows_path = outdir / f"{args.figure_stem}_plot_rows.csv"
    _write_rows(rows_path, normalized)
    record = {
        "schema": SCHEMA_ATLAS,
        "plot_kind": args.plot_kind,
        "title": args.title,
        "footnote": args.footnote,
        "source_manifest": _image_record(manifest),
        "source_images": source_images,
        "group_order": list(groups),
        "n_orbitals": len(rows),
        "outputs": outputs,
        "plot_rows": str(rows_path),
        "renderer": "matplotlib_manifest_orbital_atlas",
        "scientific_guard": (
            "The atlas arranges already-extracted orbital images. It does not infer "
            "state identity, transition character, or orbital entanglement."
        ),
    }
    _write_json(outdir / f"{args.figure_stem}_plot_dataset.json", record)
    print(f"Wrote orbital atlas: {outputs[0]}")
    print(f"Wrote frozen plot rows: {rows_path}")
    return 0


def render_orbital_pair(args: argparse.Namespace) -> int:
    manifest = args.manifest_csv.expanduser().resolve()
    rows = _read_csv(manifest)
    _required_columns(
        rows,
        {"pair_label", "left_image_path", "right_image_path", "left_label", "right_label"},
        manifest,
    )
    if args.plot_kind == "nto":
        for row in rows:
            if not (row.get("contribution") or row.get("eigenvalue")):
                raise ValueError(
                    "NTO pairs require contribution or eigenvalue metadata; "
                    "do not label a generic orbital pair as an NTO."
                )
    if args.plot_kind != "nto" and re.search(r"\bNTO\b", args.title, re.IGNORECASE):
        raise ValueError("Only --plot-kind nto may use NTO in the figure title")

    _, plt = _matplotlib()
    figure, axes = plt.subplots(
        len(rows),
        2,
        figsize=(2 * args.panel_width, len(rows) * args.panel_height + 0.8),
        squeeze=False,
    )
    normalized: list[dict[str, Any]] = []
    source_images: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        crop = _row_crop(row, args)
        left = _resolve(row["left_image_path"], manifest.parent)
        right = _resolve(row["right_image_path"], manifest.parent)
        for col, (path, label) in enumerate(
            ((left, row["left_label"]), (right, row["right_label"]))
        ):
            axes[row_index, col].imshow(_load_image(path, **crop))
            axes[row_index, col].set_axis_off()
            axes[row_index, col].set_title(label, fontsize=9, pad=2)
            source_images.append(_image_record(path))
        metadata: list[str] = [row["pair_label"]]
        if row.get("singular_value"):
            metadata.append(f"Lambda={float(row['singular_value']):.3f}")
        if row.get("eigenvalue"):
            metadata.append(f"lambda={float(row['eigenvalue']):.3f}")
        if row.get("contribution"):
            metadata.append(f"{100.0 * float(row['contribution']):.1f}%")
        axes[row_index, 0].text(
            0.0,
            1.08,
            " | ".join(metadata),
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
        normalized.append(
            dict(row)
            | {
                "resolved_left_image_path": str(left),
                "resolved_right_image_path": str(right),
                "status": _status(row),
                **{f"effective_crop_{key}": value for key, value in crop.items()},
            }
        )

    caveat = (
        "Particle/hole NTO pair; contribution is from the transition-density SVD."
        if args.plot_kind == "nto"
        else "State-orbital comparison only; this is not a particle/hole NTO pair."
    )
    figure.suptitle(args.title, fontsize=13, y=0.995)
    figure.text(
        0.5,
        0.008,
        args.footnote or caveat,
        ha="center",
        fontsize=7.2,
        color="#59636E",
    )
    figure.subplots_adjust(left=0.03, right=0.98, top=0.93, bottom=0.07, hspace=0.32)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = _save_figure(figure, outdir, args.figure_stem, args.dpi)
    plt.close(figure)
    rows_path = outdir / f"{args.figure_stem}_plot_rows.csv"
    _write_rows(rows_path, normalized)
    _write_json(
        outdir / f"{args.figure_stem}_plot_dataset.json",
        {
            "schema": SCHEMA_PAIR,
            "plot_kind": args.plot_kind,
            "title": args.title,
            "source_manifest": _image_record(manifest),
            "source_images": source_images,
            "outputs": outputs,
            "plot_rows": str(rows_path),
            "scientific_guard": caveat,
        },
    )
    print(f"Wrote orbital pair figure: {outputs[0]}")
    return 0


def _network_positions(
    nodes: list[dict[str, str]],
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    groups: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for node in sorted(
        nodes,
        key=lambda row: (
            _int(row, "family_order", 999),
            row["family"],
            _int(row, "order", 999),
            row["label"],
        ),
    ):
        groups.setdefault(node["family"], []).append(node)
    positions: dict[str, tuple[float, float]] = {}
    x_values = np.linspace(0.08, 0.92, len(groups))
    for x, group in zip(x_values, groups.values()):
        y_values = np.linspace(0.80, 0.20, len(group)) if len(group) > 1 else [0.50]
        for node, y in zip(group, y_values):
            positions[node["node_id"]] = (float(x), float(y))
    return positions, list(groups)


def render_orbital_network(args: argparse.Namespace) -> int:
    nodes_path = args.nodes_csv.expanduser().resolve()
    edges_path = args.edges_csv.expanduser().resolve()
    nodes = _read_csv(nodes_path)
    edges = _read_csv(edges_path)
    _required_columns(nodes, {"node_id", "label", "family"}, nodes_path)
    _required_columns(edges, {"source", "target", "value"}, edges_path)
    node_ids = {row["node_id"] for row in nodes}
    missing = sorted(
        {
            endpoint
            for row in edges
            for endpoint in (row["source"], row["target"])
            if endpoint not in node_ids
        }
    )
    if missing:
        raise ValueError(f"Edges reference unknown nodes: {', '.join(missing)}")
    if args.metric == "mutual_information":
        if not re.search(r"\b(DMRG|QCMaquis)\b", args.method, re.IGNORECASE):
            raise ValueError(
                "A true entanglement diagram requires a DMRG/QCMaquis method label "
                "and orbital mutual-information values."
            )
    elif re.search(r"\bentanglement\b", args.title, re.IGNORECASE):
        raise ValueError("Only --metric mutual_information may be titled an entanglement diagram")

    kept_edges = [row for row in edges if _float(row, "value") >= args.min_edge_value]
    if not kept_edges:
        raise ValueError("No edges remain after --min-edge-value filtering")
    positions, families = _network_positions(nodes)
    _, plt = _matplotlib()
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyArrowPatch, Rectangle

    figure, ax = plt.subplots(figsize=(args.width, args.height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    family_x = {
        family: positions[next(row["node_id"] for row in nodes if row["family"] == family)][0]
        for family in families
    }
    band_width = 0.82 / max(1, len(families))
    for family in families:
        x = family_x[family]
        ax.add_patch(
            Rectangle(
                (x - band_width / 2.0, 0.055),
                band_width,
                0.88,
                facecolor="#F7F8FA",
                edgecolor="#D7DADF",
                linewidth=0.7,
                zorder=0,
            )
        )
        ax.text(
            x,
            0.955,
            family,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#252B33",
        )

    values = np.asarray([_float(row, "value") for row in kept_edges], dtype=float)
    maximum = float(values.max())
    for row in sorted(kept_edges, key=lambda item: _float(item, "value")):
        value = _float(row, "value")
        fraction = value / maximum if maximum > 0.0 else 0.0
        x1, y1 = positions[row["source"]]
        x2, y2 = positions[row["target"]]
        connection = "arc3,rad=0.10" if y2 >= y1 else "arc3,rad=-0.10"
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-",
                connectionstyle=connection,
                linewidth=0.35 + args.max_edge_width * fraction,
                color="#4062A8",
                alpha=0.12 + 0.70 * fraction,
                zorder=1,
            )
        )

    source_images: list[dict[str, str]] = []
    for row in nodes:
        x, y = positions[row["node_id"]]
        status = _status(row)
        image_text = (row.get("image_path") or "").strip()
        if image_text:
            image_path = _resolve(image_text, nodes_path.parent)
            crop = _row_crop(row, args)
            image = _load_image(image_path, **crop)
            artist = AnnotationBbox(
                OffsetImage(image, zoom=args.node_image_zoom),
                (x, y),
                frameon=True,
                pad=0.05,
                bboxprops={
                    "edgecolor": STATUS_COLORS[status],
                    "linewidth": 0.8,
                    "facecolor": "white",
                },
                zorder=3,
            )
            ax.add_artist(artist)
            source_images.append(_image_record(image_path))
        else:
            ax.scatter(
                [x],
                [y],
                s=args.component_node_size,
                color=STATUS_COLORS[status],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        subtitle = _subtitle(row)
        ax.text(
            x,
            y - args.label_offset,
            row["label"] + (f"\n{subtitle}" if subtitle else ""),
            ha="center",
            va="top",
            fontsize=6.8,
            color="#252B33",
            zorder=4,
        )

    metric_label = {
        "mutual_information": "DMRG orbital mutual information",
        "ao_composition": "AO coefficient-square component weight",
        "so_parent_weight": "RASSI spin-free parent weight",
        "overlap": "orbital overlap metric",
        "custom": "user-supplied edge metric",
    }[args.metric]
    is_entanglement = args.metric == "mutual_information"
    guard = (
        "True orbital-entanglement network from DMRG mutual information."
        if is_entanglement
        else "Composition/coupling network; edge weights are not orbital entanglement."
    )
    ax.set_title(args.title, loc="left", fontsize=13, pad=12)
    figure.text(
        0.01,
        0.015,
        f"Edge metric: {metric_label}. {guard}",
        ha="left",
        fontsize=7.2,
        color="#59636E",
    )
    figure.subplots_adjust(left=0.02, right=0.985, top=0.91, bottom=0.06)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = _save_figure(figure, outdir, args.figure_stem, args.dpi)
    plt.close(figure)

    normalized_nodes = [
        dict(row)
        | {
            "status": _status(row),
            "x": positions[row["node_id"]][0],
            "y": positions[row["node_id"]][1],
        }
        for row in nodes
    ]
    nodes_out = outdir / f"{args.figure_stem}_nodes.csv"
    edges_out = outdir / f"{args.figure_stem}_edges.csv"
    _write_rows(nodes_out, normalized_nodes)
    _write_rows(edges_out, kept_edges)
    _write_json(
        outdir / f"{args.figure_stem}_plot_dataset.json",
        {
            "schema": SCHEMA_NETWORK,
            "metric": args.metric,
            "method": args.method,
            "is_true_entanglement": is_entanglement,
            "scientific_guard": guard,
            "title": args.title,
            "source_nodes": _image_record(nodes_path),
            "source_edges": _image_record(edges_path),
            "source_images": source_images,
            "min_edge_value": args.min_edge_value,
            "n_nodes": len(nodes),
            "n_edges_input": len(edges),
            "n_edges_rendered": len(kept_edges),
            "family_order": families,
            "outputs": outputs,
            "plot_nodes": str(nodes_out),
            "plot_edges": str(edges_out),
        },
    )
    print(f"Wrote orbital network: {outputs[0]}")
    print(guard)
    return 0


def _entropy(occupations: list[float]) -> float:
    return float(
        -sum(
            occupation * math.log(occupation / 2.0)
            for occupation in occupations
            if occupation > 0.0
        )
    )


def calculate_orbital_state_metrics(args: argparse.Namespace) -> int:
    source = args.occupations_csv.expanduser().resolve()
    rows = _read_csv(source)
    _required_columns(rows, {"state", "orbital_id", "occupation"}, source)
    grouped: OrderedDict[str, list[float]] = OrderedDict()
    state_status: dict[str, str] = {}
    for row in rows:
        occupation = _float(row, "occupation")
        if occupation < -args.occupation_tolerance or occupation > 2.0 + args.occupation_tolerance:
            raise ValueError(
                f"Occupation {occupation:g} for {row['state']}:{row['orbital_id']} "
                "is outside the natural-orbital range [0, 2]"
            )
        grouped.setdefault(row["state"], []).append(max(0.0, min(2.0, occupation)))
        state_status.setdefault(row["state"], _status(row))

    state_rows: list[dict[str, Any]] = []
    for state, occupations in grouped.items():
        state_rows.append(
            {
                "state": state,
                "n_orbitals": len(occupations),
                "electron_count": sum(occupations),
                "correlation_entropy": _entropy(occupations),
                "status": state_status[state],
            }
        )
    pair_rows: list[dict[str, Any]] = []
    for pair in args.pair:
        if ":" not in pair:
            raise ValueError("--pair must use STATE_A:STATE_B")
        left, right = pair.split(":", 1)
        if left not in grouped or right not in grouped:
            raise ValueError(f"Unknown state in --pair {pair}")
        left_occ = sorted(grouped[left], reverse=True)
        right_occ = sorted(grouped[right], reverse=True)
        same_dimension = len(left_occ) == len(right_occ)
        same_electrons = abs(sum(left_occ) - sum(right_occ)) <= args.electron_tolerance
        comparable = same_dimension and same_electrons
        delta_n = (
            0.5 * sum(abs(a - b) for a, b in zip(left_occ, right_occ)) if same_dimension else None
        )
        pair_rows.append(
            {
                "state_a": left,
                "state_b": right,
                "n_orbitals_a": len(left_occ),
                "n_orbitals_b": len(right_occ),
                "electron_count_a": sum(left_occ),
                "electron_count_b": sum(right_occ),
                "same_dimension_guard_pass": same_dimension,
                "electron_count_guard_pass": same_electrons,
                "comparison_accepted": comparable,
                "occupation_redistribution_half_l1": delta_n,
                "note": (
                    "Natural occupations are sorted before comparison; this metric "
                    "does not establish orbital-by-orbital identity."
                ),
            }
        )

    _, plt = _matplotlib()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    labels = [row["state"] for row in state_rows]
    colors = [STATUS_COLORS[str(row["status"])] for row in state_rows]
    axes[0].barh(labels, [row["correlation_entropy"] for row in state_rows], color=colors)
    axes[0].set_xlabel("Correlation entropy")
    axes[0].set_title(r"$-\sum_i n_i\ln(n_i/2)$", loc="left")
    axes[1].barh(labels, [row["electron_count"] for row in state_rows], color=colors)
    axes[1].set_xlabel("Selected-orbital electron count")
    axes[1].set_title("Electron-count guard", loc="left")
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].set_yticks([])
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#D9DDE2", linewidth=0.45)
        ax.set_axisbelow(True)
    figure.suptitle(args.title, fontsize=12)
    rejected_pairs = [row for row in pair_rows if not bool(row["comparison_accepted"])]
    if rejected_pairs:
        pair = rejected_pairs[0]
        comparison_note = (
            "Cross-state occupation redistribution rejected: "
            f"{pair['state_a']} has {pair['electron_count_a']:.3f} e; "
            f"{pair['state_b']} has {pair['electron_count_b']:.3f} e."
        )
    else:
        comparison_note = (
            "Cross-state occupation redistribution passes the selected-space "
            "dimension and electron-count guards."
        )
    figure.text(
        0.01,
        0.065,
        comparison_note,
        ha="left",
        fontsize=7,
        color="#B42318" if rejected_pairs else "#1B6B50",
    )
    figure.text(
        0.01,
        0.025,
        "Entropy is meaningful only for a complete, consistently defined natural-orbital set.",
        ha="left",
        fontsize=7,
        color="#59636E",
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.26, wspace=0.22)
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = _save_figure(figure, outdir, args.figure_stem, args.dpi)
    plt.close(figure)
    state_path = outdir / f"{args.figure_stem}_state_metrics.csv"
    pair_path = outdir / f"{args.figure_stem}_pair_metrics.csv"
    _write_rows(state_path, state_rows)
    if pair_rows:
        _write_rows(pair_path, pair_rows)
    _write_json(
        outdir / f"{args.figure_stem}_summary.json",
        {
            "schema": SCHEMA_METRICS,
            "source": _image_record(source),
            "state_metrics": state_rows,
            "pair_metrics": pair_rows,
            "outputs": outputs,
            "scientific_guards": [
                "Natural occupations must lie in [0, 2].",
                "Cross-state redistribution requires equal selected-space dimension and electron count.",
                "Sorted-occupation redistribution does not establish orbital identity.",
                "Density-displacement analysis requires separately supplied state-density grids.",
            ],
        },
    )
    print(f"Wrote orbital-state metrics: {outputs[0]}")
    return 0


def _add_crop_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crop-top", type=float, default=0.0)
    parser.add_argument("--crop-bottom", type=float, default=0.0)
    parser.add_argument("--crop-left", type=float, default=0.0)
    parser.add_argument("--crop-right", type=float, default=0.0)


def add_parsers(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "orbital-atlas",
        help="Build a grouped Figure-2-style atlas from a frozen orbital-image manifest.",
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("molcas_orbital_atlas"))
    parser.add_argument("--title", default="OpenMolcas orbital atlas")
    parser.add_argument(
        "--plot-kind",
        choices=("natural-orbital", "active-orbital", "nto", "generic"),
        default="natural-orbital",
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--panel-width", type=float, default=2.45)
    parser.add_argument("--panel-height", type=float, default=2.15)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-stem", default="molcas_orbital_atlas")
    parser.add_argument(
        "--footnote",
        default="Common camera and isovalue are required for comparative orbital atlases.",
    )
    _add_crop_arguments(parser)
    parser.set_defaults(func=render_orbital_atlas)

    parser = subparsers.add_parser(
        "orbital-pair",
        help="Plot particle/hole NTO pairs or explicitly labeled state-orbital pairs.",
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("molcas_orbital_pair"))
    parser.add_argument("--title", default="OpenMolcas orbital pair")
    parser.add_argument(
        "--plot-kind",
        choices=("nto", "state-natural-orbital", "generic"),
        default="state-natural-orbital",
    )
    parser.add_argument("--panel-width", type=float, default=3.2)
    parser.add_argument("--panel-height", type=float, default=2.8)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-stem", default="molcas_orbital_pair")
    parser.add_argument("--footnote", default="")
    _add_crop_arguments(parser)
    parser.set_defaults(func=render_orbital_pair)

    parser = subparsers.add_parser(
        "orbital-network",
        help="Plot a segmented orbital network with a metric-specific entanglement guard.",
    )
    parser.add_argument("--nodes-csv", type=Path, required=True)
    parser.add_argument("--edges-csv", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=(
            "mutual_information",
            "ao_composition",
            "so_parent_weight",
            "overlap",
            "custom",
        ),
        required=True,
    )
    parser.add_argument("--method", default="")
    parser.add_argument("--outdir", type=Path, default=Path("molcas_orbital_network"))
    parser.add_argument("--title", default="OpenMolcas orbital network")
    parser.add_argument("--min-edge-value", type=float, default=0.0)
    parser.add_argument("--max-edge-width", type=float, default=4.5)
    parser.add_argument("--node-image-zoom", type=float, default=0.075)
    parser.add_argument("--component-node-size", type=float, default=520.0)
    parser.add_argument("--label-offset", type=float, default=0.057)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figure-stem", default="molcas_orbital_network")
    _add_crop_arguments(parser)
    parser.set_defaults(func=render_orbital_network)

    parser = subparsers.add_parser(
        "orbital-state-metrics",
        help="Compute natural-occupation correlation entropy and guarded redistribution.",
    )
    parser.add_argument("--occupations-csv", type=Path, required=True)
    parser.add_argument("--pair", action="append", default=[])
    parser.add_argument("--electron-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--occupation-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--outdir", type=Path, default=Path("molcas_orbital_state_metrics"))
    parser.add_argument("--title", default="Natural-orbital state metrics")
    parser.add_argument("--figure-stem", default="molcas_orbital_state_metrics")
    parser.add_argument("--dpi", type=int, default=300)
    parser.set_defaults(func=calculate_orbital_state_metrics)
