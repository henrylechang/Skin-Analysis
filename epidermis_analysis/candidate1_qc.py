"""Concise production diagnostics and QC for Automatic Candidate 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

from .components import rasterize_coordinate_paths, rasterize_paths


def _normalize(image):
    values = np.asarray(image, dtype=float)
    low, high = np.percentile(values, (1, 99.8))
    return np.uint8(255 * np.clip((values - low) / max(high - low, 1e-9), 0, 1))


def candidate1_masks(result, shape):
    selected = set(result.selection.selected_nodes)
    cleaned = np.isin(result.labels, sorted(selected))
    observed = rasterize_paths(result.paths, selected, shape)
    connector_paths = [
        result.edge_paths[edge]
        for edge in sorted(result.selection.selected_edges)
        if edge in result.edge_paths
    ]
    connectors = rasterize_coordinate_paths(connector_paths, shape)
    return cleaned, observed, connectors, observed | connectors


def save_candidate1_qc(output, dapi, raw, surface, result, *, maximum_dimension=5000):
    output = Path(output) / "QC"
    output.mkdir(parents=True, exist_ok=True)
    cleaned, observed, connectors, combined = candidate1_masks(result, raw.shape)
    rejected = raw & ~cleaned
    stride = max(1, int(np.ceil(max(raw.shape) / maximum_dimension)))
    gray = _normalize(dapi[::stride, ::stride])
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[rejected[::stride, ::stride]] = (220, 45, 35)
    rgb[cleaned[::stride, ::stride]] = (30, 110, 240)
    rgb[combined[::stride, ::stride]] = (255, 255, 255)
    rgb[binary_dilation(surface[::stride, ::stride], iterations=1)] = (0, 255, 0)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 1450), 48), fill="black")
    draw.text(
        (8, 5),
        "blue accepted | red rejected | white selected graph | green tissue surface",
        fill="white",
    )
    image.save(output / "01_candidate_selection.png")
    return observed, connectors


def save_candidate1_failure_qc(
    output,
    dapi,
    raw,
    surface,
    reason,
    *,
    maximum_dimension=5000,
):
    """Persist enough context to diagnose an explicit Candidate-1 failure."""
    output = Path(output) / "QC"
    output.mkdir(parents=True, exist_ok=True)
    (output / "candidate1_QC_FAILURE.txt").write_text(
        f"Candidate 1 cleanup failed:\n{reason}\n",
        encoding="utf-8",
    )
    stride = max(1, int(np.ceil(max(raw.shape) / maximum_dimension)))
    gray = _normalize(dapi[::stride, ::stride])
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[raw[::stride, ::stride]] = (30, 110, 240)
    rgb[binary_dilation(surface[::stride, ::stride], iterations=1)] = (0, 255, 0)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 1450), 48), fill="black")
    draw.text(
        (8, 5), "QC FAILURE | blue raw components | green tissue surface", fill="white"
    )
    image.save(output / "candidate_selection_FAILURE.png")


def save_final_combined_qc(output, dapi, cleaned, epidermis, dermis, boundary):
    output = Path(output) / "QC"
    output.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(max(dapi.shape) / 5000)))
    gray = _normalize(dapi[::stride, ::stride])
    rgb = np.repeat(gray[..., None], 3, axis=2)
    rgb[dermis[::stride, ::stride]] = (70, 70, 70)
    rgb[epidermis[::stride, ::stride]] = (25, 70, 145)
    rgb[cleaned[::stride, ::stride]] = (0, 190, 220)
    rgb[binary_dilation(boundary[::stride, ::stride], iterations=1)] = (255, 220, 0)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 1500), 48), fill="black")
    draw.text(
        (8, 5),
        "navy epidermis | gray dermis | cyan cleaned band | yellow anatomical boundary",
        fill="white",
    )
    image.save(output / "02_anatomical_reconstruction.png")
