"""Read-only four-compartment BT3 partition and quantification.

This module consumes fixed production masks and skeletons. It never modifies
the anatomical boundary, reconstructed compartments, BT3 segmentation, or
the existing aggregate quantification results.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion


COMPARTMENT_NAMES = (
    "upper_epidermis",
    "basal_epidermis",
    "subbasal_dermis",
    "deep_dermis",
    "whole_epidermis",
    "whole_dermis",
)


@dataclass
class FourCompartmentResult:
    """Four-way masks plus additive whole-compartment reporting."""

    compartment_masks: dict[str, np.ndarray]
    bt3_masks: dict[str, np.ndarray]
    skeleton_masks: dict[str, np.ndarray]
    metrics: dict[str, float]


def _validate_matching_masks(named_masks):
    shapes = {name: np.asarray(mask).shape for name, mask in named_masks.items()}
    if not shapes or any(len(shape) != 2 for shape in shapes.values()):
        raise ValueError("All four-compartment masks must be two-dimensional.")
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"Four-compartment mask shapes do not match: {shapes}")
    return next(iter(unique_shapes))


def _assert_partition(upper, basal, subbasal, deep, whole_epidermis, whole_dermis):
    compartments = {
        "upper epidermis": upper,
        "basal epidermis": basal,
        "sub-basal dermis": subbasal,
        "deep dermis": deep,
    }
    for (first_name, first), (second_name, second) in combinations(
        compartments.items(), 2
    ):
        overlap = int(np.count_nonzero(first & second))
        if overlap:
            raise RuntimeError(
                f"{first_name} and {second_name} overlap by {overlap} pixels."
            )
    if not np.array_equal(upper | basal, whole_epidermis):
        raise RuntimeError("Epidermal subcompartments do not partition epidermis.")
    if not np.array_equal(subbasal | deep, whole_dermis):
        raise RuntimeError("Dermal subcompartments do not partition dermis.")
    if np.any(whole_epidermis & whole_dermis):
        raise RuntimeError("Authoritative epidermis and dermis masks overlap.")
    analyzed_tissue = whole_epidermis | whole_dermis
    combined = np.logical_or.reduce(tuple(compartments.values()))
    if not np.array_equal(combined, analyzed_tissue):
        missing = int(np.count_nonzero(analyzed_tissue & ~combined))
        extra = int(np.count_nonzero(combined & ~analyzed_tissue))
        raise RuntimeError(
            "Four compartments do not exhaust analyzed whole tissue: "
            f"{missing} missing pixels and {extra} extra pixels."
        )


def assert_aggregate_metrics_match_existing(result, existing_metrics):
    """Require direct whole-mask metrics to equal frozen production outputs."""

    comparisons = {
        "whole_epidermis_area_um2": "epidermal_area_um2",
        "whole_epidermis_bt3_area_um2": "epidermal_nerve_area_um2",
        "whole_epidermis_bt3_area_fraction": "epidermal_nerve_area_fraction",
        "whole_epidermis_bt3_skeleton_length_um": (
            "epidermal_nerve_skeleton_length_um"
        ),
        "whole_dermis_area_um2": "dermal_area_um2",
        "whole_dermis_bt3_area_um2": "dermal_nerve_area_um2",
        "whole_dermis_bt3_area_fraction": "dermal_nerve_area_fraction",
        "whole_dermis_bt3_skeleton_length_um": "dermal_nerve_skeleton_length_um",
    }
    mismatches = []
    for new_name, existing_name in comparisons.items():
        new_value = float(result.metrics[new_name])
        existing_value = float(existing_metrics[existing_name])
        if np.isnan(new_value) and np.isnan(existing_value):
            continue
        if new_value != existing_value:
            mismatches.append((new_name, existing_name, new_value, existing_value))
    if mismatches:
        raise RuntimeError(
            f"Four-compartment aggregate metrics differ from production: {mismatches}"
        )


def _normalize_grayscale(image):
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (1, 99.5))
    if high <= low:
        high = low + 1.0
    return np.rint(np.clip((values - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _blend(rgb, mask, color, alpha):
    rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * np.asarray(color)


def _outline(mask):
    values = np.asarray(mask, dtype=bool)
    return values & ~binary_erosion(values, structure=np.ones((3, 3), dtype=bool))


def save_four_compartment_qc(
    output_folder,
    background_image,
    anatomical_basal_boundary,
    subbasal_result,
    result,
    *,
    maximum_dimension=5000,
):
    """Save the single comprehensive compartment and boundary QC image."""

    output = Path(output_folder) / "QC"
    output.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(max(background_image.shape) / maximum_dimension)))
    gray = _normalize_grayscale(background_image[::stride, ::stride])
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(float)
    masks = result.compartment_masks
    _blend(rgb, masks["deep_dermis"][::stride, ::stride], (105, 80, 150), 0.12)
    _blend(rgb, masks["subbasal_dermis"][::stride, ::stride], (20, 210, 110), 0.42)
    _blend(rgb, masks["upper_epidermis"][::stride, ::stride], (40, 120, 255), 0.52)
    _blend(rgb, masks["basal_epidermis"][::stride, ::stride], (255, 175, 25), 0.62)

    epidermis_outline = _outline(masks["whole_epidermis"])[::stride, ::stride]
    rgb[binary_dilation(epidermis_outline, iterations=1)] = (255, 255, 255)
    overlays = (
        (anatomical_basal_boundary, (255, 235, 0)),
        (subbasal_result.macro_reference_mask, (0, 245, 255)),
        (subbasal_result.downweighted_boundary_mask, (90, 110, 255)),
        (subbasal_result.lower_boundary_mask, (255, 60, 40)),
    )
    for mask, color in overlays:
        sampled = binary_dilation(
            np.asarray(mask, dtype=bool)[::stride, ::stride], iterations=1
        )
        rgb[sampled] = color
    bt3 = (result.bt3_masks["whole_epidermis"] | result.bt3_masks["whole_dermis"])[
        ::stride, ::stride
    ]
    skeleton = (
        result.skeleton_masks["whole_epidermis"] | result.skeleton_masks["whole_dermis"]
    )[::stride, ::stride]
    rgb[bt3] = (255, 35, 210)
    rgb[skeleton] = (255, 255, 255)

    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 2350), 54), fill="black")
    depth_um = result.metrics["subbasal_depth_um"]
    draw.text(
        (8, 5),
        f"4 compartments | blue upper epi | orange basal epi | green 0-{depth_um:g} um dermis | "
        "purple deep dermis | yellow anatomical | cyan macro | red lower | magenta BT3 | white skeleton",
        fill="white",
    )
    image.save(output / "03_compartments_and_nerve.png")
