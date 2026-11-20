"""Repair whole-tissue masks, trace the surface, and filter superficial signal.

Masks are 2D Boolean arrays; path coordinates are (row, column) pixels.
Physical distances use micrometers. Cleanup applies the same rule to every
section and never clips attached tissue to the reference envelope.
"""

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, median_filter
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.measure import label, regionprops

from .configuration import AnalysisConfig
from .measurement import calculate_calibrated_skeleton_length

DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()


def calculate_pixel_parameters(pixel_size_um, config=None):
    """Convert whole-skin repair thresholds to integer pixel counts.

    Nerve proximity/length and macro-depth calculations use physical units
    directly and are configured separately.
    """
    settings = config or DEFAULT_ANALYSIS_CONFIG
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            f"Pixel size must be positive and finite, but received {pixel_size_um}."
        )

    pixel_area_um2 = pixel_size_um**2

    def area_to_pixels(area_um2):
        return max(
            1,
            int(round(area_um2 / pixel_area_um2)),
        )

    whole_skin_smoothing_width_pixels = max(
        1,
        int(round(settings.whole_skin_boundary_smoothing_width_um / pixel_size_um)),
    )
    if whole_skin_smoothing_width_pixels % 2 == 0:
        whole_skin_smoothing_width_pixels += 1

    return {
        "pixel_size_um": float(pixel_size_um),
        "pixel_area_um2": float(pixel_area_um2),
        "min_whole_skin_object_pixels": area_to_pixels(
            settings.min_whole_skin_object_area_um2
        ),
        "whole_skin_smoothing_width_pixels": (whole_skin_smoothing_width_pixels),
    }


def repair_whole_skin_mask(
    binary_mask,
    min_object_pixels,
    smoothing_width_pixels,
    closing_radius_pixels=0,
    config=None,
    *,
    qc_steps=None,
):
    """Apply the same whole-tissue repair to every section.

    Long superficial components define the reference envelope. Remove only
    non-reference components entirely below its median-smoothed lower edge,
    then close small gaps and fill enclosed holes. The lower edge is used to
    reject disconnected objects, never to clip the retained tissue itself.
    These are geometric operations, not a classification of tissue types.
    """
    mask = np.asarray(binary_mask, dtype=bool)
    labeled_mask = label(mask, connectivity=2)
    reference_mask, selected_labels = select_superficial_long_components(
        mask,
        min_component_area_pixels=min_object_pixels,
        labeled_mask=labeled_mask,
        config=config,
    )

    upper_boundary = extract_smoothed_mask_edge(
        reference_mask,
        edge="upper",
        smoothing_width_pixels=smoothing_width_pixels,
    )
    lower_boundary = extract_smoothed_mask_edge(
        reference_mask,
        edge="lower",
        smoothing_width_pixels=smoothing_width_pixels,
    )
    lower_boundary = np.clip(
        np.maximum(lower_boundary, upper_boundary + 1),
        0,
        mask.shape[0] - 1,
    )

    cleaned_mask = np.zeros_like(mask, dtype=bool)

    for region in regionprops(labeled_mask):
        coordinates = region.coords
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        is_reference = int(region.label) in selected_labels
        entirely_below_boundary = bool(np.all(rows > lower_boundary[columns]))
        remove_component = not is_reference and entirely_below_boundary

        if not remove_component:
            cleaned_mask[rows, columns] = True

    if qc_steps is not None:
        qc_steps.update(
            raw=mask,
            reference=reference_mask,
            upper_boundary=upper_boundary,
            lower_boundary=lower_boundary,
            filtered=cleaned_mask,
        )

    if closing_radius_pixels > 0:
        cleaned_mask = binary_closing(
            cleaned_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(closing_radius_pixels),
        )
    repaired_mask = binary_fill_holes(cleaned_mask).astype(bool)
    if qc_steps is not None:
        qc_steps.update(closed=cleaned_mask, filled=repaired_mask)
    return repaired_mask


def select_superficial_long_components(
    binary_mask,
    min_component_area_pixels,
    labeled_mask=None,
    config=None,
):
    """
    Select long components on the column-wise superficial envelope.

    Width relative to occupied tissue extent and area identify plausible
    components. Blank side margins do not change the width cutoff. In every
    column, the qualifying component with the uppermost pixel contributes to
    the envelope.
    Retain a component only when it contributes across at least
    the configured minimum fraction of its horizontal bounding-box span.
    All pixels of each selected component enter the whole-skin reference mask;
    this function does not classify those pixels as epidermis.
    """
    settings = config or DEFAULT_ANALYSIS_CONFIG
    _, width = binary_mask.shape
    if labeled_mask is None:
        labeled_mask = label(binary_mask, connectivity=2)
    regions = regionprops(labeled_mask)

    if not regions:
        raise ValueError("No positive mask pixels were found.")

    occupied_columns = np.flatnonzero(np.any(binary_mask, axis=0))
    occupied_width = int(occupied_columns[-1] - occupied_columns[0] + 1)
    minimum_span = max(
        1,
        int(round(occupied_width * settings.min_component_width_fraction)),
    )

    candidates = []
    envelope_rows = np.full(width, binary_mask.shape[0], dtype=np.int32)
    envelope_labels = np.zeros(width, dtype=labeled_mask.dtype)

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        horizontal_span = max_col - min_col

        if horizontal_span >= minimum_span and region.area >= min_component_area_pixels:
            local_component = (
                labeled_mask[min_row:max_row, min_col:max_col] == region.label
            )
            local_columns = np.arange(min_col, max_col)
            populated = np.any(local_component, axis=0)
            local_columns = local_columns[populated]
            upper_rows = min_row + np.argmax(local_component[:, populated], axis=0)
            replace = upper_rows < envelope_rows[local_columns]
            envelope_rows[local_columns[replace]] = upper_rows[replace]
            envelope_labels[local_columns[replace]] = region.label
            candidates.append((int(region.label), horizontal_span))

    if not candidates:
        raise ValueError(
            "No mask object passed the long-component filters. "
            "Check the mask and the configured minimum component area/span thresholds."
        )

    envelope_column_counts = np.bincount(
        envelope_labels.astype(np.intp, copy=False),
        minlength=int(labeled_mask.max()) + 1,
    )

    selected_labels = {
        component_id
        for component_id, span in candidates
        if float(envelope_column_counts[component_id] / span)
        >= settings.min_superficial_envelope_fraction
    }
    if not selected_labels:
        raise ValueError(
            "No long component formed enough of the superficial mask "
            "envelope. Review min_superficial_envelope_fraction in the configuration."
        )

    keep_label = np.zeros(int(labeled_mask.max()) + 1, dtype=bool)
    keep_label[list(selected_labels)] = True
    return keep_label[labeled_mask], selected_labels


def extract_smoothed_mask_edge(mask, edge, smoothing_width_pixels):
    """
    Return the interpolated, median-smoothed upper or lower mask edge.

    Parameters
    ----------
    mask : 2D Boolean array
    edge : {"upper", "lower"}
        "upper" selects the first positive row in each populated column;
        "lower" selects the last positive row.
    smoothing_width_pixels : int
        Median-filter width applied after interpolation.
    """
    if edge not in {"upper", "lower"}:
        raise ValueError("edge must be either 'upper' or 'lower'.")

    height, width = mask.shape
    populated = np.any(mask, axis=0)
    valid_columns = np.flatnonzero(populated)
    if valid_columns.size < 2:
        raise ValueError(
            f"The {edge} mask edge was found in fewer than two image columns."
        )

    if edge == "upper":
        valid_rows = np.argmax(mask[:, populated], axis=0)
    else:
        valid_rows = height - 1 - np.argmax(mask[::-1, populated], axis=0)
    all_columns = np.arange(width)
    curve = np.interp(all_columns, valid_columns, valid_rows)
    curve = median_filter(
        curve,
        size=smoothing_width_pixels,
        mode="nearest",
    )

    return np.clip(
        np.rint(curve).astype(int),
        0,
        height - 1,
    )


def filter_long_nerve_objects_near_superficial_boundary(
    nerve_positive_mask,
    superficial_boundary_mask,
    pixel_size_um,
    maximum_boundary_distance_um,
    minimum_object_length_um,
):
    """Remove intact long nerve objects close to the superficial tissue edge.

    Object length is the calibrated 8-connected skeleton length, so curved and
    diagonal objects are assessed geometrically rather than by horizontal
    bounding-box width. Short puncta and short fibers are retained even when
    close to the surface. Long objects are retained when they do not approach
    the superficial boundary.
    """
    mask = np.asarray(nerve_positive_mask, dtype=bool)
    superficial_boundary = np.asarray(superficial_boundary_mask, dtype=bool)
    if mask.shape != superficial_boundary.shape:
        raise ValueError("Nerve and superficial-boundary masks must match.")
    boundary_coordinates = np.column_stack(np.nonzero(superficial_boundary))
    if not boundary_coordinates.size:
        raise ValueError("The superficial boundary mask is empty.")
    boundary_tree = cKDTree(boundary_coordinates.astype(np.float32))
    component_labels = label(mask, connectivity=2)
    retained = mask.copy()
    removed_count = 0
    for region in regionprops(component_labels):
        coordinates = region.coords
        minimum_distance_um = float(
            boundary_tree.query(coordinates, k=1)[0].min() * pixel_size_um
        )
        min_row, min_col, max_row, max_col = region.bbox
        if minimum_distance_um <= maximum_boundary_distance_um:
            local_component = (
                component_labels[min_row:max_row, min_col:max_col] == region.label
            )
            _, skeleton_length_um = calculate_calibrated_skeleton_length(
                local_component,
                pixel_size_um,
            )
        else:
            # Exact length is irrelevant when the proximity criterion fails.
            skeleton_length_um = np.nan
        remove = bool(
            np.isfinite(skeleton_length_um)
            and skeleton_length_um >= minimum_object_length_um
            and minimum_distance_um <= maximum_boundary_distance_um
        )
        if remove:
            retained[coordinates[:, 0], coordinates[:, 1]] = False
            removed_count += 1
    return retained, removed_count


def trace_whole_tissue_superficial_boundary(
    repaired_whole_tissue_mask,
    smoothing_width_pixels,
):
    """Trace a column-wise top-edge scaffold, assuming epidermis is image-up.

    Reconstruction later selects superficial contour arcs for nerve filtering.
    """
    whole_tissue = np.asarray(repaired_whole_tissue_mask, dtype=bool)
    upper_rows = extract_smoothed_mask_edge(
        whole_tissue,
        edge="upper",
        smoothing_width_pixels=smoothing_width_pixels,
    )
    valid_columns = np.flatnonzero(np.any(whole_tissue, axis=0))
    boundary = np.zeros_like(whole_tissue)
    if valid_columns.size:
        # Keep separate tissue spans separate; never bridge a blank column gap.
        split_points = np.flatnonzero(np.diff(valid_columns) > 1) + 1
        for columns in np.split(valid_columns, split_points):
            if not columns.size:
                continue
            boundary[upper_rows[columns], columns] = True
            for index in range(1, len(columns)):
                rr, cc = line(
                    int(upper_rows[columns[index - 1]]),
                    int(columns[index - 1]),
                    int(upper_rows[columns[index]]),
                    int(columns[index]),
                )
                boundary[rr, cc] = True
    return boundary
