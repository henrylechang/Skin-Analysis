"""Compartment area, skeleton length, and boundary-normalized nerve measurements.

Whole epidermal and dermal signal masks are skeletonized separately before
assignment to daughter ROIs. Each ROI is measured directly: skeleton lengths
are not additive when a compartment boundary cuts a skeleton edge.
"""

from dataclasses import dataclass

import numpy as np
from skimage.morphology import skeletonize
from epidermis_analysis import subbasal as s, four_compartment as f
from .components import measure_skeleton_graph_length, rasterize_coordinate_paths

LEGACY_BT3_METRIC_NAMES = {
    "epidermal_nerve_area_um2": "epidermal_BT3_area_um2",
    "epidermal_nerve_area_per_boundary_length": (
        "epidermal_BT3_area_per_boundary_length"
    ),
    "epidermal_nerve_area_um2_per_boundary_mm": (
        "epidermal_BT3_area_um2_per_boundary_mm"
    ),
    "epidermal_nerve_area_fraction": "epidermal_BT3_area_fraction",
    "epidermal_nerve_skeleton_length_um": "epidermal_BT3_skeleton_length_um",
    "epidermal_nerve_skeleton_length_per_boundary_length": (
        "epidermal_BT3_skeleton_length_per_boundary_length"
    ),
    "epidermal_nerve_skeleton_length_um_per_boundary_mm": (
        "epidermal_BT3_skeleton_length_um_per_boundary_mm"
    ),
    "dermal_nerve_area_um2": "dermal_BT3_area_um2",
    "dermal_nerve_area_fraction": "dermal_BT3_area_fraction",
    "dermal_nerve_skeleton_length_um": "dermal_BT3_skeleton_length_um",
    "dermal_nerve_skeleton_length_density": "dermal_BT3_skeleton_length_density",
}


def calculate_calibrated_skeleton_length(
    binary_mask,
    pixel_size_um,
    return_full_skeleton=True,
    componentwise=False,
):
    """Skeletonize a mask and measure its calibrated 8-connected graph length.

    Each horizontal or vertical edge contributes one pixel width; each
    diagonal edge contributes ``sqrt(2)`` pixel widths only when neither
    orthogonal corner pixel is present. Edges are counted once using rightward
    and downward offsets. An isolated skeleton pixel contributes one pixel width.
    """
    binary_mask = np.asarray(binary_mask, dtype=bool)
    foreground_rows = np.flatnonzero(np.any(binary_mask, axis=1))
    foreground_columns = np.flatnonzero(np.any(binary_mask, axis=0))
    if foreground_rows.size == 0 or foreground_columns.size == 0:
        empty = np.zeros_like(binary_mask) if return_full_skeleton else None
        return empty, 0.0

    # Skeletonization is local, so disconnected components can share one
    # foreground crop without changing one another. This produces the same
    # result as the former per-component loop while avoiding a full label image
    # and repeated Python/skimage calls. Keep ``componentwise`` for API
    # compatibility with validation code and historical callers.
    row_start = max(0, int(foreground_rows[0]) - 1)
    row_stop = min(binary_mask.shape[0], int(foreground_rows[-1]) + 2)
    column_start = max(0, int(foreground_columns[0]) - 1)
    column_stop = min(binary_mask.shape[1], int(foreground_columns[-1]) + 2)
    skeleton_crop = skeletonize(
        binary_mask[row_start:row_stop, column_start:column_stop]
    )
    length_um = measure_skeleton_graph_length(skeleton_crop, pixel_size_um)
    if return_full_skeleton:
        skeleton = np.zeros_like(binary_mask)
        skeleton[
            row_start:row_stop,
            column_start:column_stop,
        ] = skeleton_crop
    else:
        skeleton = None
    return skeleton, length_um


@dataclass
class MeasuredROI:
    """Fixed tissue ROI, assigned signal/skeleton, and calibrated measurements."""

    mask: np.ndarray
    signal: np.ndarray
    skeleton: np.ndarray
    values: dict


def measure(mask, signal, skeleton, scale, *, assigned=False, length=None):
    """Measure directly, retaining upstream length arithmetic for whole ROIs."""
    inside = signal if assigned else signal & mask
    skel = skeleton if assigned else skeleton & mask
    area = float(np.count_nonzero(mask) * scale**2)
    nerve_area = float(np.count_nonzero(inside) * scale**2)
    if length is None:
        length = measure_skeleton_graph_length(skel, scale)
    return MeasuredROI(
        mask,
        inside,
        skel,
        dict(
            area_um2=area,
            bt3_area_um2=nerve_area,
            bt3_area_fraction=nerve_area / area if area > 0 else np.nan,
            bt3_skeleton_length_um=length,
            bt3_skeleton_density_um_per_mm2=length * 1_000_000.0 / area
            if area > 0
            else np.nan,
        ),
    )


def aggregate(records, boundary_length):
    """Serialize whole-compartment metrics with legacy BT3 column aliases.

    Boundary length is in micrometers. Legacy dermal skeleton density uses
    um/um², whereas compartment-specific densities use um/mm².
    """
    e, d = (records[name].values for name in ("whole_epidermis", "whole_dermis"))

    def divide(a, b):
        return a / b if b > 0 else np.nan

    values = dict(
        epidermal_area_um2=e["area_um2"],
        epidermal_boundary_length_um=boundary_length,
        epidermal_boundary_length_mm=boundary_length / 1000.0,
        epidermal_nerve_area_um2=e["bt3_area_um2"],
        epidermal_nerve_area_per_boundary_length=divide(
            e["bt3_area_um2"], boundary_length
        ),
        epidermal_nerve_area_um2_per_boundary_mm=divide(
            e["bt3_area_um2"], boundary_length / 1000.0
        ),
        epidermal_nerve_area_fraction=e["bt3_area_fraction"],
        epidermal_nerve_skeleton_length_um=e["bt3_skeleton_length_um"],
        epidermal_nerve_skeleton_length_per_boundary_length=divide(
            e["bt3_skeleton_length_um"], boundary_length
        ),
        epidermal_nerve_skeleton_length_um_per_boundary_mm=divide(
            e["bt3_skeleton_length_um"], boundary_length / 1000.0
        ),
        dermal_area_um2=d["area_um2"],
        dermal_nerve_area_um2=d["bt3_area_um2"],
        dermal_nerve_area_fraction=d["bt3_area_fraction"],
        dermal_nerve_skeleton_length_um=d["bt3_skeleton_length_um"],
        dermal_nerve_skeleton_length_density=divide(
            d["bt3_skeleton_length_um"], d["area_um2"]
        ),
    )
    values.update(
        {legacy: values[current] for current, legacy in LEGACY_BT3_METRIC_NAMES.items()}
    )
    result = {"metrics": values}
    for prefix, record in (
        ("epidermal", records["whole_epidermis"]),
        ("dermal", records["whole_dermis"]),
    ):
        for signal_name in ("nerve", "bt3"):
            result[f"{prefix}_{signal_name}_mask"] = record.signal
            result[f"{prefix}_{signal_name}_skeleton"] = record.skeleton
    return result


def macro_metrics(paths, config, scale):
    """Geometry-only reporting; original per-path summation order is retained."""
    exact_lengths = [
        np.linalg.norm(np.diff(x.anatomical_path, axis=0), axis=1).sum() * scale
        for x in paths
    ]
    macro_lengths = [
        np.linalg.norm(np.diff(x.macro_reference, axis=0), axis=1).sum() * scale
        for x in paths
    ]
    displacements = np.concatenate([x.displacement_um for x in paths])
    signed = np.concatenate([x.signed_dermal_displacement_um for x in paths])
    downweighted = np.concatenate([x.downweighted for x in paths])
    return dict(
        anatomical_basal_length_um=float(sum(exact_lengths)),
        macro_reference_length_um=float(sum(macro_lengths)),
        macro_boundary_median_offset_um=float(np.median(displacements)),
        macro_boundary_p95_offset_um=float(np.percentile(displacements, 95)),
        macro_boundary_max_deep_offset_um=float(max(0.0, np.max(signed))),
        macro_boundary_downweighted_fraction=float(np.mean(downweighted)),
        macro_smoothing_um=config.macro_smooth_um,
        dermal_depth_reference_smoothing_um=config.macro_smooth_um,
        subbasal_deep_outlier_um=config.deep_outlier_um,
        subbasal_max_appendage_width_um=config.max_appendage_width_um,
        subbasal_resample_um=config.resample_um,
        subbasal_depth_bands_um=";".join(
            f"{lo:g}-{hi:g}" for lo, hi in config.depth_bands_um
        ),
        subbasal_distance_method="8_connected_dermis_geodesic",
        subbasal_shortcut_maximum_arc_um="unbounded",
        subbasal_algorithm="global_mdl_shortcut_graph_savgol_geodesic_v5",
    )


def whole_records(epidermis, dermis, boundary, signal, scale, fixed=None):
    """Skeletonize whole compartments first and measure the anatomical boundary."""
    records = {}
    for name, roi in (("whole_epidermis", epidermis), ("whole_dermis", dermis)):
        inside = signal & roi
        if fixed is None:
            skeleton, length = calculate_calibrated_skeleton_length(
                inside, scale, componentwise=True
            )
        else:
            skeleton = fixed & roi
            length = measure_skeleton_graph_length(skeleton, scale)
        records[name] = measure(
            roi, inside, skeleton, scale, assigned=True, length=length
        )
    _, boundary_length = calculate_calibrated_skeleton_length(
        boundary, scale, return_full_skeleton=False
    )

    return records, boundary_length


def quantify_regions(
    signal, epidermis, dermis, boundary, scale, *, precomputed_nerve_skeleton=None
):
    f._validate_matching_masks(
        dict(signal=signal, epidermis=epidermis, dermis=dermis, boundary=boundary)
    )
    signal, epidermis, dermis, boundary = (
        np.asarray(x, bool) for x in (signal, epidermis, dermis, boundary)
    )
    if np.any(epidermis & dermis):
        raise ValueError("Epidermis and dermis regions must not overlap.")
    fixed = (
        None
        if precomputed_nerve_skeleton is None
        else np.asarray(precomputed_nerve_skeleton, bool)
    )
    if fixed is not None and fixed.shape != signal.shape:
        raise ValueError("Precomputed nerve skeleton shape does not match regions.")
    records, length = whole_records(epidermis, dermis, boundary, signal, scale, fixed)
    return aggregate(records, length)


def legacy_subbasal_metrics(metrics, bands):
    """Keep the established standalone subbasal CSV column order."""
    fields = (
        "roi_area_um2",
        "bt3_area_um2",
        "bt3_area_fraction",
        "skeleton_length_um",
        "skeleton_density_um_per_mm2",
    )
    prefixes = [
        "subbasal",
        *[f"subbasal_{s._depth_label(lo)}_{s._depth_label(hi)}um" for lo, hi in bands],
    ]
    first = [
        "subbasal_depth_um",
        *[f"{prefix}_{field}" for prefix in prefixes for field in fields],
    ]
    result = {name: metrics[name] for name in first}
    result.update(
        {name: value for name, value in metrics.items() if name not in result}
    )
    return result


def analyze(
    epidermis,
    dermis,
    cleaned,
    boundary,
    paths,
    signal,
    scale,
    config=s.SubbasalConfig(),
    *,
    precomputed_nerve_skeleton=None,
):
    """Return legacy aggregate/subbasal/four results from one ROI registry.

    Inputs and returned arrays must be treated as immutable. A (0, depth) band
    is the same ROI by construction, including empty dermis and disconnected
    sheets. Other depth intervals always receive their own direct measurement.
    """
    config.validate()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pixel size must be positive and finite.")
    masks = dict(
        zip(
            ("epidermis", "dermis", "cleaned", "boundary", "signal"),
            (epidermis, dermis, cleaned, boundary, signal),
        )
    )
    f._validate_matching_masks(masks)
    epidermis, dermis, cleaned, boundary, signal = (
        np.asarray(x, bool) for x in masks.values()
    )
    if np.any(epidermis & dermis):
        raise ValueError("Epidermis and dermis regions must not overlap.")
    if np.any(cleaned & dermis):
        raise RuntimeError(
            "The cleaned Ilastik basal compartment overlaps the reconstructed dermis."
        )
    fixed = (
        None
        if precomputed_nerve_skeleton is None
        else np.asarray(precomputed_nerve_skeleton, bool)
    )
    if fixed is not None and fixed.shape != epidermis.shape:
        raise ValueError("Precomputed nerve skeleton shape does not match regions.")
    records, boundary_length = whole_records(
        epidermis, dermis, boundary, signal, scale, fixed
    )

    macro = s.derive_macro_basal_reference(paths, dermis, scale, config)
    roi, bands, lower = s.build_subbasal_ribbon(
        macro, dermis, scale, config.depth_bands_um, config.depth_um
    )
    macro_mask = rasterize_coordinate_paths(
        [x.macro_reference for x in macro], dermis.shape
    )
    downweighted = s._rasterize_downweighted_paths(macro, dermis.shape)
    basal, upper = cleaned & epidermis, epidermis & ~cleaned
    deep = dermis & ~roi
    f._assert_partition(upper, basal, roi, deep, epidermis, dermis)
    epi_skel, derm_skel = (
        records["whole_epidermis"].skeleton,
        records["whole_dermis"].skeleton,
    )
    for name, mask, skeleton in (
        ("upper_epidermis", upper, epi_skel),
        ("basal_epidermis", basal, epi_skel),
        ("subbasal_dermis", roi, derm_skel),
        ("deep_dermis", deep, derm_skel),
    ):
        records[name] = measure(mask, signal, skeleton, scale)

    submetrics = {
        "subbasal_depth_um": config.depth_um,
        **macro_metrics(macro, config, scale),
    }
    sub_names = {
        "area_um2": "roi_area_um2",
        "bt3_skeleton_length_um": "skeleton_length_um",
        "bt3_skeleton_density_um_per_mm2": "skeleton_density_um_per_mm2",
    }
    depth_records = {"subbasal": records["subbasal_dermis"]}
    for band, mask in bands.items():
        prefix = f"subbasal_{s._depth_label(band[0])}_{s._depth_label(band[1])}um"
        depth_records[prefix] = (
            records["subbasal_dermis"]
            if band == (0.0, float(config.depth_um))
            else measure(mask, signal, derm_skel, scale)
        )
    for prefix, record in depth_records.items():
        submetrics.update(
            {f"{prefix}_{sub_names.get(k, k)}": v for k, v in record.values.items()}
        )
    submetrics = legacy_subbasal_metrics(submetrics, bands)
    subrecord = records["subbasal_dermis"]
    sub = s.SubbasalResult(
        macro,
        macro_mask,
        downweighted,
        lower,
        roi,
        bands,
        subrecord.signal,
        subrecord.skeleton,
        submetrics,
    )
    copied = (
        "subbasal_depth_um",
        "macro_smoothing_um",
        "anatomical_basal_length_um",
        "macro_reference_length_um",
        "macro_boundary_median_offset_um",
        "macro_boundary_p95_offset_um",
        "macro_boundary_max_deep_offset_um",
    )
    fourmetrics = dict(
        four_compartment_analyzed_tissue_area_um2=float(
            np.count_nonzero(epidermis | dermis) * scale**2
        ),
        four_compartment_excluded_tissue_pixels=0.0,
    )
    fourmetrics.update({key: submetrics[key] for key in copied})
    fourmetrics["fraction_boundary_points_downweighted"] = submetrics[
        "macro_boundary_downweighted_fraction"
    ]
    for name in f.COMPARTMENT_NAMES:
        fourmetrics.update({f"{name}_{k}": v for k, v in records[name].values.items()})
    four = f.FourCompartmentResult(
        {k: records[k].mask for k in f.COMPARTMENT_NAMES},
        {k: records[k].signal for k in f.COMPARTMENT_NAMES},
        {k: records[k].skeleton for k in f.COMPARTMENT_NAMES},
        fourmetrics,
    )
    return aggregate(records, boundary_length), sub, four
