"""Measurement-only sub-basal BT3 compartment construction.

The anatomical epidermis-dermis interface remains immutable.  This module
derives a robust, macro-scale reference from its ordered paths and uses that
reference only to construct a dermis-clipped measurement ribbon.
"""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric
from skimage.measure import label, regionprops

from .components import rasterize_coordinate_paths


@dataclass(frozen=True)
class SubbasalConfig:
    """Physical parameters controlling the derived sub-basal compartment."""

    depth_um: float = 20.0
    # Full Savitzky-Golay fitting window along resampled physical arc length.
    # Round to an odd sample count and cap at the available path length.
    macro_smooth_um: float = 40.0
    deep_outlier_um: float = 15.0
    max_appendage_width_um: float = 100.0
    resample_um: float = 1.0
    depth_bands_um: tuple[tuple[float, float], ...] = ((0.0, 20.0),)

    def validate(self):
        scalar_values = {
            "depth_um": self.depth_um,
            "macro_smooth_um": self.macro_smooth_um,
            "deep_outlier_um": self.deep_outlier_um,
            "max_appendage_width_um": self.max_appendage_width_um,
            "resample_um": self.resample_um,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        for lower, upper in self.depth_bands_um:
            if lower < 0 or upper <= lower:
                raise ValueError("Depth bands must satisfy 0 <= lower < upper.")


@dataclass
class MacroPath:
    """One exact path and its uniformly sampled macro-scale derivative."""

    anatomical_path: np.ndarray
    resampled_path: np.ndarray
    macro_reference: np.ndarray
    dermal_normals: np.ndarray
    downweighted: np.ndarray
    signed_dermal_displacement_um: np.ndarray
    displacement_um: np.ndarray


@dataclass
class SubbasalResult:
    """Masks, geometry, metrics, and diagnostics from one section."""

    macro_paths: list[MacroPath]
    macro_reference_mask: np.ndarray
    downweighted_boundary_mask: np.ndarray
    lower_boundary_mask: np.ndarray
    roi: np.ndarray
    band_rois: dict[tuple[float, float], np.ndarray]
    bt3_in_roi: np.ndarray
    skeleton_in_roi: np.ndarray
    metrics: dict[str, float | str]


def _clean_ordered_path(path):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Boundary paths must have shape (n, 2).")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return np.empty((0, 2), dtype=float)
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
    return points[keep]


def resample_ordered_path(path, pixel_size_um, spacing_um):
    """Resample an ordered ``(row, column)`` path at uniform physical spacing."""

    points = _clean_ordered_path(path)
    if len(points) < 2:
        return points
    steps_um = np.linalg.norm(np.diff(points, axis=0), axis=1) * pixel_size_um
    cumulative_um = np.r_[0.0, np.cumsum(steps_um)]
    length_um = cumulative_um[-1]
    if length_um <= 0:
        return points[:1]
    sample_um = np.arange(0.0, length_um, spacing_um)
    if not len(sample_um) or sample_um[-1] < length_um:
        sample_um = np.r_[sample_um, length_um]
    return np.column_stack(
        [np.interp(sample_um, cumulative_um, points[:, axis]) for axis in range(2)]
    )


def _local_polynomial_fairing(points, support_um, spacing_um):
    """Suppress raster noise while preserving the curve's local polynomial shape.

    Use a cubic Savitzky-Golay fit along arc length. Short paths use the
    highest supported polynomial order up to three and an odd fitting window.
    """

    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return points.copy()
    window = max(3, int(round(support_um / spacing_um)))
    if window % 2 == 0:
        window += 1
    maximum_window = len(points) if len(points) % 2 else len(points) - 1
    window = min(window, maximum_window)
    if window < 3:
        return points.copy()
    polynomial_order = min(3, window - 1)
    return savgol_filter(
        points,
        window_length=window,
        polyorder=polynomial_order,
        axis=0,
        mode="interp",
    )


def path_evidence(points, dermis, scale, offsets=None):
    """Return oriented normals and two integer vote arrays for paths of >=2 points."""
    offsets = np.array([0, len(points)]) if offsets is None else offsets
    if np.any(np.diff(offsets) < 2):
        raise ValueError("Normal evidence requires at least two points per segment")
    first, last = (offsets[:-1], offsets[1:] - 1)
    tangent = np.empty_like(points)
    tangent[1:-1] = (points[2:] - points[:-2]) / 2.0
    tangent[first] = points[first + 1] - points[first]
    tangent[last] = points[last] - points[last - 1]
    lengths = np.linalg.norm(tangent, axis=1)
    invalid = lengths < 1e-09
    tangent[invalid], lengths[invalid] = ((0.0, 1.0), 1.0)
    tangent /= lengths[:, None]
    normals = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    scores = []
    for direction in (normals, -normals):
        score = np.zeros(len(first), dtype=np.int64)
        for distance in np.maximum(1.0, np.array((2.0, 5.0, 10.0, 20.0)) / scale):
            sampled = np.rint(points + direction * distance).astype(int)
            valid = (
                (sampled[:, 0] >= 0)
                & (sampled[:, 0] < dermis.shape[0])
                & (sampled[:, 1] >= 0)
                & (sampled[:, 1] < dermis.shape[1])
            )
            hits = np.zeros(len(points), dtype=bool)
            hits[valid] = dermis[sampled[valid, 0], sampled[valid, 1]]
            score += np.add.reduceat(hits, first)
        scores.append(score)
    normals[np.repeat(scores[1] > scores[0], np.diff(offsets))] *= -1.0
    return (normals, scores)


def orient_normals_to_dermis(path, dermis_mask, pixel_size_um):
    """Compatibility entry point for one segment of the shared evidence rule."""
    return path_evidence(path, dermis_mask, pixel_size_um)[0]


def _true_runs(values):
    padded = np.r_[False, np.asarray(values, dtype=bool), False]
    changes = np.diff(padded.astype(np.int8))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def endpoint_tangents(points, span):
    """Reuse endpoint vectors; retain scalar norms to preserve score/order bits."""
    ids = np.arange(len(points))
    before = points - points[np.maximum(0, ids - span)]
    after = points[np.minimum(len(points) - 1, ids + span)] - points
    for field in (before, after):
        for vector in field:
            vector /= max(np.linalg.norm(vector), 1e-09)
    return (before, after)


def packed_depths(points, cumulative, proposals, dermis, scale):
    """Score arc-weighted chord segments through the same final-path evidence rule."""
    starts, stops = np.array([(x[0], x[1]) for x in proposals], dtype=int).T
    counts = stops - starts + 1
    offsets = np.r_[0, np.cumsum(counts)]
    owner = np.repeat(np.arange(len(starts)), counts)
    indices = np.arange(offsets[-1]) - offsets[owner] + starts[owner]
    fractions = (cumulative[indices] - cumulative[starts[owner]]) / np.maximum(
        cumulative[stops[owner]] - cumulative[starts[owner]], 1e-09
    )
    chords = points[starts[owner]] + fractions[:, None] * (
        points[stops[owner]] - points[starts[owner]]
    )
    normals, scores = path_evidence(chords, dermis, scale, offsets)
    signed = np.einsum("ij,ij->i", points[indices] - chords, normals) * scale
    depths = np.maximum.reduceat(signed, offsets[:-1])
    return depths


def _candidate_return_shortcuts(
    sampled, dermis_mask, pixel_size_um, deep_outlier_um, max_appendage_width_um
):
    if len(sampled) < 3:
        return []
    steps_um = np.linalg.norm(np.diff(sampled, axis=0), axis=1) * pixel_size_um
    cumulative = np.r_[0.0, np.cumsum(steps_um)]
    radius = max_appendage_width_um / pixel_size_um
    tree = cKDTree(sampled)
    span_um = min(10.0, max_appendage_width_um * 0.25)
    span = max(2, int(round(span_um / max(np.median(steps_um), 1e-09))))
    before, after = endpoint_tangents(sampled, span)
    proposals = []
    for start in range(len(sampled) - 2):
        best = []
        for stop in tree.query_ball_point(sampled[start], radius):
            if stop < start + 2:
                continue
            chord = sampled[stop] - sampled[start]
            length = float(np.linalg.norm(chord))
            if length < 1e-09:
                continue
            chord_um = length * pixel_size_um
            arc = cumulative[stop] - cumulative[start]
            excess = arc - chord_um
            gain = excess - chord_um - 2.0 * deep_outlier_um
            if gain <= 0:
                continue
            direction = chord / length
            alignment = min(
                float(before[start] @ direction), float(after[stop] @ direction)
            )
            if alignment < 0:
                continue
            score = gain * (0.5 + 0.5 * alignment)
            best.append((score, chord_um, stop))
        by_gain = sorted(best, reverse=True)[:8]
        by_efficiency = sorted(
            best, key=lambda x: x[0] / max(x[1], pixel_size_um), reverse=True
        )[:8]
        proposals.extend(
            (
                (start, stop, score)
                for stop, score in {
                    (stop, score) for score, _, stop in (*by_gain, *by_efficiency)
                }
            )
        )
    candidates = []
    first = 0
    # Bound transient evidence storage; a longer individual chord stays intact.
    while first < len(proposals):
        last, size = (first, 0)
        while last < len(proposals) and (
            last == first or size + proposals[last][1] - proposals[last][0] + 1 <= 32000
        ):
            size += proposals[last][1] - proposals[last][0] + 1
            last += 1
        batch = proposals[first:last]
        depths = packed_depths(sampled, cumulative, batch, dermis_mask, pixel_size_um)
        for (start, stop, score), maximum in zip(batch, depths):
            depth = float(maximum)
            if depth >= deep_outlier_um:
                candidates.append((start, stop, score + (depth - deep_outlier_um)))
        first = last
    return candidates


def _select_global_shortcuts(point_count, candidates):
    """Select the maximum-value non-overlapping set of shortcut DAG edges."""

    if not candidates:
        return [], np.zeros(point_count, dtype=bool)
    ordered = sorted(candidates, key=lambda item: (item[1], item[0]))
    stops = [item[1] for item in ordered]
    predecessors = [
        bisect_right(stops, start, hi=index) - 1
        for index, (start, _, _) in enumerate(ordered)
    ]
    optimum = np.zeros(len(ordered) + 1, dtype=float)
    choose = np.zeros(len(ordered), dtype=bool)
    for index, (_, _, score) in enumerate(ordered):
        take = score + optimum[predecessors[index] + 1]
        skip = optimum[index]
        if take > skip:
            optimum[index + 1] = take
            choose[index] = True
        else:
            optimum[index + 1] = skip

    selected = []
    index = len(ordered) - 1
    while index >= 0:
        if choose[index] and (
            ordered[index][2] + optimum[predecessors[index] + 1] > optimum[index]
        ):
            selected.append(ordered[index])
            index = predecessors[index]
        else:
            index -= 1
    selected.reverse()
    classified = np.zeros(point_count, dtype=bool)
    for start, stop, _ in selected:
        classified[start + 1 : stop] = True
    return selected, classified


def derive_macro_basal_reference(
    anatomical_paths,
    dermis_mask,
    pixel_size_um,
    config=SubbasalConfig(),
):
    """Derive a globally optimized, curvature-preserving macro reference."""

    config.validate()
    dermis = np.asarray(dermis_mask, dtype=bool)
    if dermis.ndim != 2:
        raise ValueError("Dermis mask must be two-dimensional.")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Pixel size must be positive and finite.")
    macro_paths = []
    for raw_path in anatomical_paths:
        anatomical = _clean_ordered_path(raw_path)
        sampled = resample_ordered_path(anatomical, pixel_size_um, config.resample_um)
        if len(sampled) < 2:
            continue
        candidates = _candidate_return_shortcuts(
            sampled,
            dermis,
            pixel_size_um,
            config.deep_outlier_um,
            config.max_appendage_width_um,
        )
        _, classified = _select_global_shortcuts(len(sampled), candidates)
        fitting_points = sampled[~classified]
        uniform_fitting_points = resample_ordered_path(
            fitting_points,
            pixel_size_um,
            config.resample_um,
        )
        macro = _local_polynomial_fairing(
            uniform_fitting_points,
            config.macro_smooth_um,
            config.resample_um,
        )
        normals = orient_normals_to_dermis(macro, dermis, pixel_size_um)
        nearest_macro = cKDTree(macro).query(sampled)[1]
        offsets = sampled - macro[nearest_macro]
        signed = np.einsum("ij,ij->i", offsets, normals[nearest_macro]) * pixel_size_um
        displacement = np.linalg.norm(offsets, axis=1) * pixel_size_um
        macro_paths.append(
            MacroPath(
                anatomical,
                sampled,
                macro,
                normals,
                classified,
                signed,
                displacement,
            )
        )
    if not macro_paths:
        raise ValueError("No usable ordered anatomical basal paths were supplied.")
    return macro_paths


def _rasterize_downweighted_paths(macro_paths, shape):
    paths = []
    for item in macro_paths:
        for start, stop in _true_runs(item.downweighted):
            if stop - start >= 2:
                paths.append(item.resampled_path[start:stop])
    return rasterize_coordinate_paths(paths, shape)


def _dermis_geodesic_distance(
    reference_mask,
    dermis_mask,
    pixel_size_um,
    maximum_depth_um,
):
    """Return distance from a reference while permitting travel only in dermis.

    A Euclidean prefilter supplies a mathematically safe finite search tube:
    no point farther than ``maximum_depth_um`` in straight-line distance can
    be within that depth geodesically. Within that tube, MCP propagation is
    restricted to valid dermis, so folds, holes, epidermis, and background
    cannot provide shortcuts. Reference-adjacent dermal pixels are initialized
    one calibrated pixel from the rasterized boundary.
    """

    reference = np.asarray(reference_mask, dtype=bool)
    dermis = np.asarray(dermis_mask, dtype=bool)
    if reference.shape != dermis.shape:
        raise ValueError("Reference and dermis masks must have matching shapes.")
    if not np.any(reference):
        raise ValueError("The dermal-depth reference boundary is empty.")

    straight_distance_um = distance_transform_edt(
        ~reference,
        sampling=(pixel_size_um, pixel_size_um),
    )
    candidate = dermis & (straight_distance_um <= maximum_depth_um)
    reference_neighbors = binary_dilation(
        reference,
        structure=np.ones((3, 3), dtype=bool),
    )
    seeds = reference_neighbors & candidate
    distances_um = np.full(dermis.shape, np.inf, dtype=np.float32)
    if not np.any(seeds):
        return distances_um

    candidate_labels = label(candidate, connectivity=2)
    seeded_ids = set(np.unique(candidate_labels[seeds]).tolist()) - {0}
    for region in regionprops(candidate_labels):
        if int(region.label) not in seeded_ids:
            continue
        r0, c0, r1, c1 = region.bbox
        local_candidate = candidate_labels[r0:r1, c0:c1] == region.label
        local_seeds = seeds[r0:r1, c0:c1] & local_candidate
        starts = [tuple(point) for point in np.argwhere(local_seeds)]
        costs = np.where(local_candidate, 1.0, np.inf)
        local_costs, _ = MCP_Geometric(costs, fully_connected=True).find_costs(starts)
        # The start pixels are adjacent to, rather than centered on, the
        # rasterized reference. Counting one calibrated pixel prevents the
        # band from extending one pixel beyond the requested physical depth.
        physical = local_costs * pixel_size_um + pixel_size_um
        local_output = distances_um[r0:r1, c0:c1]
        local_output[local_candidate] = physical[local_candidate].astype(np.float32)
    return distances_um


def build_subbasal_ribbon(
    macro_paths,
    dermis_mask,
    pixel_size_um,
    depth_bands_um,
    depth_um,
):
    """Build nested depth bands by geodesic propagation within valid dermis."""

    dermis = np.asarray(dermis_mask, dtype=bool)
    requested_depths = {float(depth_um)}
    requested_depths.update(float(value) for band in depth_bands_um for value in band)
    reference_mask = rasterize_coordinate_paths(
        [item.macro_reference for item in macro_paths],
        dermis.shape,
    )
    distances_um = _dermis_geodesic_distance(
        reference_mask,
        dermis,
        pixel_size_um,
        max(requested_depths),
    )
    cumulative = {
        depth: (
            np.zeros_like(dermis) if depth == 0 else (distances_um <= depth) & dermis
        )
        for depth in sorted(requested_depths)
    }
    roi = cumulative[float(depth_um)]
    band_rois = {}
    for lower, upper in depth_bands_um:
        lower = float(lower)
        upper = float(upper)
        band_rois[(lower, upper)] = cumulative[upper] & ~cumulative[lower]
    lower_boundary = roi & binary_dilation(
        dermis & ~roi,
        structure=np.ones((3, 3), dtype=bool),
    )
    return roi, band_rois, lower_boundary


def _depth_label(value):
    return f"{value:g}".replace(".", "p")
