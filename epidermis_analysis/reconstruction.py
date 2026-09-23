"""Frozen production surface-polygon epidermal compartment reconstruction.

Consumes frozen cleanup/course masks.  The medial course is used only to
associate, order, and close local polygons; it is never reported as the final
anatomical interface or normalization boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .components import (
    major_geodesic_path as _major_geodesic_path,
    outward_tangent as _outward_tangent,
    bezier_bridge as _bezier_bridge,
    outer_ordered_contour as _outer_ordered_contour,
    circular_arc as _circular_arc,
    calibrated_path_length,
    rasterize_coordinate_paths as rasterize_paths,
)


@dataclass
class InterfaceFragment:
    fragment_id: int
    polygon_id: int
    tissue_component_id: int
    trajectory_network_id: int
    coordinates: np.ndarray
    length_um: float
    s_start_um: float
    s_end_um: float
    s_mid_um: float
    fragment_class: str


@dataclass
class InterfaceReconstruction:
    epidermis_region: np.ndarray
    dermis_region: np.ndarray
    course_guide: np.ndarray
    superficial_surface: np.ndarray
    closure_mask: np.ndarray
    surface_polygon_mask: np.ndarray
    observed_interface_mask: np.ndarray
    candidate_bridge_mask: np.ndarray
    inferred_bridge_mask: np.ndarray
    final_boundary_mask: np.ndarray
    fragment_class_image: np.ndarray
    consistency_supported_mask: np.ndarray
    fragments: list[InterfaceFragment]
    final_paths: list[np.ndarray]
    bridge_rows: list[dict]
    diagnostics: dict


def _bridge_ordered_fragments(
    fragments,
    whole,
    interface_support,
    pixel_size_um,
    maximum_gap_um,
    maximum_support_p90_um=10.0,
):
    """Join compatible fragment endpoints using an evidence-weighted graph.

    A cleaned epidermal course can legitimately be split into several guide
    networks. Restricting bridges to consecutive fragments within one guide
    therefore leaves anatomical gaps at network boundaries. Here every
    accepted fragment endpoint is a graph vertex. Candidate graph edges must
    remain in one whole-tissue component, respect both endpoint tangents, stay
    close to the pre-bridge epidermis/dermis adjacency, and avoid
    existing paths. Greedy selection in increasing cost order enforces at
    most one bridge per endpoint and prevents cycles; it does not solve a
    global minimum-cost matching problem.
    """
    accepted_classes = {
        "major_anatomical_interface",
        "minor_likely_anatomical_interface",
    }
    accepted_fragments = [
        fragment
        for fragment in fragments
        if fragment.fragment_class in accepted_classes
    ]
    observed_points = (
        np.vstack([fragment.coordinates for fragment in accepted_fragments])
        if accepted_fragments
        else np.empty((0, 2))
    )
    observed_tree = cKDTree(observed_points) if len(observed_points) else None
    support_points = np.argwhere(np.asarray(interface_support, dtype=bool))
    support_tree = cKDTree(support_points) if len(support_points) else None

    endpoints = []
    for fragment in accepted_fragments:
        for endpoint_index, endpoint_name in ((0, "start"), (-1, "end")):
            endpoints.append(
                {
                    "fragment": fragment,
                    "endpoint_index": endpoint_index,
                    "endpoint_name": endpoint_name,
                    "point": fragment.coordinates[endpoint_index],
                    "tangent": _outward_tangent(fragment.coordinates, endpoint_index),
                }
            )

    candidate_paths = []
    candidate_edges = []
    rows = []
    for first_index, first_endpoint in enumerate(endpoints):
        first = first_endpoint["fragment"]
        for second_index in range(first_index + 1, len(endpoints)):
            second_endpoint = endpoints[second_index]
            second = second_endpoint["fragment"]
            if (
                first.fragment_id == second.fragment_id
                or first.tissue_component_id != second.tissue_component_id
            ):
                continue
            a = first_endpoint["point"]
            b = second_endpoint["point"]
            delta = b - a
            gap_pixels = float(np.linalg.norm(delta))
            gap_um = gap_pixels * pixel_size_um
            if not gap_pixels or gap_um > maximum_gap_um:
                continue
            direction = delta / gap_pixels
            tangent_a = first_endpoint["tangent"]
            tangent_b = second_endpoint["tangent"]
            alignment_a = float(tangent_a @ direction)
            alignment_b = float(tangent_b @ -direction)
            row = {
                "guide_id": (
                    str(first.trajectory_network_id)
                    if first.trajectory_network_id == second.trajectory_network_id
                    else f"{first.trajectory_network_id}->{second.trajectory_network_id}"
                ),
                "fragment_a": first.fragment_id,
                "fragment_b": second.fragment_id,
                "endpoint_a": first_endpoint["endpoint_name"],
                "endpoint_b": second_endpoint["endpoint_name"],
                "cross_network": bool(
                    first.trajectory_network_id != second.trajectory_network_id
                ),
                "gap_um": gap_um,
                "alignment_a": alignment_a,
                "alignment_b": alignment_b,
                "accepted": False,
                "reason": "",
            }
            if min(alignment_a, alignment_b) < 0.2:
                row["reason"] = "tangent_mismatch"
                rows.append(row)
                continue

            # Cubic endpoint tangents are unstable when endpoints are only a
            # few pixels apart. A straight sub-10-um link supplies the
            # minimum-curvature candidate; the support and topology tests
            # below still decide whether it is anatomically admissible.
            if gap_um <= 10.0:
                count = max(2, int(np.ceil(gap_pixels / 0.5)) + 1)
                bridge = np.linspace(a, b, count)
            else:
                bridge = _bezier_bridge(a, tangent_a, b, tangent_b)
            candidate_paths.append(bridge)
            differences = np.diff(bridge, axis=0)
            lengths = np.linalg.norm(differences, axis=1) * pixel_size_um
            angles = np.unwrap(np.arctan2(differences[:, 0], differences[:, 1]))
            curvature = np.abs(np.diff(angles)) / np.maximum(
                (lengths[:-1] + lengths[1:]) / 2, 1e-6
            )
            maximum_curvature = float(curvature.max()) if len(curvature) else 0.0
            row["maximum_curvature_per_um"] = maximum_curvature
            if maximum_curvature > 0.5:
                row["reason"] = "curvature_too_high"
                rows.append(row)
                continue

            rounded = np.rint(bridge).astype(int)
            rounded[:, 0] = np.clip(rounded[:, 0], 0, whole.shape[0] - 1)
            rounded[:, 1] = np.clip(rounded[:, 1], 0, whole.shape[1] - 1)
            if not np.all(whole[rounded[:, 0], rounded[:, 1]]):
                row["reason"] = "outside_whole_tissue"
                rows.append(row)
                continue
            if support_tree is None:
                row["reason"] = "no_interface_support"
                rows.append(row)
                continue
            support_distances_um = support_tree.query(bridge)[0] * pixel_size_um
            support_p90_um = float(np.percentile(support_distances_um, 90))
            row["support_mean_distance_um"] = float(np.mean(support_distances_um))
            row["support_p90_distance_um"] = support_p90_um
            if support_p90_um > maximum_support_p90_um:
                row["reason"] = "insufficient_interface_support"
                rows.append(row)
                continue

            minor_endpoint_count = int(
                first.fragment_class == "minor_likely_anatomical_interface"
            ) + int(second.fragment_class == "minor_likely_anatomical_interface")
            tangent_penalty = (2.0 - alignment_a - alignment_b) * 15.0
            score = (
                gap_um
                + 4.0 * support_p90_um
                + tangent_penalty
                + 50.0 * minor_endpoint_count
            )
            row["selection_cost"] = float(score)
            candidate_edges.append(
                {
                    "score": float(score),
                    "endpoint_a": first_index,
                    "endpoint_b": second_index,
                    "fragment_a": first.fragment_id,
                    "fragment_b": second.fragment_id,
                    "bridge": bridge,
                    "row": row,
                }
            )

    parent = {
        fragment.fragment_id: fragment.fragment_id for fragment in accepted_fragments
    }

    def find(fragment_id):
        while parent[fragment_id] != fragment_id:
            parent[fragment_id] = parent[parent[fragment_id]]
            fragment_id = parent[fragment_id]
        return fragment_id

    def union(first_id, second_id):
        first_root = find(first_id)
        second_root = find(second_id)
        if first_root != second_root:
            parent[second_root] = first_root

    used_endpoints = set()
    accepted_paths = []
    for edge in sorted(candidate_edges, key=lambda item: item["score"]):
        row = edge["row"]
        endpoint_a = edge["endpoint_a"]
        endpoint_b = edge["endpoint_b"]
        if endpoint_a in used_endpoints or endpoint_b in used_endpoints:
            row["reason"] = "endpoint_already_joined"
            rows.append(row)
            continue
        if find(edge["fragment_a"]) == find(edge["fragment_b"]):
            row["reason"] = "would_create_cycle"
            rows.append(row)
            continue
        bridge = edge["bridge"]
        interior = bridge[4:-4]
        if (
            observed_tree is not None
            and len(interior)
            and np.any(observed_tree.query(interior)[0] < 1.5)
        ):
            row["reason"] = "crosses_existing_interface"
            rows.append(row)
            continue
        if accepted_paths:
            accepted_tree = cKDTree(np.vstack(accepted_paths))
            if len(interior) and np.any(accepted_tree.query(interior)[0] < 1.5):
                row["reason"] = "crosses_existing_bridge"
                rows.append(row)
                continue
        accepted_paths.append(bridge)
        used_endpoints.update((endpoint_a, endpoint_b))
        union(edge["fragment_a"], edge["fragment_b"])
        row["accepted"] = True
        row["reason"] = "accepted"
        row["bridge_length_um"] = calibrated_path_length(bridge, pixel_size_um)
        rows.append(row)

    return (
        rasterize_paths(candidate_paths, whole.shape),
        rasterize_paths(accepted_paths, whole.shape),
        rows,
    )


def _suppress_redundant_guide_interval_excursions(
    fragments: list[InterfaceFragment],
    backbone_count: int = 2,
    maximum_excursion_length_um: float = 150.0,
    minimum_interval_overlap_fraction: float = 0.90,
) -> list[int]:
    """Reject short parallel excursions without changing reconstructed geometry.

    The dominant fragments within each course-guide network provisionally define
    the primary interface backbone.  A non-backbone fragment is suppressed only
    when it is short *and* almost its entire guide arc-length interval is already
    represented by that backbone.  Fragments extending into a previously
    uncovered guide interval remain eligible for normal bridge evaluation.

    This is deliberately an interface-graph rule: it neither changes polygons
    nor uses fragment length alone to decide anatomical validity.
    """
    accepted = {
        "major_anatomical_interface",
        "minor_likely_anatomical_interface",
    }
    suppressed: list[int] = []
    guide_ids = sorted({f.trajectory_network_id for f in fragments})
    for guide_id in guide_ids:
        candidates = [
            f
            for f in fragments
            if f.trajectory_network_id == guide_id and f.fragment_class in accepted
        ]
        if len(candidates) <= backbone_count:
            continue
        backbone = sorted(candidates, key=lambda f: f.length_um, reverse=True)[
            :backbone_count
        ]
        backbone_ids = {f.fragment_id for f in backbone}
        intervals = [(f.s_start_um, f.s_end_um) for f in backbone]
        for fragment in candidates:
            if fragment.fragment_id in backbone_ids:
                continue
            span = max(fragment.s_end_um - fragment.s_start_um, 1e-9)
            overlap = sum(
                max(0.0, min(fragment.s_end_um, end) - max(fragment.s_start_um, start))
                for start, end in intervals
            )
            overlap_fraction = min(overlap, span) / span
            if (
                fragment.length_um <= maximum_excursion_length_um
                and overlap_fraction >= minimum_interval_overlap_fraction
            ):
                fragment.fragment_class = "ambiguous_incidental_fragment"
                suppressed.append(fragment.fragment_id)
    return suppressed


def _surface_cache_for_component(
    component_crop: np.ndarray, offset: np.ndarray, seed_crop: np.ndarray
):
    contour = _outer_ordered_contour(component_crop)
    if not len(contour):
        return None
    seed_points = np.argwhere(seed_crop)
    if not len(seed_points):
        return None
    distances = cKDTree(seed_points).query(contour)[0]
    indices = np.flatnonzero(distances <= 2.0)
    if not len(indices):
        indices = np.argsort(distances)[: max(2, len(contour) // 20)]
    return contour + offset, indices, distances


def _fill_closed_ring_fast(ring: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize and fill a closed ordered ring in linear image time.

    ``skimage.draw.polygon`` becomes very slow for the tens-of-thousands of
    vertices present in stitched sections.  The ring is already an ordered
    pixel/subpixel path, so rasterizing its edges and filling the enclosed area
    preserves its geometry while avoiding vertex-by-scanline polygon work.
    """
    points = np.asarray(ring, dtype=float)
    if len(points) < 3:
        return np.zeros(shape, dtype=bool)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    edge = rasterize_paths([points], shape)
    return binary_fill_holes(edge)


def partition_regions_from_boundary_paths(
    boundary_paths: list[np.ndarray] | tuple[np.ndarray, ...],
    whole_tissue: np.ndarray,
    superficial_surface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition fixed whole tissue using explicit dermal-facing paths.

    This exposes the same surface-contour closure and fast polygon fill used
    by production reconstruction, but accepts already-traced boundary paths.
    It performs no segmentation, cleanup, bridging, or path modification.
    Paths use production ``(row, column)`` coordinates.
    """
    whole = np.asarray(whole_tissue, dtype=bool)
    surface = np.asarray(superficial_surface, dtype=bool)
    if whole.ndim != 2 or surface.shape != whole.shape:
        raise ValueError(
            "Whole tissue and superficial surface must be matching 2-D masks."
        )
    whole_labels = label(whole, connectivity=2)
    whole_regions = {int(region.label): region for region in regionprops(whole_labels)}
    surface_cache = {}
    epidermis = np.zeros_like(whole)
    accepted_path_count = 0
    for raw_path in boundary_paths:
        path = np.asarray(raw_path, dtype=float)
        path = path[np.all(np.isfinite(path), axis=1)]
        if len(path) < 2:
            continue
        rounded = np.rint(path).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, whole.shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, whole.shape[1] - 1)
        tissue_ids = whole_labels[rounded[:, 0], rounded[:, 1]]
        tissue_ids = tissue_ids[tissue_ids != 0]
        if not len(tissue_ids):
            continue
        ids, counts = np.unique(tissue_ids, return_counts=True)
        tissue_id = int(ids[int(np.argmax(counts))])
        whole_region = whole_regions.get(tissue_id)
        if whole_region is None:
            continue
        r0, c0, r1, c1 = whole_region.bbox
        component_crop = whole_labels[r0:r1, c0:c1] == tissue_id
        cached = surface_cache.get(tissue_id)
        if cached is None:
            cached = _surface_cache_for_component(
                component_crop,
                np.array([r0, c0]),
                surface[r0:r1, c0:c1] & component_crop,
            )
            surface_cache[tissue_id] = cached
        if cached is None:
            continue
        contour, surface_indices, surface_distances = cached
        surface_arc, _, _ = _choose_surface_arc(
            contour,
            surface_indices,
            surface_distances,
            path[0],
            path[-1],
        )
        ring = np.vstack((path, surface_arc[::-1]))
        pr0 = max(0, int(np.floor(ring[:, 0].min())) - 2)
        pr1 = min(whole.shape[0], int(np.ceil(ring[:, 0].max())) + 3)
        pc0 = max(0, int(np.floor(ring[:, 1].min())) - 2)
        pc1 = min(whole.shape[1], int(np.ceil(ring[:, 1].max())) + 3)
        candidate = _fill_closed_ring_fast(
            ring - np.array([pr0, pc0]),
            (pr1 - pr0, pc1 - pc0),
        )
        candidate &= whole_labels[pr0:pr1, pc0:pc1] == tissue_id
        if not np.any(candidate):
            continue
        epidermis[pr0:pr1, pc0:pc1] |= candidate
        accepted_path_count += 1
    if accepted_path_count == 0 or not np.any(epidermis):
        raise ValueError("No boundary path produced a surface-closed epidermis region.")
    epidermis &= whole
    return epidermis, whole & ~epidermis


def _choose_surface_arc(
    contour,
    surface_indices,
    distances,
    endpoint_a,
    endpoint_b,
    anatomical_support_distances=None,
):
    # Closure endpoints must land on the established superficial support.
    # An internal basal-path endpoint can be geometrically closer to the deep
    # dermal exterior than to the true surface; snapping to the unrestricted
    # contour then floods the intervening dermis into the epidermis polygon.
    # Surface-supported snapping is a hard anatomical constraint, not merely
    # a distance preference.
    supported = np.asarray(surface_indices, dtype=int)
    if not len(supported):
        raise ValueError("No superficial contour support is available.")
    tree = cKDTree(contour[supported])
    ia = int(supported[int(tree.query(endpoint_a)[1])])
    ib = int(supported[int(tree.query(endpoint_b)[1])])
    forward = _circular_arc(contour, ia, ib)
    reverse = _circular_arc(contour, ib, ia)[::-1]
    fi = np.mod(np.arange(ia, ia + len(forward)), len(contour))
    ri = np.mod(np.arange(ib, ib + len(reverse)), len(contour))
    forward_surface_score = float(np.mean(distances[fi]))
    reverse_surface_score = float(np.mean(distances[ri]))
    # The established superficial trace remains the primary evidence. When
    # both arcs are almost equally close to it, prefer the arc that stays near
    # the cleaned Ilastik epidermal band; this resolves local folds without
    # turning band proximity into a global orientation assumption.
    if (
        anatomical_support_distances is not None
        and abs(forward_surface_score - reverse_surface_score) <= 2.0
    ):
        support = np.asarray(anatomical_support_distances, dtype=float)
        choose_forward = float(np.mean(support[fi])) <= float(np.mean(support[ri]))
    else:
        choose_forward = forward_surface_score <= reverse_surface_score
    return (forward if choose_forward else reverse), ia, ib


def _path_topology(path_mask: np.ndarray) -> tuple[int, int]:
    points = np.argwhere(path_mask)
    lookup = set(map(tuple, points.tolist()))
    degrees = []
    for r, c in points:
        degrees.append(
            sum(
                (int(r + dr), int(c + dc)) in lookup
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr or dc
            )
        )
    return sum(d == 1 for d in degrees), sum(d >= 3 for d in degrees)


def _extract_paths_cropped(mask: np.ndarray, pixel_size_um: float):
    skeleton = skeletonize(np.asarray(mask, bool))
    labels = label(skeleton, connectivity=2)
    paths = []
    for region in regionprops(labels):
        r0, c0, r1, c1 = region.bbox
        local = labels[r0:r1, c0:c1] == region.label
        path = _major_geodesic_path(local)
        if len(path) >= 2:
            paths.append(path + np.array([r0, c0]))
    paths.sort(key=lambda p: calibrated_path_length(p, pixel_size_um), reverse=True)
    return paths


def _maximum_unsupported_run_um(paths, supported_mask, pixel_size_um):
    maximum = 0.0
    for path in paths:
        points = np.rint(path).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, supported_mask.shape[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, supported_mask.shape[1] - 1)
        flags = ~supported_mask[points[:, 0], points[:, 1]]
        start = None
        for index, flag in enumerate(np.r_[flags, False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                maximum = max(
                    maximum, calibrated_path_length(path[start:index], pixel_size_um)
                )
                start = None
    return float(maximum)


def _surface_seeded_topological_partition(
    whole_tissue: np.ndarray,
    basal_band: np.ndarray,
    course_guide: np.ndarray,
    closure_mask: np.ndarray,
    superficial_surface: np.ndarray,
    provisional_epidermis: np.ndarray,
):
    """Project provisional sides onto hard-seeded connected graph regions.

    Local surface polygons are useful for constructing the interface, but a
    tight fold can make an individually plausible polygon select the wrong
    anatomical side.  The global anatomy supplies a stronger constraint: the
    upper compartment must remain connected to the established superficial
    trace through pixels already classified as upper; dermis must likewise
    remain connected to the non-superficial tissue exterior through pixels
    already classified as dermis.

    Label the upper and dermal candidate regions separately, using the basal
    band and thickened guide/closures as barriers. Candidate regions without
    contact to their own seed are reassigned to the other compartment. If one
    side has no seed contact anywhere, keep its provisional labels. These
    connectivity rules do not establish biological correctness; inspect QC.
    The routine uses supplied surface seeds, while the upstream surface trace
    assumes epidermis is at the top of the image.
    """
    whole = np.asarray(whole_tissue, dtype=bool)
    basal = np.asarray(basal_band, dtype=bool) & whole
    guide = np.asarray(course_guide, dtype=bool) & whole
    closures = np.asarray(closure_mask, dtype=bool) & whole
    surface = np.asarray(superficial_surface, dtype=bool) & whole
    provisional = np.asarray(provisional_epidermis, dtype=bool) & whole
    if not (
        basal.shape
        == guide.shape
        == closures.shape
        == surface.shape
        == provisional.shape
        == whole.shape
    ):
        raise ValueError("Topological partition masks must have matching shapes.")

    # A one-pixel diagonal line does not separate an 8-connected pixel graph.
    # Thicken only the inferred guide/closures; the observed basal band already
    # has anatomical width and is retained exactly in the final compartment.
    auxiliary_barrier = (
        binary_dilation(
            guide | closures,
            structure=np.ones((3, 3), dtype=bool),
        )
        & whole
    )
    open_domain = whole & ~basal & ~auxiliary_barrier
    upper_candidate = provisional & open_domain
    dermal_candidate = (whole & ~provisional) & open_domain

    upper_labels = label(upper_candidate, connectivity=2)
    upper_component_count = int(upper_labels.max())
    upper_seed_pixels = (
        binary_dilation(surface, structure=np.ones((3, 3), dtype=bool))
        & upper_candidate
    )
    upper_component_ids = np.unique(upper_labels[upper_seed_pixels])
    upper_component_ids = upper_component_ids[upper_component_ids != 0]
    upper_lookup = np.zeros(upper_component_count + 1, dtype=bool)
    # If rasterization leaves no exact seed contact, preserving the provisional
    # label is safer than deleting the entire upper compartment.
    if len(upper_component_ids):
        upper_lookup[upper_component_ids] = True
    else:
        upper_lookup[:] = True
    supported_upper = upper_lookup[upper_labels] & upper_candidate

    tissue_exterior = whole & ~binary_erosion(
        whole,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    non_superficial_exterior = tissue_exterior & ~binary_dilation(
        surface,
        structure=np.ones((7, 7), dtype=bool),
    )
    dermal_labels = label(dermal_candidate, connectivity=2)
    dermal_component_count = int(dermal_labels.max())
    dermal_seed_pixels = non_superficial_exterior & dermal_candidate
    dermal_component_ids = np.unique(dermal_labels[dermal_seed_pixels])
    dermal_component_ids = dermal_component_ids[dermal_component_ids != 0]
    dermal_lookup = np.zeros(dermal_component_count + 1, dtype=bool)
    if len(dermal_component_ids):
        dermal_lookup[dermal_component_ids] = True
    else:
        dermal_lookup[:] = True
    supported_dermis = dermal_lookup[dermal_labels] & dermal_candidate

    # Components disconnected from their own anatomical seed are label
    # inversions: delete upper islands on the dermal side and fill dermal
    # islands on the surface side. Basal and raster-seal pixels keep their
    # authoritative/provisional assignments.
    unsupported_dermis = dermal_candidate & ~supported_dermis
    epidermis = (
        basal | supported_upper | unsupported_dermis | (auxiliary_barrier & provisional)
    )
    epidermis &= whole
    dermis = whole & ~epidermis
    diagnostics = {
        "topological_traversable_component_count": int(
            upper_component_count + dermal_component_count
        ),
        "topological_surface_seed_component_count": int(len(upper_component_ids)),
        "topological_deep_seed_component_count": int(len(dermal_component_ids)),
        "topological_unseeded_component_count": int(
            (upper_component_count - len(upper_component_ids))
            + (dermal_component_count - len(dermal_component_ids))
        ),
        "topological_barrier_pixels": int(np.count_nonzero(basal | auxiliary_barrier)),
        "topological_epidermis_to_dermis_pixels": int(
            np.count_nonzero(provisional & ~epidermis)
        ),
        "topological_dermis_to_epidermis_pixels": int(
            np.count_nonzero((whole & ~provisional) & epidermis)
        ),
    }
    return epidermis, dermis, diagnostics


def reconstruct_explicit_interface(
    cleaned_band: np.ndarray,
    observed_course: np.ndarray,
    course_connectors: np.ndarray,
    whole_tissue: np.ndarray,
    superficial_surface: np.ndarray,
    pixel_size_um: float,
    minimum_major_fragment_um: float = 20.0,
    maximum_bridge_gap_um: float = 150.0,
) -> InterfaceReconstruction:
    """Build surface-seeded local polygons and retain interface provenance.

    ``superficial_surface`` supplies the estimated top surface used to select
    contour arcs and close polygons. This routine does not retrace a column-wise
    top edge; the production caller supplies that image-up scaffold. Local
    contour following does not make the full pipeline orientation-independent.
    """
    band = np.asarray(cleaned_band, bool)
    whole = np.asarray(whole_tissue, bool)
    surface = np.asarray(superficial_surface, bool)
    if surface.shape != whole.shape:
        raise ValueError(
            "Whole tissue and superficial surface must be matching 2-D masks."
        )
    seeds = surface & whole
    if not np.any(seeds):
        raise ValueError("The supplied whole-tissue superficial surface is empty.")
    guide = skeletonize(
        np.asarray(observed_course, bool) | np.asarray(course_connectors, bool)
    )
    guide_labels = label(guide, connectivity=2)
    whole_labels = label(whole, connectivity=2)
    band_labels = label(band, connectivity=2)
    epidermis = band & whole
    polygons = np.zeros_like(whole)
    known_surface = np.zeros_like(whole)
    known_closure = np.zeros_like(whole)
    raw_fragments = []
    unresolved = 0
    polygon_id = 0
    whole_regions = {int(region.label): region for region in regionprops(whole_labels)}
    tissue_surface_cache = {}

    for network in regionprops(guide_labels):
        network_id = int(network.label)
        network_mask = guide_labels == network_id
        path = _major_geodesic_path(network_mask)
        if len(path) < 2:
            unresolved += 1
            continue
        mid = np.rint(path[len(path) // 2]).astype(int)
        tissue_id = int(
            whole_labels[
                np.clip(mid[0], 0, whole.shape[0] - 1),
                np.clip(mid[1], 0, whole.shape[1] - 1),
            ]
        )
        if not tissue_id:
            unresolved += 1
            continue
        wr = whole_regions.get(tissue_id)
        if wr is None:
            unresolved += 1
            continue
        r0, c0, r1, c1 = wr.bbox
        component_crop = whole_labels[r0:r1, c0:c1] == tissue_id
        cached = tissue_surface_cache.get(tissue_id)
        if cached is None:
            cached = _surface_cache_for_component(
                component_crop, np.array([r0, c0]), seeds[r0:r1, c0:c1] & component_crop
            )
            tissue_surface_cache[tissue_id] = cached
        if cached is None:
            unresolved += 1
            continue
        contour, surface_indices, surface_distances = cached
        local_band_support = band[r0:r1, c0:c1] & component_crop
        band_points = np.argwhere(local_band_support)
        band_distances = (
            cKDTree(band_points).query(contour - np.array([r0, c0]))[0]
            if len(band_points)
            else None
        )
        surface_arc, ia, ib = _choose_surface_arc(
            contour,
            surface_indices,
            surface_distances,
            path[0],
            path[-1],
            anatomical_support_distances=band_distances,
        )
        ring = np.vstack((path, surface_arc[::-1]))
        pr0 = max(0, int(np.floor(ring[:, 0].min())) - 2)
        pr1 = min(whole.shape[0], int(np.ceil(ring[:, 0].max())) + 3)
        pc0 = max(0, int(np.floor(ring[:, 1].min())) - 2)
        pc1 = min(whole.shape[1], int(np.ceil(ring[:, 1].max())) + 3)
        local_shape = (pr1 - pr0, pc1 - pc0)
        candidate = _fill_closed_ring_fast(ring - np.array([pr0, pc0]), local_shape)
        local_whole = whole_labels[pr0:pr1, pc0:pc1] == tissue_id
        candidate &= local_whole
        if not candidate.any():
            unresolved += 1
            continue
        # Associate exact frozen components by actual overlap with this guide;
        # no nearest-side or basal/superficial classification is used.
        expanded = binary_dilation(network_mask, structure=np.ones((5, 5), bool))
        support_ids = set(np.unique(band_labels[expanded & band]).tolist()) - {0}
        local_band = np.isin(band_labels[pr0:pr1, pc0:pc1], list(support_ids))
        local_epi = (candidate | local_band) & local_whole
        epidermis[pr0:pr1, pc0:pc1] |= local_epi
        polygons[pr0:pr1, pc0:pc1] |= candidate
        local_dermis = local_whole & ~local_epi
        adjacency = local_epi & binary_dilation(
            local_dermis, structure=np.ones((3, 3), bool)
        )
        surface_mask = rasterize_paths(
            [surface_arc - np.array([pr0, pc0])], local_shape
        )
        closure_paths = [
            np.vstack((path[0], contour[ia])) - np.array([pr0, pc0]),
            np.vstack((path[-1], contour[ib])) - np.array([pr0, pc0]),
        ]
        closure_mask = rasterize_paths(closure_paths, local_shape)
        excluded = binary_dilation(
            surface_mask | closure_mask, structure=np.ones((5, 5), bool)
        )
        candidate_interface = skeletonize(adjacency & ~excluded)
        known_surface[pr0:pr1, pc0:pc1] |= surface_mask
        known_closure[pr0:pr1, pc0:pc1] |= closure_mask
        # Record ordered local fragments before any polygon union.
        local_labels = label(candidate_interface, connectivity=2)
        guide_path = path
        guide_tree = cKDTree(guide_path)
        steps = np.linalg.norm(np.diff(guide_path, axis=0), axis=1) * pixel_size_um
        guide_s = np.concatenate(([0.0], np.cumsum(steps)))
        for fr in regionprops(local_labels):
            fm = local_labels == fr.label
            ordered = _major_geodesic_path(fm)
            if len(ordered) < 2:
                continue
            ordered += np.array([pr0, pc0])
            projected = guide_s[guide_tree.query(ordered)[1]]
            if projected[0] > projected[-1]:
                ordered = ordered[::-1].copy()
                projected = projected[::-1]
            endpoints, _ = _path_topology(fm)
            length = calibrated_path_length(ordered, pixel_size_um)
            # ``candidate_interface`` is a raster boundary.  At diagonal turns,
            # its 8-connected pixel graph can contain tiny 2x2 cycles/spurs even
            # though the extracted major geodesic is one unambiguous open path.
            # Do not discard a long anatomical interface solely because of that
            # raster artifact.  Closed components remain holes; open components
            # are classified by the ordered major path retained above.
            cls = (
                "internal_hole_edge"
                if endpoints == 0
                else (
                    "major_anatomical_interface"
                    if length >= minimum_major_fragment_um
                    else "minor_likely_anatomical_interface"
                )
            )
            raw_fragments.append(
                dict(
                    polygon_id=polygon_id,
                    tissue_component_id=tissue_id,
                    trajectory_network_id=network_id,
                    coordinates=ordered,
                    length_um=length,
                    s_start_um=float(projected.min()),
                    s_end_um=float(projected.max()),
                    s_mid_um=float(np.median(projected)),
                    fragment_class=cls,
                )
            )
        polygon_id += 1

    epidermis &= whole
    epidermis, dermis, topology_diagnostics = _surface_seeded_topological_partition(
        whole,
        band,
        guide,
        known_closure,
        seeds,
        epidermis,
    )
    final_adjacency = epidermis & binary_dilation(
        dermis, structure=np.ones((3, 3), bool)
    )
    fragments = []
    class_image = np.zeros_like(whole, np.uint8)
    fid = 1
    class_codes = {
        "major_anatomical_interface": 1,
        "minor_likely_anatomical_interface": 2,
        "internal_hole_edge": 3,
        "polygon_seam": 4,
        "ambiguous_incidental_fragment": 6,
    }
    supported_domain = binary_dilation(final_adjacency, structure=np.ones((3, 3), bool))
    for item in raw_fragments:
        rounded = np.rint(item["coordinates"]).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, whole.shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, whole.shape[1] - 1)
        support = (
            float(np.mean(supported_domain[rounded[:, 0], rounded[:, 1]]))
            if len(rounded)
            else 0.0
        )
        if support < 0.5:
            item["fragment_class"] = "polygon_seam"
        fragment = InterfaceFragment(fid, **item)
        fragments.append(fragment)
        fid += 1
    suppressed_fragment_ids = _suppress_redundant_guide_interval_excursions(fragments)
    accepted_paths = []
    for fragment in fragments:
        rounded = np.rint(fragment.coordinates).astype(int)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, whole.shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, whole.shape[1] - 1)
        class_image[rounded[:, 0], rounded[:, 1]] = class_codes[fragment.fragment_class]
        if fragment.fragment_class in {
            "major_anatomical_interface",
            "minor_likely_anatomical_interface",
        }:
            accepted_paths.append(fragment.coordinates)
    observed_mask = rasterize_paths(accepted_paths, whole.shape)
    anatomical_support = final_adjacency & ~binary_dilation(
        known_surface | known_closure,
        structure=np.ones((5, 5), bool),
    )
    candidate_bridge_mask, bridge_mask, bridge_rows = _bridge_ordered_fragments(
        fragments,
        whole,
        anatomical_support,
        pixel_size_um,
        maximum_bridge_gap_um,
    )
    final_mask = observed_mask | bridge_mask
    final_paths = _extract_paths_cropped(final_mask, pixel_size_um)
    supported = final_mask & supported_domain
    unsupported = final_mask & ~supported_domain
    observed_length = sum(
        f.length_um
        for f in fragments
        if f.fragment_class
        in {"major_anatomical_interface", "minor_likely_anatomical_interface"}
    )
    bridge_length = sum(
        float(r.get("bridge_length_um", 0)) for r in bridge_rows if r.get("accepted")
    )
    diagnostics = {
        "trajectory_network_count": int(guide_labels.max()),
        "surface_polygon_count": polygon_id,
        "unresolved_surface_polygon_count": unresolved,
        "raw_explicit_fragment_count": len(fragments),
        "major_fragment_count": sum(
            f.fragment_class == "major_anatomical_interface" for f in fragments
        ),
        "minor_fragment_count": sum(
            f.fragment_class == "minor_likely_anatomical_interface" for f in fragments
        ),
        "hole_fragment_count": sum(
            f.fragment_class == "internal_hole_edge" for f in fragments
        ),
        "seam_fragment_count": sum(
            f.fragment_class == "polygon_seam" for f in fragments
        ),
        "suppressed_redundant_excursion_count": len(suppressed_fragment_ids),
        "suppressed_redundant_excursion_ids": ";".join(
            map(str, suppressed_fragment_ids)
        ),
        "accepted_bridge_count": sum(bool(r.get("accepted")) for r in bridge_rows),
        "observed_interface_length_um": observed_length,
        "inferred_bridge_length_um": bridge_length,
        "inferred_boundary_fraction": bridge_length
        / max(observed_length + bridge_length, 1e-9),
        "maximum_accepted_bridge_length_um": max(
            (
                float(r.get("bridge_length_um", 0))
                for r in bridge_rows
                if r.get("accepted")
            ),
            default=0.0,
        ),
        "final_boundary_path_count": len(final_paths),
        "epidermis_area_pixels": int(epidermis.sum()),
        "dermis_area_pixels": int(dermis.sum()),
        "whole_tissue_area_pixels": int(whole.sum()),
        "compartment_consistency_error_pixels": int(
            np.count_nonzero((epidermis | dermis) ^ whole)
        )
        + int(np.count_nonzero(epidermis & dermis)),
        "boundary_compartment_support_fraction": float(
            supported.sum() / max(final_mask.sum(), 1)
        ),
        "unsupported_boundary_pixels": int(unsupported.sum()),
        "maximum_unsupported_run_um": _maximum_unsupported_run_um(
            final_paths, supported_domain, pixel_size_um
        ),
        "active_basal_side_dependency_count": 0,
    }
    diagnostics.update(topology_diagnostics)
    return InterfaceReconstruction(
        epidermis,
        dermis,
        guide,
        known_surface,
        known_closure,
        polygons,
        observed_mask,
        candidate_bridge_mask,
        bridge_mask,
        final_mask,
        class_image,
        supported,
        fragments,
        final_paths,
        bridge_rows,
        diagnostics,
    )
