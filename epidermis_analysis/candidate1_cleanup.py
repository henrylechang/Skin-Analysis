"""Automatic full-section graph construction for structured Candidate 1.

Candidate 1 is the retained name of the production component-selection method.
At inference it uses:

* raw improved-Ilastik component pixels;
* ordered principal component paths;
* DAPI intensity/texture and orientation;
* repaired whole-tissue membership and its superficial contour.

No per-section manual graph labels are supplied to this step. The upstream
Ilastik projects are trained classifiers, and the graph uses fixed parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    label as ndi_label,
    uniform_filter,
)
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from .components import component_features, measure_skeleton_graph_length

from .candidate1_config import CANDIDATE1_CONFIG, Candidate1Config
from .candidate1_graph import (
    StructuredEdge,
    StructuredNode,
    StructuredSelection,
    select_rooted_path_forest,
)


@dataclass(frozen=True)
class Candidate1CleanupResult:
    labels: np.ndarray
    features: pd.DataFrame
    paths: dict[int, np.ndarray]
    nodes: list[StructuredNode]
    edges: list[StructuredEdge]
    edge_table: pd.DataFrame
    edge_paths: dict[tuple[int, int], np.ndarray]
    selection: StructuredSelection
    config: Candidate1Config


class Candidate1Failure(RuntimeError):
    """An explicit per-sample Candidate-1 QC failure."""


def _observed_path_prize(length_um, superficial_fraction, config):
    support_weight = 0.25 + 0.75 * float(np.clip(superficial_fraction, 0, 1))
    return float(length_um) * support_weight / config.evidence_length_scale_um


def _component_surface_course_um(
    labels,
    component_id,
    bbox,
    distance_to_surface_um,
    surface,
    tissue_labels,
    tissue_piece,
    pixel_size_um,
    maximum_surface_distance_um,
):
    """Measure corroborated course length without choosing one diameter path.

    The component and superficial contour must each sustain the minimum course:
    taking the shorter of those two lengths prevents a branched object touching
    the surface at one point, or a tiny object beside a long surface, from
    manufacturing root evidence.
    """

    r0, c0, r1, c1 = (int(value) for value in bbox)
    padding = int(np.ceil(maximum_surface_distance_um / pixel_size_um)) + 1
    r0 = max(0, r0 - padding)
    c0 = max(0, c0 - padding)
    r1 = min(labels.shape[0], r1 + padding)
    c1 = min(labels.shape[1], c1 + padding)
    component = labels[r0:r1, c0:c1] == component_id
    component_course = skeletonize(component) & (
        distance_to_surface_um[r0:r1, c0:c1] <= maximum_surface_distance_um
    )
    component_course_um = measure_skeleton_graph_length(component_course, pixel_size_um)
    if component_course_um <= 0:
        return 0.0

    distance_to_component_um = distance_transform_edt(~component).astype(np.float32)
    distance_to_component_um *= pixel_size_um
    supported_surface = (
        np.asarray(surface[r0:r1, c0:c1], dtype=bool)
        & (tissue_labels[r0:r1, c0:c1] == tissue_piece)
        & (distance_to_component_um <= maximum_surface_distance_um)
    )
    surface_course_um = measure_skeleton_graph_length(
        skeletonize(supported_surface), pixel_size_um
    )
    return float(min(component_course_um, surface_course_um))


def _surface_context(labels, features, paths, whole, surface, pixel_size_um, config):
    tissue_labels, _ = ndi_label(np.asarray(whole, bool), structure=np.ones((3, 3)))
    distance = distance_transform_edt(~np.asarray(surface, bool)).astype(np.float32)
    distance *= pixel_size_um
    rows = []
    for row in features.itertuples(index=False):
        component_id = int(row.component_id)
        r0, c0, r1, c1 = (
            int(row.bbox_min_row),
            int(row.bbox_min_col),
            int(row.bbox_max_row),
            int(row.bbox_max_col),
        )
        component = labels[r0:r1, c0:c1] == component_id
        tissue_values = tissue_labels[r0:r1, c0:c1][component]
        positive = tissue_values[tissue_values > 0]
        piece = int(np.bincount(positive).argmax()) if len(positive) else 0
        inside = float(np.mean(tissue_values > 0)) if len(tissue_values) else 0.0
        path = np.rint(paths[component_id]).astype(int)
        valid = (
            (path[:, 0] >= 0)
            & (path[:, 0] < labels.shape[0])
            & (path[:, 1] >= 0)
            & (path[:, 1] < labels.shape[1])
        )
        path = path[valid]
        values = distance[path[:, 0], path[:, 1]] if len(path) else np.array([np.inf])
        within_surface_100um = values <= config.root_surface_distance_um
        surface_course_um = 0.0
        if (
            float(row.principal_geodesic_length_um) >= config.minimum_root_geodesic_um
            and piece > 0
        ):
            surface_course_um = _component_surface_course_um(
                labels,
                component_id,
                (r0, c0, r1, c1),
                distance,
                surface,
                tissue_labels,
                piece,
                pixel_size_um,
                config.root_surface_distance_um,
            )
        rows.append(
            {
                "component_id": component_id,
                "component_whole_tissue_piece": piece,
                "component_inside_whole_tissue_fraction": inside,
                "path_fraction_within_surface_100um": float(
                    np.mean(within_surface_100um)
                ),
                "surface_distance_median_um": float(np.median(values)),
                "component_surface_course_um": surface_course_um,
            }
        )
    return (
        features.merge(
            pd.DataFrame(rows), on="component_id", how="left", validate="one_to_one"
        ),
        tissue_labels,
    )


def _support_field(dapi, raw_mask, downsample):
    """Return a compact DAPI/Ilastik support and local ridge orientation field."""

    image = np.asarray(dapi[::downsample, ::downsample], dtype=np.float32)
    smooth = gaussian_filter(image, 1.2)
    low, high = np.percentile(smooth, (10, 99))
    intensity = np.clip((smooth - low) / max(float(high - low), 1e-6), 0, 1)
    mean = uniform_filter(image, 9)
    variance = np.maximum(uniform_filter(image * image, 9) - mean * mean, 0)
    low, high = np.percentile(variance, (20, 99))
    texture = np.clip((variance - low) / max(float(high - low), 1e-6), 0, 1)
    hard = gaussian_filter(raw_mask[::downsample, ::downsample].astype(np.float32), 2)
    support = np.clip(
        0.45 * intensity + 0.35 * texture + 0.20 * np.clip(hard, 0, 1), 0, 1
    )
    gy, gx = np.gradient(gaussian_filter(image, 1.5))
    jxx = gaussian_filter(gx * gx, 2)
    jyy = gaussian_filter(gy * gy, 2)
    jxy = gaussian_filter(gx * gy, 2)
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy * jxy) / np.maximum(jxx + jyy, 1e-6)
    return (
        support.astype(np.float32),
        theta.astype(np.float32),
        np.clip(coherence, 0, 1).astype(np.float32),
    )


def _endpoint_inward_tangent(path, endpoint_index, radius_pixels):
    ordered = path if endpoint_index == 0 else path[::-1]
    if len(ordered) < 2:
        return np.array([0.0, 0.0])
    cumulative = np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))
    index = min(int(np.searchsorted(cumulative, radius_pixels)) + 1, len(ordered) - 1)
    vector = ordered[index] - ordered[0]
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def enumerate_candidate_pairs(paths, component_ids, pixel_size_um, config):
    """Use an endpoint KD-tree to avoid an all-to-all component graph."""

    records = []
    for component_id in sorted(component_ids):
        path = paths[component_id]
        if len(path) < 2:
            continue
        for endpoint_index, point in enumerate(path[[0, -1]]):
            records.append(
                (component_id, endpoint_index, float(point[0]), float(point[1]))
            )
    if not records:
        return []
    points = np.asarray([[row[2], row[3]] for row in records])
    tree = cKDTree(points)
    maximum_pixels = config.maximum_endpoint_gap_um / pixel_size_um
    pairs = {}
    for index, record in enumerate(records):
        distances, neighbors = tree.query(
            points[index],
            k=min(len(records), config.nearest_neighbors_per_endpoint + 1),
            distance_upper_bound=maximum_pixels,
        )
        for distance, other_index in zip(
            np.atleast_1d(distances), np.atleast_1d(neighbors)
        ):
            if not np.isfinite(distance) or other_index >= len(records):
                continue
            other = records[int(other_index)]
            if record[0] == other[0]:
                continue
            key = tuple(sorted((int(record[0]), int(other[0]))))
            candidate = (float(distance), int(record[1]), int(other[1]))
            if record[0] > other[0]:
                candidate = (candidate[0], candidate[2], candidate[1])
            if key not in pairs or candidate[0] < pairs[key][0]:
                pairs[key] = candidate
    return [(a, b, *pairs[(a, b)]) for a, b in sorted(pairs)]


def _curve_candidates(start, end, start_derivative, end_derivative):
    gap = float(np.linalg.norm(end - start))
    count = max(8, int(np.ceil(gap / 2.0)) + 1)
    t = np.linspace(0, 1, count)[:, None]
    curves = []
    for tangent_scale in (0.0, 0.25, 0.50, 1.0):
        m0 = start_derivative * gap * tangent_scale
        m1 = end_derivative * gap * tangent_scale
        curve = (
            (2 * t**3 - 3 * t**2 + 1) * start
            + (t**3 - 2 * t**2 + t) * m0
            + (-2 * t**3 + 3 * t**2) * end
            + (t**3 - t**2) * m1
        )
        curves.append(curve)
    chord = end - start
    normal = np.array([-chord[1], chord[0]]) / max(gap, 1e-9)
    for offset in (-0.50, -0.25, 0.25, 0.50):
        control = (start + end) / 2 + normal * gap * offset
        curves.append((1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end)
    return curves


def score_candidate_edge(
    a,
    b,
    endpoint_a,
    endpoint_b,
    paths,
    pixel_size_um,
    tissue_labels,
    support,
    theta,
    coherence,
    config,
):
    """Score the best smooth, image-supported connector between two path ends."""

    path_a, path_b = paths[a], paths[b]
    start = path_a[0 if endpoint_a == 0 else -1]
    end = path_b[0 if endpoint_b == 0 else -1]
    radius = config.tangent_radius_um / pixel_size_um
    # The connector leaves A opposite its inward tangent and arrives along B's
    # inward tangent. Alternative curved candidates allow real fold turns.
    tangent_a = -_endpoint_inward_tangent(path_a, endpoint_a, radius)
    tangent_b = _endpoint_inward_tangent(path_b, endpoint_b, radius)
    downsample = config.support_downsample
    piece_a = int(tissue_labels[int(round(start[0])), int(round(start[1]))])
    piece_b = int(tissue_labels[int(round(end[0])), int(round(end[1]))])
    if piece_a <= 0 or piece_a != piece_b:
        return None
    best = None
    for curve in _curve_candidates(start, end, tangent_a, tangent_b):
        rounded = np.rint(curve).astype(int)
        valid = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < tissue_labels.shape[0])
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < tissue_labels.shape[1])
        )
        if not valid.all():
            continue
        if np.mean(tissue_labels[rounded[:, 0], rounded[:, 1]] == piece_a) < 0.95:
            continue
        delta = np.diff(curve, axis=0)
        segment_pixels = np.linalg.norm(delta, axis=1)
        keep = segment_pixels > 1e-9
        if not keep.any():
            continue
        delta = delta[keep]
        segment_pixels = segment_pixels[keep]
        mid = (curve[:-1][keep] + curve[1:][keep]) / 2
        sampled = np.rint(mid / downsample).astype(int)
        sampled[:, 0] = np.clip(sampled[:, 0], 0, support.shape[0] - 1)
        sampled[:, 1] = np.clip(sampled[:, 1], 0, support.shape[1] - 1)
        values = support[sampled[:, 0], sampled[:, 1]]
        angles = np.arctan2(delta[:, 0], delta[:, 1])
        local_theta = theta[sampled[:, 0], sampled[:, 1]]
        local_coherence = coherence[sampled[:, 0], sampled[:, 1]]
        mismatch = (1 - np.abs(np.cos(angles - local_theta))) * local_coherence
        segment_um = segment_pixels * pixel_size_um
        integrated = float(
            np.sum(segment_um * (1 + 3.2 * (1 - values) + 1.5 * mismatch))
        )
        unit = delta / segment_pixels[:, None]
        turns = (
            np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
            if len(unit) > 1
            else np.array([])
        )
        integrated += config.curvature_penalty_um * float(np.sum(turns * turns))
        record = {
            "component_a": a,
            "component_b": b,
            "endpoint_a": endpoint_a,
            "endpoint_b": endpoint_b,
            "endpoint_gap_um": float(np.linalg.norm(end - start) * pixel_size_um),
            "inferred_length_um": float(np.sum(segment_um)),
            "integrated_cost_um": integrated,
            "mean_image_support": float(np.average(values, weights=segment_um)),
            "strong_support_fraction": float(
                np.average(values >= 0.5, weights=segment_um)
            ),
            "total_turn_degrees": float(np.degrees(turns).sum()),
            "coordinates": curve,
        }
        if best is None or record["integrated_cost_um"] < best["integrated_cost_um"]:
            best = record
    return best


def run_candidate1_cleanup(
    dapi, raw_mask, whole, surface, pixel_size_um, config=CANDIDATE1_CONFIG
):
    labels, features, paths = component_features(raw_mask, pixel_size_um)
    features, tissue_labels = _surface_context(
        labels, features, paths, whole, surface, pixel_size_um, config
    )
    eligible = features[
        features.principal_geodesic_length_um >= config.minimum_node_geodesic_um
    ].copy()
    root_eligible = (
        (eligible.principal_geodesic_length_um >= config.minimum_root_geodesic_um)
        & (eligible.component_surface_course_um >= config.minimum_root_geodesic_um)
        & (
            eligible.component_inside_whole_tissue_fraction
            >= config.minimum_inside_tissue_fraction
        )
        & (eligible.component_whole_tissue_piece > 0)
    )
    eligible["automatic_root_eligible"] = root_eligible
    eligible["automatic_root_eligibility_mode"] = np.where(
        root_eligible, "component_surface_course", ""
    )
    nodes = [
        StructuredNode(
            int(row.component_id),
            _observed_path_prize(
                row.principal_geodesic_length_um,
                row.path_fraction_within_surface_100um,
                config,
            ),
            bool(row.automatic_root_eligible),
            int(row.component_whole_tissue_piece),
        )
        for row in eligible.itertuples(index=False)
    ]
    if not nodes:
        raise Candidate1Failure(
            "Candidate 1 found no components above the frozen minimum node length."
        )
    if not any(node.root_eligible for node in nodes):
        raise Candidate1Failure(
            "Candidate 1 found no credible rooted epidermal course."
        )
    ids = {node.node_id for node in nodes}
    pairs = enumerate_candidate_pairs(paths, ids, pixel_size_um, config)
    support, theta, coherence = _support_field(
        dapi, raw_mask, config.support_downsample
    )
    feature_piece = dict(
        zip(
            eligible.component_id.astype(int),
            eligible.component_whole_tissue_piece.astype(int),
        )
    )
    records = []
    for a, b, gap_pixels, endpoint_a, endpoint_b in pairs:
        if feature_piece[a] <= 0 or feature_piece[a] != feature_piece[b]:
            continue
        record = score_candidate_edge(
            a,
            b,
            endpoint_a,
            endpoint_b,
            paths,
            pixel_size_um,
            tissue_labels,
            support,
            theta,
            coherence,
            config,
        )
        if record is not None:
            records.append(record)
    edge_paths = {
        tuple(
            sorted((int(record["component_a"]), int(record["component_b"])))
        ): np.asarray(record.pop("coordinates"), dtype=float)
        for record in records
    }
    edge_table = pd.DataFrame(records)
    edges = [
        StructuredEdge(
            int(row.component_a),
            int(row.component_b),
            float(row.integrated_cost_um) / config.evidence_length_scale_um,
            float(row.inferred_length_um),
        )
        for row in edge_table.itertuples(index=False)
    ]
    selection = select_rooted_path_forest(nodes, edges, root_cost=config.root_cost)
    if not selection.success:
        raise Candidate1Failure(f"Candidate 1 graph solver failed: {selection.message}")
    if not selection.roots:
        raise Candidate1Failure(
            "Candidate 1 found no credible rooted epidermal course."
        )
    if not selection.selected_nodes:
        raise Candidate1Failure("Candidate 1 accepted no epidermis components.")
    return Candidate1CleanupResult(
        labels=labels,
        features=eligible,
        paths=paths,
        nodes=nodes,
        edges=edges,
        edge_table=edge_table,
        edge_paths=edge_paths,
        selection=selection,
        config=config,
    )


def selected_network_ids(selection):
    neighbors = {node: set() for node in selection.selected_nodes}
    for first, second in selection.selected_edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    network_ids = {}
    network = 0
    for start in sorted(neighbors):
        if start in network_ids:
            continue
        network += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in network_ids:
                continue
            network_ids[current] = network
            stack.extend(neighbors[current] - network_ids.keys())
    return network_ids


def assert_candidate1_invariants(result, raw_mask):
    selected = set(result.selection.selected_nodes)
    cleaned = np.isin(result.labels, sorted(selected))
    raw = np.asarray(raw_mask, dtype=bool)
    if np.any(cleaned & ~raw):
        raise Candidate1Failure(
            "Cleaned epidermis contains invented foreground pixels."
        )
    for component_id in selected:
        component = result.labels == component_id
        if not np.array_equal(cleaned & component, component):
            raise Candidate1Failure(
                f"Accepted component {component_id} was partially clipped."
            )
    degree = {node: 0 for node in selected}
    for first, second in result.selection.selected_edges:
        degree[first] += 1
        degree[second] += 1
        pieces = {node.node_id: node.tissue_piece for node in result.nodes}
        if pieces[first] != pieces[second]:
            raise Candidate1Failure("Selected graph contains a cross-tissue edge.")
    if degree and max(degree.values()) > result.config.maximum_graph_degree:
        raise Candidate1Failure("Selected graph contains a branch above degree two.")
    if len(result.selection.selected_edges) > len(selected) - len(
        result.selection.roots
    ):
        raise Candidate1Failure("Selected graph contains a cycle.")
    networks = selected_network_ids(result.selection)
    roots_by_network = {network: 0 for network in set(networks.values())}
    for root in result.selection.roots:
        roots_by_network[networks[root]] += 1
    if any(count < 1 for count in roots_by_network.values()):
        raise Candidate1Failure(
            "Selected graph contains an unrooted component network."
        )
    return cleaned
