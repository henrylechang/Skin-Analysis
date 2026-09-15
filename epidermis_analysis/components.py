"""Geometry-preserving connected-component and principal-path measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.draw import line
from skimage.measure import find_contours
from skimage.measure import label, perimeter, regionprops
from skimage.morphology import skeletonize


@dataclass
class GeodesicResult:
    skeleton_length_pixels: float
    principal_length_pixels: float
    ordered_path: np.ndarray
    endpoint_count: int
    branch_count: int
    method: str
    exact: bool


def _skeleton_graph(skeleton):
    """Build an 8-connected sparse graph without per-pixel Python lookups."""

    skeleton = np.asarray(skeleton, dtype=bool)
    points = np.argwhere(skeleton)
    point_count = len(points)
    if point_count == 0:
        return points, csr_matrix((0, 0), dtype=float), np.empty(0, dtype=np.int32)

    node_ids = np.full(skeleton.shape, -1, dtype=np.int32)
    node_ids[points[:, 0], points[:, 1]] = np.arange(point_count, dtype=np.int32)
    diagonal_weight = np.sqrt(2.0)
    neighbor_slices = (
        (
            (slice(None), slice(None, -1)),
            (slice(None), slice(1, None)),
            1.0,
        ),
        (
            (slice(None, -1), slice(None)),
            (slice(1, None), slice(None)),
            1.0,
        ),
        (
            (slice(None, -1), slice(None, -1)),
            (slice(1, None), slice(1, None)),
            diagonal_weight,
        ),
        (
            (slice(None, -1), slice(1, None)),
            (slice(1, None), slice(None, -1)),
            diagonal_weight,
        ),
    )
    first_nodes = []
    second_nodes = []
    edge_weights = []
    for first_slice, second_slice, weight in neighbor_slices:
        connected = skeleton[first_slice] & skeleton[second_slice]
        count = int(np.count_nonzero(connected))
        if count:
            first_nodes.append(node_ids[first_slice][connected])
            second_nodes.append(node_ids[second_slice][connected])
            edge_weights.append(np.full(count, weight, dtype=float))

    if not first_nodes:
        return (
            points,
            csr_matrix((point_count, point_count), dtype=float),
            np.zeros(point_count, dtype=np.int32),
        )

    first = np.concatenate(first_nodes)
    second = np.concatenate(second_nodes)
    weights = np.concatenate(edge_weights)
    rows = np.concatenate((first, second))
    columns = np.concatenate((second, first))
    symmetric_weights = np.concatenate((weights, weights))
    degrees = np.bincount(rows, minlength=point_count).astype(np.int32, copy=False)
    graph = csr_matrix(
        (symmetric_weights, (rows, columns)), shape=(point_count, point_count)
    )
    return points, graph, degrees


def _farthest(graph, source):
    distances, predecessors = dijkstra(
        graph, directed=False, indices=int(source), return_predecessors=True
    )
    finite = np.isfinite(distances)
    target = int(np.argmax(np.where(finite, distances, -1)))
    return target, float(distances[target]), predecessors


def _restore_path(points, source, target, predecessors):
    indices = [int(target)]
    while indices[-1] != source:
        previous = int(predecessors[indices[-1]])
        if previous < 0:
            return np.empty((0, 2), dtype=float)
        indices.append(previous)
    indices.reverse()
    return points[np.asarray(indices)].astype(float)


def principal_geodesic(component, *, skeleton=None):
    component = np.asarray(component, dtype=bool)
    skeleton = (
        skeletonize(component) if skeleton is None else np.asarray(skeleton, bool)
    )
    if skeleton.shape != component.shape:
        raise ValueError("Precomputed skeleton shape must match the component.")
    points, graph, degrees = _skeleton_graph(skeleton)
    if not len(points):
        return GeodesicResult(0, 0, np.empty((0, 2)), 0, 0, "empty", True)
    edge_sum = float(graph.data.sum() / 2.0)
    endpoints = np.flatnonzero(degrees == 1)
    branches = int(np.count_nonzero(degrees >= 3))
    undirected_edges = graph.nnz // 2
    if undirected_edges == len(points) - 1:
        first, _, _ = _farthest(graph, 0)
        second, length, predecessor = _farthest(graph, first)
        return GeodesicResult(
            edge_sum,
            length,
            _restore_path(points, first, second, predecessor),
            len(endpoints),
            branches,
            "exact_tree_diameter",
            True,
        )
    if len(endpoints) >= 2 and len(endpoints) <= 128 and len(points) <= 100_000:
        sources, method, exact = endpoints, "exact_endpoint_pair_diameter", True
    elif len(endpoints) < 2 and len(points) <= 2_000:
        sources, method, exact = np.arange(len(points)), "exact_all_node_diameter", True
    else:
        pool = endpoints if len(endpoints) else np.arange(len(points))
        sources = pool[np.linspace(0, len(pool) - 1, min(64, len(pool))).astype(int)]
        method, exact = "robust_64_source_diameter", False
    best = (-1.0, 0, 0, None)
    for source in sources:
        target, length, predecessor = _farthest(graph, int(source))
        if length > best[0]:
            best = (length, int(source), target, predecessor)
    return GeodesicResult(
        edge_sum,
        best[0],
        _restore_path(points, best[1], best[2], best[3]),
        len(endpoints),
        branches,
        method,
        exact,
    )


def component_features(mask, pixel_size_um):
    """Measure components while retaining every accepted component's raw pixels."""

    labels = label(mask, connectivity=2)
    rows, paths = [], {}
    for region in regionprops(labels):
        r0, c0, r1, c1 = region.bbox
        local = labels[r0:r1, c0:c1] == region.label
        skeleton = skeletonize(local)
        geodesic = principal_geodesic(local, skeleton=skeleton)
        distance = distance_transform_edt(local)
        thickness = 2.0 * distance[skeleton]
        median_thickness = float(np.median(thickness)) if len(thickness) else 0.0
        mean_thickness = float(np.mean(thickness)) if len(thickness) else 0.0
        thickness_cv = (
            float(np.std(thickness) / mean_thickness) if mean_thickness else np.nan
        )
        principal_um = geodesic.principal_length_pixels * pixel_size_um
        area_um2 = float(region.area) * pixel_size_um**2
        path = (
            geodesic.ordered_path + np.array([r0, c0])
            if len(geodesic.ordered_path)
            else geodesic.ordered_path
        )
        paths[int(region.label)] = path
        perimeter_pixels = float(perimeter(local, neighborhood=8))
        rows.append(
            {
                "component_id": int(region.label),
                "area_pixels": int(region.area),
                "area_um2": area_um2,
                "perimeter_pixels": perimeter_pixels,
                "perimeter_um": perimeter_pixels * pixel_size_um,
                "bbox_min_row": r0,
                "bbox_min_col": c0,
                "bbox_max_row": r1,
                "bbox_max_col": c1,
                "x_span_pixels": c1 - c0,
                "y_span_pixels": r1 - r0,
                "x_span_um": (c1 - c0) * pixel_size_um,
                "y_span_um": (r1 - r0) * pixel_size_um,
                "major_axis_um": float(region.axis_major_length * pixel_size_um),
                "minor_axis_um": float(region.axis_minor_length * pixel_size_um),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "skeleton_length_um": geodesic.skeleton_length_pixels * pixel_size_um,
                "skeleton_endpoint_count": geodesic.endpoint_count,
                "skeleton_branch_count": geodesic.branch_count,
                "principal_geodesic_length_um": principal_um,
                "principal_geodesic_method": geodesic.method,
                "principal_geodesic_exact": geodesic.exact,
                "median_thickness_um": median_thickness * pixel_size_um,
                "mean_thickness_um": mean_thickness * pixel_size_um,
                "thickness_cv": thickness_cv,
                "area_over_geodesic_um": (
                    area_um2 / principal_um if principal_um else np.nan
                ),
                "geodesic_over_median_thickness": (
                    principal_um / (median_thickness * pixel_size_um)
                    if median_thickness
                    else np.nan
                ),
            }
        )
    return labels, pd.DataFrame(rows), paths


def rasterize_paths(paths, accepted_ids, shape):
    output = np.zeros(shape, dtype=bool)
    for component_id in accepted_ids:
        path = paths.get(int(component_id), np.empty((0, 2)))
        rounded = np.rint(path).astype(int)
        if len(rounded):
            output[rounded[:, 0], rounded[:, 1]] = True
    return output


def rasterize_coordinate_paths(paths, shape):
    mask = np.zeros(shape, dtype=bool)
    for path in paths:
        rounded = np.rint(path).astype(int)
        if not len(rounded):
            continue
        rounded[:, 0] = np.clip(rounded[:, 0], 0, shape[0] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, shape[1] - 1)
        for start, stop in zip(rounded[:-1], rounded[1:]):
            rr, cc = line(int(start[0]), int(start[1]), int(stop[0]), int(stop[1]))
            mask[rr, cc] = True
        mask[rounded[:, 0], rounded[:, 1]] = True
    return mask


def calibrated_path_length(path, pixel_size_um):
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    return float(
        np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])).sum() * pixel_size_um
    )


def measure_skeleton_graph_length(skeleton_mask, pixel_size_um):
    """Measure an existing 8-connected skeleton using production conventions."""

    skeleton = np.asarray(skeleton_mask, dtype=bool)
    if skeleton.ndim != 2:
        raise ValueError("Skeleton mask must be two-dimensional.")
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError("Pixel size must be positive and finite.")
    foreground_rows = np.flatnonzero(np.any(skeleton, axis=1))
    foreground_columns = np.flatnonzero(np.any(skeleton, axis=0))
    if not foreground_rows.size or not foreground_columns.size:
        return 0.0
    skeleton = skeleton[
        foreground_rows[0] : foreground_rows[-1] + 1,
        foreground_columns[0] : foreground_columns[-1] + 1,
    ]
    horizontal_edges = np.count_nonzero(skeleton[:, :-1] & skeleton[:, 1:])
    vertical_edges = np.count_nonzero(skeleton[:-1, :] & skeleton[1:, :])
    diagonal_down_right = (
        skeleton[:-1, :-1] & skeleton[1:, 1:] & ~(skeleton[:-1, 1:] | skeleton[1:, :-1])
    )
    diagonal_down_left = (
        skeleton[:-1, 1:] & skeleton[1:, :-1] & ~(skeleton[:-1, :-1] | skeleton[1:, 1:])
    )
    diagonal_edges = np.count_nonzero(diagonal_down_right) + np.count_nonzero(
        diagonal_down_left
    )
    connected = np.zeros_like(skeleton)
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
        ((slice(None, -1), slice(None, -1)), (slice(1, None), slice(1, None))),
        ((slice(None, -1), slice(1, None)), (slice(1, None), slice(None, -1))),
    ):
        pairs = skeleton[first] & skeleton[second]
        connected[first] |= pairs
        connected[second] |= pairs
    isolated_pixels = np.count_nonzero(skeleton & ~connected)
    length_pixels = (
        horizontal_edges
        + vertical_edges
        + np.sqrt(2.0) * diagonal_edges
        + isolated_pixels
    )
    return float(length_pixels * pixel_size_um)


def major_geodesic_path(component):
    return principal_geodesic(np.asarray(component, dtype=bool)).ordered_path


def outward_tangent(path, endpoint, samples=12):
    if endpoint == 0:
        vector = path[0] - path[min(samples, len(path) - 1)]
    else:
        vector = path[-1] - path[max(0, len(path) - 1 - samples)]
    norm = np.linalg.norm(vector)
    return vector / norm if norm else np.zeros(2)


def bezier_bridge(p0, tangent0, p3, tangent3, spacing_pixels=0.5):
    distance = float(np.linalg.norm(p3 - p0))
    control_length = distance / 3.0
    p1 = p0 + tangent0 * control_length
    p2 = p3 + tangent3 * control_length
    count = max(2, int(np.ceil(distance / spacing_pixels)) + 1)
    t = np.linspace(0, 1, count)[:, None]
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def outer_ordered_contour(component):
    filled = np.asarray(component, dtype=bool)
    from scipy.ndimage import binary_fill_holes

    padded = np.pad(binary_fill_holes(filled), 1)
    contours = find_contours(padded.astype(np.uint8), 0.5)
    return max(contours, key=len).astype(float) - 1 if contours else np.empty((0, 2))


def circular_arc(contour, start, stop):
    return (
        contour[start : stop + 1]
        if start <= stop
        else np.vstack((contour[start:], contour[: stop + 1]))
    )
