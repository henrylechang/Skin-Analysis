"""Rooted, nonbranching global forest optimizer for Automatic Candidate 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class StructuredNode:
    node_id: int
    prize: float
    root_eligible: bool = False
    tissue_piece: int = 1


@dataclass(frozen=True)
class StructuredEdge:
    first: int
    second: int
    cost: float
    inferred_length_um: float = 0.0


@dataclass(frozen=True)
class StructuredSelection:
    selected_nodes: frozenset[int]
    selected_edges: frozenset[tuple[int, int]]
    roots: frozenset[int]
    objective: float
    success: bool
    message: str


def connected_root_groups(nodes, edges):
    eligible = {node.node_id for node in nodes if node.root_eligible}
    neighbors = {node_id: set() for node_id in eligible}
    for edge in edges:
        if edge.first in eligible and edge.second in eligible:
            neighbors[edge.first].add(edge.second)
            neighbors[edge.second].add(edge.first)
    groups = {}
    visited = set()
    for start in sorted(eligible):
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        frozen = frozenset(component)
        visited.update(component)
        for member in component:
            groups[member] = frozen
    return groups


def select_rooted_path_forest(
    nodes,
    edges,
    *,
    root_cost=0.25,
    maximum_roots_per_tissue_piece=None,
    one_root_per_connected_eligible_group=True,
):
    """Select components and connections with a rooted degree-two forest MILP.

    Minimize edge and root costs minus selected-node prizes over the supplied
    candidate graph. Optimality applies to that graph and objective, not to
    anatomical correctness or all possible image paths.
    """

    if not nodes:
        return StructuredSelection(
            frozenset(), frozenset(), frozenset(), 0, True, "empty"
        )
    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Structured node IDs must be unique")
    index = {node_id: i for i, node_id in enumerate(ids)}
    for edge in edges:
        if (
            edge.first not in index
            or edge.second not in index
            or edge.first == edge.second
        ):
            raise ValueError(f"Invalid structured edge {edge}")
        if (
            nodes[index[edge.first]].tissue_piece
            != nodes[index[edge.second]].tissue_piece
        ):
            raise ValueError("Edges may not cross repaired whole-tissue pieces")
    n, m = len(nodes), len(edges)
    x0, y0, z0, f0, g0 = 0, n, n + m, 2 * n + m, 2 * n + 2 * m
    size = 2 * n + 3 * m
    objective = np.zeros(size)
    objective[x0 : x0 + n] = -np.asarray([node.prize for node in nodes])
    objective[y0 : y0 + m] = np.asarray([edge.cost for edge in edges])
    objective[z0 : z0 + n] = root_cost
    integrality = np.zeros(size, dtype=int)
    integrality[: 2 * n + m] = 1
    lower = np.zeros(size)
    upper = np.full(size, float(n))
    upper[: 2 * n + m] = 1
    upper[z0 : z0 + n] = np.asarray([node.root_eligible for node in nodes], dtype=float)
    rows, lb, ub = [], [], []

    def constraint(coefficients, low=-np.inf, high=np.inf):
        rows.append(coefficients)
        lb.append(low)
        ub.append(high)

    incident = {node_id: [] for node_id in ids}
    for e, edge in enumerate(edges):
        a, b = index[edge.first], index[edge.second]
        incident[edge.first].append(e)
        incident[edge.second].append(e)
        constraint({y0 + e: 1, x0 + a: -1}, high=0)
        constraint({y0 + e: 1, x0 + b: -1}, high=0)
        constraint({f0 + e: 1, y0 + e: -n}, high=0)
        constraint({g0 + e: 1, y0 + e: -n}, high=0)
    for i, node in enumerate(nodes):
        constraint({z0 + i: 1, x0 + i: -1}, high=0)
        constraint({y0 + e: 1 for e in incident[node.node_id]}, high=2)
        coefficients = {x0 + i: -1, z0 + i: n}
        for e in incident[node.node_id]:
            edge = edges[e]
            if edge.second == node.node_id:
                coefficients[f0 + e] = coefficients.get(f0 + e, 0) + 1
                coefficients[g0 + e] = coefficients.get(g0 + e, 0) - 1
            else:
                coefficients[g0 + e] = coefficients.get(g0 + e, 0) + 1
                coefficients[f0 + e] = coefficients.get(f0 + e, 0) - 1
        constraint(coefficients, low=0)
    forest = {y0 + e: 1 for e in range(m)}
    forest.update({x0 + i: -1 for i in range(n)})
    forest.update({z0 + i: 1 for i in range(n)})
    constraint(forest, high=0)
    constraint({z0 + i: 1 for i in range(n)}, low=1)
    if maximum_roots_per_tissue_piece is not None:
        for piece in sorted(set(node.tissue_piece for node in nodes)):
            constraint(
                {
                    z0 + i: 1
                    for i, node in enumerate(nodes)
                    if node.tissue_piece == piece
                },
                high=maximum_roots_per_tissue_piece,
            )
    if one_root_per_connected_eligible_group:
        groups = connected_root_groups(nodes, edges)
        for group in sorted(
            set(groups.values()), key=lambda values: tuple(sorted(values))
        ):
            constraint({z0 + index[node_id]: 1 for node_id in group}, high=1)
    matrix = lil_matrix((len(rows), size), dtype=float)
    for row_index, coefficients in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsr(), np.asarray(lb), np.asarray(ub)),
        options={"presolve": True},
    )
    if not result.success:
        return StructuredSelection(
            frozenset(), frozenset(), frozenset(), np.inf, False, result.message
        )
    solution = result.x
    selected_nodes = frozenset(ids[i] for i in range(n) if solution[x0 + i] > 0.5)
    selected_edges = frozenset(
        tuple(sorted((edge.first, edge.second)))
        for e, edge in enumerate(edges)
        if solution[y0 + e] > 0.5
    )
    roots = frozenset(ids[i] for i in range(n) if solution[z0 + i] > 0.5)
    return StructuredSelection(
        selected_nodes, selected_edges, roots, float(result.fun), True, result.message
    )
