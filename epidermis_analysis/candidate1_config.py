"""Parameters for epidermal-band component selection (historical name: Candidate 1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate1Config:
    """Shared physical thresholds for automatic graph construction."""

    minimum_node_geodesic_um: float = 25.0
    minimum_root_geodesic_um: float = 300.0
    root_surface_distance_um: float = 100.0
    minimum_inside_tissue_fraction: float = 0.90
    maximum_endpoint_gap_um: float = 750.0
    nearest_neighbors_per_endpoint: int = 24
    tangent_radius_um: float = 40.0
    support_downsample: int = 3
    curvature_penalty_um: float = 10.0
    root_cost: float = 0.05
    evidence_length_scale_um: float = 500.0
    # Invariant check; the optimizer itself enforces degree <= 2.
    maximum_graph_degree: int = 2


CANDIDATE1_CONFIG = Candidate1Config()
