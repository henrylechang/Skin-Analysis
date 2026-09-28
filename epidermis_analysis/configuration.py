"""Explicit, validated numerical settings; JSON needs no extra parser dependency."""

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path

from .candidate1_config import Candidate1Config
from .subbasal import SubbasalConfig


@dataclass(frozen=True)
class AnalysisConfig:
    fallback_pixel_size_um: float = 0.621504
    epidermis_label: int = 1
    whole_skin_label: int = 1
    min_whole_skin_object_area_um2: float = 3862.67222016
    whole_skin_boundary_smoothing_width_um: float = 20.0
    whole_skin_context_closing_radius_um: float = 8.0
    whole_skin_disconnected_vertical_margin_um: float = 50.0
    whole_skin_basal_smoothing_width_um: float = 100.0
    whole_skin_basal_percentile: float = 35.0
    min_component_width_fraction: float = 0.025
    min_superficial_envelope_fraction: float = 0.5
    superficial_nerve_exclusion_distance_um: float = 5.0
    min_superficial_nerve_object_length_um: float = 20.0
    manual_nerve_threshold: float = 1500
    minimum_major_fragment_um: float = 20.0
    maximum_bridge_gap_um: float = 150.0
    whole_skin_cleanup: str = "legacy"
    candidate1: Candidate1Config = Candidate1Config()
    subbasal: SubbasalConfig = SubbasalConfig()

    def validate(self):
        if not isinstance(
            self.whole_skin_cleanup, str
        ) or self.whole_skin_cleanup not in {"legacy", "on", "off"}:
            raise ValueError("whole_skin_cleanup must be legacy, on, or off.")
        for name, value in asdict(self).items():
            if name in {"candidate1", "subbasal", "whole_skin_cleanup"}:
                continue
            _number(name, value)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in (
            "fallback_pixel_size_um",
            "minimum_major_fragment_um",
            "maximum_bridge_gap_um",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("epidermis_label", "whole_skin_label"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "min_component_width_fraction",
            "min_superficial_envelope_fraction",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if not 0 <= self.whole_skin_basal_percentile <= 100:
            raise ValueError("whole_skin_basal_percentile must be between 0 and 100.")
        for name, value in asdict(self.candidate1).items():
            _number(f"candidate1.{name}", value)
            if value < 0:
                raise ValueError(f"candidate1.{name} must be nonnegative.")
        for name in ("nearest_neighbors_per_endpoint", "support_downsample"):
            value = getattr(self.candidate1, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"candidate1.{name} must be a positive integer.")
        if self.candidate1.evidence_length_scale_um <= 0:
            raise ValueError("candidate1.evidence_length_scale_um must be positive.")
        if not 0 <= self.candidate1.minimum_inside_tissue_fraction <= 1:
            raise ValueError(
                "candidate1.minimum_inside_tissue_fraction must be between 0 and 1."
            )
        if self.candidate1.maximum_graph_degree != 2:
            raise ValueError("The current optimizer requires maximum_graph_degree=2.")
        for name, value in asdict(self.subbasal).items():
            if name != "depth_bands_um":
                _number(f"subbasal.{name}", value)
        for band in self.subbasal.depth_bands_um:
            if not isinstance(band, (tuple, list)) or len(band) != 2:
                raise ValueError("Each depth band must contain two limits.")
            for value in band:
                _number("depth band", value)
        self.subbasal.validate()
        return self

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def cleanup_for(self, group):
        if self.whole_skin_cleanup != "legacy":
            return self.whole_skin_cleanup == "on"
        return str(group).casefold() in {
            "oldmice",
            "youngmice",
            "oldmice_validation",
            "youngmice_validation",
        }


def _number(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")


def _update(instance, values):
    if not isinstance(values, dict):
        raise ValueError("Configuration sections must be JSON objects.")
    unknown = values.keys() - {field.name for field in fields(instance)}
    if unknown:
        raise ValueError(f"Unknown configuration settings: {sorted(unknown)}")
    values = values.copy()
    for name in ("candidate1", "subbasal"):
        if name in values:
            values[name] = _update(getattr(instance, name), values[name])
    if "depth_bands_um" in values:
        bands = values["depth_bands_um"]
        if not isinstance(bands, list) or any(
            not isinstance(b, list) or len(b) != 2 for b in bands
        ):
            raise ValueError("depth_bands_um must be an array of two-limit arrays.")
        values["depth_bands_um"] = tuple(tuple(b) for b in bands)
    return replace(instance, **values)


def load_config(path):
    """Apply a partial JSON configuration; reject misspellings instead of ignoring them."""
    config = AnalysisConfig()
    if path is not None:
        config = _update(config, json.loads(Path(path).read_text(encoding="utf-8")))
    return config.validate()
