"""Root selection invariance and disclosure of omitted surface evidence."""

from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from epidermis_analysis.candidate1_cleanup import (
    _surface_ownership,
    run_candidate1_cleanup,
)
from epidermis_analysis.candidate1_config import Candidate1Config
from epidermis_analysis.candidate1_qc import save_selection_decisions
from epidermis_analysis.configuration import AnalysisConfig


class SurfaceOwnershipTests(unittest.TestCase):
    def test_padding_and_separate_pieces(self):
        labels = np.zeros((40, 80), int)
        labels[10, 5:35] = 1
        labels[25, 5:15] = 2
        labels[15, 50:75] = 3
        pieces = np.zeros_like(labels)
        pieces[5:35, 3:38] = 1
        pieces[5:35, 45:78] = 2
        surface = np.zeros_like(labels, dtype=bool)
        surface[5, 3:38] = True
        surface[5, 45:78] = True
        features = pd.DataFrame(
            {
                "component_id": [1, 2, 3],
                "component_whole_tissue_piece": [1, 1, 2],
                "principal_geodesic_length_um": [30.0, 10.0, 25.0],
            }
        )
        before = _surface_ownership(labels, features, pieces, surface)
        after = _surface_ownership(
            np.pad(labels, 20), features, np.pad(pieces, 20), np.pad(surface, 20)
        )
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(
            set(before.loc[before.surface_ownership_root, "component_id"]), {1, 3}
        )
        self.assertEqual(before.surface_ownership_pixels.sum(), surface.sum())

    def test_legacy_root_thresholds_do_not_gate_surface_roots(self):
        raw = np.zeros((160, 240), bool)
        raw[30:35, 10:150] = True
        raw[30:35, 170:225] = True
        whole = np.zeros_like(raw)
        whole[10:145, 5:230] = True
        surface = np.zeros_like(raw)
        surface[10, 5:230] = True
        cfg = replace(
            Candidate1Config(),
            minimum_root_geodesic_um=10000.0,
            maximum_endpoint_gap_um=0.0,
        )
        result = run_candidate1_cleanup(
            raw.astype(float), raw, whole, surface, 1.0, cfg
        )
        self.assertEqual(len(result.selection.roots), 1)
        with tempfile.TemporaryDirectory() as folder:
            save_selection_decisions(folder, result)
            rows = pd.read_csv(Path(folder) / "QC/component_decisions.csv")
            self.assertFalse(rows.legacy_root_thresholds_applied.any())
            self.assertFalse(rows.root_length_pass.any())
            report = json.loads((Path(folder) / "QC/surface_coverage.json").read_text())
            self.assertEqual(report["root_selection"], "surface_ownership")
            self.assertGreater(report["unselected_surface_fraction"], 0)
            self.assertEqual(len(report["unselected_surface_component_ids"]), 1)

    def test_invalid_rule_rejected(self):
        with self.assertRaises(ValueError):
            AnalysisConfig(
                candidate1=replace(Candidate1Config(), root_selection="guess")
            ).validate()
