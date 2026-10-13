"""Regression checks for complete, auditable component decisions."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from epidermis_analysis.candidate1_cleanup import run_candidate1_cleanup
from epidermis_analysis.candidate1_qc import save_selection_decisions


class SelectionDecisionTests(unittest.TestCase):
    def test_rejected_small_component_is_not_omitted_from_ledger(self):
        raw = np.zeros((180, 440), bool)
        raw[30:45, 10:430] = True
        raw[100:103, 100:103] = True
        whole = np.zeros_like(raw)
        whole[10:160, 5:435] = True
        surface = np.zeros_like(raw)
        surface[10, 5:435] = True
        result = run_candidate1_cleanup(
            raw.astype(np.uint16) * 2000, raw, whole, surface, 1.0
        )
        with tempfile.TemporaryDirectory() as folder:
            save_selection_decisions(folder, result)
            rows = pd.read_csv(Path(folder) / "QC/component_decisions.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                set(rows.component_id), set(np.unique(result.labels)) - {0}
            )
            selected = rows.loc[rows.selected, "component_id"]
            self.assertEqual(set(selected), set(result.selection.selected_nodes))
            rejected = rows.loc[~rows.node_eligible]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected.iloc[0].decision, "excluded_by_node_length")
            self.assertFalse(rejected.iloc[0].automatic_root_eligible)
            edges = pd.read_csv(Path(folder) / "QC/candidate_edge_decisions.csv")
            self.assertTrue(edges.empty)
            self.assertIn("selected", edges.columns)
