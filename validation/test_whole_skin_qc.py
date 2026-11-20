"""Cleanup diagnostics must expose exact stages without changing the masks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

import Skin_Section_Analysis as pipeline
from epidermis_analysis.whole_skin_qc import save_whole_skin_qc


class WholeSkinQCTests(unittest.TestCase):
    def test_exact_stages_changes_and_noninterference(self):
        raw = np.zeros((150, 200), bool)
        raw[20:90, 10:190] = True
        raw[85:125, 95:99] = True
        raw[45:60, 40:55] = False  # A hole too wide for the closing step.
        raw[30:65, 75:77] = False  # A narrow gap that closing can fill.
        raw[135:140, 30:35] = True  # Detached lower debris.
        original = raw.copy()
        args = dict(
            min_object_pixels=100, smoothing_width_pixels=21, closing_radius_pixels=2
        )
        expected = pipeline.repair_whole_skin_mask(raw, **args)
        steps = {}
        actual = pipeline.repair_whole_skin_mask(raw, qc_steps=steps, **args)
        np.testing.assert_array_equal(expected, actual)
        self.assertTrue(steps["raw"][135:140, 30:35].all())
        self.assertFalse(steps["filtered"][135:140, 30:35].any())
        self.assertFalse(steps["filtered"][30:65, 75:77].any())
        self.assertTrue(steps["closed"][30:65, 75:77].all())
        self.assertFalse(steps["closed"][45:60, 40:55].any())
        self.assertTrue(steps["filled"][45:60, 40:55].all())
        for key in ("raw", "filtered", "closed", "filled"):
            self.assertTrue(steps[key][90:125, 95:99].all())
        saved = {k: v.copy() for k, v in steps.items()}
        with tempfile.TemporaryDirectory() as tmp:
            rows = save_whole_skin_qc(tmp, raw.astype(np.uint16) * 2000, steps, 0.5)
            qc = Path(tmp) / "QC"
            self.assertTrue((qc / "00_whole_skin_cleanup.png").is_file())
            self.assertTrue((qc / "whole_skin/index.html").is_file())
            self.assertEqual(len(list((qc / "whole_skin").glob("*.tif"))), 5)
            np.testing.assert_array_equal(
                tifffile.imread(qc / "whole_skin/05_filled.tif") > 0, expected
            )
            self.assertAlmostEqual(
                pipeline.read_pixel_size_um(qc / "whole_skin/05_filled.tif")[0], 0.5
            )
            self.assertEqual(rows[1]["removed_pixels"], 25)
            self.assertEqual(rows[1]["added_pixels"], 0)
            self.assertEqual(
                sum(r["added_pixels"] - r["removed_pixels"] for r in rows),
                int(expected.sum()) - int(raw.sum()),
            )
        np.testing.assert_array_equal(raw, original)
        for key in steps:
            np.testing.assert_array_equal(steps[key], saved[key])


if __name__ == "__main__":
    unittest.main()
