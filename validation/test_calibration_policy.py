"""Calibration must come from metadata or an explicit acquisition setting."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
import Skin_Section_Analysis as pipeline
from epidermis_analysis.configuration import AnalysisConfig


class CalibrationPolicyTests(unittest.TestCase):
    def test_missing_metadata_requires_explicit_scale(self):
        self.assertIsNone(AnalysisConfig().validate().fallback_pixel_size_um)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "uncalibrated.tif"
            tifffile.imwrite(path, np.zeros((12, 16), np.uint16))
            with self.assertRaisesRegex(ValueError, "No usable pixel calibration"):
                pipeline.read_pixel_size_um(path)
            self.assertEqual(pipeline.read_pixel_size_um(path, 0.7), (0.7, "fallback"))
            for invalid in (0, -1, np.nan, np.inf, True):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    pipeline.read_pixel_size_um(path, invalid)

    def test_metadata_precedes_explicit_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "calibrated.tif"
            tifffile.imwrite(
                path,
                np.zeros((12, 16), np.uint16),
                imagej=True,
                resolution=(2, 2),
                metadata={"unit": "um"},
            )
            self.assertEqual(
                pipeline.read_pixel_size_um(path), (0.5, "ImageJ TIFF metadata")
            )
            self.assertEqual(
                pipeline.read_pixel_size_um(path, 0.7), (0.5, "ImageJ TIFF metadata")
            )

    def test_uncalibrated_pair_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("dapi.tif", "nerve.tif"):
                tifffile.imwrite(root / name, np.zeros((12, 16), np.uint16))
            with patch.object(pipeline, "run_ilastik_segmentation") as inference:
                with self.assertRaisesRegex(ValueError, "No usable pixel calibration"):
                    pipeline.process_sample(
                        "section",
                        root / "nerve.tif",
                        root / "dapi.tif",
                        root / "output",
                        root / "launcher",
                    )
                inference.assert_not_called()
