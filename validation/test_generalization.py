"""Anatomical decisions must not depend on blank canvas or implicit group names."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

import Skin_Section_Analysis as pipeline
from epidermis_analysis.configuration import AnalysisConfig


class GeneralizationTests(unittest.TestCase):
    def test_reference_and_repaired_tissue_ignore_blank_horizontal_padding(self):
        mask = np.zeros((600, 1000), bool)
        mask[100:200, 100:600] = True
        mask[100:500, 700:730] = True
        cfg = AnalysisConfig()
        scale = 0.621504  # Explicit synthetic geometry, not an acquisition fallback.
        px = pipeline.calculate_pixel_parameters(scale, cfg)
        reference, _ = pipeline.select_superficial_long_components(
            mask, 10000, config=cfg
        )
        for left, right in ((0, 1000), (1000, 0), (400, 600)):
            padded = np.pad(mask, ((0, 0), (left, right)))
            with self.subTest(left=left, right=right, step="reference"):
                actual, _ = pipeline.select_superficial_long_components(
                    padded, 10000, config=cfg
                )
                np.testing.assert_array_equal(
                    reference, actual[:, left : left + mask.shape[1]]
                )
            kwargs = dict(
                min_object_pixels=px["min_whole_skin_object_pixels"],
                smoothing_width_pixels=px["whole_skin_smoothing_width_pixels"],
                closing_radius_pixels=max(
                    1, round(cfg.whole_skin_context_closing_radius_um / scale)
                ),
                config=cfg,
            )
            expected = pipeline.repair_whole_skin_mask(mask, **kwargs)
            actual = pipeline.repair_whole_skin_mask(padded, **kwargs)
            np.testing.assert_array_equal(
                expected, actual[:, left : left + mask.shape[1]]
            )

    def test_attached_deep_tissue_is_preserved_but_lower_debris_is_removed(self):
        mask = np.zeros((120, 180), bool)
        mask[20:60, 10:170] = True
        mask[55:100, 80:84] = True  # Attached extension below the median edge.
        mask[105:110, 30:35] = True  # Detached debris below that edge.
        actual = pipeline.repair_whole_skin_mask(
            mask,
            min_object_pixels=100,
            smoothing_width_pixels=21,
            closing_radius_pixels=2,
        )
        self.assertTrue(actual[60:100, 80:84].all())
        self.assertFalse(actual[105:110, 30:35].any())
        self.assertTrue(actual[20:60, 10:170].all())

    def test_cleanup_choice_is_not_a_configuration_parameter(self):
        self.assertNotIn("whole_skin_cleanup", AnalysisConfig().to_dict())
        with self.assertRaises(TypeError):
            AnalysisConfig(whole_skin_cleanup="off")

    def test_direct_api_rejects_retired_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / "new_outputs"
            with self.assertRaises(TypeError):
                pipeline.process_sample(
                    "section",
                    root / "nerve.tif",
                    root / "dapi.tif",
                    output,
                    root / "launcher",
                    apply_mouse_whole_skin_cleanup=False,
                )
            self.assertFalse(output.exists())

    def test_direct_api_always_applies_the_same_basic_cleanup(self):
        class StopAfterCleanupArguments(Exception):
            pass

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dapi, nerve = root / "dapi.tif", root / "nerve.tif"
            image = np.ones((8, 12), np.uint16)
            for path in (dapi, nerve):
                tifffile.imwrite(path, image)
            for name in ("oldmice", "renamed", "conventionalized"):
                with (
                    self.subTest(name=name),
                    patch.object(pipeline, "run_ilastik_segmentation"),
                    patch.object(pipeline, "load_2d_image", return_value=image),
                    patch.object(
                        pipeline,
                        "repair_whole_skin_mask",
                        side_effect=StopAfterCleanupArguments,
                    ) as cleanup,
                ):
                    with self.assertRaises(StopAfterCleanupArguments):
                        pipeline.process_sample(
                            name,
                            nerve,
                            dapi,
                            root / "outputs",
                            root / "launcher",
                            config=AnalysisConfig(fallback_pixel_size_um=1.0),
                        )
                    self.assertEqual(
                        set(cleanup.call_args.kwargs),
                        {
                            "min_object_pixels",
                            "smoothing_width_pixels",
                            "closing_radius_pixels",
                            "config",
                        },
                    )
                    self.assertEqual(
                        cleanup.call_args.kwargs["smoothing_width_pixels"], 21
                    )
                    self.assertEqual(
                        cleanup.call_args.kwargs["closing_radius_pixels"], 8
                    )

    def test_empty_tissue_remains_an_explicit_failure(self):
        with self.assertRaisesRegex(ValueError, "No positive"):
            pipeline.select_superficial_long_components(np.zeros((20, 30), bool), 1)


if __name__ == "__main__":
    unittest.main()
