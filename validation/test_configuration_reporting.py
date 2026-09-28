"""Portable settings, sample identity, lossless masks, and computation-free reports."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

import Skin_Section_Analysis as pipeline
from epidermis_analysis.configuration import AnalysisConfig, load_config
from epidermis_analysis.samples import (
    MANIFEST_COLUMNS,
    load_samples,
    write_resolved_samples,
)
from epidermis_analysis.reporting import rebuild_reports
import test_reliability


class ConfigurationReportingTests(unittest.TestCase):
    batch_fixture = test_reliability.ReliabilityTests.batch_fixture

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def manifest(self, inputs, **overrides):
        row = dict(
            zip(
                MANIFEST_COLUMNS,
                (
                    "001",
                    "007",
                    "renamed/group",
                    "mouse_section1_DAPI.tif",
                    "mouse_section1_BT3.tif",
                    "on",
                ),
            )
        )
        row.update(overrides)
        path = self.root / "samples.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_config_roundtrip_partial_overrides_and_validation(self):
        path = self.root / "settings.json"
        defaults = AnalysisConfig()
        self.assertEqual(defaults, pipeline.current_analysis_config())
        path.write_text(json.dumps(defaults.to_dict()))
        self.assertEqual(load_config(path), defaults)
        path.write_text(
            json.dumps(
                {
                    "manual_nerve_threshold": 1700,
                    "candidate1": {"nearest_neighbors_per_endpoint": 3},
                    "subbasal": {"depth_um": 15},
                }
            )
        )
        config = load_config(path)
        self.assertEqual(config.manual_nerve_threshold, 1700)
        self.assertEqual(config.candidate1.nearest_neighbors_per_endpoint, 3)
        self.assertEqual(config.subbasal.depth_um, 15)
        self.assertEqual(config.fallback_pixel_size_um, defaults.fallback_pixel_size_um)
        for value in (
            {"typo": 3},
            {"candidate1": {"typo": 3}},
            {"fallback_pixel_size_um": 0},
            {"manual_nerve_threshold": True},
            {"whole_skin_cleanup": {}},
            {"manual_nerve_threshold": float("nan")},
            {"epidermis_label": 1.5},
            {"candidate1": {"nearest_neighbors_per_endpoint": 0}},
            {"subbasal": {"depth_bands_um": [[20, 10]]}},
        ):
            with self.subTest(value=value):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_manifest_roundtrip_and_independent_cleanup(self):
        inputs, _, _ = self.batch_fixture()
        path = self.manifest(inputs)
        samples = load_samples(path, inputs)
        self.assertEqual(samples[0]["sample_name"], "001")
        self.assertEqual(samples[0]["mouse_id"], "007")
        self.assertTrue(samples[0]["apply_mouse_whole_skin_cleanup"])
        self.assertFalse(AnalysisConfig().cleanup_for("renamed/group"))
        resolved = self.root / "resolved.csv"
        write_resolved_samples(resolved, samples, inputs)
        self.assertEqual(load_samples(resolved, inputs), samples)
        # Header order does not alter the meaning of named columns.
        table = pd.read_csv(path, dtype=str)
        table[list(reversed(MANIFEST_COLUMNS))].to_csv(path, index=False)
        self.assertEqual(load_samples(path, inputs), samples)

    def test_manifest_rejects_escape_duplicate_and_implicit_cleanup(self):
        inputs, _, _ = self.batch_fixture()
        for override in (
            {"sample_id": "../escape"},
            {"group": "../escape"},
            {"dapi": "../outside.tif"},
            {"dapi": "/absolute.tif"},
            {"whole_skin_cleanup": "legacy"},
            {"nerve": "mouse_section1_DAPI.tif"},
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    load_samples(self.manifest(inputs, **override), inputs)
        path = self.manifest(inputs)
        lines = path.read_text().splitlines()
        path.write_text("\n".join([*lines, lines[1]]) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_samples(path, inputs)

    def test_lossless_tiff_compression(self):
        mask = np.zeros((512, 768), bool)
        mask[100:400, 200:500] = True
        path = self.root / "mask.tif"
        pipeline.save_binary_image(path, mask)
        decoded = tifffile.imread(path)
        self.assertEqual(decoded.dtype, np.uint8)
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8) * 255)
        self.assertLess(path.stat().st_size, mask.size // 10)

    def test_explicit_mouse_ids_drive_section_means(self):
        table = pd.DataFrame(
            {
                "Group": ["g", "g", "g"],
                "Sample": ["unrelated", "names", "third"],
                "mouse_id": ["007", "007", "008"],
                "epidermal_nerve_area_um2_per_boundary_mm": [10, 30, 99],
                "epidermal_nerve_skeleton_length_um_per_boundary_mm": [2, 4, 9],
            }
        )
        result = pipeline.build_biological_replicate_averages(table).set_index(
            "Biological replicate"
        )
        self.assertEqual(result.loc["007", "Number of sections"], 2)
        self.assertEqual(
            result.loc["007", "Mean epidermal nerve area (um2 per boundary mm)"], 20
        )

    def test_explicit_settings_and_identity_invalidate_skip(self):
        inputs, outputs, process = self.batch_fixture()
        manifest = self.manifest(inputs)
        config = replace(
            AnalysisConfig(), whole_skin_cleanup="off", manual_nerve_threshold=1700
        )
        pipeline.main(inputs, outputs, config=config, samples_manifest=manifest)
        self.assertEqual(process.call_args.kwargs["config"], config)
        self.assertTrue(process.call_args.kwargs["apply_mouse_whole_skin_cleanup"])
        self.assertEqual(process.call_args.kwargs["mouse_id"], "007")
        result = pipeline.main(
            inputs,
            outputs,
            config=config,
            samples_manifest=manifest,
            skip_already_processed=True,
        )
        self.assertEqual(result["skipped"], 1)
        config = replace(config, manual_nerve_threshold=1800)
        result = pipeline.main(
            inputs,
            outputs,
            config=config,
            samples_manifest=manifest,
            skip_already_processed=True,
        )
        self.assertEqual(result["completed"], 1)
        self.manifest(inputs, mouse_id="008")
        result = pipeline.main(
            inputs,
            outputs,
            config=config,
            samples_manifest=manifest,
            skip_already_processed=True,
        )
        self.assertEqual(result["completed"], 1)
        self.assertEqual(
            json.loads((outputs / "configuration.resolved.json").read_text()),
            config.to_dict(),
        )
        self.assertEqual(
            load_samples(outputs / "samples.resolved.csv", inputs)[0]["mouse_id"], "008"
        )

    def test_reports_rebuild_without_analysis_or_input_files(self):
        inputs, outputs, process = self.batch_fixture()
        pipeline.main(inputs, outputs)
        section_files = {
            p: p.read_bytes()
            for p in outputs.rglob("*")
            if p.is_file() and p.parent.name == "mouse_section1"
        }
        combined = outputs / "combined_BT3_quantification_results.csv"
        expected = pd.read_csv(combined, float_precision="round_trip")
        # The archived report must not depend on current models, inputs, or source.
        for path in inputs.rglob("*.tif"):
            path.unlink()  # Temporary synthetic fixtures only.
        process.reset_mock()
        with patch.object(
            pipeline, "find_ilastik_executable", side_effect=AssertionError("inference")
        ):
            self.assertEqual(
                pipeline.cli(
                    ["--reports-only", "--output-dir", str(outputs), "--no-excel"]
                ),
                0,
            )
        process.assert_not_called()
        rebuilt = pd.read_csv(combined, float_precision="round_trip")
        pd.testing.assert_frame_equal(
            expected, rebuilt[expected.columns], check_exact=True
        )
        self.assertFalse(list(outputs.rglob("*.xlsx")))
        self.assertTrue(
            all(p.read_bytes() == original for p, original in section_files.items())
        )
        # Tampered measurements are rejected before replacing a valid report.
        protected_report = combined.read_bytes()
        (outputs / "group/mouse_section1/BT3_quantification_results.csv").write_text(
            "changed"
        )
        with self.assertRaisesRegex(ValueError, "unverified"):
            rebuild_reports(outputs)
        self.assertEqual(combined.read_bytes(), protected_report)

    def test_reports_use_latest_batch_membership_and_preserve_string_ids(self):
        inputs, outputs, _ = self.batch_fixture()
        pipeline.main(inputs, outputs, samples_manifest=self.manifest(inputs))
        self.assertEqual(rebuild_reports(outputs), 1)
        table = pd.read_excel(
            outputs / "BT3_quantification_by_biological_replicate.xlsx", dtype=str
        )
        self.assertEqual(table.iloc[0]["Biological replicate"], "007")
        self.assertEqual(table.iloc[0]["Sample"], "001")
        log = outputs / "batch_run_log.csv"
        table = pd.read_csv(log, dtype=str, keep_default_na=False)
        table["Status"] = "Failed"
        table.to_csv(log, index=False)
        self.assertEqual(rebuild_reports(outputs), 0)
        self.assertFalse((outputs / "combined_BT3_quantification_results.csv").exists())
        self.assertTrue((outputs / "renamed/group/001/completion.json").exists())
