"""Public regression checks using synthetic inputs and temporary outputs only."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

import Skin_Section_Analysis as pipeline
from epidermis_analysis import provenance
from epidermis_analysis import measurement
from epidermis_analysis.components import measure_skeleton_graph_length


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_anisotropic_calibration_is_rejected(self):
        for kind in ("imagej", "ome", "standard"):
            with self.subTest(kind=kind):
                path = self.root / f"{kind}.tif"
                options = {
                    "imagej": dict(
                        imagej=True, resolution=(2, 4), metadata={"unit": "um"}
                    ),
                    "ome": dict(
                        ome=True, metadata={"PhysicalSizeX": 0.5, "PhysicalSizeY": 0.25}
                    ),
                    "standard": dict(
                        resolution=(20000, 40000), resolutionunit="CENTIMETER"
                    ),
                }[kind]
                tifffile.imwrite(path, np.zeros((8, 12), np.uint16), **options)
                with self.assertRaisesRegex(ValueError, "Anisotropic"):
                    pipeline.read_pixel_size_um(path)

    def test_calibration_accepts_roundoff_and_rejects_nonfinite(self):
        self.assertAlmostEqual(pipeline._isotropic_pixel_size([0.5, 0.50001]), 0.500005)
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "positive and finite"):
                pipeline._isotropic_pixel_size([value, 0.5])

    def test_output_protection_including_symlinks(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        for output in (inputs, inputs / "outputs", self.root):
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                provenance.validate_output_location(output, inputs)
        outputs = self.root / "outputs"
        outputs.mkdir()
        (outputs / "redirect").symlink_to(inputs, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            provenance.validate_output_location(outputs, inputs)

    def test_both_unmatched_channels_are_reported(self):
        for name in ("one_DAPI.tif", "two_BT3.tif"):
            tifffile.imwrite(self.root / name, np.zeros((8, 12), np.uint16))
        samples, missing = pipeline.discover_samples(self.root)
        self.assertEqual(samples, [])
        self.assertEqual({row["Sample"] for row in missing}, {"one", "two"})

    def test_skeleton_length_and_density_units(self):
        diagonal = np.eye(4, dtype=bool)
        self.assertAlmostEqual(
            measure_skeleton_graph_length(diagonal, 0.5), 3 * np.sqrt(2) * 0.5
        )
        roi = np.ones((4, 4), bool)
        result = measurement.measure(roi, diagonal, diagonal, 0.5).values
        self.assertEqual(result["area_um2"], 4)
        self.assertEqual(result["bt3_area_fraction"], 0.25)
        self.assertAlmostEqual(
            result["bt3_skeleton_density_um_per_mm2"], 3 * np.sqrt(2) * 0.5 * 1e6 / 4
        )
        corner = np.array([[True, True], [False, True]])
        self.assertEqual(measure_skeleton_graph_length(corner, 1), 2)

    def test_compartments_partition_tissue_and_preserve_input_masks(self):
        epidermis, dermis, basal, boundary, signal = (
            np.zeros((60, 80), bool) for _ in range(5)
        )
        epidermis[5:20, 5:75] = True
        dermis[20:55, 5:75] = True
        basal[17:20, 5:75] = True
        boundary[19, 5:75] = True
        signal[10:50, 30] = True
        before = signal.copy()
        path = np.column_stack((np.full(70, 19), np.arange(5, 75))).astype(float)
        aggregate, subbasal, four = measurement.analyze(
            epidermis, dermis, basal, boundary, [path], signal, 1
        )
        masks = four.compartment_masks
        np.testing.assert_array_equal(
            masks["upper_epidermis"] | masks["basal_epidermis"], epidermis
        )
        np.testing.assert_array_equal(
            masks["subbasal_dermis"] | masks["deep_dermis"], dermis
        )
        self.assertFalse(np.any(masks["subbasal_dermis"] & masks["deep_dermis"]))
        np.testing.assert_array_equal(signal, before)
        self.assertEqual(aggregate["metrics"]["epidermal_nerve_area_um2"], 10)
        self.assertEqual(aggregate["metrics"]["dermal_nerve_area_um2"], 30)
        self.assertEqual(
            subbasal.metrics["subbasal_roi_area_um2"], masks["subbasal_dermis"].sum()
        )

    def test_mixed_legacy_rows_contribute_to_mouse_mean(self):
        table = pd.DataFrame(
            {
                "Sample": ["mouse_section1", "mouse_section2", "mouse_section3"],
                "epidermal_nerve_area_um2_per_boundary_mm": [10, np.nan, np.nan],
                "epidermal_BT3_area_um2_per_boundary_mm": [999, 20, np.nan],
                "epidermal_BT3_area_per_boundary_length": [999, 999, 0.03],
                "epidermal_nerve_skeleton_length_um_per_boundary_mm": [
                    1,
                    np.nan,
                    np.nan,
                ],
                "epidermal_BT3_skeleton_length_um_per_boundary_mm": [999, 2, np.nan],
                "epidermal_BT3_skeleton_length_per_boundary_length": [999, 999, 0.003],
            }
        )
        result = pipeline.build_biological_replicate_averages(table).iloc[0]
        self.assertEqual(result["Number of sections"], 3)
        self.assertEqual(result["Mean epidermal nerve area (um2 per boundary mm)"], 20)
        self.assertEqual(
            result["Mean epidermal nerve skeleton length (um per boundary mm)"], 2
        )

    def test_segmentation_cache_checks_content_and_process_success(self):
        dapi, project, launcher = (
            self.root / name for name in ("dapi.tif", "model.ilp", "launcher")
        )
        tifffile.imwrite(dapi, np.zeros((8, 12), np.uint16))
        project.write_bytes(b"model1")
        launcher.write_bytes(b"launcher1")
        output = self.root / "output.tif"

        def export(command, **kwargs):
            for name, value in provenance.ILASTIK_ENVIRONMENT.items():
                self.assertEqual(kwargs["env"][name], value)
            destination = next(
                x.split("=", 1)[1]
                for x in command
                if x.startswith("--output_filename_format=")
            )
            tifffile.imwrite(destination, np.ones((8, 12), np.uint8))
            return subprocess.CompletedProcess(command, 0, "", "")

        def run():
            pipeline.run_ilastik_segmentation(
                dapi, output, launcher, project, "simple segmentation stage 2"
            )

        with (
            patch.object(pipeline.subprocess, "run", side_effect=export) as process,
            redirect_stdout(io.StringIO()),
        ):
            run()
            run()
            self.assertEqual(process.call_count, 1)
            # Byte changes invalidate even when filenames are unchanged.
            for path in (project, launcher, output):
                with path.open("ab") as stream:
                    stream.write(b"changed")
                run()
            tifffile.imwrite(dapi, np.ones((8, 12), np.uint16))
            run()
            self.assertEqual(process.call_count, 5)
            Path(str(output) + ".json").write_text("{}")
            run()
            self.assertEqual(process.call_count, 6)

        def failed_export(command, **kwargs):
            export(command, **kwargs)
            return subprocess.CompletedProcess(command, 1, "", "failed after export")

        original = output.read_bytes()
        with (
            patch.object(pipeline.subprocess, "run", side_effect=failed_export),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "return code 1"):
                pipeline.run_ilastik_segmentation(
                    dapi, output, launcher, project, "other export"
                )
        self.assertEqual(output.read_bytes(), original)

    def batch_fixture(self):
        inputs, outputs = self.root / "inputs", self.root / "outputs"
        for group in ("", "group", "nan"):
            directory = inputs / group
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ("DAPI", "BT3"):
                tifffile.imwrite(
                    directory / f"mouse_section1_{suffix}.tif",
                    np.zeros((8, 12), np.uint16),
                )
        model = self.root / "model.ilp"
        model.write_bytes(b"synthetic model")

        def process(sample_name, sample_output_folder, **kwargs):
            empty = np.zeros((8, 12), bool)
            metrics = pipeline.quantify_nerve_by_regions(empty, empty, empty, empty, 1)[
                "metrics"
            ]
            row = {
                "Sample": sample_name,
                "sample_id": sample_name,
                "pixel_size_um": 1,
                "nerve_segmentation_method": "test",
                "nerve_threshold": 1500,
                **metrics,
            }
            sample_output_folder.mkdir(parents=True, exist_ok=True)
            for name in provenance.RESULT_FILES:
                if name.endswith(".csv"):
                    pd.DataFrame([row]).to_csv(sample_output_folder / name, index=False)
                else:
                    (sample_output_folder / name).write_text("{}")
            return [row]

        stack = self.enterContext(ExitStack())
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch.object(pipeline, "EPIDERMIS_ILASTIK_PROJECT", model))
        stack.enter_context(patch.object(pipeline, "WHOLE_SKIN_ILASTIK_PROJECT", model))
        stack.enter_context(
            patch.object(pipeline, "find_ilastik_executable", return_value=model)
        )
        mock = stack.enter_context(
            patch.object(pipeline, "process_sample", side_effect=process)
        )
        return inputs, outputs, mock

    def test_root_summary_and_log_keep_all_groups(self):
        inputs, outputs, _ = self.batch_fixture()
        self.assertEqual(pipeline.main(inputs, outputs)["completed"], 3)
        for name in ("combined_BT3_quantification_results.csv", "batch_run_log.csv"):
            table = pd.read_csv(outputs / name, keep_default_na=False)
            self.assertEqual(set(table.Group), {".", "group", "nan"})
        workbook = pd.read_excel(
            outputs / "BT3_quantification_by_biological_replicate.xlsx",
            keep_default_na=False,
        )
        self.assertEqual(len(workbook), 3)
        self.assertTrue((outputs / "nan" / "batch_run_log.csv").exists())

    def test_skip_checks_inputs_settings_models_and_result_integrity(self):
        inputs, outputs, process = self.batch_fixture()
        pipeline.main(inputs, outputs)
        self.assertEqual(
            pipeline.main(inputs, outputs, skip_already_processed=True)["skipped"], 3
        )
        self.assertEqual(process.call_count, 3)
        (outputs / "group/mouse_section1/analysis_parameters.json").write_text(
            '{"changed": true}'
        )
        self.assertEqual(
            pipeline.main(inputs, outputs, skip_already_processed=True)["completed"], 1
        )
        tifffile.imwrite(inputs / "mouse_section1_BT3.tif", np.ones((8, 12), np.uint16))
        self.assertEqual(
            pipeline.main(inputs, outputs, skip_already_processed=True)["completed"], 1
        )
        self.assertEqual(
            pipeline.main(
                inputs, outputs, skip_already_processed=True, whole_skin_cleanup=True
            )["completed"],
            3,
        )
        (self.root / "model.ilp").write_bytes(b"new model")
        self.assertEqual(
            pipeline.main(
                inputs, outputs, skip_already_processed=True, whole_skin_cleanup=True
            )["completed"],
            3,
        )

    def test_failed_rerun_removes_stale_success_and_summaries(self):
        inputs, outputs, process = self.batch_fixture()
        pipeline.main(inputs, outputs)
        (outputs / "missing_file_pairs.csv").write_text("old missing files")
        process.side_effect = RuntimeError("synthetic failure")
        self.assertEqual(pipeline.main(inputs, outputs)["failed"], 3)
        self.assertFalse((outputs / "combined_BT3_quantification_results.csv").exists())
        self.assertFalse((outputs / "missing_file_pairs.csv").exists())
        self.assertEqual(list(outputs.rglob("completion.json")), [])
        self.assertEqual(
            list(outputs.rglob(pipeline.NERVE_QUANTIFICATION_FILENAME)), []
        )
        self.assertEqual(len(pd.read_csv(outputs / "batch_run_log.csv")), 3)


if __name__ == "__main__":
    unittest.main()
