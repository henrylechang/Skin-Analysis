# -*- coding: utf-8 -*-
"""
Batch analysis of skin-section nerve-signal images with Ilastik segmentation.

Expected input naming (configurable below):
    Sample01_nerve.tif
    Sample01_DAPI.tif

Legacy ``Sample01_BT3.tif`` nerve-signal names remain supported.

For every sample, this script:
    1. Runs the DAPI image through the trained epidermis Ilastik classifier.
    2. Runs the same DAPI image through the trained whole-skin Ilastik classifier.
    3. Exports both Simple Segmentation TIFF label images automatically.
    4. Reads TIFF pixel calibration and converts physical thresholds to pixels.
    5. Repairs whole tissue and traces its superficial edge.
    6. Cleans the epidermal band and reconstructs the anatomical basal boundary.
    7. Thresholds nerve signal and removes long objects near the tissue surface.
    8. Quantifies nerve area and skeleton length within four tissue compartments.
    9. Saves three essential per-sample QC images plus quantitative outputs.

IMPORTANT:
    Keep the trained .ilp files in the Ilastic_models folder.
    If automatic Ilastik detection fails, set ILASTIK_EXE manually.
"""

import argparse
import gc
import json
from dataclasses import asdict
import os
import re
import shutil
import sys
import subprocess
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    median_filter,
    percentile_filter,
)
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.measure import label, regionprops

from epidermis_analysis.candidate1_cleanup import (
    Candidate1Failure,
    assert_candidate1_invariants,
    run_candidate1_cleanup,
)
from epidermis_analysis.candidate1_qc import (
    save_candidate1_qc,
    save_candidate1_failure_qc,
    save_final_combined_qc,
)
from epidermis_analysis.reconstruction import (
    reconstruct_explicit_interface,
)
from epidermis_analysis.four_compartment import (
    assert_aggregate_metrics_match_existing,
    save_four_compartment_qc,
)
from epidermis_analysis.measurement import (
    analyze as analyze_measurements,
    calculate_calibrated_skeleton_length,
    quantify_regions,
    LEGACY_BT3_METRIC_NAMES,
)
from epidermis_analysis.subbasal import (
    SubbasalConfig,
)
from epidermis_analysis.candidate1_config import CANDIDATE1_CONFIG
from epidermis_analysis.provenance import (
    ILASTIK_ENVIRONMENT,
    cached_segmentation_matches,
    completed_sample_matches,
    file_sha256,
    mark_sample_complete,
    runtime_provenance,
    segmentation_identity,
    validate_output_location,
    write_json,
)


# ============================================================
# BATCH FILE PATHS AND NAMING
# ============================================================

# Resolve every project path relative to this Python script. This lets the
# repository run after being cloned or moved without editing personal paths.
PROJECT_ROOT = Path(__file__).resolve().parent

# Place matched TIFF pairs here, for example:
#   Mouse1_DAPI.tif
#   Mouse1_nerve.tif
INPUT_FOLDER = PROJECT_ROOT / "Input_Data"

# Trained Ilastik two-stage Autocontext projects; both consume DAPI.
EPIDERMIS_ILASTIK_PROJECT = (
    PROJECT_ROOT / "Ilastic_models" / "Epidermis_test_Improved.ilp"
)

WHOLE_SKIN_ILASTIK_PROJECT = PROJECT_ROOT / "Ilastic_models" / "Wholeskin_test.ilp"

# Optional explicit Ilastik launcher or macOS .app bundle path.
# Leaving this as None enables portable detection; see README.md.
ILASTIK_EXE = None

# The script creates this folder automatically if it does not exist.
OUTPUT_ROOT = PROJECT_ROOT / "Output_Data"

# File endings used to match files belonging to one sample.
# Example:
#   Mouse1_nerve.tif  (preferred)
#   Mouse1_BT3.tif    (legacy, still accepted)
#   Mouse1_DAPI.tif
#
# Both masks are generated automatically from DAPI using the two
# Ilastik projects above.
NERVE_SIGNAL_SUFFIX = "_nerve"
LEGACY_NERVE_SIGNAL_SUFFIXES = ("_BT3",)
DAPI_SUFFIX = "_DAPI"

# Accepted TIFF extensions. Matching is case-insensitive.
TIFF_EXTENSIONS = {".tif", ".tiff"}

# When True, skip completed samples only when their provenance still matches.
SKIP_ALREADY_PROCESSED = False

# Reuse existing model outputs when rerunning downstream analysis.
REUSE_EXISTING_ILASTIK_SEGMENTATIONS = True

# ============================================================
# ILASTIK SETTINGS
# ============================================================

# Both current Ilastik projects are two-stage Autocontext workflows.
EPIDERMIS_ILASTIK_EXPORT_SOURCE = "simple segmentation stage 2"
WHOLE_SKIN_ILASTIK_EXPORT_SOURCE = "simple segmentation stage 2"

# Label number assigned to your dense epidermis DAPI band in Ilastik.
EPIDERMIS_LABEL = 1

# Whole-skin segmentation labels:
#   1 = whole skin
#   2 = background
WHOLE_SKIN_LABEL = 1


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# The script reads pixel size from each original TIFF. This value is used only
# when usable spatial calibration cannot be found in the TIFF metadata.
FALLBACK_PIXEL_SIZE_UM = 0.621504

# Minimum area required for a long whole-skin component to help define the
# trusted skin surface and its lower boundary.
MIN_WHOLE_SKIN_OBJECT_AREA_UM2 = 3862.67222016

# Median-filter width for the traced bottom of the trusted whole-skin objects.
WHOLE_SKIN_BOUNDARY_SMOOTHING_WIDTH_UM = 20.0
# Close gaps of up to approximately this physical radius before filling
# enclosed holes. This repaired mask supplies compartment context only.
WHOLE_SKIN_CONTEXT_CLOSING_RADIUS_UM = 8.0
# Mouse-paw whole-tissue cleanup. Disconnected detections must lie completely
# beyond this vertical margin from the long superficial skin envelope before
# they are rejected. The lower dermal contour uses a lower rolling percentile
# so narrow fat projections cannot pull the whole-skin boundary downward.
WHOLE_SKIN_DISCONNECTED_VERTICAL_MARGIN_UM = 50.0
WHOLE_SKIN_BASAL_SMOOTHING_WIDTH_UM = 100.0
WHOLE_SKIN_BASAL_PERCENTILE = 35.0
# Minimum horizontal span as a fraction of total image width.
MIN_COMPONENT_WIDTH_FRACTION = 0.025

# A long component must be the uppermost qualifying object across at least this
# fraction of its own horizontal span to define the superficial epidermis.
# This rejects deeper objects that appear only briefly inside epidermis gaps.
MIN_SUPERFICIAL_ENVELOPE_FRACTION = 0.5

# Active epidermal area/skeleton artifact filter. A threshold-positive connected
# object is removed intact only when it is both long and close to the true
# superficial boundary traced from the repaired whole-tissue mask. This avoids
# imposing a blanket maximum epidermal height.
SUPERFICIAL_NERVE_EXCLUSION_DISTANCE_UM = 5.0
MIN_SUPERFICIAL_NERVE_OBJECT_LENGTH_UM = 20.0

# Fixed grayscale nerve-signal threshold applied identically to every sample
# for area and skeleton-length quantification.
MANUAL_NERVE_THRESHOLD = 1500

# Measurement-only sub-basal BT3 compartment. A global shortcut DAG removes
# narrow dermal-facing return detours, and a cubic local polynomial fit over a
# 40 um arc-length support suppresses raster noise without flattening genuine
# basal curvature. The derived reference never replaces the anatomical boundary.
SUBBASAL_DEPTH_UM = 20.0
DERMAL_DEPTH_REFERENCE_SMOOTHING_UM = 40.0
# Backward-compatible name for downstream configuration imports.
SUBBASAL_MACRO_SMOOTH_UM = DERMAL_DEPTH_REFERENCE_SMOOTHING_UM
SUBBASAL_DEEP_OUTLIER_UM = 15.0
SUBBASAL_MAX_APPENDAGE_WIDTH_UM = 100.0
SUBBASAL_RESAMPLE_UM = 1.0
SUBBASAL_DEPTH_BANDS_UM = ((0.0, 20.0),)

# Compact production artifacts reused by read-only downstream validation.
EPIDERMIS_REGION_FILENAME = "06_reconstructed_epidermis_region.tif"
DERMIS_REGION_FILENAME = "07_dermis_region.tif"
EPIDERMIS_DERMIS_BOUNDARY_FILENAME = "08_final_epidermis_dermis_boundary.tif"
FILTERED_EPIDERMAL_NERVE_FILENAME = "11_epidermal_BT3_mask.tif"
FILTERED_DERMAL_NERVE_FILENAME = "12_dermal_BT3_mask.tif"
EPIDERMAL_NERVE_SKELETON_FILENAME = "13_epidermal_BT3_skeleton.tif"
DERMAL_NERVE_SKELETON_FILENAME = "14_dermal_BT3_skeleton.tif"
NERVE_QUANTIFICATION_FILENAME = "BT3_quantification_results.csv"
SUBBASAL_QUANTIFICATION_FILENAME = "subbasal_BT3_quantification_results.csv"
FOUR_COMPARTMENT_QUANTIFICATION_FILENAME = (
    "four_compartment_BT3_quantification_results.csv"
)

# Compatibility aliases for downstream tools that consume the historical BT3
# API and output schema. New production code uses nerve terminology above.
BT3_SUFFIX = LEGACY_NERVE_SIGNAL_SUFFIXES[0]
SUPERFICIAL_BT3_EXCLUSION_DISTANCE_UM = SUPERFICIAL_NERVE_EXCLUSION_DISTANCE_UM
MIN_SUPERFICIAL_BT3_OBJECT_LENGTH_UM = MIN_SUPERFICIAL_NERVE_OBJECT_LENGTH_UM
MANUAL_BT3_THRESHOLD = MANUAL_NERVE_THRESHOLD
BT3_QUANTIFICATION_FILENAME = NERVE_QUANTIFICATION_FILENAME
FILTERED_EPIDERMAL_BT3_FILENAME = FILTERED_EPIDERMAL_NERVE_FILENAME
FILTERED_DERMAL_BT3_FILENAME = FILTERED_DERMAL_NERVE_FILENAME
EPIDERMAL_BT3_SKELETON_FILENAME = EPIDERMAL_NERVE_SKELETON_FILENAME
DERMAL_BT3_SKELETON_FILENAME = DERMAL_NERVE_SKELETON_FILENAME


# ============================================================
# ILASTIK FUNCTIONS
# ============================================================


def _ilastik_launcher(path):
    """Normalize an executable or macOS app bundle to a runnable launcher."""
    path = Path(path).expanduser()
    if path.is_dir() and path.suffix.lower() == ".app":
        path = path / "Contents" / "ilastik-release" / "run_ilastik.sh"
    if path.is_file() and (sys.platform == "win32" or os.access(path, os.X_OK)):
        return path.resolve()
    return None


def _ilastik_search_roots():
    """Ordered, shallow installation locations; never scan input data."""
    home = Path.home()
    if sys.platform == "darwin":
        return [Path("/Applications"), home / "Applications", home / "Downloads", home]
    if sys.platform == "win32":
        return [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            home,
        ]
    return [Path("/opt"), home / ".local" / "opt", home]


def _unique_ilastik_launcher(candidates, source):
    paths = sorted({p for p in candidates if p is not None}, key=str)
    if len(paths) > 1:
        listing = "\n".join(f"  {p}" for p in paths)
        raise RuntimeError(
            f"Multiple Ilastik installations found in {source}:\n{listing}\n"
            "Select one explicitly with --ilastik-exe or the ILASTIK_EXE "
            "environment variable."
        )
    return paths[0] if paths else None


def find_ilastik_executable(configured_path=None):
    """Resolve CLI > script setting > environment > PATH > install folders.

    Explicit settings must be valid; ambiguous automatic matches are errors.
    Symlinks to the same launcher count as a single installation.
    """
    for requested_path in (configured_path, ILASTIK_EXE, os.environ.get("ILASTIK_EXE")):
        if requested_path is not None:
            launcher = _ilastik_launcher(requested_path)
            if launcher is None:
                raise FileNotFoundError(
                    "The configured Ilastik path is not a runnable launcher "
                    f"or supported .app bundle:\n{requested_path}"
                )
            return launcher

    names = (
        ("ilastik.exe", "run-ilastik.bat")
        if sys.platform == "win32"
        else ("run_ilastik.sh", "ilastik")
    )
    path_candidates = []
    for name in names:
        found = shutil.which(name)
        if found:
            path_candidates.append(_ilastik_launcher(found))
    launcher = _unique_ilastik_launcher(path_candidates, "PATH")
    if launcher is not None:
        return launcher

    roots = _ilastik_search_roots()
    candidates = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory in sorted(root.iterdir(), key=lambda p: p.name):
            if not directory.name.lower().startswith("ilastik"):
                continue
            if sys.platform == "darwin" and directory.suffix.lower() == ".app":
                candidates.append(_ilastik_launcher(directory))
            elif directory.is_dir():
                candidates.extend(_ilastik_launcher(directory / name) for name in names)
    launcher = _unique_ilastik_launcher(candidates, "installation folders")
    if launcher is not None:
        return launcher

    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(
        "Could not automatically find Ilastik.\n"
        f"Searched PATH and Ilastik folders directly inside: {searched}.\n"
        "Install Ilastik 1.4.2 separately, then pass --ilastik-exe PATH "
        "or set the ILASTIK_EXE environment variable.\n"
        "On macOS, PATH may be the .app bundle or its "
        "Contents/ilastik-release/run_ilastik.sh launcher; "
        "on Windows use ilastik.exe; on Linux use run_ilastik.sh."
    )


def run_ilastik_segmentation(
    dapi_path,
    output_path,
    ilastik_executable,
    project_file,
    export_source,
    reuse_existing=REUSE_EXISTING_ILASTIK_SEGMENTATIONS,
):
    """
    Run a two-stage Ilastik Autocontext project headlessly and export
    one Simple Segmentation Stage 2 label TIFF.

    The original .ilp file stays in place so relative training-image paths
    remain valid. The input image and exported segmentation are stored in a
    temporary job directory during the run (its parent may contain spaces).
    """
    if not project_file.exists():
        raise FileNotFoundError(
            "The Ilastik project file does not exist:\n"
            f"{project_file}\n\n"
            "Change the corresponding Ilastik project path near the top of this script."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    identity = segmentation_identity(
        dapi_path, project_file, ilastik_executable, export_source
    )
    if reuse_existing and cached_segmentation_matches(output_path, identity):
        print(f"  Reusing verified Ilastik segmentation: {output_path.name}")
        return output_path

    # Keep headless jobs inside the writable project tree. A machine-level
    # C:\ilastik_batch_temp directory can have a different owner/ACL, causing
    # Ilastik logging or output creation to stall or fail without a useful
    # return code.
    temp_root = Path(__file__).resolve().parent / ".ilastik_batch_temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    job_dir = Path(
        tempfile.mkdtemp(
            prefix="job_",
            dir=str(temp_root),
        )
    )

    try:
        temp_dapi = job_dir / "input.tif"
        shutil.copy2(dapi_path, temp_dapi)

        original_project = project_file.resolve()

        # Ilastik generally normalizes TIFF exports to the .tiff extension.
        requested_output = job_dir / "exported_segmentation.tiff"

        command = [
            str(ilastik_executable),
            "--headless",
            "--readonly",
            f"--logfile={job_dir / 'ilastik_internal.log'}",
            f"--project={original_project}",
            "--input_axes=yx",
            f"--export_source={export_source}",
            "--output_format=tiff",
            f"--output_filename_format={requested_output}",
            str(temp_dapi),
        ]

        print("  Running Ilastik on:", dapi_path.name)

        ilastik_env = os.environ.copy()
        ilastik_env.pop("MPLBACKEND", None)
        for environment_name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_PROMPT_MODIFIER",
            "CONDA_EXE",
            "CONDA_PYTHON_EXE",
            "_CE_CONDA",
            "_CE_M",
        ):
            ilastik_env.pop(environment_name, None)
        ilastik_env.update(ILASTIK_ENVIRONMENT)
        isolated_local_appdata = job_dir / "local_appdata"
        isolated_local_appdata.mkdir(parents=True, exist_ok=True)
        (isolated_local_appdata / "ilastik" / "Logs").mkdir(parents=True, exist_ok=True)
        ilastik_env["LOCALAPPDATA"] = str(isolated_local_appdata)
        ilastik_env["WIN_PD_OVERRIDE_LOCAL_APPDATA"] = str(isolated_local_appdata)

        completed = subprocess.run(
            command,
            cwd=str(original_project.parent),
            check=False,
            capture_output=True,
            text=True,
            env=ilastik_env,
        )

        log_path = output_path.parent / (output_path.stem + "_ilastik_console_log.txt")

        log_text = (
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nWORKING DIRECTORY:\n"
            + str(original_project.parent)
            + "\n\nSTDOUT:\n"
            + (completed.stdout or "")
            + "\n\nSTDERR:\n"
            + (completed.stderr or "")
        )
        log_path.write_text(log_text, encoding="utf-8")

        combined_console_text = (
            (completed.stdout or "") + "\n" + (completed.stderr or "")
        )

        # A partial export from a failed process is never a valid cache entry.
        if completed.returncode != 0:
            raise RuntimeError(
                f"Ilastik failed for {dapi_path.name} with return code "
                f"{completed.returncode}. See: {log_path}"
            )

        # Warnings alone do not indicate failure; validate the exported image.
        candidates = []

        for candidate in (
            requested_output,
            requested_output.with_suffix(".tif"),
        ):
            if candidate.exists() and candidate.resolve() != temp_dapi.resolve():
                candidates.append(candidate)

        candidates.extend(
            p
            for p in job_dir.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in {".tif", ".tiff"}
                and p.resolve() != temp_dapi.resolve()
                and p not in candidates
            )
        )

        candidates.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        actual_output = candidates[0] if candidates else None

        if actual_output is not None:
            labels = load_2d_image(actual_output)
            expected_shape = load_2d_image(temp_dapi).shape
            if labels.shape != expected_shape or labels.dtype.kind not in "iu":
                raise ValueError(
                    "Ilastik export must be an integer label image matching DAPI dimensions."
                )
            del labels
            # Write beside the destination so replacement is atomic across filesystems.
            with tempfile.NamedTemporaryFile(
                dir=output_path.parent, delete=False
            ) as stream:
                staged_output = Path(stream.name)
            try:
                shutil.copy2(actual_output, staged_output)
                os.replace(staged_output, output_path)
            finally:
                staged_output.unlink(missing_ok=True)
            write_json(
                Path(str(output_path) + ".json"),
                {
                    "identity": identity,
                    "output_sha256": file_sha256(output_path),
                },
            )

            if not output_path.exists():
                raise FileNotFoundError(
                    "The Ilastik TIFF was produced, but it could not be copied to:\n"
                    f"{output_path}"
                )

            print(
                "  Ilastik Simple Segmentation saved:",
                output_path.name,
            )
            return output_path

        # No output was produced. Only now interpret console errors.
        fatal_classifier_markers = (
            "Tried to access placeholder dataset",
            "Without access to training data, the classifier cannot be retrained",
            "Failed to request data from `OpMissingDataSource.Output`",
        )

        if any(marker in combined_console_text for marker in fatal_classifier_markers):
            raise RuntimeError(
                "Ilastik could not access the training data needed by the project. "
                "The script used the original project path:\n"
                f"{original_project}\n\n"
                "Open that project in Ilastik, confirm Live Prediction works, "
                "save the project, and verify that its training images remain at "
                "the expected relative locations.\n"
                f"See the full console log: {log_path}"
            )

        directory_contents = sorted(
            str(p.relative_to(job_dir)) for p in job_dir.rglob("*") if p.is_file()
        )
        console_lines = combined_console_text.strip().splitlines()
        console_tail = "\n".join(console_lines[-40:])

        raise RuntimeError(
            "Ilastik completed without creating a detectable Simple "
            "Segmentation TIFF.\n\n"
            "Files created in the temporary Ilastik folder:\n"
            + (
                "\n".join(directory_contents)
                if directory_contents
                else "(none besides the copied input)"
            )
            + "\n\nLast console lines:\n"
            + console_tail
            + "\n\nSee the full Ilastik log:\n"
            + str(log_path)
        )

    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def load_2d_image(path):
    """Load an image and confirm that it is two-dimensional."""
    image = tifffile.imread(path)

    # Ilastik can sometimes export singleton dimensions.
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"{path.name} must be a 2D image after squeezing singleton "
            f"dimensions, but its shape is {image.shape}."
        )

    return image


def _resolution_value_as_float(value):
    """Convert a TIFF rational resolution value to a floating-point number."""
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        denominator = float(value.denominator)
        if denominator == 0:
            return None
        return float(value.numerator) / denominator

    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        denominator = float(denominator)
        if denominator == 0:
            return None
        return float(numerator) / denominator

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unit_to_micrometers(unit):
    """Return the number of micrometers represented by one metadata unit."""
    if unit is None:
        return None

    normalized = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")

    unit_factors = {
        "um": 1.0,
        "micron": 1.0,
        "microns": 1.0,
        "micrometer": 1.0,
        "micrometers": 1.0,
        "micrometre": 1.0,
        "micrometres": 1.0,
        "nm": 0.001,
        "nanometer": 0.001,
        "nanometers": 0.001,
        "mm": 1000.0,
        "millimeter": 1000.0,
        "millimeters": 1000.0,
        "cm": 10_000.0,
        "centimeter": 10_000.0,
        "centimeters": 10_000.0,
        "inch": 25_400.0,
        "inches": 25_400.0,
    }

    return unit_factors.get(normalized)


def _isotropic_pixel_size(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Pixel calibration must be positive and finite.")
    if len(values) == 2 and not np.isclose(values[0], values[1], rtol=0.001, atol=0):
        raise ValueError(
            f"Anisotropic pixels are unsupported: X/Y sizes are {values} um. "
            "Supply isotropically calibrated images."
        )
    return float(np.mean(values))


def read_pixel_size_um(path):
    """
    Read the lateral pixel size from ImageJ, OME-TIFF, or standard TIFF
    resolution metadata.

    Returns
    -------
    pixel_size_um : float
        Isotropic lateral pixel size in micrometers per pixel (0.1% tolerance).
    source : str
        Description of the metadata source, or "fallback".
    """
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]

        # First try OME-TIFF PhysicalSizeX and PhysicalSizeY metadata.
        if tif.ome_metadata:
            ome_values = []
            try:
                root = ET.fromstring(tif.ome_metadata)
                pixels_element = next(
                    element for element in root.iter() if element.tag.endswith("Pixels")
                )

                x_value = pixels_element.attrib.get("PhysicalSizeX")
                y_value = pixels_element.attrib.get("PhysicalSizeY")
                x_unit = pixels_element.attrib.get(
                    "PhysicalSizeXUnit",
                    "um",
                )
                y_unit = pixels_element.attrib.get(
                    "PhysicalSizeYUnit",
                    x_unit,
                )

                x_factor = _unit_to_micrometers(x_unit)
                y_factor = _unit_to_micrometers(y_unit)

                x_um = (
                    float(x_value) * x_factor
                    if x_value is not None and x_factor is not None
                    else None
                )
                y_um = (
                    float(y_value) * y_factor
                    if y_value is not None and y_factor is not None
                    else None
                )

                ome_values = [x_um, y_um]
            except (ET.ParseError, StopIteration, TypeError, ValueError):
                pass
            size = _isotropic_pixel_size(ome_values)
            if size is not None:
                return size, "OME-TIFF metadata"

        # ImageJ TIFFs commonly store the unit in ImageDescription and
        # pixels-per-unit in XResolution/YResolution.
        imagej_metadata = tif.imagej_metadata or {}
        imagej_unit = imagej_metadata.get("unit")
        imagej_factor = _unit_to_micrometers(imagej_unit)

        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")

        x_resolution = (
            _resolution_value_as_float(x_tag.value) if x_tag is not None else None
        )
        y_resolution = (
            _resolution_value_as_float(y_tag.value) if y_tag is not None else None
        )

        if imagej_factor is not None:
            sizes_um = [
                imagej_factor / value if value > 0 else np.nan
                for value in (x_resolution, y_resolution)
                if value is not None
            ]
            if sizes_um:
                return _isotropic_pixel_size(sizes_um), "ImageJ TIFF metadata"

        # Finally try standard TIFF ResolutionUnit.
        resolution_unit_tag = page.tags.get("ResolutionUnit")
        resolution_unit = (
            resolution_unit_tag.value if resolution_unit_tag is not None else None
        )

        # TIFF ResolutionUnit: 2 = inch, 3 = centimeter.
        standard_factor = None
        unit_name = str(resolution_unit).lower()

        if resolution_unit == 2 or "inch" in unit_name:
            standard_factor = 25_400.0
        elif resolution_unit == 3 or "centimeter" in unit_name:
            standard_factor = 10_000.0

        if standard_factor is not None:
            sizes_um = [
                standard_factor / value if value > 0 else np.nan
                for value in (x_resolution, y_resolution)
                if value is not None
            ]
            if sizes_um:
                return _isotropic_pixel_size(sizes_um), "standard TIFF metadata"

    return float(FALLBACK_PIXEL_SIZE_UM), "fallback"


def calculate_pixel_parameters(pixel_size_um):
    """Convert all physical thresholds to integer pixel thresholds."""
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(
            f"Pixel size must be positive and finite, but received {pixel_size_um}."
        )

    pixel_area_um2 = pixel_size_um**2

    def area_to_pixels(area_um2):
        return max(
            1,
            int(round(area_um2 / pixel_area_um2)),
        )

    whole_skin_smoothing_width_pixels = max(
        1,
        int(round(WHOLE_SKIN_BOUNDARY_SMOOTHING_WIDTH_UM / pixel_size_um)),
    )
    if whole_skin_smoothing_width_pixels % 2 == 0:
        whole_skin_smoothing_width_pixels += 1

    whole_skin_basal_smoothing_width_pixels = max(
        3,
        int(round(WHOLE_SKIN_BASAL_SMOOTHING_WIDTH_UM / pixel_size_um)),
    )
    if whole_skin_basal_smoothing_width_pixels % 2 == 0:
        whole_skin_basal_smoothing_width_pixels += 1

    return {
        "pixel_size_um": float(pixel_size_um),
        "pixel_area_um2": float(pixel_area_um2),
        "min_whole_skin_object_pixels": area_to_pixels(MIN_WHOLE_SKIN_OBJECT_AREA_UM2),
        "whole_skin_smoothing_width_pixels": (whole_skin_smoothing_width_pixels),
        "whole_skin_disconnected_vertical_margin_pixels": max(
            1,
            int(round(WHOLE_SKIN_DISCONNECTED_VERTICAL_MARGIN_UM / pixel_size_um)),
        ),
        "whole_skin_basal_smoothing_width_pixels": (
            whole_skin_basal_smoothing_width_pixels
        ),
    }


def repair_whole_skin_mask(
    binary_mask,
    min_object_pixels,
    smoothing_width_pixels,
    closing_radius_pixels=0,
    disconnected_vertical_margin_pixels=0,
    basal_smoothing_width_pixels=0,
    basal_percentile=50.0,
):
    """
    Remove remote disconnected objects, smooth the true dermal bottom, and fill holes.

    Long superficial components define the connected whole-skin reference.
    Their upper and lower edges define the trusted vertical skin envelope. A
    non-reference component is removed only when every pixel lies beyond the
    configured margin above or below that envelope. The lower reference edge
    can additionally use a rolling lower percentile to reject narrow downward
    fat projections without altering the superficial edge. All enclosed holes
    in the retained mask are then filled, irrespective of size.
    """
    mask = np.asarray(binary_mask, dtype=bool)
    labeled_mask = label(mask, connectivity=2)
    enhanced_cleanup_enabled = bool(
        disconnected_vertical_margin_pixels > 0 or basal_smoothing_width_pixels > 1
    )

    (
        reference_mask,
        candidates,
        selected,
        minimum_span,
    ) = select_superficial_long_components(
        mask,
        min_component_area_pixels=min_object_pixels,
        labeled_mask=labeled_mask,
    )

    upper_boundary = extract_smoothed_mask_edge(
        reference_mask,
        edge="upper",
        smoothing_width_pixels=smoothing_width_pixels,
    )
    raw_lower_boundary = extract_smoothed_mask_edge(
        reference_mask,
        edge="lower",
        smoothing_width_pixels=1,
    )
    if basal_smoothing_width_pixels > 1:
        lower_boundary = percentile_filter(
            raw_lower_boundary.astype(float),
            percentile=float(basal_percentile),
            size=int(basal_smoothing_width_pixels),
            mode="nearest",
        )
        lower_boundary = median_filter(
            lower_boundary,
            size=int(smoothing_width_pixels),
            mode="nearest",
        )
        lower_boundary = np.rint(lower_boundary).astype(int)
    else:
        lower_boundary = extract_smoothed_mask_edge(
            reference_mask,
            edge="lower",
            smoothing_width_pixels=smoothing_width_pixels,
        )
    lower_boundary = np.clip(
        np.maximum(lower_boundary, upper_boundary + 1),
        0,
        mask.shape[0] - 1,
    )

    selected_labels = {int(component["label"]) for component in selected}
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    removed_mask = np.zeros_like(mask, dtype=bool)
    component_rows = []

    for region in regionprops(labeled_mask):
        coordinates = region.coords
        rows = coordinates[:, 0]
        columns = coordinates[:, 1]
        is_reference = int(region.label) in selected_labels
        entirely_below_boundary = bool(np.all(rows > lower_boundary[columns]))
        entirely_far_below_boundary = bool(
            np.all(rows > lower_boundary[columns] + disconnected_vertical_margin_pixels)
        )
        entirely_far_above_boundary = bool(
            np.all(rows < upper_boundary[columns] - disconnected_vertical_margin_pixels)
        )
        if enhanced_cleanup_enabled:
            remove_component = not is_reference and (
                entirely_far_below_boundary or entirely_far_above_boundary
            )
        else:
            # Preserve the established behavior outside the mouse-paw groups.
            remove_component = not is_reference and entirely_below_boundary

        target = removed_mask if remove_component else cleaned_mask
        target[rows, columns] = True

        min_row, min_col, max_row, max_col = region.bbox
        component_rows.append(
            {
                "label": int(region.label),
                "reference_component": bool(is_reference),
                "area_pixels": int(region.area),
                "horizontal_span_pixels": int(max_col - min_col),
                "entirely_below_traced_bottom": entirely_below_boundary,
                "entirely_far_below_traced_bottom": entirely_far_below_boundary,
                "entirely_far_above_superficial_trace": entirely_far_above_boundary,
                "removed": bool(remove_component),
            }
        )

    if closing_radius_pixels > 0:
        cleaned_mask = binary_closing(
            cleaned_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(closing_radius_pixels),
        )
    repaired_mask = binary_fill_holes(cleaned_mask).astype(bool)
    if basal_smoothing_width_pixels > 1:
        row_grid = np.arange(mask.shape[0])[:, None]
        basal_protrusions = repaired_mask & (row_grid > lower_boundary[None, :])
        repaired_mask &= ~basal_protrusions
        removed_mask |= basal_protrusions
    repaired_reference_mask = repaired_mask & reference_mask
    filled_holes_mask = repaired_mask & ~cleaned_mask

    return (
        repaired_mask,
        removed_mask,
        filled_holes_mask,
        repaired_reference_mask,
        lower_boundary,
        component_rows,
        candidates,
        selected,
        minimum_span,
    )


def select_superficial_long_components(
    binary_mask,
    min_component_area_pixels,
    labeled_mask=None,
):
    """
    Select long components on the column-wise superficial envelope.

    Width and area first identify plausible components. In every column, the
    qualifying component with the uppermost pixel contributes to the epidermis
    envelope. If a component contributes in any column, all connected pixels
    belonging to it are retained as epidermis.
    """
    _, width = binary_mask.shape
    if labeled_mask is None:
        labeled_mask = label(binary_mask, connectivity=2)
    regions = regionprops(labeled_mask)

    if not regions:
        raise ValueError("No positive mask pixels were found.")

    minimum_span = max(
        1,
        int(round(width * MIN_COMPONENT_WIDTH_FRACTION)),
    )

    candidates = []
    envelope_rows = np.full(width, binary_mask.shape[0], dtype=np.int32)
    envelope_labels = np.zeros(width, dtype=labeled_mask.dtype)

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        horizontal_span = max_col - min_col

        if horizontal_span >= minimum_span and region.area >= min_component_area_pixels:
            local_component = (
                labeled_mask[min_row:max_row, min_col:max_col] == region.label
            )
            local_columns = np.arange(min_col, max_col)
            populated = np.any(local_component, axis=0)
            local_columns = local_columns[populated]
            upper_rows = min_row + np.argmax(local_component[:, populated], axis=0)
            replace = upper_rows < envelope_rows[local_columns]
            envelope_rows[local_columns[replace]] = upper_rows[replace]
            envelope_labels[local_columns[replace]] = region.label
            candidates.append(
                {
                    "label": region.label,
                    "horizontal_span": horizontal_span,
                    "area": int(region.area),
                    "min_col": min_col,
                    "max_col": max_col,
                    "min_row": min_row,
                    "max_row": max_row,
                    "median_row": float(np.median(region.coords[:, 0])),
                }
            )

    if not candidates:
        raise ValueError(
            "No mask object passed the long-component filters. "
            "Lower MIN_COMPONENT_WIDTH_FRACTION or MIN_COMPONENT_AREA_UM2."
        )

    envelope_column_counts = np.bincount(
        envelope_labels.astype(np.intp, copy=False),
        minlength=int(labeled_mask.max()) + 1,
    )

    for candidate in candidates:
        contribution = envelope_column_counts[int(candidate["label"])]
        candidate["superficial_envelope_columns"] = int(contribution)
        candidate["superficial_envelope_fraction"] = float(
            contribution / candidate["horizontal_span"]
        )

    selected = [
        candidate
        for candidate in candidates
        if (
            candidate["superficial_envelope_fraction"]
            >= MIN_SUPERFICIAL_ENVELOPE_FRACTION
        )
    ]
    if not selected:
        raise ValueError(
            "No long component formed enough of the superficial mask "
            "envelope. Lower MIN_SUPERFICIAL_ENVELOPE_FRACTION."
        )

    selected.sort(
        key=lambda item: (
            item["superficial_envelope_columns"],
            item["horizontal_span"],
            item["area"],
        ),
        reverse=True,
    )

    keep_label = np.zeros(int(labeled_mask.max()) + 1, dtype=bool)
    keep_label[[int(component["label"]) for component in selected]] = True
    retained_mask = keep_label[labeled_mask]

    return retained_mask, candidates, selected, minimum_span


def extract_smoothed_mask_edge(mask, edge, smoothing_width_pixels):
    """
    Return the interpolated, median-smoothed upper or lower mask edge.

    Parameters
    ----------
    mask : 2D Boolean array
    edge : {"upper", "lower"}
        "upper" selects the first positive row in each populated column;
        "lower" selects the last positive row.
    smoothing_width_pixels : int
        Median-filter width applied after interpolation.
    """
    if edge not in {"upper", "lower"}:
        raise ValueError("edge must be either 'upper' or 'lower'.")

    height, width = mask.shape
    populated = np.any(mask, axis=0)
    valid_columns = np.flatnonzero(populated)
    if valid_columns.size < 2:
        raise ValueError(
            f"The {edge} mask edge was found in fewer than two image columns."
        )

    if edge == "upper":
        valid_rows = np.argmax(mask[:, populated], axis=0)
    else:
        valid_rows = height - 1 - np.argmax(mask[::-1, populated], axis=0)
    all_columns = np.arange(width)
    curve = np.interp(all_columns, valid_columns, valid_rows)
    curve = median_filter(
        curve,
        size=smoothing_width_pixels,
        mode="nearest",
    )

    return np.clip(
        np.rint(curve).astype(int),
        0,
        height - 1,
    )


def filter_long_nerve_objects_near_superficial_boundary(
    nerve_positive_mask,
    superficial_boundary_mask,
    pixel_size_um,
    maximum_boundary_distance_um,
    minimum_object_length_um,
):
    """Remove intact long nerve objects close to the superficial tissue edge.

    Object length is the calibrated 8-connected skeleton length, so curved and
    diagonal objects are assessed geometrically rather than by horizontal
    bounding-box width. Short puncta and short fibers are retained even when
    close to the surface. Long objects are retained when they do not approach
    the superficial boundary.
    """
    mask = np.asarray(nerve_positive_mask, dtype=bool)
    superficial_boundary = np.asarray(superficial_boundary_mask, dtype=bool)
    if mask.shape != superficial_boundary.shape:
        raise ValueError("Nerve and superficial-boundary masks must match.")
    boundary_coordinates = np.column_stack(np.nonzero(superficial_boundary))
    if not boundary_coordinates.size:
        raise ValueError("The superficial boundary mask is empty.")
    boundary_tree = cKDTree(boundary_coordinates.astype(np.float32))
    component_labels = label(mask, connectivity=2)
    retained = mask.copy()
    removed_count = 0
    for region in regionprops(component_labels):
        coordinates = region.coords
        minimum_distance_um = float(
            boundary_tree.query(coordinates, k=1)[0].min() * pixel_size_um
        )
        min_row, min_col, max_row, max_col = region.bbox
        if minimum_distance_um <= maximum_boundary_distance_um:
            local_component = (
                component_labels[min_row:max_row, min_col:max_col] == region.label
            )
            _, skeleton_length_um = calculate_calibrated_skeleton_length(
                local_component,
                pixel_size_um,
                componentwise=False,
            )
        else:
            # Exact length is irrelevant when the proximity criterion fails.
            skeleton_length_um = np.nan
        remove = bool(
            np.isfinite(skeleton_length_um)
            and skeleton_length_um >= minimum_object_length_um
            and minimum_distance_um <= maximum_boundary_distance_um
        )
        if remove:
            retained[coordinates[:, 0], coordinates[:, 1]] = False
            removed_count += 1
    return retained, removed_count


def trace_whole_tissue_superficial_boundary(
    repaired_whole_tissue_mask,
    smoothing_width_pixels,
):
    """Trace the top edge of repaired whole tissue for epidermis QC/context."""
    whole_tissue = np.asarray(repaired_whole_tissue_mask, dtype=bool)
    upper_rows = extract_smoothed_mask_edge(
        whole_tissue,
        edge="upper",
        smoothing_width_pixels=smoothing_width_pixels,
    )
    valid_columns = np.flatnonzero(np.any(whole_tissue, axis=0))
    boundary = np.zeros_like(whole_tissue)
    if valid_columns.size:
        # Keep separate tissue spans separate; never bridge a blank column gap.
        split_points = np.flatnonzero(np.diff(valid_columns) > 1) + 1
        for columns in np.split(valid_columns, split_points):
            if not columns.size:
                continue
            boundary[upper_rows[columns], columns] = True
            for index in range(1, len(columns)):
                rr, cc = line(
                    int(upper_rows[columns[index - 1]]),
                    int(columns[index - 1]),
                    int(upper_rows[columns[index]]),
                    int(columns[index]),
                )
                boundary[rr, cc] = True
    return boundary


def quantify_nerve_by_regions(*args, **kwargs):
    """Compatibility entry point for the canonical fixed-ROI measurement engine."""
    return quantify_regions(*args, **kwargs)


def quantify_bt3_by_regions(*args, precomputed_bt3_skeleton=None, **kwargs):
    """Compatibility wrapper for the historical BT3-named public function."""
    if "precomputed_nerve_skeleton" in kwargs:
        if precomputed_bt3_skeleton is not None:
            raise TypeError("Specify only one precomputed nerve skeleton.")
        precomputed_bt3_skeleton = kwargs.pop("precomputed_nerve_skeleton")
    return quantify_nerve_by_regions(
        *args,
        precomputed_nerve_skeleton=precomputed_bt3_skeleton,
        **kwargs,
    )


# Compatibility alias for callers of the historical filter name.
filter_long_bt3_objects_near_superficial_boundary = (
    filter_long_nerve_objects_near_superficial_boundary
)


def save_binary_image(path, mask):
    """Save a Boolean mask as a contiguous 8-bit TIFF."""
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_image = np.ascontiguousarray(mask, dtype=np.uint8)
    output_image *= 255

    tifffile.imwrite(
        str(path),
        output_image,
        photometric="minisblack",
    )


def biological_replicate_from_sample(sample_name):
    """Collapse case-insensitive section suffixes to one biological replicate."""
    sample_name = str(sample_name)
    replicate = re.sub(
        r"(?i)[_-]section[-_ ]*\d.*$",
        "",
        sample_name,
    ).rstrip("_- ")
    return replicate or sample_name


def standardize_skeleton_density_columns(table):
    """Convert old density fields to um/mm² without changing source tables."""
    table = table.copy()
    target = "whole_dermis_bt3_skeleton_density_um_per_mm2"
    old_columns = (
        "dermal_nerve_skeleton_length_density",
        "dermal_BT3_skeleton_length_density",
    )
    for source in old_columns:
        if source not in table.columns:
            continue
        converted = pd.to_numeric(table[source], errors="raise") * 1_000_000.0
        if target not in table.columns:
            table[target] = converted
        else:
            table[target] = table[target].fillna(converted)
    return table.drop(columns=list(old_columns), errors="ignore")


def build_biological_replicate_averages(section_results_df):
    """Average normalized innervation metrics across sections per animal."""
    table = normalize_legacy_results(section_results_df)
    table["Biological replicate"] = table["Sample"].map(
        biological_replicate_from_sample
    )
    grouping_columns = ["Biological replicate"]
    if "Group" in table.columns:
        grouping_columns.insert(0, "Group")
    metrics = [
        "epidermal_nerve_area_um2_per_boundary_mm",
        "epidermal_nerve_skeleton_length_um_per_boundary_mm",
    ]
    missing = [column for column in metrics if column not in table.columns]
    if missing:
        raise ValueError(
            "Cannot calculate biological-replicate averages; missing columns: "
            f"{missing}"
        )
    averages = (
        table.groupby(grouping_columns, dropna=False)[metrics].mean().reset_index()
    )
    section_counts = (
        table.groupby(grouping_columns, dropna=False)
        .size()
        .rename("Number of sections")
        .reset_index()
    )
    averages = section_counts.merge(averages, on=grouping_columns, how="left")
    return averages.rename(
        columns={
            "epidermal_nerve_area_um2_per_boundary_mm": (
                "Mean epidermal nerve area (um2 per boundary mm)"
            ),
            "epidermal_nerve_skeleton_length_um_per_boundary_mm": (
                "Mean epidermal nerve skeleton length (um per boundary mm)"
            ),
        }
    )


def normalize_legacy_results(table):
    """Backfill each historical row without dropping it from mixed-schema means."""
    table = standardize_skeleton_density_columns(table)
    aliases = {
        **LEGACY_BT3_METRIC_NAMES,
        "nerve_segmentation_method": "BT3_segmentation_method",
        "nerve_threshold": "BT3_threshold",
    }
    for target, source in aliases.items():
        if source in table:
            table[target] = (
                table[target].fillna(table[source])
                if target in table
                else table[source]
            )
    for target, source in {
        "epidermal_nerve_area_um2_per_boundary_mm": "epidermal_nerve_area_per_boundary_length",
        "epidermal_nerve_skeleton_length_um_per_boundary_mm": "epidermal_nerve_skeleton_length_per_boundary_length",
    }.items():
        if source in table:
            converted = pd.to_numeric(table[source], errors="raise") * 1000.0
            table[target] = (
                table[target].fillna(converted) if target in table else converted
            )
    return table


def write_quantification_excel(section_results_df, output_path):
    """Write section results and biological-replicate means to two sheets."""
    section_table = section_results_df.copy()
    section_table["Biological replicate"] = section_table["Sample"].map(
        biological_replicate_from_sample
    )
    replicate_table = build_biological_replicate_averages(section_table)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        section_table.to_excel(writer, sheet_name="Section Results", index=False)
        replicate_table.to_excel(
            writer, sheet_name="Biological Replicates", index=False
        )


def find_input_files(input_folder):
    """Return case-insensitive TIFF lookups grouped by relative folder."""
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    grouped_lookups = {}

    for path in input_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in TIFF_EXTENSIONS:
            relative_group = path.parent.relative_to(input_folder)
            group_key = relative_group.as_posix()
            lookup = grouped_lookups.setdefault(group_key, {})
            key = path.stem.lower()

            if key in lookup:
                raise ValueError(
                    "Two TIFF files in the same input group have the same "
                    "stem when case is ignored: "
                    f"{lookup[key].name} and {path.name}"
                )

            lookup[key] = path

    return grouped_lookups


def discover_samples(input_folder):
    """
    Match each nerve-signal image to its DAPI image.

    ``_nerve`` is the canonical suffix. Historical ``_BT3`` names are still
    accepted, but the canonical file wins when both variants exist for the
    same sample. No whole-tissue input channel is required.
    """
    grouped_lookups = find_input_files(input_folder)

    samples = []
    missing = []
    suffixes = (
        (NERVE_SIGNAL_SUFFIX, "nerve"),
        *((suffix, "legacy_BT3") for suffix in LEGACY_NERVE_SIGNAL_SUFFIXES),
    )

    for group_path, lookup in sorted(grouped_lookups.items()):
        nerve_candidates = {}
        for stem_lower, path in sorted(lookup.items()):
            for priority, (suffix, convention) in enumerate(suffixes):
                if not stem_lower.endswith(suffix.lower()):
                    continue
                sample_stem = path.stem[: -len(suffix)]
                sample_key = sample_stem.casefold()
                current = nerve_candidates.get(sample_key)
                if current is None or priority < current[0]:
                    nerve_candidates[sample_key] = (
                        priority,
                        sample_stem,
                        path,
                        convention,
                    )
                break

        for _, sample_stem, nerve_signal_path, convention in sorted(
            nerve_candidates.values(), key=lambda candidate: candidate[1].casefold()
        ):
            dapi_key = (sample_stem + DAPI_SUFFIX).lower()
            dapi_path = lookup.get(dapi_key)

            if dapi_path is None:
                missing.append(
                    {
                        "Group": group_path,
                        "Sample": sample_stem,
                        "Nerve signal file": nerve_signal_path.name,
                        "Missing DAPI image": True,
                    }
                )
                continue

            samples.append(
                {
                    "group_path": group_path,
                    "sample_name": sample_stem,
                    "nerve_signal_path": nerve_signal_path,
                    "nerve_input_convention": convention,
                    "dapi_path": dapi_path,
                }
            )

        for stem_lower, path in sorted(lookup.items()):
            if stem_lower.endswith(DAPI_SUFFIX.lower()):
                sample_stem = path.stem[: -len(DAPI_SUFFIX)]
                if sample_stem.casefold() not in nerve_candidates:
                    missing.append(
                        {
                            "Group": group_path,
                            "Sample": sample_stem,
                            "DAPI file": path.name,
                            "Missing nerve image": True,
                        }
                    )

    return samples, missing


def group_output_folder(output_root, group_path):
    """Return the output directory that mirrors one relative input group."""
    return (
        Path(output_root)
        if str(group_path) in {"", "."}
        else Path(output_root) / Path(str(group_path))
    )


# ============================================================
# ACTIVE EPIDERMIS-ONLY SAMPLE WORKFLOW
# ============================================================


def process_sample(
    sample_name,
    nerve_signal_path,
    dapi_path,
    sample_output_folder,
    ilastik_executable,
    nerve_input_convention="nerve",
    apply_mouse_whole_skin_cleanup=False,
    reuse_existing_segmentations=REUSE_EXISTING_ILASTIK_SEGMENTATIONS,
):
    """Segment DAPI, reconstruct tissue, and quantify four nerve compartments.

    The input nerve image must already be preprocessed. This function applies
    a strict grayscale threshold, then removes whole long superficial objects;
    the measurement engine clips the retained signal to the tissue ROIs.
    """
    sample_output_folder.mkdir(parents=True, exist_ok=True)
    ilastik_output_folder = sample_output_folder / "ilastik"
    ilastik_output_folder.mkdir(parents=True, exist_ok=True)
    # Separate content-verified cache entries for the two classifiers.
    epidermis_path = ilastik_output_folder / f"{sample_name}_improved_epimask.tif"
    whole_skin_path = ilastik_output_folder / f"{sample_name}_wholemask.tif"

    dapi_pixel_size_um, pixel_size_source = read_pixel_size_um(dapi_path)
    nerve_pixel_size_um, _ = read_pixel_size_um(nerve_signal_path)
    if not np.isclose(dapi_pixel_size_um, nerve_pixel_size_um, rtol=0.001, atol=0):
        raise ValueError(
            "DAPI and nerve-signal calibrations differ: "
            f"{dapi_pixel_size_um} vs {nerve_pixel_size_um} um/pixel."
        )
    pixel_parameters = calculate_pixel_parameters(dapi_pixel_size_um)

    print("  Running epidermis classifier...")
    ilastik_started = time.perf_counter()
    run_ilastik_segmentation(
        dapi_path=dapi_path,
        output_path=epidermis_path,
        ilastik_executable=ilastik_executable,
        project_file=EPIDERMIS_ILASTIK_PROJECT,
        export_source=EPIDERMIS_ILASTIK_EXPORT_SOURCE,
        reuse_existing=reuse_existing_segmentations,
    )
    epidermis_ilastik_seconds = time.perf_counter() - ilastik_started
    print("  Running whole-tissue context classifier...")
    ilastik_started = time.perf_counter()
    run_ilastik_segmentation(
        dapi_path=dapi_path,
        output_path=whole_skin_path,
        ilastik_executable=ilastik_executable,
        project_file=WHOLE_SKIN_ILASTIK_PROJECT,
        export_source=WHOLE_SKIN_ILASTIK_EXPORT_SOURCE,
        reuse_existing=reuse_existing_segmentations,
    )
    whole_tissue_ilastik_seconds = time.perf_counter() - ilastik_started
    nerve_signal = load_2d_image(nerve_signal_path)
    dapi = load_2d_image(dapi_path)
    epidermis_labels = load_2d_image(epidermis_path)
    whole_skin_labels = load_2d_image(whole_skin_path)
    if (
        nerve_signal.shape != dapi.shape
        or nerve_signal.shape != epidermis_labels.shape
        or nerve_signal.shape != whole_skin_labels.shape
    ):
        raise ValueError(
            "Nerve signal, DAPI, epidermis segmentation, and whole-tissue "
            "segmentation must have identical dimensions; found DAPI "
            f"{dapi.shape}, nerve signal {nerve_signal.shape}, "
            f"epidermis {epidermis_labels.shape}, whole tissue "
            f"{whole_skin_labels.shape}."
        )

    raw_epidermis_band_mask = epidermis_labels == EPIDERMIS_LABEL
    raw_whole_skin_mask = whole_skin_labels == WHOLE_SKIN_LABEL
    if not np.any(raw_epidermis_band_mask):
        raise ValueError(
            f"Epidermis label {EPIDERMIS_LABEL} is absent from Ilastik output."
        )
    if not np.any(raw_whole_skin_mask):
        raise ValueError(
            f"Whole-tissue label {WHOLE_SKIN_LABEL} is absent from Ilastik output."
        )
    del epidermis_labels, whole_skin_labels
    gc.collect()

    (
        repaired_whole_skin_mask,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = repair_whole_skin_mask(
        raw_whole_skin_mask,
        min_object_pixels=pixel_parameters["min_whole_skin_object_pixels"],
        smoothing_width_pixels=pixel_parameters["whole_skin_smoothing_width_pixels"],
        closing_radius_pixels=max(
            1,
            int(round(WHOLE_SKIN_CONTEXT_CLOSING_RADIUS_UM / dapi_pixel_size_um)),
        ),
        disconnected_vertical_margin_pixels=(
            pixel_parameters["whole_skin_disconnected_vertical_margin_pixels"]
            if apply_mouse_whole_skin_cleanup
            else 0
        ),
        basal_smoothing_width_pixels=(
            pixel_parameters["whole_skin_basal_smoothing_width_pixels"]
            if apply_mouse_whole_skin_cleanup
            else 0
        ),
        basal_percentile=WHOLE_SKIN_BASAL_PERCENTILE,
    )
    # The true superficial tissue boundary is computed once and reused by
    # Candidate 1, compartment reconstruction, and the nerve artifact filter.
    upper_boundary_mask = trace_whole_tissue_superficial_boundary(
        repaired_whole_skin_mask,
        pixel_parameters["whole_skin_smoothing_width_pixels"],
    )

    candidate1_started = time.perf_counter()
    try:
        candidate1_result = run_candidate1_cleanup(
            dapi,
            raw_epidermis_band_mask,
            repaired_whole_skin_mask,
            upper_boundary_mask,
            dapi_pixel_size_um,
        )
    except Candidate1Failure as error:
        save_candidate1_failure_qc(
            sample_output_folder,
            dapi,
            raw_epidermis_band_mask,
            upper_boundary_mask,
            str(error),
        )
        raise
    candidate1_seconds = time.perf_counter() - candidate1_started
    clean_epidermis_band_mask = assert_candidate1_invariants(
        candidate1_result, raw_epidermis_band_mask
    )
    observed_epidermal_course, inferred_epidermal_connectors = save_candidate1_qc(
        sample_output_folder,
        dapi,
        raw_epidermis_band_mask,
        upper_boundary_mask,
        candidate1_result,
    )
    candidate1_selected_component_count = len(
        candidate1_result.selection.selected_nodes
    )
    candidate1_selected_edge_count = len(candidate1_result.selection.selected_edges)
    del candidate1_result, raw_whole_skin_mask
    # Optimized explicit-interface reconstruction. The course guide is an
    # internal ordering scaffold only; the final quantitative boundary comes
    # from polygon epidermis/dermis adjacency after removing surface/closure
    # edges and redundant folded-region excursions.
    cleaned_epidermis_mask = np.asarray(clean_epidermis_band_mask, dtype=bool)
    if not np.any(cleaned_epidermis_mask):
        raise ValueError(
            "The cleaned epidermis band does not overlap repaired whole tissue."
        )
    reconstruction_started = time.perf_counter()
    interface_reconstruction = reconstruct_explicit_interface(
        cleaned_epidermis_mask,
        observed_epidermal_course,
        inferred_epidermal_connectors,
        repaired_whole_skin_mask,
        upper_boundary_mask,
        dapi_pixel_size_um,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    epidermis_region = interface_reconstruction.epidermis_region
    dermal_context_mask = interface_reconstruction.dermis_region
    # Reuse the exact surface arcs that actually closed the topological
    # epidermis polygons. This keeps downstream superficial filtering aligned
    # with the locally rotated tissue anatomy rather than the initial tracing
    # scaffold used by Candidate 1.
    upper_boundary_mask = interface_reconstruction.superficial_surface
    epidermal_boundary_normalization_mask = interface_reconstruction.final_boundary_mask
    anatomical_boundary_paths = interface_reconstruction.final_paths
    reconstruction_diagnostics = interface_reconstruction.diagnostics
    if not np.any(epidermal_boundary_normalization_mask):
        raise ValueError("The reconstructed epidermis-dermis interface is empty.")
    if np.any(epidermis_region & dermal_context_mask):
        raise RuntimeError("Compartment invariant failed: epidermis overlaps dermis.")
    if np.any((epidermis_region | dermal_context_mask) ^ repaired_whole_skin_mask):
        raise RuntimeError(
            "Compartment invariant failed: compartments do not partition whole tissue."
        )
    if np.any(epidermis_region & ~repaired_whole_skin_mask):
        raise RuntimeError(
            "Compartment invariant failed: epidermis lies outside whole tissue."
        )
    if (
        interface_reconstruction.diagnostics["compartment_consistency_error_pixels"]
        != 0
    ):
        raise RuntimeError(
            "Frozen reconstruction reported compartment consistency errors."
        )
    if (
        interface_reconstruction.diagnostics["boundary_compartment_support_fraction"]
        < 0.95
    ):
        raise RuntimeError(
            "Final anatomical boundary has insufficient epidermis-dermis adjacency support."
        )
    if interface_reconstruction.diagnostics["active_basal_side_dependency_count"] != 0:
        raise RuntimeError("Frozen reconstruction reported a basal-side dependency.")
    expected_final_boundary = (
        interface_reconstruction.observed_interface_mask
        | interface_reconstruction.inferred_bridge_mask
    )
    if not np.array_equal(
        epidermal_boundary_normalization_mask, expected_final_boundary
    ):
        raise RuntimeError(
            "Boundary provenance invariant failed: final boundary contains a "
            "non-anatomical source."
        )
    del interface_reconstruction

    nerve_threshold = float(MANUAL_NERVE_THRESHOLD)
    # Area and skeleton-length quantification use one fixed-threshold mask from
    # the supplied grayscale nerve-signal image. The measurement engine clips
    # retained signal to the reconstructed epidermis and dermis separately.
    nerve_threshold_mask = nerve_signal > nerve_threshold
    filtered_nerve_mask, removed_superficial_object_count = (
        filter_long_nerve_objects_near_superficial_boundary(
            nerve_threshold_mask,
            upper_boundary_mask,
            dapi_pixel_size_um,
            SUPERFICIAL_NERVE_EXCLUSION_DISTANCE_UM,
            MIN_SUPERFICIAL_NERVE_OBJECT_LENGTH_UM,
        )
    )
    print(
        f"  Long superficial nerve objects excluded: {removed_superficial_object_count}"
    )
    subbasal_config = SubbasalConfig(
        depth_um=SUBBASAL_DEPTH_UM,
        macro_smooth_um=DERMAL_DEPTH_REFERENCE_SMOOTHING_UM,
        deep_outlier_um=SUBBASAL_DEEP_OUTLIER_UM,
        max_appendage_width_um=SUBBASAL_MAX_APPENDAGE_WIDTH_UM,
        resample_um=SUBBASAL_RESAMPLE_UM,
        depth_bands_um=SUBBASAL_DEPTH_BANDS_UM,
    )
    (sample_output_folder / "analysis_parameters.json").write_text(
        json.dumps(
            {
                "pixel_size_um": dapi_pixel_size_um,
                "pixel_size_source": pixel_size_source,
                "apply_mouse_whole_skin_cleanup": apply_mouse_whole_skin_cleanup,
                "pixel_parameters": pixel_parameters,
                "nerve_threshold": nerve_threshold,
                "superficial_distance_um": SUPERFICIAL_NERVE_EXCLUSION_DISTANCE_UM,
                "superficial_minimum_length_um": MIN_SUPERFICIAL_NERVE_OBJECT_LENGTH_UM,
                "subbasal": asdict(subbasal_config),
                "candidate1": asdict(CANDIDATE1_CONFIG),
                "runtime": runtime_provenance(PROJECT_ROOT),
                "inputs_sha256": {
                    "dapi": file_sha256(dapi_path),
                    "nerve": file_sha256(nerve_signal_path),
                },
                "segmentation_provenance": {
                    "epidermis": json.loads(
                        Path(str(epidermis_path) + ".json").read_text(encoding="utf-8")
                    ),
                    "whole_skin": json.loads(
                        Path(str(whole_skin_path) + ".json").read_text(encoding="utf-8")
                    ),
                },
                "segmentation_reuse_requested": reuse_existing_segmentations,
            },
            indent=2,
        )
    )
    nerve_quantification, subbasal_result, four_compartment_result = (
        analyze_measurements(
            epidermis_region,
            dermal_context_mask,
            cleaned_epidermis_mask,
            epidermal_boundary_normalization_mask,
            anatomical_boundary_paths,
            filtered_nerve_mask,
            dapi_pixel_size_um,
            subbasal_config,
        )
    )
    nerve_metrics = nerve_quantification["metrics"]
    epidermal_nerve_mask = nerve_quantification["epidermal_nerve_mask"]
    dermal_nerve_mask = nerve_quantification["dermal_nerve_mask"]
    epidermal_nerve_skeleton = nerve_quantification["epidermal_nerve_skeleton"]
    dermal_nerve_skeleton = nerve_quantification["dermal_nerve_skeleton"]
    assert_aggregate_metrics_match_existing(
        four_compartment_result,
        nerve_metrics,
    )
    summary = {
        "Sample": sample_name,
        "sample_id": sample_name,
        "pixel_size_um": dapi_pixel_size_um,
        "pixel_size_source": pixel_size_source,
        "nerve_signal_file": Path(nerve_signal_path).name,
        "nerve_input_convention": nerve_input_convention,
        "nerve_segmentation_method": "fixed_grayscale_1500",
        "nerve_threshold": nerve_threshold,
        "epidermis_segmentation_model": EPIDERMIS_ILASTIK_PROJECT.name,
        "epidermis_cleanup_method": "candidate1_global_forest_surface_course_roots",
        "epidermis_boundary_method": "optimized_explicit_polygon_interface",
        "epidermis_ilastik_seconds": epidermis_ilastik_seconds,
        "whole_tissue_ilastik_seconds": whole_tissue_ilastik_seconds,
        "candidate1_cleanup_seconds": candidate1_seconds,
        "epidermis_reconstruction_seconds": reconstruction_seconds,
        "candidate1_selected_component_count": candidate1_selected_component_count,
        "candidate1_selected_edge_count": candidate1_selected_edge_count,
        "epidermis_boundary_path_count": reconstruction_diagnostics[
            "final_boundary_path_count"
        ],
        "epidermis_boundary_inferred_fraction": reconstruction_diagnostics[
            "inferred_boundary_fraction"
        ],
    }
    summary.update(nerve_metrics)
    overlapping_subbasal_columns = set(summary).intersection(subbasal_result.metrics)
    if overlapping_subbasal_columns:
        raise RuntimeError(
            "Sub-basal analysis attempted to overwrite pre-existing summary "
            f"columns: {sorted(overlapping_subbasal_columns)}"
        )
    summary.update(subbasal_result.metrics)
    overlapping_four_compartment_columns = set(summary).intersection(
        four_compartment_result.metrics
    )
    for column in overlapping_four_compartment_columns:
        existing_value = summary[column]
        compartment_value = four_compartment_result.metrics[column]
        if not (
            existing_value == compartment_value
            or (np.isnan(existing_value) and np.isnan(compartment_value))
        ):
            raise RuntimeError(
                "Four-compartment analysis disagrees with an existing summary "
                f"column: {column}"
            )
    summary.update(
        {
            column: value
            for column, value in four_compartment_result.metrics.items()
            if column not in summary
        }
    )
    # Preserve the two historical metadata fields alongside the canonical
    # nerve-named fields for existing result readers.
    summary["BT3_segmentation_method"] = summary["nerve_segmentation_method"]
    summary["BT3_threshold"] = summary["nerve_threshold"]

    masks = {
        "01_raw_epidermis_ilastik.tif": raw_epidermis_band_mask,
        "04_cleaned_epidermis_mask.tif": cleaned_epidermis_mask,
        EPIDERMIS_REGION_FILENAME: epidermis_region,
        DERMIS_REGION_FILENAME: dermal_context_mask,
        EPIDERMIS_DERMIS_BOUNDARY_FILENAME: (epidermal_boundary_normalization_mask),
        "10_BT3_fixed_threshold_binary_mask.tif": nerve_threshold_mask,
        FILTERED_EPIDERMAL_NERVE_FILENAME: epidermal_nerve_mask,
        FILTERED_DERMAL_NERVE_FILENAME: dermal_nerve_mask,
        EPIDERMAL_NERVE_SKELETON_FILENAME: epidermal_nerve_skeleton,
        DERMAL_NERVE_SKELETON_FILENAME: dermal_nerve_skeleton,
        "16_macro_basal_reference.tif": subbasal_result.macro_reference_mask,
        "18_subbasal_ROI.tif": subbasal_result.roi,
        "23_upper_epidermis_region.tif": four_compartment_result.compartment_masks[
            "upper_epidermis"
        ],
        "24_basal_epidermis_region.tif": four_compartment_result.compartment_masks[
            "basal_epidermis"
        ],
        "25_subbasal_dermis_region.tif": four_compartment_result.compartment_masks[
            "subbasal_dermis"
        ],
        "26_deep_dermis_region.tif": four_compartment_result.compartment_masks[
            "deep_dermis"
        ],
    }
    for (lower, upper), band_roi in subbasal_result.band_rois.items():
        if np.isclose(lower, 0.0) and np.isclose(upper, subbasal_config.depth_um):
            continue
        label_text = f"{lower:g}_{upper:g}um".replace(".", "p")
        masks[f"18_subbasal_ROI_{label_text}.tif"] = band_roi
    for filename, mask in masks.items():
        save_binary_image(sample_output_folder / filename, mask)
    save_final_combined_qc(
        sample_output_folder,
        dapi,
        cleaned_epidermis_mask,
        epidermis_region,
        dermal_context_mask,
        epidermal_boundary_normalization_mask,
    )
    save_four_compartment_qc(
        sample_output_folder,
        dapi,
        epidermal_boundary_normalization_mask,
        subbasal_result,
        four_compartment_result,
    )
    subbasal_summary = {
        "Sample": sample_name,
        "sample_id": sample_name,
        "pixel_size_um": dapi_pixel_size_um,
        **subbasal_result.metrics,
    }
    pd.DataFrame([subbasal_summary]).to_csv(
        sample_output_folder / SUBBASAL_QUANTIFICATION_FILENAME,
        index=False,
    )
    four_compartment_summary = {
        "Sample": sample_name,
        "sample_id": sample_name,
        "pixel_size_um": dapi_pixel_size_um,
        **four_compartment_result.metrics,
    }
    pd.DataFrame([four_compartment_summary]).to_csv(
        sample_output_folder / FOUR_COMPARTMENT_QUANTIFICATION_FILENAME,
        index=False,
    )
    # Write the main completion table last, after all other section artifacts.
    pd.DataFrame([summary]).to_csv(
        sample_output_folder / NERVE_QUANTIFICATION_FILENAME, index=False
    )
    return [summary]


# ============================================================
# BATCH RUNNER
# ============================================================


def main(
    input_folder=INPUT_FOLDER,
    output_root=OUTPUT_ROOT,
    ilastik_executable=None,
    *,
    skip_already_processed=SKIP_ALREADY_PROCESSED,
    reuse_existing_segmentations=REUSE_EXISTING_ILASTIK_SEGMENTATIONS,
    whole_skin_cleanup=None,
):
    """Run the batch pipeline with configurable input and output paths."""
    input_folder = Path(input_folder).expanduser()
    output_root = Path(output_root).expanduser()
    validate_output_location(output_root, input_folder, INPUT_FOLDER)
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        "Using epidermis Ilastik project:",
        EPIDERMIS_ILASTIK_PROJECT,
    )

    print(
        "Nerve area/skeleton segmentation: fixed grayscale threshold "
        f"{MANUAL_NERVE_THRESHOLD}"
    )

    samples, missing = discover_samples(input_folder)

    # Retire summaries from the previous invocation, including groups whose
    # inputs were removed. Only these generated summary names are touched.
    for filename in (
        "combined_BT3_quantification_results.csv",
        "BT3_quantification_by_biological_replicate.xlsx",
        "batch_run_log.csv",
        "missing_file_pairs.csv",
    ):
        for path in output_root.rglob(filename):
            path.unlink()

    if missing:
        pd.DataFrame(missing).to_csv(
            output_root / "missing_file_pairs.csv",
            index=False,
        )

    if not samples:
        raise FileNotFoundError(
            "No complete image sets were found.\n"
            "Expected names like:\n"
            "Sample01_nerve.tif (preferred) or Sample01_BT3.tif (legacy)\n"
            "Sample01_DAPI.tif"
        )

    all_results = []
    run_log = []

    print(f"Found {len(samples)} complete sample set(s).")

    print(f"Batch output folder: {output_root.resolve()}")

    resolved_ilastik_executable = None
    run_identity = {
        "runtime": runtime_provenance(PROJECT_ROOT),
        "models": {
            "epidermis": file_sha256(EPIDERMIS_ILASTIK_PROJECT),
            "whole_skin": file_sha256(WHOLE_SKIN_ILASTIK_PROJECT),
        },
        "settings": {
            name: value
            for name, value in globals().items()
            if name.isupper() and isinstance(value, (str, int, float, bool, tuple))
        },
        "candidate1": asdict(CANDIDATE1_CONFIG),
        "whole_skin_cleanup": whole_skin_cleanup,
        "ilastik_override": str(
            ilastik_executable or ILASTIK_EXE or os.environ.get("ILASTIK_EXE", "auto")
        ),
    }
    # Normalize tuples to JSON arrays so identities compare after serialization.
    run_identity = json.loads(json.dumps(run_identity))
    for sample in samples:
        sample["analysis_identity"] = {
            **run_identity,
            "group": sample.get("group_path", "."),
            "sample": sample["sample_name"],
            "nerve_input_convention": sample["nerve_input_convention"],
            "input_names": {
                key: sample[key].name for key in ("dapi_path", "nerve_signal_path")
            },
            "inputs": {
                key: file_sha256(sample[key])
                for key in ("dapi_path", "nerve_signal_path")
            },
        }
        folder = (
            group_output_folder(output_root, sample.get("group_path", "."))
            / sample["sample_name"]
        )
        sample["can_skip"] = skip_already_processed and completed_sample_matches(
            folder, sample["analysis_identity"]
        )
    processing_required = any(not sample["can_skip"] for sample in samples)
    if processing_required:
        resolved_ilastik_executable = find_ilastik_executable(ilastik_executable)
        print("Using Ilastik:", resolved_ilastik_executable)

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        sample_name = sample["sample_name"]
        group_path = sample.get("group_path", ".")
        group_folder = group_output_folder(output_root, group_path)

        sample_output_folder = group_folder / sample_name

        existing_results = sample_output_folder / NERVE_QUANTIFICATION_FILENAME

        print()
        print(f"[{index}/{len(samples)}] Processing {sample_name}")

        if sample["can_skip"]:
            print("  Skipped because completed results and provenance match.")

            existing_df = pd.read_csv(
                existing_results, converters={"Sample": str, "sample_id": str}
            )
            existing_df["Group"] = group_path

            all_results.extend(existing_df.to_dict("records"))

            run_log.append(
                {
                    "Group": group_path,
                    "Sample": sample_name,
                    "Status": ("Skipped - existing results"),
                    "Error": "",
                }
            )

            continue

        try:
            # An interrupted or failed rerun must not retain old completion CSVs.
            for filename in (
                NERVE_QUANTIFICATION_FILENAME,
                SUBBASAL_QUANTIFICATION_FILENAME,
                FOUR_COMPARTMENT_QUANTIFICATION_FILENAME,
                "analysis_parameters.json",
                "error_traceback.txt",
                "completion.json",
            ):
                (sample_output_folder / filename).unlink(missing_ok=True)
            sample_results = process_sample(
                sample_name=sample_name,
                nerve_signal_path=sample["nerve_signal_path"],
                dapi_path=sample["dapi_path"],
                sample_output_folder=(sample_output_folder),
                ilastik_executable=resolved_ilastik_executable,
                nerve_input_convention=sample["nerve_input_convention"],
                reuse_existing_segmentations=reuse_existing_segmentations,
                apply_mouse_whole_skin_cleanup=(
                    whole_skin_cleanup
                    if whole_skin_cleanup is not None
                    else str(group_path).casefold()
                    in {
                        "oldmice",
                        "youngmice",
                        "oldmice_validation",
                        "youngmice_validation",
                    }
                ),
            )
            for result_row in sample_results:
                result_row["Group"] = group_path

            mark_sample_complete(sample_output_folder, sample["analysis_identity"])

            all_results.extend(sample_results)

            run_log.append(
                {
                    "Group": group_path,
                    "Sample": sample_name,
                    "Status": "Completed",
                    "Error": "",
                }
            )

            print("  Completed.")

        except Exception as error:
            run_log.append(
                {
                    "Group": group_path,
                    "Sample": sample_name,
                    "Status": "Failed",
                    "Error": str(error),
                }
            )

            print(f"  FAILED: {error}")

            sample_output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            error_file = sample_output_folder / "error_traceback.txt"

            error_file.write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

        finally:
            # Ensure image-sized temporaries from one section are released
            # before the next biological sample is loaded.
            gc.collect()

    pd.DataFrame(run_log).to_csv(
        output_root / "batch_run_log.csv",
        index=False,
    )
    if run_log:
        run_log_df = pd.DataFrame(run_log)
        for group_path, group_log_df in run_log_df.groupby("Group", dropna=False):
            group_folder = group_output_folder(output_root, group_path)
            if group_folder.resolve() == output_root.resolve():
                continue
            group_folder.mkdir(parents=True, exist_ok=True)
            group_log_df.to_csv(group_folder / "batch_run_log.csv", index=False)

    # --------------------------------------------------------
    # SAVE COMBINED RESULTS
    # --------------------------------------------------------

    if all_results:
        combined_results_df = normalize_legacy_results(pd.DataFrame(all_results))
        if "nerve_signal_file" not in combined_results_df.columns:
            combined_results_df["nerve_signal_file"] = ""
        if "nerve_input_convention" not in combined_results_df.columns:
            combined_results_df["nerve_input_convention"] = "legacy_BT3"

        normalized_columns = [
            "Group",
            "Sample",
            "sample_id",
            "nerve_signal_file",
            "nerve_input_convention",
            "nerve_segmentation_method",
            "nerve_threshold",
            "pixel_size_um",
            "epidermal_area_um2",
            "epidermal_boundary_length_um",
            "epidermal_boundary_length_mm",
            "epidermal_nerve_area_um2",
            "epidermal_nerve_area_per_boundary_length",
            "epidermal_nerve_area_um2_per_boundary_mm",
            "epidermal_nerve_area_fraction",
            "epidermal_nerve_skeleton_length_um",
            "epidermal_nerve_skeleton_length_per_boundary_length",
            "epidermal_nerve_skeleton_length_um_per_boundary_mm",
            "dermal_area_um2",
            "dermal_nerve_area_um2",
            "dermal_nerve_area_fraction",
            "dermal_nerve_skeleton_length_um",
        ]
        subbasal_columns = [
            column
            for column in combined_results_df.columns
            if column.startswith("subbasal_")
            or column
            in {
                "anatomical_basal_length_um",
                "macro_reference_length_um",
                "macro_boundary_median_offset_um",
                "macro_boundary_p95_offset_um",
                "macro_boundary_max_deep_offset_um",
                "macro_boundary_downweighted_fraction",
                "macro_smoothing_um",
            }
        ]
        normalized_columns.extend(
            column for column in subbasal_columns if column not in normalized_columns
        )
        four_compartment_prefixes = (
            "upper_epidermis_",
            "basal_epidermis_",
            "subbasal_dermis_",
            "deep_dermis_",
            "whole_epidermis_",
            "whole_dermis_",
        )
        four_compartment_columns = [
            column
            for column in combined_results_df.columns
            if column.startswith(four_compartment_prefixes)
            or column == "fraction_boundary_points_downweighted"
        ]
        normalized_columns.extend(
            column
            for column in four_compartment_columns
            if column not in normalized_columns
        )
        (
            combined_results_df[normalized_columns]
            .drop_duplicates(subset=["Group", "Sample"])
            .to_csv(
                output_root / "combined_BT3_quantification_results.csv",
                index=False,
            )
        )
        root_section_results = combined_results_df[normalized_columns].drop_duplicates(
            subset=["Group", "Sample"]
        )
        write_quantification_excel(
            root_section_results,
            output_root / "BT3_quantification_by_biological_replicate.xlsx",
        )

        # Mirror the input hierarchy and give every biological group its own
        # clean combined table in addition to the cross-group root summary.
        for group_path, group_results_df in combined_results_df.groupby(
            "Group", dropna=False
        ):
            group_folder = group_output_folder(output_root, group_path)
            if group_folder.resolve() == output_root.resolve():
                continue  # The root summary already contains every group.
            group_folder.mkdir(parents=True, exist_ok=True)
            (
                group_results_df[normalized_columns]
                .drop_duplicates(subset=["Group", "Sample"])
                .to_csv(
                    group_folder / "combined_BT3_quantification_results.csv",
                    index=False,
                )
            )
            group_section_results = group_results_df[
                normalized_columns
            ].drop_duplicates(subset=["Group", "Sample"])
            write_quantification_excel(
                group_section_results,
                group_folder / "BT3_quantification_by_biological_replicate.xlsx",
            )

    completed = sum(row["Status"] == "Completed" for row in run_log)

    failed = sum(row["Status"] == "Failed" for row in run_log)

    skipped = sum(row["Status"].startswith("Skipped") for row in run_log)

    print()
    print("Batch processing finished.")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Outputs saved to: {output_root.resolve()}")
    return {"completed": completed, "failed": failed, "skipped": skipped}


def parse_args(argv=None):
    """Parse command-line options without changing the historical defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Segment skin-section TIFF pairs and quantify nerve signal by "
            "reconstructed tissue compartment."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_FOLDER,
        help=f"Input TIFF directory (default: {INPUT_FOLDER})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--ilastik-exe",
        type=Path,
        default=None,
        help="Explicit Ilastik executable; otherwise use ILASTIK_EXE or auto-detect.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=SKIP_ALREADY_PROCESSED,
        help="Skip only complete results with matching inputs, models, source, and settings.",
    )
    parser.add_argument(
        "--rerun-ilastik",
        action="store_true",
        help="Regenerate segmentations even when reusable TIFF outputs exist.",
    )
    parser.add_argument(
        "--whole-skin-cleanup",
        choices=("legacy", "on", "off"),
        default="legacy",
        help="Explicit whole-skin cleanup mode; legacy preserves historical group defaults.",
    )
    return parser.parse_args(argv)


def cli(argv=None):
    """Command-line entry point."""
    args = parse_args(argv)
    result = main(
        input_folder=args.input_dir,
        output_root=args.output_dir,
        ilastik_executable=args.ilastik_exe,
        skip_already_processed=args.skip_existing,
        reuse_existing_segmentations=not args.rerun_ilastik,
        whole_skin_cleanup={"legacy": None, "on": True, "off": False}[
            args.whole_skin_cleanup
        ],
    )
    return int(result["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(cli())
