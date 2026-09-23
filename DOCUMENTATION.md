# Methods and technical documentation

For a short introduction and example run, see [README](README.md). This guide
describes the implemented methods, settings, output fields, and verification
limits. Default values below refer to the configuration supplied with the code.

Quantifies βIII-tubulin (TUBB3/BT3) innervation in mouse plantar-skin sections.
Two Ilastik Autocontext models segment DAPI images; Python reconstructs tissue
boundaries and measures nerve signal in upper epidermis, basal epidermis,
subepidermal dermis, and deep dermis.

## Installation

Install Git, Conda, and [Ilastik 1.4.2](https://www.ilastik.org/documentation/basics/installation).
The Conda environment supplies Python 3.13.9, Git LFS, and pinned runtime dependencies.

```bash
git clone https://github.com/henrylechang/Skin-Analysis.git
cd Skin-Analysis
conda --no-plugins env create --solver classic -f environment.yml
conda activate skin-analysis
git lfs install
git lfs pull
git lfs fsck
python -m pip check
```

If cloning reports missing Git LFS but creates the checkout, continue with the
remaining commands. The two models in `Ilastic_models/` total approximately
1.49 GB; small text pointers are not usable models. Both are two-stage
Autocontext projects with saved classifiers and embedded training images.

The pipeline has been tested on macOS Apple Silicon. Launcher discovery is
implemented for Windows and Linux; full-pipeline operation on those platforms
has not been verified here. Ilastik runs one job at a time, configured with one
worker and an 8192 MB lazyflow RAM budget. This is not a limit on total process
memory; allow extra RAM for Python and the operating system. Numerical-library
thread limits are set to one in the Ilastik subprocess only. Serial inference
reduces parallel scheduling variation but can be slower. Exact output equality
across software versions and platforms is not guaranteed.

## Input images

Use aligned, equally sized 2D grayscale TIFFs, oriented with epidermis at the
top and dermis at the bottom. The supplied preparation protocol uses 12-bit
acquisition (0–4095), saved as 16-bit grayscale TIFFs with the original intensity
scale preserved, and Fiji (ImageJ v1.54p) for image preparation.
Do not rescale intensities to 0–65535: nerve segmentation uses a fixed threshold.

Pair each `_DAPI` image with a `_nerve` image containing TUBB3.
The suffix `_BT3` is also accepted;
`_nerve` takes priority if both exist. Discovery is recursive, matches filenames
case-insensitively, and supports `.tif` and `.tiff`. Each pair must share a
folder and sample prefix. The pipeline does not register channels, rotate
images, normalize intensities, or process stacks.

Pixel size is read from OME, ImageJ, or standard TIFF resolution metadata, in
that order, with a fallback of 0.621504 µm/pixel when none is usable.
Pixels must be isotropic: X/Y calibration differing by more than 0.1% is
rejected; values within tolerance are averaged. If only one axis is calibrated,
its value is used for both axes. DAPI and nerve-channel pixel sizes must also
agree within 0.1%; DAPI calibration controls all measurements. Check
`pixel_size_source` and `nerve_pixel_size_source` in the per-section results
to identify fallback use. Alignment, orientation, and acquisition intensity
scale require manual verification. Keep inputs read-only and results separate.

## Included example

The repository includes one section (approximately 222 MiB, downloaded through
Git LFS):

```text
Input_Data/
└── Oldmice_Validation/
    ├── valid_oldmouse-1_section1_DAPI.tif
    ├── valid_oldmouse-1_section1_BT3.tif
    └── valid_oldmouse-1_section1_CD49F.tif
```

CD49F is provided for visual boundary validation, not as a pipeline input.
Other standalone input images are excluded from Git.

## Running the pipeline

From the repository root:

```bash
conda activate skin-analysis
python Skin_Section_Analysis.py --input-dir Input_Data --output-dir Output_Data
```

In a fresh checkout, this runs the included example. On a workstation with
additional inputs, it processes those too. Keep the `Oldmice_Validation`
parent folder under the input root: default cleanup depends on the relative
group name.

Specify the Ilastik launcher if automatic discovery fails:

```bash
python Skin_Section_Analysis.py --ilastik-exe "/Applications/ilastik-1.4.2-arm64-OSX.app"
```

On Windows, supply `ilastik.exe`; on Linux, supply `run_ilastik.sh`.
Use `python Skin_Section_Analysis.py --help` for all options.

- `--rerun-ilastik`: regenerate segmentation masks. Otherwise, reuse requires
  matching SHA-256 hashes for the DAPI input, model, launcher, and exported mask,
  plus matching export settings. Old masks without provenance regenerate once.
- `--skip-existing`: skip sections only when `completion.json` verifies matching
  inputs, models, Python source, runtime versions, settings, measurement CSVs,
  and the parameter file. This check does not verify masks, QC images, or the
  installed Ilastik executable's contents. Inspect masks and QC separately.
  Old results without this manifest regenerate. A verified skip takes precedence
  over `--rerun-ilastik`; omit `--skip-existing` to force a complete rerun.
- `--whole-skin-cleanup legacy|on|off`: select enhanced whole-skin cleanup. The default,
  `legacy`, enables enhanced cleanup for relative group paths `oldmice`,
  `youngmice`, `oldmice_validation`, and `youngmice_validation`
  (case-insensitive, exact relative-path match), and disables it for other paths.
  For example, `experiment/Oldmice_Validation` does not match.
  Basic component filtering, gap closing, and hole filling still run in `off` mode.

Renaming a group can therefore change results. Use a new output directory for
an independent run, especially after changing inputs, models, or settings.
Use `--whole-skin-cleanup on` or `off` to make cleanup independent of folder names.
Input and output directories must not overlap; output trees containing symbolic
links or hard-linked files are rejected to protect input files. Use a dedicated
output directory.
Do not change input images, models, code, or settings while a batch is running.
Run only one batch at a time against a given output directory.

## Analysis

TUBB3-positive pixels have intensity strictly greater than 1500.
An entire 8-connected positive object is excluded if any of its pixels is
within 5 µm Euclidean distance of the reconstructed superficial surface and
its skeleton length is at least 20 µm. Filtering happens before tissue clipping,
so deeper signal connected to a qualifying superficial object is also removed.
Remaining signal is clipped to the reconstructed tissue. These are geometric
segmentation rules; inspect QC to decide whether the result fits the anatomy.

| Compartment | Definition |
| --- | --- |
| Upper epidermis | Reconstructed epidermis outside the cleaned Ilastik band |
| Basal epidermis | Cleaned Ilastik band within the epidermis; not a fixed-depth strip |
| Subepidermal dermis (`subbasal_dermis`) | Dermis reached within 20 µm by the discrete depth rule below |
| Deep dermis | Remaining dermis |

Epidermal-band cleanup (called **Candidate 1** in the code) selects whole
Ilastik components using a rooted, nonbranching graph. Component length,
proximity to the superficial surface, whole-tissue membership, and DAPI
support guide selection. Reconstruction closes local polygons against the
surface, assigns epidermis and dermis, and extracts their interface. Inferred
bridges can connect interface fragments; this boundary is an image-derived
estimate that requires visual inspection.

The macro reference is a smoothed version of the reconstructed basal paths
with selected return detours removed. Depth propagates through 8-connected
dermal pixels near this reference, with horizontal/vertical steps of one pixel
width and diagonal steps of √2 pixel widths. Dermal pixels on or touching the
reference start at one pixel width of depth. This is a raster approximation;
dermal pieces with no reference contact fall into deep dermis.

The reconstructed anatomical boundary is used for epidermal normalization;
the macro reference is used only for dermal depth. Parameters are defined at the top of
`Skin_Section_Analysis.py`, in `epidermis_analysis/candidate1_config.py`,
and in `SubbasalConfig` in `epidermis_analysis/subbasal.py`.

## Measurements

Measurements describe positive-pixel area and skeleton length, not integrated
fluorescence intensity or intraepidermal nerve fiber density (IENFD).

| Measurement | Interpretation |
| --- | --- |
| Nerve area fraction | Positive area / tissue area, reported as a fraction (0–1) |
| Epidermal nerve area per boundary mm | Positive area in µm² / anatomical basal boundary length in mm |
| Epidermal skeleton length per boundary mm | Skeleton length in µm / anatomical basal boundary length in mm |
| Skeleton length density | Skeleton length / tissue area, in µm/mm², for each compartment and whole epidermis/dermis |

All skeleton length densities use µm/mm² and column names ending in
`_skeleton_density_um_per_mm2`. Whole-dermis density is reported as
`whole_dermis_bt3_skeleton_density_um_per_mm2`; the duplicate older density
columns are no longer exported. Zero denominators produce empty CSV cells (NaN).
Whole epidermal and dermal skeleton lengths are
measured separately; daughter-compartment lengths need not sum to whole lengths.
Legacy `BT3` columns in the main table alias the corresponding `nerve` metrics.
The older `_per_boundary_length` fields divide by boundary length in **µm**;
the explicit `_per_boundary_mm` fields divide by boundary length in **mm**
and are 1000 times larger. `epidermal_boundary_length_um` is measured from the
raster boundary skeleton. `anatomical_basal_length_um` is the sum of ordered
path lengths used in depth analysis and is not the normalization denominator.

The table normalization helpers convert historical µm/µm² densities to µm/mm²
and backfill legacy column names row by row, including mixed old/new tables.

## Results and quality control

Each section's output folder contains:

- `BT3_quantification_results.csv`: measurements and section diagnostics.
- `subbasal_BT3_quantification_results.csv`: dermal-depth measurements.
- `four_compartment_BT3_quantification_results.csv`: compartment measurements.
- TIFF masks, cached Ilastik exports with hash sidecars, `analysis_parameters.json`,
  and a `completion.json` manifest for successful sections.
- Three overlays in `QC/`: component selection, anatomical reconstruction,
  and compartments with nerve signal.

QC overlays are display-normalized and may be downsampled to at most 5000 pixels
on the longest side. Inspect the overlays and full-resolution masks before
accepting measurements; display normalization does not change measurements.
Analysis-mask TIFFs do not embed spatial calibration; use `pixel_size_um`
from the CSV or JSON. Check `batch_run_log.csv` for failures and
`missing_file_pairs.csv`, when present, for either unmatched channel.
Batch summaries are rebuilt for the current inputs; failed sections are excluded.
A rerun invalidates prior completion tables before processing. Masks left in a
failed section's directory are not evidence of a successful current analysis.
Use the current batch log to distinguish success, failure, and verified skips;
old section folders can remain after their inputs are removed. A batch with
failed paired sections exits with status 1. Unmatched files are reported and
omitted; they do not by themselves cause a nonzero exit when complete pairs exist.

Group and root folders contain combined CSVs and
`BT3_quantification_by_biological_replicate.xlsx`. The workbook averages only
epidermal nerve area and skeleton length per boundary mm within each mouse.
These are unweighted means of section-level ratios, grouped by input folder
and inferred mouse ID, rather than ratios pooled across all pixels or boundary
lengths. Missing values are omitted separately for each mean; the section count
includes every section row, even if a metric is undefined.
Mouse IDs are inferred by removing the numbered `_section` or `-section`
suffix; verify IDs and section counts. Other metrics require downstream
aggregation. No automatic biological QC exclusion or inferential testing is performed.

Keep the Git commit, model LFS identifiers (`git lfs ls-files --long`),
software versions, command, and section parameter files with each analysis.
The default `Output_Data/` directory is ignored by Git. Custom output paths
inside the repository may not be ignored; keep analysis outputs out of source control.

Section parameter files record input/model hashes, Python source hashes, runtime
versions, and cleanup configuration automatically. Ilastik cache validation hashes
the launcher, not its entire installation: after upgrading Ilastik in place, run
with `--rerun-ilastik` and without `--skip-existing`.

## Validation

Run the public synthetic regression checks without Ilastik or example data:

```bash
python -m unittest discover -s validation -v
```

These check cache invalidation, calibration, output protection, mixed-schema
averages, batch summaries, invalid inputs, and a synthetic section through
reconstruction and export. Ilastik inference is simulated in the tests. They
do not measure segmentation accuracy, establish biological validity, or prove
repeatability across platforms. Inspect biological QC separately.

## Parameter and output reference

Coordinates are `(row, column)` in pixels. The scalar calibration `s` is in
µm/pixel; each pixel has area `s²`. Source constants and function defaults are
authoritative when configurations are changed.

### Parameter locations

| Setting in `Skin_Section_Analysis.py` | Current value | Role |
| --- | --- | --- |
| `FALLBACK_PIXEL_SIZE_UM` | 0.621504 | Used only when readable spatial metadata is absent |
| `MANUAL_NERVE_THRESHOLD` | 1500 | Positive nerve pixels satisfy intensity **>** threshold |
| `EPIDERMIS_LABEL`, `WHOLE_SKIN_LABEL` | 1, 1 | Positive class in Stage 2 label exports |
| `MIN_WHOLE_SKIN_OBJECT_AREA_UM2` | 3862.67222016 | Minimum area of a candidate whole-skin reference object |
| `MIN_COMPONENT_WIDTH_FRACTION` | 0.025 | Minimum span as fraction of image width |
| `MIN_SUPERFICIAL_ENVELOPE_FRACTION` | 0.5 | Fraction of candidate span contributing to the top envelope |
| `WHOLE_SKIN_BOUNDARY_SMOOTHING_WIDTH_UM` | 20 | Median smoothing width of whole-skin context traces |
| `WHOLE_SKIN_CONTEXT_CLOSING_RADIUS_UM` | 8 | Converted to iteration count for closing with a 3×3 square |
| `WHOLE_SKIN_DISCONNECTED_VERTICAL_MARGIN_UM` | 50 | Enhanced-mode margin for remote components |
| `WHOLE_SKIN_BASAL_SMOOTHING_WIDTH_UM` | 100 | Enhanced-mode lower-contour percentile window |
| `WHOLE_SKIN_BASAL_PERCENTILE` | 35 | Enhanced lower-contour percentile |
| `SUPERFICIAL_NERVE_EXCLUSION_DISTANCE_UM` | 5 | Minimum object-to-surface distance at or below this value qualifies for exclusion |
| `MIN_SUPERFICIAL_NERVE_OBJECT_LENGTH_UM` | 20 | Qualifying object must also have skeleton length at or above this value |
| `SUBBASAL_DEPTH_UM` | 20 | Maximum geodesic depth in dermis |
| `DERMAL_DEPTH_REFERENCE_SMOOTHING_UM` | 40 | Arc-length support for Savitzky–Golay smoothing |
| `SUBBASAL_DEEP_OUTLIER_UM` | 15 | Deep return-detour criterion |
| `SUBBASAL_MAX_APPENDAGE_WIDTH_UM` | 100 | Candidate shortcut endpoint separation limit |
| `SUBBASAL_RESAMPLE_UM` | 1 | Ordered-path sampling interval |
| `SUBBASAL_DEPTH_BANDS_UM` | `((0.0, 20.0),)` | Additional reported band, identical to the default overall ROI |

Whole-skin repair lengths convert using `round(distance/s)`, areas using
`round(area/s²)`, with
the minimum pixel counts and odd smoothing-window adjustments in
`calculate_pixel_parameters`. Closing uses a repeated square, not a disk.
Whole-skin repair fills all enclosed holes in the retained mask. Basic filtering,
closing, and hole filling run even with `--whole-skin-cleanup off`; the switch
controls the additional disconnected-object and lower-contour cleanup only.

`epidermis_analysis/candidate1_config.py` holds the immutable `Candidate1Config`:
25 µm minimum candidate geodesic length; 300 µm minimum root geodesic and
surface-course length; 100 µm root surface distance; 0.90 inside-tissue fraction;
750 µm endpoint gap; 24 neighbors per endpoint; 40 µm tangent radius;
support downsampling factor 3; curvature penalty 10; root cost 0.05;
evidence scale 500 µm. The optimizer enforces graph degree at most 2;
`maximum_graph_degree` is an invariant check, not an optimizer setting.
Additional image-support weights remain explicit in `candidate1_cleanup.py`.

`reconstruct_explicit_interface` defaults to 20 µm major-fragment length and
150 µm bridge gap. Its bridge and topology rules are in `reconstruction.py`.
`SubbasalConfig` in `subbasal.py` defines physical macro/depth settings;
the entry point constructs it from the constants above.

### Measurement conventions

Let `A_E`, `A_D` be epidermis/dermis areas; `N_E`, `N_D` the retained
nerve-positive areas; `L_E`, `L_D` the measured skeleton lengths; and `B`
the skeletonized anatomical basal boundary length. Areas are µm² and lengths
are µm. Zero denominators give NaN (empty CSV cells), not zero.

| Main CSV column | Definition | Units |
| --- | --- | --- |
| `epidermal_area_um2` | `A_E` | µm² |
| `epidermal_boundary_length_um` | `B` | µm |
| `epidermal_boundary_length_mm` | `B / 1000` | mm |
| `epidermal_nerve_area_um2` | `N_E` | µm² |
| `epidermal_nerve_area_fraction` | `N_E / A_E` | Fraction, 0–1 |
| `epidermal_nerve_area_per_boundary_length` | `N_E / B` | µm²/µm |
| `epidermal_nerve_area_um2_per_boundary_mm` | `N_E / (B / 1000)` | µm²/mm |
| `epidermal_nerve_skeleton_length_um` | `L_E` | µm |
| `epidermal_nerve_skeleton_length_per_boundary_length` | `L_E / B` | µm/µm |
| `epidermal_nerve_skeleton_length_um_per_boundary_mm` | `L_E / (B / 1000)` | µm/mm |
| `dermal_area_um2` | `A_D` | µm² |
| `dermal_nerve_area_um2` | `N_D` | µm² |
| `dermal_nerve_area_fraction` | `N_D / A_D` | Fraction, 0–1 |
| `dermal_nerve_skeleton_length_um` | `L_D` | µm |
| `whole_dermis_bt3_skeleton_density_um_per_mm2` | `L_D × 1,000,000 / A_D` | **µm/mm²** |

Legacy columns replacing `nerve` with `BT3` are aliases of the corresponding
nerve-named metrics above. They remain in per-section exports for compatibility.
These are positive-pixel areas, not sums of fluorescence intensities.

For each prefix `upper_epidermis`, `basal_epidermis`, `subbasal_dermis`,
`deep_dermis`, `whole_epidermis`, `whole_dermis`, the four-compartment table uses:

| Suffix after the prefix | Definition | Units |
| --- | --- | --- |
| `_area_um2` | ROI pixel count × `s²` | µm² |
| `_bt3_area_um2` | Retained positive pixel count in ROI × `s²` | µm² |
| `_bt3_area_fraction` | Positive area / ROI area | Fraction, 0–1 |
| `_bt3_skeleton_length_um` | Direct length of assigned skeleton | µm |
| `_bt3_skeleton_density_um_per_mm2` | Length × 1,000,000 / ROI area | **µm/mm²** |

All current skeleton densities use µm/mm². The old
`dermal_nerve_skeleton_length_density` and `dermal_BT3_skeleton_length_density`
fields used µm/µm² and are no longer exported. Table normalization multiplies
those historical values by 1,000,000, backfills the explicit-unit field row by
row, and removes the old fields. It does not rewrite saved source CSVs.

The subbasal table uses prefixes `subbasal` and `subbasal_0_20um` with suffixes
`_roi_area_um2`, `_bt3_area_um2`, `_bt3_area_fraction`, `_skeleton_length_um`, and
`_skeleton_density_um_per_mm2`. These correspond to the default
`subbasal_dermis` values. Other configured bands use `(lower, upper]` depth
intervals, encoded as `subbasal_<lower>_<upper>um` (decimal points become `p`).

Depth uses a discrete 8-connected dermal graph within a Euclidean search tube
around the raster macro reference. Seeds on or touching the reference all
start at `s`, including diagonal neighbors; subsequent orthogonal/diagonal
steps cost `s`/`sqrt(2) × s`. Seedless dermal regions remain outside the
subbasal ROI. This convention approximates physical depth; it is not an exact
continuous distance to the fitted curve.

Horizontal/vertical skeleton edges contribute `s`; a diagonal contributes
`sqrt(2) × s` only if neither orthogonal corner pixel is present. Isolated
skeleton pixels contribute `s`. Whole epidermis/dermis are skeletonized
separately before daughter assignment. Cutting a skeleton at a compartment
boundary can change edge and isolated-pixel counts, so daughter lengths are
not summed to obtain whole lengths.

### Macro geometry and QC columns

| Column | Meaning |
| --- | --- |
| `anatomical_basal_length_um` | Sum of segment lengths of ordered anatomical paths; differs in definition from raster boundary skeleton length `B` |
| `macro_reference_length_um` | Sum of ordered macro path lengths in µm |
| `macro_boundary_median_offset_um`, `macro_boundary_p95_offset_um` | Median / 95th percentile displacement of resampled anatomical points from nearest macro points |
| `macro_boundary_max_deep_offset_um` | Maximum signed dermal-normal displacement, clamped at zero |
| `macro_boundary_downweighted_fraction` | Fraction of resampled points marked by selected shortcut intervals |
| `fraction_boundary_points_downweighted` | Alias of that fraction in the compartment table |
| `subbasal_depth_um`, `macro_smoothing_um` | Configured physical depth and fitting support |
| `dermal_depth_reference_smoothing_um` | Alias of fitting support in per-section subbasal outputs |
| `subbasal_deep_outlier_um`, `subbasal_max_appendage_width_um`, `subbasal_resample_um` | Effective physical macro parameters |
| `subbasal_depth_bands_um` | Semicolon-separated depth intervals |
| `subbasal_distance_method`, `subbasal_shortcut_maximum_arc_um`, `subbasal_algorithm` | Algorithm identifiers; maximum shortcut arc is reported as unbounded |
| `four_compartment_analyzed_tissue_area_um2` | Area of epidermis union dermis |
| `four_compartment_excluded_tissue_pixels` | Zero when the mandatory partition checks pass |

Section metadata includes `Sample`/`sample_id`, DAPI calibration/source and
`nerve_pixel_size_um`/`nerve_pixel_size_source`, nerve
filename/convention, segmentation method/threshold, epidermis model and method
names, stage runtimes (seconds), selected component/edge counts, final boundary
path count, and inferred boundary fraction. `Group` is added by the batch
aggregator. Runtime columns vary even when scientific outputs are identical.

### Per-section files

Binary analysis TIFFs contain 0/255 uint8 pixels. They do not embed spatial
calibration; use `pixel_size_um` from the CSV/JSON when opening them externally.

| File | Contents |
| --- | --- |
| `01_raw_epidermis_ilastik.tif` | Positive class from epidermal segmentation |
| `04_cleaned_epidermis_mask.tif` | Accepted epidermal-band pixels |
| `06_reconstructed_epidermis_region.tif` | Whole reconstructed epidermis |
| `07_dermis_region.tif` | Whole reconstructed dermis |
| `08_final_epidermis_dermis_boundary.tif` | Quantitative anatomical boundary |
| `10_BT3_fixed_threshold_binary_mask.tif` | Unfiltered threshold result; can contain signal outside tissue |
| `11_epidermal_BT3_mask.tif`, `12_dermal_BT3_mask.tif` | Filtered, tissue-clipped positive signal |
| `13_epidermal_BT3_skeleton.tif`, `14_dermal_BT3_skeleton.tif` | Whole-compartment signal skeletons |
| `16_macro_basal_reference.tif` | Derived depth reference |
| `18_subbasal_ROI.tif` | Default dermal-depth ROI |
| `23_upper_epidermis_region.tif`, `24_basal_epidermis_region.tif` | Epidermal daughter ROIs |
| `25_subbasal_dermis_region.tif`, `26_deep_dermis_region.tif` | Dermal daughter ROIs |
| `18_subbasal_ROI_<lower>_<upper>um.tif` | Only written for additional nondefault depth bands |
| `ilastik/*_improved_epimask.tif`, `ilastik/*_wholemask.tif` | Cached class-label exports, not 0/255 analysis masks |
| `ilastik/*_ilastik_console_log.txt` | Command and stdout/stderr from new inference |
| `analysis_parameters.json` | Section settings, both calibration sources, runtime versions, source/input/model hashes, and segmentation provenance |
| `ilastik/*.tif.json` | Cache identity and exported label-image hash |
| `completion.json` | Analysis identity and hashes of three CSVs plus the parameter file |
| `BT3_quantification_results.csv` | One merged section record |
| `subbasal_BT3_quantification_results.csv` | One depth-analysis record |
| `four_compartment_BT3_quantification_results.csv` | One four/whole-compartment record |

The three QC files are `QC/01_candidate_selection.png`,
`QC/02_anatomical_reconstruction.png`, and `QC/03_compartments_and_nerve.png`.
The first shows accepted/rejected components and the course graph; the second
shows the anatomical reconstruction; the third overlays daughter compartments,
anatomical/macro/lower boundaries, signal, and skeletons. Read each image's legend.
Display intensity normalization affects QC images only, never quantitative masks.

On failure, inspect `error_traceback.txt` and, when produced,
`QC/candidate1_QC_FAILURE.txt` and `QC/candidate_selection_FAILURE.png`.
Before processing a section again, the batch removes its old completion
manifest, three result tables, parameter file, traceback, and Candidate-1
failure-QC markers. A failure can leave partial new files or older masks/QC
overlays; use the current batch log and completion manifest to assess status.
Completion checks verify tables and parameters, not masks or QC images.
Use a new output folder for an independent run.
