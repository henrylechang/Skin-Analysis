# Methods and settings

For installation and a first run, see [README](README.md).

## Input images

The pipeline was developed for Olympus FLUOVIEW FV3000 confocal images with a
20× objective: 0.621504 µm/pixel, 1040 × 1040-pixel fields stitched across an
intact section. Sections were 50 µm thick and longitudinal, imaged at 3 µm depth
intervals, then combined into one maximum-intensity projection per channel.
Acquisition settings were held constant within each comparison. Other imaging
setups need separate validation.

Use aligned 2D grayscale DAPI/BT3 TIFFs with epidermis up and dermis below.
Preserve the 12-bit acquisition range (0–4095) when saving as 16-bit TIFFs;
rescaling to 0–65535 changes which pixels pass the fixed nerve threshold.
The pipeline does not align channels, rotate images, normalize intensities,
or process stacks.

Files are paired by sample name within each folder, recursively and without
regard to case. `.tif` and `.tiff` are accepted. `_DAPI` pairs with `_BT3` or
`_nerve`; `_nerve` takes priority if both exist.

Pixel size comes from OME, ImageJ, then standard TIFF metadata, in that order.
X/Y calibration and the two channels must agree within 0.1%. A single calibrated
axis is used for both axes; two valid axes are averaged. Missing calibration
requires an explicit, verified `fallback_pixel_size_um` in your configuration.
DAPI calibration controls measurements. Check both `pixel_size_source` and
`nerve_pixel_size_source` in the results.

## Settings and sample lists

Use JSON to change settings without editing Python:

```bash
python Skin_Section_Analysis.py --config examples/analysis.json --samples examples/samples.csv
```

[analysis.json](examples/analysis.json) shows common settings at their defaults.
Partial files are allowed; omitted settings keep their defaults. Unknown keys
and invalid values are rejected. Full defaults live in
[AnalysisConfig](epidermis_analysis/configuration.py),
[Candidate1Config](epidermis_analysis/candidate1_config.py), and
[SubbasalConfig](epidermis_analysis/subbasal.py).

A sample CSV selects exactly which sections to process and sets their mouse IDs:

| Column | Meaning |
| --- | --- |
| `sample_id` | Section name, unique within its group |
| `mouse_id` | Animal ID used for section averages |
| `group` | Relative output folder, or `.` for the root |
| `dapi`, `nerve` | TIFF paths relative to `--input-dir`, using `/` separators |

All fields are required; see [samples.csv](examples/samples.csv). Without a CSV,
mouse IDs are inferred by removing the numbered `_section` or `-section` suffix.
Verify these IDs before using mouse averages. Every section uses the same cleanup.

Each run saves `configuration.resolved.json` and `samples.resolved.csv` at the
output root. Copy them outside that folder before using them as inputs to a new
run. For all command options, run `python Skin_Section_Analysis.py --help`.

## How measurements are made

1. Two Ilastik Autocontext models segment DAPI into an epidermal band and whole
   tissue. Both use class 1 from their Stage 2 label exports.
2. Long superficial tissue components define a smoothed reference envelope.
   Detached objects entirely below it are removed, small gaps are closed, and
   enclosed holes are filled. Attached tissue is not clipped to the envelope.
3. Epidermal components are joined into nonbranching paths using their length,
   surface proximity, DAPI support, and tissue membership. By default, the
   component nearest the most surface pixels anchors each tissue piece.
   The code calls this selection method **Candidate 1**.
4. Paths and the tissue surface define the epidermis, dermis, and their boundary.
   Bridges across gaps are inferred; check them in the overlays.
5. BT3 pixels with intensity **greater than 1500** are nerve-positive. An entire
   connected object is removed if any part is within 5 µm of the reconstructed
   surface and its skeleton is at least 20 µm long. This happens before tissue
   clipping, so connected deeper signal can also be removed.
6. Retained signal is measured in the four compartments below.

| Compartment | Definition |
| --- | --- |
| Upper epidermis | Reconstructed epidermis outside the cleaned Ilastik band |
| Basal epidermis | Cleaned Ilastik band inside the epidermis |
| Subepidermal dermis (`subbasal_dermis`) | Dermis within 20 µm depth of a smoothed basal reference |
| Deep dermis | Remaining dermis |

The smoothed reference is used only for dermal depth. Depth follows connected
dermal pixels, with straight/diagonal steps of one/√2 pixel widths. Pixels on or
touching the reference start at one pixel width; disconnected, unreached dermis
falls into deep dermis. This is an approximation to physical depth.

| Measurement | Calculation |
| --- | --- |
| Nerve area fraction | Positive area / tissue area (0–1) |
| Epidermal nerve area per boundary mm | Positive area in µm² / basal boundary length in mm |
| Epidermal skeleton length per boundary mm | Skeleton length in µm / basal boundary length in mm |
| Skeleton length density | Skeleton length / tissue area, in µm/mm² |

Boundary normalization uses the skeletonized anatomical boundary
(`epidermal_boundary_length_um`), not the smoothed depth reference or the ordered
path length (`anatomical_basal_length_um`). Skeleton length counts each horizontal
or vertical edge once; a diagonal counts √2 pixel widths only when neither
orthogonal corner pixel is present. An isolated pixel counts one pixel width.
Whole compartments are skeletonized separately before assignment to smaller
compartments, so smaller-compartment lengths need not sum to whole lengths.
Zero denominators produce empty CSV cells.

Columns ending in `_per_boundary_mm` use millimeters; older `_per_boundary_length`
columns use micrometers and are 1000 times smaller. Legacy `BT3` columns alias
corresponding `nerve` columns. These measurements describe staining area and
skeleton length, not fluorescence intensity or individual nerve counts (IENFD).

The mouse workbook reports unweighted means of section-level epidermal area and
skeleton length per boundary mm, grouped by group and mouse ID. Missing values
are omitted per measurement; section counts include all section rows. Other
metrics require downstream aggregation. There is no automatic biological QC
exclusion or statistical testing.

## Outputs and review

| Location | Contents |
| --- | --- |
| `batch_run_log.csv` | Current batch status; failed sections are excluded from summaries |
| `combined_BT3_quantification_results.csv` | Section measurements, at root and group levels |
| `BT3_quantification_by_biological_replicate.xlsx` | Mouse averages |
| Each section's three `*quantification_results.csv` files | Main, dermal-depth, and compartment measurements |
| `QC/00_whole_skin_cleanup.png`, `QC/whole_skin/index.html` | Whole-tissue cleanup overview and before/after views |
| `QC/01_candidate_selection.png` | Accepted and rejected epidermal components |
| `QC/02_anatomical_reconstruction.png` | Reconstructed tissue boundaries |
| `QC/03_compartments_and_nerve.png` | Compartments, nerve signal, and skeletons |
| `QC/component_decisions.csv`, `QC/candidate_edge_decisions.csv` | Component evidence and evaluated bridge selections |
| `QC/surface_coverage.json` | Surface assigned to unselected components; diagnostic only |
| Numbered TIFFs | Full-resolution masks, saved losslessly as 0/255 pixels |
| `analysis_parameters.json`, `completion.json` | Settings, provenance, and result-table integrity records |
| `ilastik/` | Cached label exports, hash records, and inference logs |

Review tissue coverage, inferred boundaries, and retained nerve signal before
accepting measurements. QC previews may be downsampled and contrast-adjusted;
quantitative masks are unchanged. Main analysis TIFFs do not embed calibration;
use `pixel_size_um` from the results. The extra `QC/whole_skin/` TIFFs do embed it.
That folder also contains exact pixel-change counts in `step_changes.csv`.

## Reruns and reports

Use a separate output folder for each independent analysis. Input and output
folders must not overlap; linked output files are rejected. Do not run two
batches into the same folder or change inputs, models, or code during a run.

- Ilastik exports are reused only when input, model, launcher, settings, and
  export hashes match. `--rerun-ilastik` regenerates them.
- `--skip-existing` skips completed sections only when inputs, models, source,
  runtime, settings, result tables, and parameters match. It takes precedence
  over `--rerun-ilastik`. It does not verify masks, QC, or the full Ilastik
  installation. After an Ilastik upgrade, use `--rerun-ilastik` without
  `--skip-existing`.
- Failed reruns can leave older masks or partial files. Use the current batch
  log to judge completion. Unmatched inputs appear in `missing_file_pairs.csv`;
  they are omitted, while failures of paired sections cause a nonzero exit.

Rebuild reports from verified results listed in the latest batch log, without
running image analysis:

```bash
python Skin_Section_Analysis.py --output-dir My_Results --reports-only
```

Add `--no-excel` to either mode for CSV-only reports; this removes stale generated
workbooks. Keep the code version and parameter files with your results. Default
example and `My_Images`/`My_Results` folders are ignored by Git; other custom
folders may need their own ignore rules.

## Troubleshooting

| Problem | Action |
| --- | --- |
| Model is a small text file, or clone reports missing Git LFS | Finish creating and activating the environment, then run `git lfs install` and `git lfs pull` |
| Ilastik is missing or multiple installations are found | Pass `--ilastik-exe PATH`; use the `.app` bundle on macOS, `ilastik.exe` on Windows, or `run_ilastik.sh` on Linux |
| No usable calibration | Export calibrated TIFFs or set a verified `fallback_pixel_size_um` |
| A section fails | Read its `error_traceback.txt` and Ilastik console log, then inspect QC |
| Old configuration is rejected | Start from the current example; retired settings are not silently reused |

Ilastik runs one job at a time with one worker and an 8192 MB lazyflow budget.
Allow additional RAM for Python and the operating system. The budget is not a
limit on total memory, and exact results across platforms are not guaranteed.

## Code and checks

| File or module | Responsibility |
| --- | --- |
| `Skin_Section_Analysis.py` | Command line, Ilastik execution, section workflow, and batch orchestration |
| `image_io.py`, `samples.py`, `configuration.py` | TIFF calibration, sample lists, and settings |
| `whole_skin.py`, `whole_skin_qc.py` | Tissue cleanup and its review outputs |
| `candidate1_*.py`, `components.py`, `reconstruction.py` | Epidermal selection, geometry, and boundaries |
| `subbasal.py`, `four_compartment.py`, `measurement.py` | Depth, tissue compartments, and measurements |
| `reporting.py`, `provenance.py` | Summaries, cache checks, and run records |

Modules are in `epidermis_analysis/`. Coordinates use `(row, column)` pixels;
physical quantities use micrometers unless a name specifies otherwise. Python
callers can pass an `AnalysisConfig` to `main()` or `process_sample()`.

```bash
python -m unittest discover -s validation -v
```

Tests cover calibration, input protection, caching, configuration, grouping,
report exports, and synthetic tissue analysis. Ilastik inference is simulated;
these checks do not establish biological accuracy or cross-platform repeatability.
