# Skin Section Analysis

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

The pipeline has been tested on macOS Apple Silicon. Launcher discovery also
supports Windows and Linux. Each Ilastik process uses one worker and an
8192 MB RAM budget; allow additional memory for Python and the operating system.
Numerical-library threads are also fixed to one. Parallel random-forest
aggregation in Ilastik 1.4.2 can change a few labels between fresh runs because
floating-point sums depend on completion order. Serial inference avoids that
source of variation, but can be slower. Outputs from older parallel runs
can differ slightly; do not assume bitwise equality across software platforms.

## Input images

Use aligned, equally sized 2D grayscale TIFFs, oriented with epidermis at the
top and dermis at the bottom. Images were acquired at 12-bit depth
(0–4095) and saved as 16-bit grayscale TIFFs with the original intensity
scale preserved. Image preparation was performed in Fiji (ImageJ v1.54p).
Do not rescale intensities to 0–65535: nerve segmentation uses a fixed threshold.

Pair each `_DAPI` image with a `_nerve` image containing TUBB3.
The suffix `_BT3` is also accepted;
`_nerve` takes priority if both exist. Discovery is recursive and supports
`.tif` and `.tiff`. The pipeline does not register channels or process stacks.

Pixel size is read from TIFF metadata, with a fallback of 0.621504 µm/pixel.
Pixels must be isotropic: X/Y calibration differing by more than 0.1% is
rejected instead of averaged. Check `pixel_size_source` in the results to
identify fallback use. Keep input files read-only and write results separately.

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
  and the parameter file. Inspect masks and QC separately.
  Old results without this manifest regenerate. A verified skip takes precedence
  over `--rerun-ilastik`; omit `--skip-existing` to force a complete rerun.
- `--whole-skin-cleanup legacy|on|off`: select whole-skin cleanup. The default,
  `legacy`, enables enhanced cleanup for relative group paths `oldmice`,
  `youngmice`, `oldmice_validation`, and `youngmice_validation`
  (case-insensitive), and disables it for other paths.

Renaming a group can therefore change results. Use a new output directory for
an independent run, especially after changing inputs, models, or settings.
Use `--whole-skin-cleanup on` or `off` to make cleanup independent of folder names.
Input and output directories must not overlap; output trees containing symbolic
links are rejected to protect input files. Use a dedicated output directory.
Do not change input images, models, code, or settings while a batch is running.

## Analysis

TUBB3-positive pixels have intensity strictly greater than 1500.
An entire positive object is excluded if its distance from the superficial
surface is at most 5 µm and its skeleton length is at least 20 µm.
Remaining signal is clipped to the reconstructed tissue.

| Compartment | Definition |
| --- | --- |
| Upper epidermis | Reconstructed epidermis outside the cleaned Ilastik band |
| Basal epidermis | Cleaned Ilastik band within the epidermis; not a fixed-depth strip |
| Subepidermal dermis (`subbasal_dermis`) | Dermis within 20 µm geodesic depth of the macro basal reference |
| Deep dermis | Remaining dermis |

The macro reference defines dermal depth; the anatomical basal boundary is used
for epidermal normalization. Parameters are defined at the top of
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

Inspect the overlays and full-resolution masks before accepting measurements.
Analysis-mask TIFFs do not embed spatial calibration; use `pixel_size_um`
from the CSV or JSON. Check `batch_run_log.csv` for failures and
`missing_file_pairs.csv`, when present, for either unmatched channel.
Batch summaries are rebuilt for the current inputs; failed sections are excluded.
A rerun invalidates prior completion tables before processing. Masks left in a
failed section's directory are not evidence of a successful current analysis.

Group and root folders contain combined CSVs and
`BT3_quantification_by_biological_replicate.xlsx`. The workbook averages only
epidermal nerve area and skeleton length per boundary mm within each mouse.
Mouse IDs are inferred by removing the numbered `_section` or `-section`
suffix; verify IDs and section counts. Other metrics require downstream
aggregation. No automatic biological QC exclusion or inferential testing is performed.

Keep the Git commit, model LFS identifiers (`git lfs ls-files --long`),
software versions, command, and section parameter files with each analysis.
Outputs are ignored by Git.

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
averages, and batch summaries. They do not replace inspection of biological QC.
