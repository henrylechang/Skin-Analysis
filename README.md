# Skin Section Analysis

Measures βIII-tubulin (TUBB3/BT3) staining in mouse plantar-skin sections using
paired DAPI and BT3 images. Reports nerve-positive area and skeleton length in
upper epidermis, basal epidermis, subepidermal dermis (20 µm), and deeper dermis.
These are staining measurements, not counts of individual nerve fibers.

## Setup

Install Git, Conda, and [Ilastik 1.4.2](https://www.ilastik.org/documentation/basics/installation), then run:

```bash
git clone https://github.com/henrylechang/Skin-Analysis.git
cd Skin-Analysis
conda --no-plugins env create --solver classic -f environment.yml
conda activate skin-analysis
git lfs install
git lfs pull
```

Tested on macOS Apple Silicon. The model download is approximately 1.49 GB.
See [installation details](DOCUMENTATION.md#installation) for memory needs and troubleshooting.

## Images

Place matched images together under `Input_Data/`, for example:

```text
mouse1_section1_DAPI.tif
mouse1_section1_BT3.tif
```

Use aligned, equally sized 2D grayscale TIFFs, with epidermis at the top.
Preserve the original 12-bit intensity scale (0–4095) when saving as 16-bit
TIFFs; nerve detection uses a fixed threshold of **>1500**. Verify pixel
calibration; missing calibration defaults to **0.621504 µm/pixel**.

One example section is included. Its CD49F image is for visual comparison only.
Use the sample manifest below to record its mouse ID and tissue-cleanup choice
explicitly. Without a manifest, default cleanup depends on the group-folder name.

## Run

From the repository folder:

```bash
conda activate skin-analysis
python Skin_Section_Analysis.py --input-dir Input_Data --output-dir Output_Data
```

This processes all matched sections under `Input_Data/`. Inputs remain unchanged.
For an explicit, repeatable batch, use the supplied settings and sample manifest:

```bash
python Skin_Section_Analysis.py --config examples/analysis.json --samples examples/samples.csv
```

The example manifest selects only the included section. For your own batch, copy
and edit these files: each CSV row supplies a section ID, mouse ID, group, two
input-root-relative TIFF paths, and an explicit cleanup choice. See
[configuration and manifests](DOCUMENTATION.md#configuration-and-sample-manifests).

Rebuild combined CSVs and workbooks from verified section results without
rerunning analysis:

```bash
python Skin_Section_Analysis.py --output-dir Output_Data --reports-only
```

Add `--no-excel` to either workflow to produce CSV reports without workbooks.

## Results

Under `Output_Data/`, look for:

- **`QC/`** in each section folder: overlays of tissue boundaries and nerve detection.
- **`BT3_quantification_results.csv`** in each section folder: measurements.
- **`BT3_quantification_by_biological_replicate.xlsx`** in the group/root folder:
  section results and mouse averages for the two epidermal measurements per
  millimeter of basal boundary.
- **`batch_run_log.csv`**: completed, skipped, and failed sections.
- **`configuration.resolved.json` and `samples.resolved.csv`**: effective settings
  and sample identities for the batch.

Inspect QC overlays and full-resolution masks before using the measurements.
Masks use lossless TIFF compression. Verify mouse IDs before using mouse averages;
IDs are inferred from filenames only when no manifest is supplied.

See [Methods and technical documentation](DOCUMENTATION.md) for measurement
formulas, all output files, parameters, rerun options, and validation limits.
