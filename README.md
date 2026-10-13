# Skin Section Analysis

Measure nerve staining in mouse plantar-skin sections from paired DAPI and
βIII-tubulin (BT3) images. The pipeline creates tissue overlays, measurements for
each section, and averages for each mouse.

**Work in progress:** This pipeline is under active development. For questions
or feedback, email [henrylechang45@gmail.com](mailto:henrylechang45@gmail.com).

## Before you start: imaging requirements

Use intact sections with the whole tissue visible. This pipeline was designed
for mouse plantar skin acquired with the following imaging setup:

- **Microscope:** Olympus FLUOVIEW FV3000 confocal microscope with a **20× objective**.
- **Sampling:** **0.621504 µm/pixel**; 1040 × 1040-pixel fields stitched into a whole-section image.
- **Sections and depth:** 50 µm longitudinal sections; images acquired at 3 µm depth intervals and combined into one maximum-intensity projection per channel.
- **Channels and files:** matched DAPI and βIII-tubulin images, acquired at 12-bit depth and saved as 16-bit grayscale TIFFs without changing the intensity scale.
- **Orientation:** epidermis facing **up**, with dermis below. Acquisition settings were held constant within each experimental comparison.

Other imaging setups require separate validation.

## 1. Install once

Install **Git**, **Conda** (for example, Miniforge), and
[**Ilastik 1.4.2**](https://www.ilastik.org/documentation/basics/installation).
Tested on macOS Apple Silicon. See the
[installation guide](DOCUMENTATION.md#installation) if you need help.

Open **Terminal**, paste these commands, and keep that window open for the next steps:

```bash
git clone https://github.com/henrylechang/Skin-Analysis.git
cd Skin-Analysis
conda --no-plugins env create --solver classic -f environment.yml
conda activate skin-analysis
git lfs install
git lfs pull
```

The model download is approximately **1.49 GB**.

## 2. Try the included example

```bash
python Skin_Section_Analysis.py --samples examples/samples.csv
```

This runs one included section and saves its results in `Output_Data`.
The example's CD49F image is for visual comparison; only DAPI and BT3 are analyzed.

## 3. Add your images

Create a folder called **`My_Images`** inside `Skin-Analysis`. Copy your paired
TIFFs into it, using matching names:

```text
mouse1_section1_DAPI.tif
mouse1_section1_BT3.tif
mouse1_section2_DAPI.tif
mouse1_section2_BT3.tif
mouse2_section1_DAPI.tif
mouse2_section1_BT3.tif
```

Use a unique mouse name for each animal. Keep that name the same across its
sections. Each DAPI/BT3 pair must be aligned, have the same dimensions, and
retain its pixel calibration. Keep the epidermis facing up.

## 4. Run your images

In the same Terminal window, run:

```bash
python Skin_Section_Analysis.py --input-dir My_Images --output-dir My_Results
```

This processes every matching image pair in `My_Images` and saves the results
in `My_Results`. Your original images are not changed.

If you closed Terminal, open it again, return to the `Skin-Analysis` folder
with `cd`, and run `conda activate skin-analysis` before running the command.

## 5. Check your results

Open **`My_Results`** in Finder:

- **`batch_run_log.csv`** tells you which sections completed or failed.
- Each section's **`QC`** folder contains three overlay images. Check that the
  tissue boundaries and nerve staining were identified correctly.
- Each section's **`BT3_quantification_results.csv`** contains its measurements.
- **`BT3_quantification_by_biological_replicate.xlsx`** contains the two epidermal measurements averaged for
  each mouse (within each group folder, if you used subfolders). Check that sections are grouped under the correct mouse names.

Review the overlays before interpreting the measurements. These measure
nerve-positive staining area and skeleton length. They do not count individual nerves.
For a fresh analysis, choose a new results-folder name to keep earlier results.

[Methods, settings, and troubleshooting](DOCUMENTATION.md)
