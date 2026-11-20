# Skin Section Analysis

Measure nerve-positive area and skeleton length in mouse plantar-skin sections
from paired DAPI and βIII-tubulin (BT3) images. Results include tissue overlays,
measurements for each section, and averages for each mouse.

Work in progress. Review the overlays before using the measurements; the
pipeline does not count individual nerves.

## Install

Install Git, Conda, and [Ilastik 1.4.2](https://www.ilastik.org/documentation/basics/installation).
Tested on macOS Apple Silicon; Windows and Linux are not yet fully verified.

```bash
git clone https://github.com/henrylechang/Skin-Analysis.git
cd Skin-Analysis
conda --no-plugins env create --solver classic -f environment.yml
conda activate skin-analysis
git lfs install
git lfs pull
```

The two models are about 1.49 GB; the example images add about 222 MiB.

## Try the example

```bash
python Skin_Section_Analysis.py --samples examples/samples.csv
```

Results go to `Output_Data/`. The included CD49F image is for visual comparison;
only DAPI and BT3 are analyzed.

## Run your images

Use intact, aligned, equally sized 2D grayscale TIFFs with pixel calibration
and the epidermis facing up. Preserve the original 12-bit intensity scale when
saving as 16-bit TIFFs. Other acquisition setups need separate validation.

Place matching pairs in `My_Images/`:

```text
mouse1_section1_DAPI.tif
mouse1_section1_BT3.tif
mouse1_section2_DAPI.tif
mouse1_section2_BT3.tif
```

Keep each mouse's name consistent across its sections, then run:

```bash
python Skin_Section_Analysis.py --input-dir My_Images --output-dir My_Results
```

Inputs stay unchanged. Use a new output folder to preserve earlier results.

## Check the results

- `batch_run_log.csv`: which sections completed or failed.
- Each section's `QC/` folder: tissue boundaries and nerve overlays. Start with
  `00_whole_skin_cleanup.png`; open `whole_skin/index.html` for cleanup details.
- `BT3_quantification_results.csv`: measurements for one section.
- `BT3_quantification_by_biological_replicate.xlsx`: the two epidermal measurements
  averaged per mouse. Check mouse names and section counts.

[Imaging requirements, methods, and settings](DOCUMENTATION.md)
