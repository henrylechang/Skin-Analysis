"""Step-by-step whole-skin diagnostics, without changing cleanup decisions."""

import csv
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, maximum_filter
import tifffile

from .candidate1_qc import _normalize


STAGES = (
    ("raw", "01_raw", "Raw Ilastik whole-skin mask"),
    ("filtered", "03_filtered", "After removing detached lower objects"),
    ("closed", "04_closed", "After closing small gaps"),
    ("filled", "05_filled", "After filling enclosed holes - final mask"),
)


def save_whole_skin_qc(output, dapi, steps, pixel_size_um, *, maximum_width=2400):
    """Save exact masks, DAPI overlays, per-step changes, and a review page.

    All views use the same image extent and DAPI contrast. Colored changes
    are max-pooled for display so a small edit is not lost by downsampling;
    only the TIFF masks and pixel counts represent exact pixel geometry.
    Reference components are a diagnostic, not a tissue-removal step.
    """
    output = Path(output) / "QC"
    folder = output / "whole_skin"
    folder.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(np.ceil(dapi.shape[1] / maximum_width)))
    gray = _normalize(dapi[::stride, ::stride])
    background = np.repeat(gray[..., None], 3, axis=2)
    font = ImageFont.load_default(size=20)
    small_font = ImageFont.load_default(size=16)
    sample = Path(output).parent.name

    def visible(mask):
        if stride == 1:
            return mask
        return maximum_filter(mask, size=stride, mode="constant")[::stride, ::stride]

    def overlay(mask, *, added=None, removed=None):
        rgb = background.copy()
        outline = mask & ~binary_erosion(mask, border_value=0)
        rgb[visible(outline)] = (255, 220, 0)
        if added is not None:
            rgb[visible(added)] = (0, 230, 255)
        if removed is not None:
            rgb[visible(removed)] = (255, 65, 65)
        return Image.fromarray(rgb)

    def caption(image, title, subtitle=""):
        canvas = Image.new("RGB", (max(image.width, 900), image.height + 72), "#15191f")
        canvas.paste(image, (0, 72))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), title, fill="white", font=font)
        draw.text((12, 38), subtitle, fill="#d7dce3", font=small_font)
        return canvas

    def stack(images):
        canvas = Image.new(
            "RGB",
            (max(i.width for i in images), sum(i.height for i in images)),
            "#15191f",
        )
        y = 0
        for image in images:
            canvas.paste(image, (0, y))
            y += image.height
        return canvas

    def save_mask(name, mask):
        tifffile.imwrite(
            folder / f"{name}.tif",
            np.asarray(mask, dtype=np.uint8) * 255,
            compression="deflate",
            imagej=True,
            resolution=(1 / pixel_size_um, 1 / pixel_size_um),
            metadata={"unit": "um", "axes": "YX"},
        )

    overview = [
        caption(
            Image.fromarray(background),
            f"{sample} | DAPI",
            "Same contrast and full image extent in every panel",
        )
    ]
    rows = []
    previous = None
    links = []
    for key, filename, title in STAGES:
        mask = steps[key]
        save_mask(filename, mask)
        before = mask if previous is None else previous
        added, removed = mask & ~before, before & ~mask
        count = int(mask.sum())
        add_count, remove_count = int(added.sum()), int(removed.sum())
        before_count = int(before.sum())
        details = f"Added: {add_count:,} px | Removed: {remove_count:,} px | Retained mask: {count:,} px"
        rows.append(
            dict(
                stage=key,
                before_pixels=before_count,
                after_pixels=count,
                added_pixels=add_count,
                removed_pixels=remove_count,
                added_area_um2=add_count * pixel_size_um**2,
                removed_area_um2=remove_count * pixel_size_um**2,
                removed_percent_of_before=100 * remove_count / before_count
                if before_count
                else 0,
            )
        )
        current = caption(overlay(mask), title, "Yellow: mask outline | " + details)
        if previous is None:
            current.save(folder / f"{filename}.png")
            overview.append(current)
            reference = steps["reference"]
            save_mask("02_reference_components", reference)
            ref = np.array(overlay(reference))
            columns = np.arange(ref.shape[1])
            for edge in ("upper_boundary", "lower_boundary"):
                ys = np.clip(
                    np.rint(steps[edge][::stride] / stride).astype(int),
                    0,
                    ref.shape[0] - 1,
                )
                ref[ys, columns] = (0, 230, 255)
            reference_panel = caption(
                Image.fromarray(ref),
                "Reference components and smoothed limits (diagnostic only)",
                "Yellow: reference outline | Cyan: limits used to test detached objects; these do not clip tissue",
            )
            reference_panel.save(folder / "02_reference_components.png")
            overview.append(reference_panel)
            with (folder / "reference_limits.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["column_px", "upper_row_px", "lower_row_px"])
                writer.writerows(
                    zip(
                        range(mask.shape[1]),
                        steps["upper_boundary"],
                        steps["lower_boundary"],
                    )
                )
        else:
            change = caption(
                overlay(mask, added=added, removed=removed),
                title + " | Changes",
                "Red: removed | Cyan: added | Yellow: resulting outline | " + details,
            )
            stack(
                [
                    caption(
                        overlay(previous),
                        title + " | Before",
                        "Yellow: previous mask outline",
                    ),
                    current,
                    change,
                ]
            ).save(folder / f"{filename}.png")
            overview.append(change)
        links.append((filename, title))
        previous = mask
    with (folder / "step_changes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    overview_image = stack(overview)
    if overview_image.width > 1800:
        overview_image = overview_image.resize(
            (1800, round(overview_image.height * 1800 / overview_image.width)),
            Image.Resampling.LANCZOS,
        )
    overview_image.save(output / "00_whole_skin_cleanup.png")
    links.insert(
        1, ("02_reference_components", "Reference components (diagnostic only)")
    )
    sections = "".join(
        f'<section><h2>{html.escape(title)}</h2><p><a href="{name}.tif">Full-resolution mask (TIFF)</a> · <a href="{name}.png">Open comparison at full size</a></p><a href="{name}.png"><img src="{name}.png" alt="{html.escape(title)}"></a></section>'
        for name, title in links
    )
    (folder / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(sample)} - whole skin QC</title>"
        "<style>body{margin:24px auto;max-width:1400px;padding:0 20px;background:#15191f;color:#eef2f7;font:18px/1.5 system-ui}a{color:#76d8ff}img{width:100%;height:auto}section{margin:40px 0}h2{font-size:24px}</style>"
        f"<h1>{html.escape(sample)}: whole-skin cleanup</h1>"
        "<p>Read from top to bottom. Each cleanup comparison shows before, after, and changes. Yellow marks the mask outline; red marks removed pixels; cyan marks added pixels. The reference view instead uses cyan for the smoothed limits.</p>"
        "<p>The full image extent and DAPI contrast stay fixed. Small changes are enlarged for visibility in previews; TIFF masks contain the exact original-resolution pixels. Reference components guide debris removal and are not an intermediate tissue mask.</p>"
        '<p><a href="../00_whole_skin_cleanup.png">Overview</a> · <a href="step_changes.csv">Exact changes at each step</a> · <a href="reference_limits.csv">Reference limits</a></p>'
        + sections,
        encoding="utf-8",
    )
    return rows
