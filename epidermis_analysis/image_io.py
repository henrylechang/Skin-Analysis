"""Read 2D TIFFs and pixel calibration; write lossless binary masks.

Calibration is in micrometers per pixel. OME metadata takes precedence over
ImageJ and standard TIFF tags. Input files are only opened for reading.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import tifffile


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


def read_pixel_size_um(path, fallback_pixel_size_um=None):
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

    if fallback_pixel_size_um is None:
        raise ValueError(
            f"No usable pixel calibration in {Path(path).name}. "
            "Provide calibrated TIFF metadata or explicitly set "
            "fallback_pixel_size_um to the verified acquisition scale."
        )
    if (
        isinstance(fallback_pixel_size_um, (bool, np.bool_))
        or not isinstance(fallback_pixel_size_um, (int, float, np.integer, np.floating))
        or not np.isfinite(fallback_pixel_size_um)
        or fallback_pixel_size_um <= 0
    ):
        raise ValueError("fallback_pixel_size_um must be a positive finite number.")
    return float(fallback_pixel_size_um), "fallback"


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
        compression="deflate",
    )
