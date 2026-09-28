"""Portable sample manifests with explicit biological identities and cleanup choices."""

import csv
from pathlib import Path, PurePosixPath

MANIFEST_COLUMNS = (
    "sample_id",
    "mouse_id",
    "group",
    "dapi",
    "nerve",
    "whole_skin_cleanup",
)


def validate_sample_location(group, sample_id):
    """Validate names before mapping them to output directories on any platform."""
    group = group or "."
    if group != ".":
        parts = group.split("/")
        for part in parts:
            _safe_name(part)
        if PurePosixPath(group).is_absolute():
            raise ValueError("Group must be a relative folder path.")
    _safe_name(sample_id)
    return group


def _safe_name(value):
    if not value or value in {".", ".."} or any(c in value for c in "/\\:\x00<>|?*"):
        raise ValueError(f"Unsafe sample/group name: {value!r}")
    if value != value.strip() or value.endswith("."):
        raise ValueError(
            f"Sample/group names cannot end in dots or whitespace: {value!r}"
        )


def load_samples(path, input_root):
    """Paths are relative to input_root; the manifest selects the exact batch."""
    input_root = Path(input_root).resolve()
    samples, locations = [], set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(MANIFEST_COLUMNS)
            or set(reader.fieldnames) != set(MANIFEST_COLUMNS)
        ):
            raise ValueError(f"Manifest header must be: {','.join(MANIFEST_COLUMNS)}")
        for number, row in enumerate(reader, start=2):
            if None in row or any(
                value is None or not value.strip() for value in row.values()
            ):
                raise ValueError(
                    f"Manifest row {number} must fill every column (use . for the root group)."
                )
            group = validate_sample_location(row["group"], row["sample_id"])
            location = (group.casefold(), row["sample_id"].casefold())
            if location in locations:
                raise ValueError(
                    f"Duplicate manifest sample at row {number}: {location}"
                )
            locations.add(location)
            if row["whole_skin_cleanup"] not in {"on", "off"}:
                raise ValueError(
                    f"Manifest row {number} cleanup must be on or off, never legacy."
                )
            paths = {}
            for channel in ("dapi", "nerve"):
                relative = PurePosixPath(row[channel])
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in row[channel]
                    or ":" in row[channel]
                ):
                    raise ValueError(
                        f"Manifest {channel} paths must be relative to the input root."
                    )
                resolved = (input_root / relative).resolve()
                if not resolved.is_relative_to(input_root) or not resolved.is_file():
                    raise ValueError(
                        f"Manifest row {number}: {channel} is not a file inside the input root."
                    )
                if resolved.suffix.lower() not in {".tif", ".tiff"}:
                    raise ValueError("Manifest inputs must be TIFF files.")
                paths[channel] = resolved
            if paths["dapi"] == paths["nerve"]:
                raise ValueError("DAPI and nerve inputs must be distinct files.")
            samples.append(
                {
                    "sample_name": row["sample_id"],
                    "mouse_id": row["mouse_id"],
                    "group_path": group,
                    "dapi_path": paths["dapi"],
                    "nerve_signal_path": paths["nerve"],
                    "nerve_input_convention": "legacy_BT3"
                    if paths["nerve"].stem.lower().endswith("_bt3")
                    else "nerve",
                    "apply_mouse_whole_skin_cleanup": row["whole_skin_cleanup"] == "on",
                }
            )
    if not samples:
        raise ValueError("The sample manifest contains no sections.")
    return samples


def write_resolved_samples(path, samples, input_root):
    """Write a reusable manifest outside the read-only input tree."""
    root = Path(input_root).resolve()
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_name"],
                    "mouse_id": sample["mouse_id"],
                    "group": sample["group_path"],
                    "dapi": Path(sample["dapi_path"])
                    .resolve()
                    .relative_to(root)
                    .as_posix(),
                    "nerve": Path(sample["nerve_signal_path"])
                    .resolve()
                    .relative_to(root)
                    .as_posix(),
                    "whole_skin_cleanup": "on"
                    if sample["apply_mouse_whole_skin_cleanup"]
                    else "off",
                }
            )
