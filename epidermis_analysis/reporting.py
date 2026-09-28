"""Batch tables and workbook exports, independent of segmentation execution."""

import json
from pathlib import Path
import re

import pandas as pd

from .measurement import LEGACY_BT3_METRIC_NAMES
from .provenance import completed_sample_matches
from .samples import validate_sample_location


def rebuild_reports(output_root, *, excel=True):
    """Re-export the last batch's verified tables without inputs, models, or inference.

    Verification checks saved tables against their original completion record;
    it intentionally does not require the current analysis source to match.
    """
    output_root = Path(output_root)
    log = pd.read_csv(
        output_root / "batch_run_log.csv",
        keep_default_na=False,
        converters={"Group": str, "Sample": str},
    )
    if not {"Group", "Sample", "Status"}.issubset(log.columns):
        raise ValueError("Batch log is missing Group, Sample, or Status.")
    if log.duplicated(["Group", "Sample"]).any():
        raise ValueError("Batch log contains duplicate sections.")
    records = []
    for row in log.to_dict("records"):
        if row["Status"] != "Completed" and not row["Status"].startswith("Skipped"):
            continue
        group = validate_sample_location(row["Group"], row["Sample"])
        folder = group_output_folder(output_root, group) / row["Sample"]
        try:
            stored = json.loads(
                (folder / "completion.json").read_text(encoding="utf-8")
            )
            identity = stored["identity"]
            valid = (
                identity["group"] == group
                and identity["sample"] == row["Sample"]
                and completed_sample_matches(folder, identity)
            )
        except (OSError, ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            raise ValueError(
                f"Cannot report unverified results: {group}/{row['Sample']}"
            )
        table = pd.read_csv(
            folder / "BT3_quantification_results.csv",
            float_precision="round_trip",
            converters={"Sample": str, "sample_id": str, "mouse_id": str},
        )
        if len(table) != 1 or table.iloc[0]["Sample"] != row["Sample"]:
            raise ValueError(
                f"Section table identity mismatch: {group}/{row['Sample']}"
            )
        table["Group"] = group
        if "mouse_id" in identity:
            table["mouse_id"] = identity["mouse_id"]
        records.extend(table.to_dict("records"))
    # Validate the entire batch before replacing any existing report.
    write_batch_reports(records, output_root, excel=excel)
    return len(records)


def biological_replicate_from_sample(sample_name):
    """Collapse case-insensitive section suffixes to one biological replicate."""
    sample_name = str(sample_name)
    replicate = re.sub(
        r"(?i)[_-]section[-_ ]*\d.*$",
        "",
        sample_name,
    ).rstrip("_- ")
    return replicate or sample_name


def biological_replicates(table):
    inferred = table["Sample"].map(biological_replicate_from_sample)
    if "mouse_id" not in table:
        return inferred
    explicit = table["mouse_id"].replace("", pd.NA)
    return explicit.fillna(inferred)


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
    """Compute two unweighted section means per explicit (or inferred) animal and group.

    Means omit NaN values independently for each metric. The section count
    includes every row, including rows with an undefined metric.
    """
    table = normalize_legacy_results(section_results_df)
    table["Biological replicate"] = biological_replicates(table)
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
    section_table["Biological replicate"] = biological_replicates(section_table)
    replicate_table = build_biological_replicate_averages(section_table)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        section_table.to_excel(writer, sheet_name="Section Results", index=False)
        replicate_table.to_excel(
            writer, sheet_name="Biological Replicates", index=False
        )


def group_output_folder(output_root, group_path):
    """Return the output directory that mirrors one relative input group."""
    return (
        Path(output_root)
        if str(group_path) in {"", "."}
        else Path(output_root) / Path(str(group_path))
    )


def write_batch_reports(all_results, output_root, *, excel=True):
    """Export the supplied section records; do not discover or process images."""
    output_root = Path(output_root)
    for name in (
        "combined_BT3_quantification_results.csv",
        "BT3_quantification_by_biological_replicate.xlsx",
    ):
        for path in output_root.rglob(name):
            path.unlink()
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
        if "mouse_id" in combined_results_df:
            normalized_columns.insert(3, "mouse_id")
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
        if excel:
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
            if excel:
                write_quantification_excel(
                    group_section_results,
                    group_folder / "BT3_quantification_by_biological_replicate.xlsx",
                )
