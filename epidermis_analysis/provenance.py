"""Content fingerprints and atomic metadata for reproducible analysis runs."""

from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile


ILASTIK_ENVIRONMENT = {
    # Applied only to the Ilastik subprocess, not the parent Python process.
    # Serial workers reduce parallel scheduling variation; this is not a
    # guarantee of identical results across installations or platforms.
    # The lazyflow RAM setting is a budget, not a process-wide memory limit.
    "LAZYFLOW_THREADS": "1",
    "LAZYFLOW_TOTAL_RAM_MB": "8192",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Replace metadata only after its complete contents have been written."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".metadata-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def segmentation_identity(dapi, project, launcher, export_source):
    return {
        "schema": 2,
        "dapi_sha256": file_sha256(dapi),
        "project_sha256": file_sha256(project),
        "launcher_sha256": file_sha256(launcher),
        "launcher_path": str(Path(launcher).resolve()),
        "export_source": export_source,
        "input_axes": "yx",
        "environment": ILASTIK_ENVIRONMENT.copy(),
    }


def cached_segmentation_matches(output, identity):
    try:
        stored = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
        return stored["identity"] == identity and stored[
            "output_sha256"
        ] == file_sha256(output)
    except (OSError, ValueError, KeyError, TypeError):
        return False


RESULT_FILES = (
    "BT3_quantification_results.csv",
    "subbasal_BT3_quantification_results.csv",
    "four_compartment_BT3_quantification_results.csv",
    "analysis_parameters.json",
)


def completed_sample_matches(folder, identity):
    try:
        stored = json.loads((folder / "completion.json").read_text(encoding="utf-8"))
        return stored["identity"] == identity and all(
            stored["files"][name] == file_sha256(folder / name) for name in RESULT_FILES
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def mark_sample_complete(folder, identity):
    write_json(
        folder / "completion.json",
        {
            "identity": identity,
            "files": {name: file_sha256(folder / name) for name in RESULT_FILES},
        },
    )


def runtime_provenance(root):
    root = Path(root)
    sources = [
        root / "Skin_Section_Analysis.py",
        root / "requirements.txt",
        root / "environment.yml",
        *sorted((root / "epidermis_analysis").glob("*.py")),
    ]
    packages = [
        line.split("==", 1)[0]
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in packages},
        "source_sha256": {
            str(path.relative_to(root)): file_sha256(path) for path in sources
        },
    }


def validate_output_location(output, *input_roots):
    """Reject overlapping roots and linked output files before batch writes.

    This preflight assumes no concurrent changes to the input/output trees.
    """
    output = Path(output)
    resolved = output.resolve()
    for root in input_roots:
        root = Path(root).resolve()
        if (
            resolved == root
            or resolved.is_relative_to(root)
            or root.is_relative_to(resolved)
        ):
            raise ValueError(
                "Input and output directories must be separate and non-overlapping."
            )
    # Either link type can make an output write modify a file outside the root.
    if output.exists():
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("Output directory must not contain symbolic links.")
            if path.is_file() and path.stat().st_nlink > 1:
                raise ValueError("Output directory must not contain hard-linked files.")
