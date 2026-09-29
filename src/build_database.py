import os

# ---------------------------------------------------------------------
# Constrain native threading before importing numerical libraries.
# ---------------------------------------------------------------------

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


import argparse
import json
import uuid
from pathlib import Path
from typing import Literal

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from utils import (
    calculate_image_hash,
    calculate_image_statistics,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EXPECTED_HEIGHT = CONFIG.qc.expected_height
EXPECTED_WIDTH = CONFIG.qc.expected_width
ALLOWED_LABELS = {0, 1}
LOW_VARIANCE_QUANTILE = CONFIG.qc.low_variance_quantile

DATASET_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "medmnist:pneumoniamnist",
)

SplitName = Literal[
    "train",
    "val",
    "test",
]


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------


class ImageMetadata(BaseModel):
    """
    Metadata derived independently from one source image.

    Dataset-level QC fields are added after all records have been
    assembled.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    image_id: uuid.UUID
    split: SplitName
    split_index: int = Field(ge=0)

    label: int = Field(
        ge=0,
        le=1,
    )

    width: int = Field(gt=0)
    height: int = Field(gt=0)

    mean_intensity: float
    std_intensity: float = Field(ge=0)
    min_intensity: float
    max_intensity: float

    image_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )


# ---------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------


def inspect_npz(
    data: np.lib.npyio.NpzFile,
) -> None:
    """
    Inspect the NPZ file before making assumptions about its contents.
    """

    logger.info("Inspecting NPZ contents")

    for key in data.files:
        array = data[key]

        logger.info(
            "{}: shape={}, dtype={}",
            key,
            array.shape,
            array.dtype,
        )


def validate_expected_keys(
    data: np.lib.npyio.NpzFile,
) -> None:
    """
    Verify that the expected dataset arrays exist.
    """

    required_keys = {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }

    missing = required_keys - set(data.files)

    if missing:
        raise ValueError("NPZ file is missing required arrays: " f"{sorted(missing)}")

    logger.success("All required NPZ arrays are present")


# ---------------------------------------------------------------------
# Image ID
# ---------------------------------------------------------------------


def create_image_id(
    split: str,
    split_index: int,
    image_hash: str,
) -> uuid.UUID:
    """
    Create a deterministic UUID5 for one dataset record.
    """

    identity = f"{split}:" f"{split_index}:" f"{image_hash}"

    return uuid.uuid5(
        DATASET_NAMESPACE,
        identity,
    )


# ---------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------


def extract_split_metadata(
    images: np.ndarray,
    labels: np.ndarray,
    split: SplitName,
) -> list[ImageMetadata]:
    """
    Extract image-level metadata for one dataset split.
    """

    labels = np.asarray(labels).reshape(-1)

    if len(images) != len(labels):
        raise ValueError(
            f"{split}: image count ({len(images)}) "
            f"does not match label count ({len(labels)})"
        )

    logger.info(
        "Processing {} images from '{}' split",
        len(images),
        split,
    )

    records: list[ImageMetadata] = []

    for index, (
        image,
        raw_label,
    ) in enumerate(zip(images, labels)):
        label = int(raw_label)

        image_hash = calculate_image_hash(image)

        statistics = calculate_image_statistics(image)

        record = ImageMetadata(
            image_id=create_image_id(
                split=split,
                split_index=index,
                image_hash=image_hash,
            ),
            split=split,
            split_index=index,
            label=label,
            image_hash=image_hash,
            **statistics,
        )

        records.append(record)

    logger.success(
        "Finished '{}' split: {} records",
        split,
        len(records),
    )

    return records


# ---------------------------------------------------------------------
# Dataset-level QC
# ---------------------------------------------------------------------


def add_qc_fields(
    metadata: pd.DataFrame,
    low_variance_quantile: float = LOW_VARIANCE_QUANTILE,
) -> pd.DataFrame:
    """
    Add dataset-level quality-control fields.

    These are informational flags. Images are not removed.
    """

    metadata = metadata.copy()

    low_variance_threshold = metadata["std_intensity"].quantile(low_variance_quantile)

    metadata["low_variance_flag"] = metadata["std_intensity"] <= low_variance_threshold

    metadata["duplicate_count"] = (
        metadata.groupby("image_hash")["image_hash"].transform("size").astype(int)
    )

    metadata["exact_duplicate_flag"] = metadata["duplicate_count"] > 1

    # Store the image IDs of the complete exact-duplicate group as a JSON
    # array string. JSON is unambiguous in both DuckDB and CSV and
    # avoids relying on an arbitrary delimiter.
    duplicate_ids_by_hash = (
        metadata.groupby(
            "image_hash",
            sort=False,
        )["image_id"]
        .agg(list)
        .to_dict()
    )

    duplicate_image_ids = []

    for image_hash in metadata["image_hash"]:
        group_ids = sorted(
            str(image_id) for image_id in duplicate_ids_by_hash[image_hash]
        )

        if len(group_ids) < 2:
            group_ids = []

        duplicate_image_ids.append(
            json.dumps(
                group_ids,
                separators=(",", ":"),
            )
        )

    metadata["duplicate_image_ids"] = duplicate_image_ids

    split_count_by_hash = metadata.groupby("image_hash")["split"].transform("nunique")

    metadata["cross_split_duplicate_flag"] = split_count_by_hash > 1

    logger.info(
        "Low-variance threshold: " "std_intensity <= {:.4f} " "(bottom {:.1%})",
        low_variance_threshold,
        low_variance_quantile,
    )

    return metadata


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def validate_metadata(
    metadata: pd.DataFrame,
) -> None:
    """
    Run the required Part 1 validation checks.
    """

    logger.info("Running dataset validation checks")

    split_counts = (
        metadata.groupby("split")
        .size()
        .reindex(
            [
                "train",
                "val",
                "test",
            ]
        )
    )

    logger.info("Image counts by split:")

    for split, count in split_counts.items():
        logger.info(
            "  {}: {}",
            split,
            count,
        )

    observed_labels = set(metadata["label"].unique())

    if not observed_labels.issubset(ALLOWED_LABELS):
        raise ValueError("Unexpected labels found: " f"{sorted(observed_labels)}")

    logger.success(
        "Label validation passed: {}",
        sorted(observed_labels),
    )

    dimensions = set(
        zip(
            metadata["height"],
            metadata["width"],
        )
    )

    expected_dimensions = {
        (
            EXPECTED_HEIGHT,
            EXPECTED_WIDTH,
        )
    }

    if dimensions != expected_dimensions:
        raise ValueError("Unexpected image dimensions found: " f"{sorted(dimensions)}")

    logger.success(
        "Dimension validation passed: " "all images are {}x{}",
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
    )

    missing_count = int(metadata.isna().sum().sum())

    numeric_columns = [
        "split_index",
        "label",
        "width",
        "height",
        "mean_intensity",
        "std_intensity",
        "min_intensity",
        "max_intensity",
        "duplicate_count",
    ]

    numeric_values = metadata[numeric_columns].to_numpy(dtype=float)

    if missing_count > 0:
        raise ValueError("Metadata contains " f"{missing_count} missing values")

    if not np.isfinite(numeric_values).all():
        raise ValueError("Metadata contains non-finite numeric values")

    logger.success("Missing/invalid value validation passed")

    # Validate duplicate-group image IDs.
    for row in metadata.itertuples(index=False):
        try:
            peer_ids = json.loads(row.duplicate_image_ids)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Invalid duplicate_image_ids JSON for " f"image_id={row.image_id}"
            ) from exc

        if not isinstance(
            peer_ids,
            list,
        ):
            raise ValueError(
                "duplicate_image_ids must decode to a list "
                f"for image_id={row.image_id}"
            )

        duplicate_count = int(row.duplicate_count)

        if duplicate_count < 2:
            if peer_ids:
                raise ValueError(
                    "duplicate_image_ids must be an empty list "
                    "when no exact duplicate exists for "
                    f"image_id={row.image_id}"
                )

        else:
            if str(row.image_id) not in peer_ids:
                raise ValueError(
                    "duplicate_image_ids must include the "
                    f"current image_id when duplicates exist: "
                    f"{row.image_id}"
                )

            if len(peer_ids) != duplicate_count:
                raise ValueError(
                    "duplicate_image_ids count does not match "
                    "duplicate_count for "
                    f"image_id={row.image_id}: "
                    f"expected {duplicate_count}, "
                    f"found {len(peer_ids)}"
                )

            if len(peer_ids) < 2:
                raise ValueError(
                    "duplicate_image_ids must contain at least "
                    "two IDs when duplicates exist for "
                    f"image_id={row.image_id}"
                )

        if len(peer_ids) != len(set(peer_ids)):
            raise ValueError(
                "duplicate_image_ids contains repeated IDs "
                f"for image_id={row.image_id}"
            )

    logger.success("Duplicate peer-ID validation passed")

    low_variance_rows = metadata[metadata["low_variance_flag"]]

    logger.warning(
        "{} images flagged as unusually low variance",
        len(low_variance_rows),
    )

    if not low_variance_rows.empty:
        sample = (
            low_variance_rows[
                [
                    "image_id",
                    "split",
                    "split_index",
                    "label",
                    "mean_intensity",
                    "std_intensity",
                ]
            ]
            .sort_values("std_intensity")
            .head(5)
        )

        logger.info(
            "Lowest-variance sample:\n{}",
            sample.to_string(index=False),
        )

    duplicate_rows = metadata[metadata["exact_duplicate_flag"]]

    duplicate_group_count = duplicate_rows["image_hash"].nunique()

    logger.info(
        "Exact duplicate validation: "
        "{} duplicate groups, "
        "{} image records involved",
        duplicate_group_count,
        len(duplicate_rows),
    )

    cross_split_rows = metadata[metadata["cross_split_duplicate_flag"]]

    cross_split_group_count = cross_split_rows["image_hash"].nunique()

    if cross_split_group_count:
        logger.warning(
            "{} exact duplicate image groups occur "
            "across different dataset splits "
            "({} records involved)",
            cross_split_group_count,
            len(cross_split_rows),
        )

    else:
        logger.success("No exact duplicates found " "across train/val/test splits")


# ---------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------


def write_outputs(
    metadata: pd.DataFrame,
    database_path: Path,
    csv_path: Path,
) -> None:
    """
    Write canonical metadata to DuckDB, then export the persisted table
    to CSV.

    Exporting the CSV from DuckDB ensures that both outputs represent
    the same persisted table contents.
    """

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Writing metadata to DuckDB: {}",
        database_path,
    )

    with duckdb.connect(
        str(database_path),
        config={
            "threads": 1,
        },
    ) as connection:

        logger.info("DuckDB configured with 1 thread")

        connection.register(
            "metadata_df",
            metadata,
        )

        connection.execute("""
            CREATE OR REPLACE TABLE main.image_metadata AS
            SELECT *
            FROM metadata_df
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_hash
            ON main.image_metadata(image_hash)
            """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_split_label
            ON main.image_metadata(split, label)
            """)

        persisted_metadata = connection.execute("""
                SELECT *
                FROM main.image_metadata
                ORDER BY
                    CASE split
                        WHEN 'train' THEN 1
                        WHEN 'val' THEN 2
                        WHEN 'test' THEN 3
                    END,
                    split_index
                """).fetchdf()

    persisted_metadata.to_csv(
        csv_path,
        index=False,
    )

    logger.success(
        "Stored {} records in {}",
        len(persisted_metadata),
        database_path,
    )

    logger.success(
        "Exported {} records to {}",
        len(persisted_metadata),
        csv_path,
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def build_database(
    input_path: Path,
    database_path: Path,
    csv_path: Path,
) -> None:
    """
    Inspect, extract, validate, and store PneumoniaMNIST metadata.
    """

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    logger.info(
        "Loading PneumoniaMNIST dataset from {}",
        input_path,
    )

    with np.load(
        input_path,
        allow_pickle=False,
    ) as data:

        inspect_npz(data)

        validate_expected_keys(data)

        records: list[ImageMetadata] = []

        split_config: list[
            tuple[
                SplitName,
                str,
                str,
            ]
        ] = [
            (
                "train",
                "train_images",
                "train_labels",
            ),
            (
                "val",
                "val_images",
                "val_labels",
            ),
            (
                "test",
                "test_images",
                "test_labels",
            ),
        ]

        for (
            split,
            image_key,
            label_key,
        ) in split_config:

            records.extend(
                extract_split_metadata(
                    images=data[image_key],
                    labels=data[label_key],
                    split=split,
                )
            )

    logger.info(
        "Extracted metadata for {} total images",
        len(records),
    )

    metadata = pd.DataFrame([record.model_dump(mode="json") for record in records])

    metadata = add_qc_fields(metadata)

    validate_metadata(metadata)

    write_outputs(
        metadata=metadata,
        database_path=database_path,
        csv_path=csv_path,
    )

    logger.success("Part 1 pipeline completed successfully")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate PneumoniaMNIST metadata "
            "and store it in DuckDB and CSV."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PATHS.dataset,
        help="Path to pneumoniamnist.npz",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PATHS.database,
        help="Output DuckDB path",
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PATHS.metadata_csv,
        help=(
            "CSV export of main.image_metadata "
            "(default: data/pneumoniamnist_metadata.csv)"
        ),
    )

    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=[
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        default="SUCCESS",
        help=("Console logging verbosity " "(default: SUCCESS)"),
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=PATHS.log_file("build_database"),
        help=("Detailed log file " "(default: output/logs/build_database.log)"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ensure_output_directories(CONFIG)

    configure_logging(
        console_level=args.log_level,
        log_file=args.log_file,
    )

    try:
        build_database(
            input_path=args.input,
            database_path=args.output,
            csv_path=args.csv_output,
        )

    except Exception:
        logger.exception("Dataset pipeline failed")
        raise
