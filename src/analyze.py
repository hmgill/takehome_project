from runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from utils import get_image_from_npz

REQUIRED_ANALYSIS_VIEWS = {
    "class_balance",
    "image_characteristics",
    "image_characteristics_by_split",
    "qc_summary",
    "cross_split_duplicates",
    "duplicate_label_conflicts",
    "unusual_images",
}


def validate_source_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    result = connection.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = 'image_metadata'
        """).fetchone()

    if result is None or result[0] != 1:
        raise RuntimeError(
            "Required table 'main.image_metadata' does not exist. "
            "Run src/build_database.py first."
        )

    logger.success("Found source table main.image_metadata")


def validate_analysis_views(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    rows = connection.execute("""
        SELECT table_name
        FROM information_schema.views
        WHERE table_schema = 'analysis'
        ORDER BY table_name
        """).fetchall()

    actual_views = {row[0] for row in rows}

    missing = REQUIRED_ANALYSIS_VIEWS - actual_views

    if missing:
        raise RuntimeError(
            "analysis.sql did not create all expected views. "
            f"Missing: {sorted(missing)}"
        )

    logger.success(
        "Verified {} analysis views",
        len(REQUIRED_ANALYSIS_VIEWS),
    )


def create_analysis_views(
    connection: duckdb.DuckDBPyConnection,
    sql_file: Path,
) -> None:
    if not sql_file.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    connection.execute(sql_file.read_text(encoding="utf-8"))

    validate_analysis_views(connection)

    logger.success("SQL analysis views created successfully")


def query_dataframe(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> pd.DataFrame:
    return connection.execute(query).fetchdf()


def export_analysis_tables(
    connection: duckdb.DuckDBPyConnection,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    queries = {
        "class_balance": """
            SELECT *
            FROM analysis.class_balance
            ORDER BY
                CASE split
                    WHEN 'train' THEN 1
                    WHEN 'val' THEN 2
                    WHEN 'test' THEN 3
                END
        """,
        "image_characteristics": """
            SELECT *
            FROM analysis.image_characteristics
            ORDER BY label
        """,
        "image_characteristics_by_split": """
            SELECT *
            FROM analysis.image_characteristics_by_split
            ORDER BY
                CASE split
                    WHEN 'train' THEN 1
                    WHEN 'val' THEN 2
                    WHEN 'test' THEN 3
                END,
                label
        """,
        "qc_summary": """
            SELECT *
            FROM analysis.qc_summary
        """,
        "cross_split_duplicates": """
            SELECT *
            FROM analysis.cross_split_duplicates
            ORDER BY image_hash, split, split_index
        """,
        "duplicate_label_conflicts": """
            SELECT *
            FROM analysis.duplicate_label_conflicts
            ORDER BY image_hash
        """,
        "unusual_images": """
            SELECT *
            FROM analysis.unusual_images
            ORDER BY
                CASE criterion
                    WHEN 'darkest' THEN 1
                    WHEN 'brightest' THEN 2
                    WHEN 'lowest_contrast' THEN 3
                END,
                selection_rank
        """,
    }

    results = {}

    for name, query in queries.items():
        logger.info(
            "Running analysis query: {}",
            name,
        )

        dataframe = query_dataframe(
            connection,
            query,
        )

        output_path = output_dir / f"{name}.csv"

        dataframe.to_csv(
            output_path,
            index=False,
        )

        results[name] = dataframe

        logger.info(
            "Saved {} rows to {}",
            len(dataframe),
            output_path,
        )

    return results


def plot_unusual_images(
    unusual_images: pd.DataFrame,
    dataset_path: Path,
    output_path: Path,
) -> None:
    if unusual_images.empty:
        logger.warning("No unusual images selected; skipping figure")
        return

    criteria = [
        "darkest",
        "brightest",
        "lowest_contrast",
    ]

    criterion_titles = {
        "darkest": "Darkest",
        "brightest": "Brightest",
        "lowest_contrast": "Lowest contrast",
    }

    label_names = {
        0: "Normal",
        1: "Pneumonia",
    }

    figure, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(
            9,
            9,
        ),
    )

    with np.load(
        dataset_path,
        allow_pickle=False,
    ) as dataset:

        for row_number, criterion in enumerate(criteria):
            selected = (
                unusual_images[unusual_images["criterion"] == criterion]
                .sort_values("selection_rank")
                .head(3)
            )

            for column_number, (
                _,
                record,
            ) in enumerate(selected.iterrows()):
                image = get_image_from_npz(
                    dataset=dataset,
                    split=str(record["split"]),
                    split_index=int(record["split_index"]),
                )

                axis = axes[
                    row_number,
                    column_number,
                ]

                axis.imshow(
                    image,
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                )

                metric_name = str(record["metric_name"])

                metric_value = float(record["metric_value"])

                metric_text = (
                    f"mean={metric_value:.1f}"
                    if metric_name == "mean_intensity"
                    else f"std={metric_value:.1f}"
                )

                axis.set_title(
                    (
                        f"{criterion_titles[criterion]}\n"
                        f"{record['split']} #{int(record['split_index'])} | "
                        f"{label_names[int(record['label'])]}\n"
                        f"{metric_text}"
                    ),
                    fontsize=9,
                )

                axis.axis("off")

    figure.suptitle(
        "Quantitatively Unusual PneumoniaMNIST Images",
        fontsize=14,
    )

    figure.text(
        0.5,
        0.01,
        (
            "Selected using pixel statistics only; these are QC "
            "examples, not assessments of medical abnormality."
        ),
        ha="center",
        fontsize=9,
    )

    figure.tight_layout(
        rect=(
            0,
            0.04,
            1,
            0.96,
        )
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    logger.success(
        "Saved unusual-image figure to {}",
        output_path,
    )


def run_analysis(
    database_path: Path,
    dataset_path: Path,
    sql_file: Path,
    output_dir: Path,
) -> None:
    if not database_path.exists():
        raise FileNotFoundError(f"DuckDB database not found: {database_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with duckdb.connect(
        str(database_path),
        config={
            "threads": CONFIG.project.threads,
        },
    ) as connection:

        validate_source_table(connection)

        create_analysis_views(
            connection,
            sql_file,
        )

        results = export_analysis_tables(
            connection,
            output_dir,
        )

    plot_unusual_images(
        unusual_images=results["unusual_images"],
        dataset_path=dataset_path,
        output_path=(output_dir / "unusual_images.png"),
    )

    logger.success("Part 2 analysis completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run PneumoniaMNIST SQL analysis and generate Part 2 outputs.")
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=PATHS.database,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PATHS.dataset,
    )

    parser.add_argument(
        "--sql",
        type=Path,
        default=PATHS.analysis_sql,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.analysis_dir,
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
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=PATHS.log_file("analyze"),
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
        run_analysis(
            database_path=args.database,
            dataset_path=args.dataset,
            sql_file=args.sql,
            output_dir=args.output_dir,
        )

    except Exception:
        logger.exception("Part 2 analysis failed")
        raise
