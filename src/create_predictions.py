from runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from model_utils import load_model_provenance
from data_splits import SplitData, load_dataset_splits
from utils import flatten_images

MODEL_CONFIG = {
    "logistic": {
        "model_path": PATHS.model_file("logistic"),
        "metrics_path": PATHS.model_metrics_file("logistic"),
        "score_type": "pneumonia_probability",
    },
    "svm": {
        "model_path": PATHS.model_file("svm"),
        "metrics_path": PATHS.model_metrics_file("svm"),
        "score_type": "decision_function",
    },
    "xgboost": {
        "model_path": PATHS.model_file("xgboost"),
        "metrics_path": PATHS.model_metrics_file("xgboost"),
        "score_type": "pneumonia_probability",
    },
}

LABEL_NAMES = {
    0: "normal",
    1: "pneumonia",
}


def parse_model_names(value: str) -> list[str]:
    """Parse a comma-delimited model list or 'all'."""

    value = value.strip().lower()

    if value == "all":
        return list(MODEL_CONFIG)

    names = []

    for raw_name in value.split(","):
        name = raw_name.strip().lower()

        if not name:
            continue

        if name not in MODEL_CONFIG:
            raise ValueError(
                f"Unknown model '{name}'. "
                f"Choose from {sorted(MODEL_CONFIG)} or 'all'."
            )

        if name not in names:
            names.append(name)

    if not names:
        raise ValueError("No models selected")

    return names


def load_split(
    dataset_path: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray, SplitData]:
    """
    Load and flatten one split of the shared deduplicated partition.

    Also returns the split's provenance so rows can be joined back to
    DuckDB by their source (official split, index) coordinates.
    """

    data = load_dataset_splits(dataset_path).get(split)

    return (
        flatten_images(data.images),
        data.labels,
        data,
    )


def load_metadata(
    database_path: Path,
    split: str,
    split_data: SplitData,
) -> pd.DataFrame:
    """
    Retrieve image IDs for the records in one assigned split.

    Rows are returned in the same order as ``split_data``. ``split`` is
    the assigned (deduplicated) split; ``source_split`` and
    ``source_index`` locate the image in the official NPZ and DuckDB.
    """

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    with duckdb.connect(
        str(database_path),
        read_only=True,
        config={
            "threads": 1,
        },
    ) as connection:
        database = connection.execute("""
            SELECT
                CAST(image_id AS VARCHAR) AS image_id,
                split AS source_split,
                split_index AS source_index,
                label,
                image_hash
            FROM main.image_metadata
            """).fetchdf()

    wanted = pd.DataFrame(
        {
            "source_split": split_data.source_split.astype(str),
            "source_index": split_data.source_index,
            "expected_hash": split_data.image_hash,
        }
    )

    metadata = wanted.merge(
        database,
        on=["source_split", "source_index"],
        how="left",
        validate="one_to_one",
    )

    if metadata["image_id"].isna().any():
        missing = int(metadata["image_id"].isna().sum())
        raise RuntimeError(
            f"{missing} '{split}' images have no DuckDB metadata row. "
            "Rebuild the database from the same NPZ."
        )

    if not (metadata["image_hash"] == metadata["expected_hash"]).all():
        raise RuntimeError(
            "DuckDB image hashes do not match the NPZ. "
            "Rebuild the database from the same NPZ."
        )

    metadata.insert(1, "split", split)

    return metadata.drop(columns=["expected_hash", "image_hash"])


def load_threshold(
    metrics_path: Path,
) -> float:
    """Load the validation-selected threshold saved by training."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)

    if metrics.empty or "threshold" not in metrics.columns:
        raise ValueError(f"No threshold found in {metrics_path}")

    return float(metrics.iloc[0]["threshold"])


def prediction_scores(
    model,
    model_name: str,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return the continuous score used by that model's saved threshold.
    """

    if model_name == "svm":
        return np.asarray(model.decision_function(X)).reshape(-1)

    return np.asarray(model.predict_proba(X)[:, 1]).reshape(-1)


def outcome_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return TP, TN, FP, or FN for each record."""

    return np.select(
        [
            (y_true == 1) & (y_pred == 1),
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="",
    )


def create_prediction_table(
    model_name: str,
    X: np.ndarray,
    y_true: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate one row per image for one fitted model.
    """

    config = MODEL_CONFIG[model_name]

    model_path = config["model_path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    provenance = load_model_provenance(model_path)

    threshold = load_threshold(config["metrics_path"])

    scores = prediction_scores(
        model=model,
        model_name=model_name,
        X=X,
    )

    predictions = (scores >= threshold).astype(int)

    if len(scores) != len(metadata):
        raise RuntimeError(
            f"{model_name}: prediction count does not " "match metadata count."
        )

    if not np.array_equal(
        metadata["label"].to_numpy(),
        y_true,
    ):
        raise RuntimeError("Labels in DuckDB do not match labels in NPZ.")

    result = metadata.copy()

    result["true_class"] = [LABEL_NAMES[int(label)] for label in y_true]

    result["model"] = model_name

    result["prediction_score"] = scores

    result["score_type"] = config["score_type"]

    result["threshold"] = threshold

    result["predicted_label"] = predictions

    result["predicted_class"] = [LABEL_NAMES[int(label)] for label in predictions]

    result["model_file"] = str(model_path)

    result["mlflow_experiment_name"] = provenance.get("mlflow_experiment_name")

    result["mlflow_experiment_id"] = provenance.get("mlflow_experiment_id")

    result["mlflow_run_id"] = provenance.get("mlflow_run_id")

    result["mlflow_run_name"] = provenance.get("mlflow_run_name")

    result["mlflow_tracking_uri"] = provenance.get("mlflow_tracking_uri")

    result["outcome"] = outcome_labels(
        y_true,
        predictions,
    )

    result["correct"] = y_true == predictions

    return result[
        [
            "image_id",
            "split",
            "source_split",
            "source_index",
            "label",
            "true_class",
            "model",
            "prediction_score",
            "score_type",
            "threshold",
            "predicted_label",
            "predicted_class",
            "model_file",
            "mlflow_experiment_name",
            "mlflow_experiment_id",
            "mlflow_run_id",
            "mlflow_run_name",
            "mlflow_tracking_uri",
            "outcome",
            "correct",
        ]
    ]


def run_predictions(
    database_path: Path,
    dataset_path: Path,
    output_dir: Path,
    split: str,
    model_names: list[str],
) -> None:
    """
    Generate one prediction CSV per requested model.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y_true, split_data = load_split(
        dataset_path=dataset_path,
        split=split,
    )

    metadata = load_metadata(
        database_path=database_path,
        split=split,
        split_data=split_data,
    )

    generated = 0

    for model_name in model_names:
        model_path = MODEL_CONFIG[model_name]["model_path"]

        if not model_path.exists():
            logger.warning(
                "Skipping {} because {} does not exist",
                model_name,
                model_path,
            )
            continue

        logger.info(
            "Generating {} predictions for {} images",
            model_name,
            len(y_true),
        )

        predictions = create_prediction_table(
            model_name=model_name,
            X=X,
            y_true=y_true,
            metadata=metadata,
        )

        output_path = output_dir / f"{model_name}_predictions_{split}.csv"

        predictions.to_csv(
            output_path,
            index=False,
        )

        generated += 1

        logger.success(
            "Saved {} predictions to {}",
            len(predictions),
            output_path,
        )

        logger.info(
            "{} outcome counts:\n{}",
            model_name,
            predictions["outcome"].value_counts().to_string(),
        )

    if generated == 0:
        raise RuntimeError(
            "No prediction CSVs were generated. "
            "Train at least one requested model first."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-image prediction CSVs for fitted " "PneumoniaMNIST models."
        )
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
        "--output-dir",
        type=Path,
        default=PATHS.predictions_dir,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "test",
        ],
        default="test",
        help="Deduplicated split to score (default: test)",
    )

    parser.add_argument(
        "--models",
        default="all",
        help=("Comma-delimited models or 'all'. " "Choices: logistic, svm, xgboost."),
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
        default=PATHS.log_file("create_predictions"),
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
        run_predictions(
            database_path=args.database,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            split=args.split,
            model_names=parse_model_names(args.models),
        )

    except Exception:
        logger.exception("Prediction export failed")
        raise
