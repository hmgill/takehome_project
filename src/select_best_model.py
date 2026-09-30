from runtime import configure_runtime

configure_runtime()

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from model_utils import evaluate_scores
from data_splits import load_dataset_splits
from utils import flatten_images

MODEL_CONFIG = {
    "logistic": {
        "model_path": PATHS.model_file("logistic"),
        "metrics_path": PATHS.model_metrics_file("logistic"),
    },
    "svm": {
        "model_path": PATHS.model_file("svm"),
        "metrics_path": PATHS.model_metrics_file("svm"),
    },
    "xgboost": {
        "model_path": PATHS.model_file("xgboost"),
        "metrics_path": PATHS.model_metrics_file("xgboost"),
    },
}


def load_threshold(
    metrics_path: Path,
) -> float:
    """Read the validation-selected threshold for a fitted model."""

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)

    if metrics.empty or "threshold" not in metrics.columns:
        raise ValueError(f"No threshold found in {metrics_path}")

    return float(metrics.iloc[0]["threshold"])


def load_training_cv_auc(
    metrics_path: Path,
) -> float:
    """
    Read training-only CV ROC AUC when the model used hyperparameter
    search. The required logistic baseline does not have this value.
    """

    if not metrics_path.exists():
        return float("nan")

    metrics = pd.read_csv(metrics_path)

    if metrics.empty or "training_cv_roc_auc" not in metrics.columns:
        return float("nan")

    return float(metrics.iloc[0]["training_cv_roc_auc"])


def prediction_scores(
    model,
    model_name: str,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return continuous validation scores.

    ROC AUC does not require all models to expose calibrated
    probabilities, so SVM decision-function scores are valid here.
    """

    if model_name == "svm":
        return np.asarray(model.decision_function(X)).reshape(-1)

    return np.asarray(model.predict_proba(X)[:, 1]).reshape(-1)


def evaluate_candidate(
    model_name: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Evaluate one already-fitted candidate on the held-out validation
    split.

    Model selection uses validation ROC AUC. Threshold-dependent
    validation metrics are reported only as supporting information.
    """

    config = MODEL_CONFIG[model_name]

    model_path = config["model_path"]

    metrics_path = config["metrics_path"]

    model = joblib.load(model_path)

    threshold = load_threshold(metrics_path)

    scores = prediction_scores(
        model=model,
        model_name=model_name,
        X=X_val,
    )

    validation_auc = float(
        roc_auc_score(
            y_val,
            scores,
        )
    )

    threshold_metrics, _ = evaluate_scores(
        y_true=y_val,
        scores=scores,
        threshold=threshold,
    )

    return {
        "model": model_name,
        "model_path": str(model_path),
        "threshold": threshold,
        "training_cv_roc_auc": load_training_cv_auc(metrics_path),
        "validation_roc_auc": validation_auc,
        "validation_accuracy": threshold_metrics["accuracy"],
        "validation_balanced_accuracy": threshold_metrics["balanced_accuracy"],
        "validation_sensitivity": threshold_metrics["sensitivity"],
        "validation_specificity": threshold_metrics["specificity"],
    }


def run_selection(
    dataset_path: Path,
    output_dir: Path,
) -> None:
    """
    Compare all available fitted candidate models.

    Hyperparameter tuning may use training-fold cross-validation, but
    final candidate selection is based on held-out validation ROC AUC.
    The test split is deliberately not read by this script.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same deduplicated partition the models were trained on.
    splits = load_dataset_splits(dataset_path)

    X_val = flatten_images(splits.val.images)
    y_val = splits.val.labels

    candidates = []

    for model_name, config in MODEL_CONFIG.items():
        if not config["model_path"].exists():
            logger.warning(
                "Skipping {} because {} does not exist",
                model_name,
                config["model_path"],
            )
            continue

        logger.info(
            "Evaluating {} on the validation split",
            model_name,
        )

        candidates.append(
            evaluate_candidate(
                model_name=model_name,
                X_val=X_val,
                y_val=y_val,
            )
        )

    if not candidates:
        raise RuntimeError("No fitted candidate models were found.")

    comparison = (
        pd.DataFrame(candidates)
        .sort_values(
            [
                "validation_roc_auc",
                "model",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    comparison.insert(
        0,
        "rank",
        np.arange(
            1,
            len(comparison) + 1,
        ),
    )

    comparison["selected"] = comparison["rank"] == 1

    comparison_path = output_dir / "model_selection.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    selected = comparison.iloc[0]

    manifest = {
        "selected_model": selected["model"],
        "model_path": selected["model_path"],
        "selection_metric": "validation_roc_auc",
        "selection_score": float(selected["validation_roc_auc"]),
        "decision_threshold": float(selected["threshold"]),
        "selection_rule": (
            "Highest ROC AUC on the held-out validation split. "
            "Training-fold cross-validation is used for hyperparameter "
            "tuning where applicable; test performance is not used for "
            "model selection."
        ),
    }

    manifest_path = output_dir / "selected_model.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Model comparison:\n{}",
        comparison.to_string(index=False),
    )

    logger.success(
        "Selected model: {} " "(validation ROC AUC = {:.4f})",
        manifest["selected_model"],
        manifest["selection_score"],
    )

    logger.success(
        "Saved comparison to {}",
        comparison_path,
    )

    logger.success(
        "Saved selection manifest to {}",
        manifest_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best available PneumoniaMNIST candidate "
            "using held-out validation ROC AUC."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=PATHS.dataset,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.model_selection_dir,
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
        default=PATHS.log_file("select_best_model"),
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
        run_selection(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
        )

    except Exception:
        logger.exception("Model selection failed")
        raise
