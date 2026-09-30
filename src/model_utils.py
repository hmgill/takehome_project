from contextlib import nullcontext
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from data_splits import load_dataset_splits, save_split_manifest
from project_config import CONFIG, PATHS
from utils import flatten_images


def load_model_data(
    dataset_path: Path,
    augment_train: bool = False,
    manifest_dir: Path | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load, deduplicate, split, and flatten the dataset for training.

    Exact duplicate images are removed from the pooled dataset *before*
    the train/val/test split (see ``data_splits``), so no image can occur
    in more than one split. The resulting split assignment is written to
    ``manifest_dir`` (default ``PATHS.splits_dir``) for auditing.

    Optional augmentation is applied only to the training split, after
    splitting. Validation and test arrays are always left unchanged.
    """

    splits = load_dataset_splits(dataset_path)

    save_split_manifest(
        splits,
        PATHS.splits_dir if manifest_dir is None else manifest_dir,
    )

    train_images, y_train = splits.train.images, splits.train.labels

    if augment_train:
        from augmentation_utils import augment_training_split

        original_count = len(train_images)

        train_images, y_train = augment_training_split(
            train_images,
            y_train,
        )

        logger.info(
            "Training augmentation expanded {} images to {}",
            original_count,
            len(train_images),
        )

    return (
        flatten_images(train_images),
        y_train,
        flatten_images(splits.val.images),
        splits.val.labels,
        flatten_images(splits.test.images),
        splits.test.labels,
    )


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float]:
    """Select a validation-set threshold using Youden's J statistic."""

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)

    fpr = fpr[finite]
    tpr = tpr[finite]
    thresholds = thresholds[finite]

    if len(thresholds) == 0:
        raise RuntimeError("No finite thresholds available")

    youden_j = tpr - fpr
    best_index = int(np.argmax(youden_j))

    return float(thresholds[best_index]), float(youden_j[best_index])


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate continuous model scores using a fixed decision threshold."""

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "sensitivity": float(recall_score(y_true, predictions, pos_label=1)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }

    return metrics, matrix


def save_search_results(search, output_path: Path) -> Path:
    """Save the useful portion of RandomizedSearchCV results."""

    results = pd.DataFrame(search.cv_results_)
    parameter_columns = [
        column for column in results.columns if column.startswith("param_")
    ]

    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        *parameter_columns,
    ]

    results[columns].sort_values("rank_test_score").to_csv(
        output_path,
        index=False,
    )

    logger.info("Saved search results to {}", output_path)
    return output_path


def save_evaluation_outputs(
    name: str,
    title: str,
    metrics: dict[str, float],
    matrix: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    """Save metrics, confusion matrix, and a confusion-matrix figure."""

    metrics_path = output_dir / f"{name}_metrics.csv"
    matrix_path = output_dir / f"{name}_confusion_matrix.csv"
    figure_path = output_dir / f"{name}_evaluation.png"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    pd.DataFrame(
        matrix,
        index=["actual_normal", "actual_pneumonia"],
        columns=["predicted_normal", "predicted_pneumonia"],
    ).to_csv(matrix_path)

    figure, axis = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Normal", "Pneumonia"],
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.success("Saved {} evaluation outputs", name)
    return [metrics_path, matrix_path, figure_path]


def save_model(model, output_path: Path) -> Path:
    """Persist a fitted model locally with joblib."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.success("Saved fitted model to {}", output_path)
    return output_path


def get_mlflow_context(
    enabled: bool,
    tracking_dir: Path,
    run_name: str,
    experiment_name: str | None = None,
):
    """Start an optional local MLflow run."""

    if not enabled:
        return nullcontext()

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow requested but not installed. "
            "Install requirements-experiments.txt."
        ) from exc

    tracking_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    mlflow.set_experiment(experiment_name or CONFIG.mlflow.experiments_experiment)

    return mlflow.start_run(run_name=run_name)


def log_mlflow_experiment(
    enabled: bool,
    parameters: dict,
    metrics: dict[str, float],
    artifacts: list[Path],
) -> None:
    """Log parameters, scalar metrics, and generated artifacts to MLflow."""

    if not enabled:
        return

    import mlflow

    mlflow.log_params({key: str(value) for key, value in parameters.items()})
    mlflow.log_metrics(
        {key: float(value) for key, value in metrics.items() if np.isfinite(value)}
    )

    for artifact in artifacts:
        mlflow.log_artifact(str(artifact))


def log_mlflow_model(
    enabled: bool,
    model,
    model_type: str,
) -> None:
    """Log the fitted model itself to the active MLflow run."""

    if not enabled:
        return

    if model_type == "xgboost":
        import mlflow.xgboost

        mlflow.xgboost.log_model(
            xgb_model=model,
            name="model",
            model_format="json",
        )

    elif model_type in {"logistic_regression", "svm"}:
        import mlflow.sklearn

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logger.info("Logged fitted {} model to MLflow", model_type)


def model_provenance_path(
    model_path: Path,
) -> Path:
    """
    Return the JSON sidecar path used to persist model provenance.
    """

    return model_path.with_name(f"{model_path.stem}_metadata.json")


def _default_model_provenance(
    model_path: Path | None = None,
) -> dict:
    """
    Return an explicit local-only provenance record.
    """

    return {
        "provenance_available": False,
        "model_source": "local_joblib",
        "model_path": (str(model_path) if model_path is not None else None),
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": None,
        "mlflow_experiment_id": None,
        "mlflow_run_id": None,
        "mlflow_run_name": None,
        "mlflow_model_uri": None,
    }


def get_mlflow_run_provenance(
    run_id: str,
    tracking_dir: Path,
) -> dict:
    """
    Retrieve experiment/run provenance for an existing MLflow run.
    """

    try:
        import mlflow
        from mlflow import MlflowClient

    except ImportError as exc:
        raise RuntimeError(
            "MLflow provenance was requested but mlflow is not installed."
        ) from exc

    tracking_uri = tracking_dir.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    run = client.get_run(run_id)

    experiment = client.get_experiment(run.info.experiment_id)

    return {
        "provenance_available": True,
        "model_source": "mlflow_run",
        "model_path": None,
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_experiment_name": (experiment.name if experiment is not None else None),
        "mlflow_experiment_id": str(run.info.experiment_id),
        "mlflow_run_id": str(run.info.run_id),
        "mlflow_run_name": run.data.tags.get("mlflow.runName"),
        "mlflow_model_uri": (f"runs:/{run.info.run_id}/model"),
    }


def get_active_mlflow_provenance(
    enabled: bool,
    model_path: Path,
) -> dict:
    """
    Capture provenance from the active MLflow run.
    """

    provenance = _default_model_provenance(model_path)

    if not enabled:
        return provenance

    import mlflow
    from mlflow import MlflowClient

    active_run = mlflow.active_run()

    if active_run is None:
        raise RuntimeError("MLflow is enabled but no active run exists.")

    client = MlflowClient()

    run = client.get_run(active_run.info.run_id)

    experiment = client.get_experiment(run.info.experiment_id)

    provenance.update(
        {
            "provenance_available": True,
            "model_source": "local_joblib",
            "model_path": str(model_path),
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "mlflow_experiment_name": (
                experiment.name if experiment is not None else None
            ),
            "mlflow_experiment_id": str(run.info.experiment_id),
            "mlflow_run_id": str(run.info.run_id),
            "mlflow_run_name": run.data.tags.get("mlflow.runName"),
            "mlflow_model_uri": (f"runs:/{run.info.run_id}/model"),
        }
    )

    return provenance


def save_model_provenance(
    model_type: str,
    model_path: Path,
    mlflow_enabled: bool,
) -> Path:
    """
    Save model provenance next to the persisted joblib model.

    The file is overwritten on every training run to prevent stale
    MLflow provenance from being associated with a newly trained model.
    """

    provenance = get_active_mlflow_provenance(
        enabled=mlflow_enabled,
        model_path=model_path,
    )

    provenance["model_type"] = model_type

    output_path = model_provenance_path(model_path)

    output_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if provenance["mlflow_run_id"]:
        logger.info(
            "Model provenance: experiment='{}' ({}), run='{}' ({})",
            provenance["mlflow_experiment_name"],
            provenance["mlflow_experiment_id"],
            provenance["mlflow_run_name"],
            provenance["mlflow_run_id"],
        )

    else:
        logger.info("Model provenance saved without MLflow association")

    logger.success(
        "Saved model provenance to {}",
        output_path,
    )

    return output_path


def load_model_provenance(
    model_path: Path,
) -> dict:
    """
    Load provenance from the model's JSON sidecar.

    Older joblib files remain usable. Missing provenance is represented
    explicitly rather than inferred.
    """

    provenance_path = model_provenance_path(model_path)

    if not provenance_path.exists():
        logger.warning(
            "No model provenance sidecar found for {}. "
            "Retrain with the updated training script to record "
            "the MLflow experiment/run association.",
            model_path,
        )

        return _default_model_provenance(model_path)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    defaults = _default_model_provenance(model_path)

    defaults.update(provenance)

    return defaults
