from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

# Environment variable that points at the user-supplied dataset archive.
# The NPZ is never shipped with the repository or the Docker image; users
# provide their own copy and point to it with this variable (or with the
# per-script --dataset/--input/--npz flags).
DATASET_ENV_VAR = "PNEUMONIAMNIST_NPZ"


def _resolve_path(value: str) -> Path:
    """
    Resolve a config path relative to the project root.
    """

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    random_seed: int
    threads: int


@dataclass(frozen=True)
class PathSettings:
    data_dir: Path
    output_dir: Path
    dataset: Path
    database: Path
    metadata_csv: Path
    analysis_sql: Path
    logs_dir: Path
    analysis_dir: Path
    models_dir: Path
    predictions_dir: Path
    model_selection_dir: Path
    shap_dir: Path
    mlflow_dir: Path
    temp_dir: Path
    splits_dir: Path

    def log_file(self, name: str) -> Path:
        return self.logs_dir / f"{name}.log"

    def model_dir(self, model_name: str) -> Path:
        return self.models_dir / model_name

    def model_file(self, model_name: str) -> Path:
        return self.model_dir(model_name) / "model.joblib"

    def _model_output_prefix(self, model_name: str) -> str:
        prefixes = {
            "logistic": "model",
            "svm": "svm",
            "xgboost": "xgboost",
        }

        try:
            return prefixes[model_name]

        except KeyError as exc:
            raise ValueError(f"Unknown model name: {model_name}") from exc

    def model_metrics_file(self, model_name: str) -> Path:
        prefix = self._model_output_prefix(model_name)
        return self.model_dir(model_name) / f"{prefix}_metrics.csv"

    def model_confusion_matrix_file(self, model_name: str) -> Path:
        prefix = self._model_output_prefix(model_name)
        return self.model_dir(model_name) / f"{prefix}_confusion_matrix.csv"

    def model_evaluation_file(self, model_name: str) -> Path:
        prefix = self._model_output_prefix(model_name)
        return self.model_dir(model_name) / f"{prefix}_evaluation.png"

    def model_search_results_file(self, model_name: str) -> Path:
        prefix = self._model_output_prefix(model_name)
        return self.model_dir(model_name) / f"{prefix}_search_results.csv"

    def model_metadata_file(self, model_name: str) -> Path:
        return self.model_dir(model_name) / "model_metadata.json"

    def shap_model_dir(self, model_name: str) -> Path:
        return self.shap_dir / model_name


@dataclass(frozen=True)
class QCSettings:
    expected_height: int
    expected_width: int
    low_variance_quantile: float


@dataclass(frozen=True)
class ModelingSettings:
    cv_folds: int
    search_iterations: int
    threshold_method: str


@dataclass(frozen=True)
class SplitSettings:
    deduplicate: bool
    label_conflict_policy: str
    preserve_source_proportions: bool
    val_fraction: float
    test_fraction: float


@dataclass(frozen=True)
class MLflowSettings:
    enabled: bool
    baseline_experiment: str
    experiments_experiment: str


@dataclass(frozen=True)
class ShapSettings:
    background_size: int
    default_output_format: str


@dataclass(frozen=True)
class AugmentationSettings:
    enabled: bool
    copies_per_image: int
    rotate_limit_degrees: float
    translate_limit_fraction: float
    scale_limit_fraction: float
    brightness_limit: float
    contrast_limit: float
    probability: float


@dataclass(frozen=True)
class AppConfig:
    project: ProjectSettings
    paths: PathSettings
    qc: QCSettings
    modeling: ModelingSettings
    splits: SplitSettings
    mlflow: MLflowSettings
    shap: ShapSettings
    augmentation: AugmentationSettings


def _require_section(
    config: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    try:
        value = config[name]
    except KeyError as exc:
        raise ValueError(f"Missing [{name}] section in config") from exc

    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table")

    return value


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    """
    Load the project configuration from TOML.

    Python 3.11 includes tomllib, so no additional config dependency
    is required.
    """

    if not config_path.exists():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    with config_path.open("rb") as file:
        raw = tomllib.load(file)

    project = _require_section(
        raw,
        "project",
    )

    paths = _require_section(
        raw,
        "paths",
    )

    outputs = _require_section(
        raw,
        "outputs",
    )

    qc = _require_section(
        raw,
        "qc",
    )

    modeling = _require_section(
        raw,
        "modeling",
    )

    splits = _require_section(
        raw,
        "splits",
    )

    mlflow = _require_section(
        raw,
        "mlflow",
    )

    shap_settings = _require_section(
        raw,
        "shap",
    )

    augmentation = _require_section(
        raw,
        "augmentation",
    )

    return AppConfig(
        project=ProjectSettings(
            name=str(project["name"]),
            random_seed=int(project["random_seed"]),
            threads=int(project["threads"]),
        ),
        paths=PathSettings(
            data_dir=_resolve_path(paths["data_dir"]),
            output_dir=_resolve_path(paths["output_dir"]),
            dataset=_resolve_path(os.environ.get(DATASET_ENV_VAR) or paths["dataset"]),
            database=_resolve_path(paths["database"]),
            metadata_csv=_resolve_path(paths["metadata_csv"]),
            analysis_sql=_resolve_path(paths["analysis_sql"]),
            logs_dir=_resolve_path(outputs["logs"]),
            analysis_dir=_resolve_path(outputs["analysis"]),
            models_dir=_resolve_path(outputs["models"]),
            predictions_dir=_resolve_path(outputs["predictions"]),
            model_selection_dir=_resolve_path(outputs["model_selection"]),
            shap_dir=_resolve_path(outputs["shap"]),
            mlflow_dir=_resolve_path(outputs["mlflow"]),
            temp_dir=_resolve_path(outputs["temp"]),
            splits_dir=_resolve_path(outputs["splits"]),
        ),
        qc=QCSettings(
            expected_height=int(qc["expected_height"]),
            expected_width=int(qc["expected_width"]),
            low_variance_quantile=float(qc["low_variance_quantile"]),
        ),
        modeling=ModelingSettings(
            cv_folds=int(modeling["cv_folds"]),
            search_iterations=int(modeling["search_iterations"]),
            threshold_method=str(modeling["threshold_method"]),
        ),
        splits=SplitSettings(
            deduplicate=bool(splits["deduplicate"]),
            label_conflict_policy=str(splits["label_conflict_policy"]),
            preserve_source_proportions=bool(splits["preserve_source_proportions"]),
            val_fraction=float(splits["val_fraction"]),
            test_fraction=float(splits["test_fraction"]),
        ),
        mlflow=MLflowSettings(
            enabled=bool(mlflow.get("enabled", False)),
            baseline_experiment=str(mlflow["baseline_experiment"]),
            experiments_experiment=str(mlflow["experiments_experiment"]),
        ),
        shap=ShapSettings(
            background_size=int(shap_settings["background_size"]),
            default_output_format=str(shap_settings["default_output_format"]),
        ),
        augmentation=AugmentationSettings(
            enabled=bool(augmentation["enabled"]),
            copies_per_image=int(augmentation["copies_per_image"]),
            rotate_limit_degrees=float(augmentation["rotate_limit_degrees"]),
            translate_limit_fraction=float(augmentation["translate_limit_fraction"]),
            scale_limit_fraction=float(augmentation["scale_limit_fraction"]),
            brightness_limit=float(augmentation["brightness_limit"]),
            contrast_limit=float(augmentation["contrast_limit"]),
            probability=float(augmentation["probability"]),
        ),
    )


def ensure_output_directories(
    config: AppConfig,
) -> None:
    """
    Create the common generated-output directories.
    """

    paths = config.paths

    directories = [
        paths.output_dir,
        paths.logs_dir,
        paths.analysis_dir,
        paths.models_dir,
        paths.predictions_dir,
        paths.model_selection_dir,
        paths.shap_dir,
        paths.mlflow_dir,
        paths.temp_dir,
        paths.splits_dir,
    ]

    for model_name in (
        "logistic",
        "svm",
        "xgboost",
    ):
        directories.append(paths.model_dir(model_name))

    for model_name in (
        "logistic",
        "xgboost",
    ):
        directories.append(paths.shap_model_dir(model_name))

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


CONFIG = load_config()
PATHS = CONFIG.paths
