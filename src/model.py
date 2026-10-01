from runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from model_utils import (
    evaluate_scores,
    get_mlflow_context,
    load_model_data,
    log_mlflow_experiment,
    log_mlflow_model,
    save_evaluation_outputs,
    save_model,
    save_model_provenance,
    save_search_results,
    select_threshold,
    tune_and_fit,
    tuning_metrics,
)


def create_model() -> Pipeline:
    """Create the required simple CPU baseline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=CONFIG.project.random_seed,
                ),
            ),
        ]
    )


# Tuned exactly like the SVM and XGBoost experiments (same sampler, seed,
# iteration count and ROC AUC scoring) so the candidates compare like for
# like. liblinear supports both penalties, and the fitted model stays
# StandardScaler -> LogisticRegression, so the exact linear SHAP explainer
# still applies.
SEARCH_PARAMETERS = {
    "classifier__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__class_weight": [None, "balanced"],
}


def run_model(
    dataset_path: Path,
    output_dir: Path,
    n_iter: int,
    use_cv: bool,
    cv_folds: int,
    use_mlflow: bool,
    mlflow_dir: Path,
    use_augmentation: bool,
) -> None:
    """Tune (holdout or k-fold CV), pick the threshold on validation, test once."""

    output_dir.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = load_model_data(
        dataset_path,
        augment_train=use_augmentation,
    )

    logger.info("Training data: X={}, y={}", X_train.shape, y_train.shape)
    logger.info("Validation data: X={}, y={}", X_val.shape, y_val.shape)
    logger.info("Test data: X={}, y={}", X_test.shape, y_test.shape)

    with get_mlflow_context(
        enabled=use_mlflow,
        tracking_dir=mlflow_dir,
        run_name="logistic_regression_baseline",
        experiment_name=CONFIG.mlflow.baseline_experiment,
    ) as active_run:
        if active_run is not None:
            logger.info("MLflow run ID: {}", active_run.info.run_id)

        tuning = tune_and_fit(
            create_model(),
            SEARCH_PARAMETERS,
            X_train,
            y_train,
            X_val,
            y_val,
            n_iter=n_iter,
            use_cv=use_cv,
            cv_folds=cv_folds,
        )
        logger.success("Logistic regression tuning complete")
        search_path = save_search_results(
            tuning.search,
            output_dir / "model_search_results.csv",
        )
        model = tuning.model

        validation_scores = model.predict_proba(X_val)[:, 1]
        threshold, validation_j = select_threshold(y_val, validation_scores)
        validation_auc = float(roc_auc_score(y_val, validation_scores))

        test_scores = model.predict_proba(X_test)[:, 1]
        metrics, matrix = evaluate_scores(y_test, test_scores, threshold)
        metrics.update(
            {
                **tuning_metrics(tuning, use_cv),
                "validation_roc_auc": validation_auc,
                "validation_youden_j": float(validation_j),
            }
        )

        output_paths = save_evaluation_outputs(
            name="model",
            title="Logistic Regression – Test Set",
            metrics=metrics,
            matrix=matrix,
            output_dir=output_dir,
        )

        model_path = save_model(
            model,
            output_dir / "model.joblib",
        )

        provenance_path = save_model_provenance(
            model_type="logistic_regression",
            model_path=model_path,
            mlflow_enabled=use_mlflow,
        )
        output_paths.extend([search_path, model_path, provenance_path])

        logger.info(
            "Final test metrics:\n{}",
            pd.DataFrame([metrics]).to_string(index=False),
        )

        log_mlflow_experiment(
            enabled=use_mlflow,
            parameters={
                "model": "logistic_regression",
                "features": "flattened_pixels",
                "scaler": "standard_scaler",
                "solver": "liblinear",
                "max_iter": 2000,
                "random_state": CONFIG.project.random_seed,
                "search_iterations": n_iter,
                "tuning_method": tuning.method,
                "cv_folds": cv_folds if use_cv else None,
                "search_metric": "roc_auc",
                "threshold_method": "validation_youden_j",
                "augmentation_enabled": use_augmentation,
                **{f"best_{key}": value for key, value in tuning.best_params.items()},
            },
            metrics=metrics,
            artifacts=output_paths,
        )

        log_mlflow_model(
            enabled=use_mlflow,
            model=model,
            model_type="logistic_regression",
            input_example=X_train,
        )

    logger.success("Part 3 baseline completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the PneumoniaMNIST " "logistic-regression baseline."
        )
    )

    parser.add_argument("--dataset", type=Path, default=PATHS.dataset)
    parser.add_argument("--output-dir", type=Path, default=PATHS.model_dir("logistic"))
    parser.add_argument("--n-iter", type=int, default=CONFIG.modeling.search_iterations)
    parser.add_argument(
        "--cv",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.modeling.cross_validation,
        help=(
            "Tune with stratified k-fold CV inside the training split instead of "
            "the default train -> validation holdout "
            "(default from config.toml [modeling].cross_validation)"
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=CONFIG.modeling.cv_folds,
        help="Number of folds when --cv is on",
    )
    parser.add_argument(
        "--mlflow",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.mlflow.enabled,
        help=(
            "Track the run in the local MLflow store "
            "(default from config.toml [mlflow].enabled)"
        ),
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=CONFIG.augmentation.enabled,
        help=(
            "Append mild augmented copies of the training split only. "
            "Disabled by default."
        ),
    )
    parser.add_argument("--mlflow-dir", type=Path, default=PATHS.mlflow_dir)
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="SUCCESS",
    )
    parser.add_argument("--log-file", type=Path, default=PATHS.log_file("model"))

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_output_directories(CONFIG)
    configure_logging(
        console_level=args.log_level,
        log_file=args.log_file,
    )

    try:
        run_model(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            n_iter=args.n_iter,
            use_cv=args.cv,
            cv_folds=args.cv_folds,
            use_mlflow=args.mlflow,
            mlflow_dir=args.mlflow_dir,
            use_augmentation=args.augment,
        )
    except Exception:
        logger.exception("Part 3 baseline failed")
        raise
