from runtime import configure_runtime

configure_runtime()

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
)


def create_search(n_iter: int, cv_folds: int) -> RandomizedSearchCV:
    """Create a small training-only RBF-SVM hyperparameter search."""

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    probability=False,
                    cache_size=1000,
                ),
            ),
        ]
    )

    parameters = {
        "svc__C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0],
        "svc__gamma": ["scale", 0.0005, 0.001, 0.005, 0.01],
        "svc__class_weight": [None, "balanced"],
    }

    cross_validation = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=CONFIG.project.random_seed,
    )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cross_validation,
        refit=True,
        n_jobs=1,
        random_state=CONFIG.project.random_seed,
        return_train_score=False,
        error_score="raise",
    )


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    n_iter: int,
    cv_folds: int,
    use_mlflow: bool,
    mlflow_dir: Path,
    use_augmentation: bool,
) -> None:
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

    logger.info(
        "Training RBF SVM using {} randomized parameter combinations and {}-fold CV",
        n_iter,
        cv_folds,
    )

    search = create_search(n_iter=n_iter, cv_folds=cv_folds)

    with get_mlflow_context(
        enabled=use_mlflow,
        tracking_dir=mlflow_dir,
        run_name="rbf_svm",
        experiment_name=CONFIG.mlflow.experiments_experiment,
    ) as active_run:
        if active_run is not None:
            logger.info("MLflow run ID: {}", active_run.info.run_id)

        search.fit(X_train, y_train)
        logger.success("SVM search complete")
        logger.info("Best training CV ROC AUC: {:.4f}", search.best_score_)
        logger.info("Best parameters: {}", search.best_params_)

        search_path = save_search_results(
            search,
            output_dir / "svm_search_results.csv",
        )

        model = search.best_estimator_
        model_path = save_model(model, output_dir / "model.joblib")

        provenance_path = save_model_provenance(
            model_type="svm",
            model_path=model_path,
            mlflow_enabled=use_mlflow,
        )

        validation_scores = model.decision_function(X_val)
        threshold, validation_j = select_threshold(y_val, validation_scores)
        validation_auc = float(roc_auc_score(y_val, validation_scores))

        test_scores = model.decision_function(X_test)
        metrics, matrix = evaluate_scores(y_test, test_scores, threshold)
        metrics.update(
            {
                "training_cv_roc_auc": float(search.best_score_),
                "validation_roc_auc": validation_auc,
                "validation_youden_j": float(validation_j),
            }
        )

        output_paths = save_evaluation_outputs(
            name="svm",
            title="RBF SVM – Test Set",
            metrics=metrics,
            matrix=matrix,
            output_dir=output_dir,
        )
        output_paths.extend([search_path, model_path, provenance_path])

        logger.info(
            "SVM test metrics:\n{}",
            pd.DataFrame([metrics]).to_string(index=False),
        )

        log_mlflow_experiment(
            enabled=use_mlflow,
            parameters={
                "model": "rbf_svm",
                "search_iterations": n_iter,
                "cv_folds": cv_folds,
                "search_metric": "roc_auc",
                "threshold_method": "validation_youden_j",
                "augmentation_enabled": use_augmentation,
                **{f"best_{key}": value for key, value in search.best_params_.items()},
            },
            metrics=metrics,
            artifacts=output_paths,
        )

        log_mlflow_model(
            enabled=use_mlflow,
            model=model,
            model_type="svm",
            input_example=X_train,
        )

    logger.success("SVM experiment completed successfully")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an optional RBF-SVM experiment on PneumoniaMNIST."
    )
    parser.add_argument("--dataset", type=Path, default=PATHS.dataset)
    parser.add_argument("--output-dir", type=Path, default=PATHS.model_dir("svm"))
    parser.add_argument("--n-iter", type=int, default=CONFIG.modeling.search_iterations)
    parser.add_argument("--cv-folds", type=int, default=CONFIG.modeling.cv_folds)
    parser.add_argument(
        "--augment",
        action="store_true",
        default=CONFIG.augmentation.enabled,
        help=(
            "Append mild augmented copies of the training split only. "
            "Disabled by default."
        ),
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
    parser.add_argument("--mlflow-dir", type=Path, default=PATHS.mlflow_dir)
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        default="SUCCESS",
    )
    parser.add_argument("--log-file", type=Path, default=PATHS.log_file("model_svm"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_output_directories(CONFIG)
    configure_logging(
        console_level=args.log_level,
        log_file=args.log_file,
    )

    try:
        run_experiment(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
            use_mlflow=args.mlflow,
            mlflow_dir=args.mlflow_dir,
            use_augmentation=args.augment,
        )
    except Exception:
        logger.exception("SVM experiment failed")
        raise
