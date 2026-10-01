import dataclasses
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from streamlit.testing.v1 import AppTest
from xgboost import XGBClassifier

import image_pipeline as pipeline
import linear_shap
import project_config
from model_utils import tune_and_fit, tuning_metrics
from test_pipeline import png

PARAMS = {
    "logisticregression__C": [0.01, 0.1, 1.0],
    "logisticregression__penalty": ["l2"],
}


def data(n=120, features=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, features))
    y = (X[:, 0] + rng.normal(scale=0.8, size=n) > 0).astype(int)
    return X, y


def save_model(tmp_path, model, threshold, name="model"):
    model_file = tmp_path / f"{name}.joblib"
    metrics_file = tmp_path / f"{name}_metrics.csv"
    joblib.dump(model, model_file)
    pd.DataFrame([{"threshold": threshold}]).to_csv(metrics_file, index=False)
    return model_file, metrics_file


def pixels(n=40, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 256, (n, 784)).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    X[y == 1, :100] += 40  # learnable signal
    return X, y


# ----------------------------------------------------------------------------
# Tuning: holdout by default, k-fold CV on request
# ----------------------------------------------------------------------------
def test_holdout_tuning_scores_on_validation_and_fits_on_train_only():
    X, y = data()
    Xtr, ytr, Xva, yva = X[:80], y[:80], X[80:], y[80:]
    base = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    result = tune_and_fit(
        base, PARAMS, Xtr, ytr, Xva, yva, n_iter=3, use_cv=False, cv_folds=3
    )
    assert result.method.startswith("holdout")
    assert result.search.n_splits_ == 1
    # Final model == the same settings fit on train alone (not train + val).
    assert result.model[0].n_samples_seen_ == 80
    expected = clone(base).set_params(**result.best_params).fit(Xtr, ytr)
    np.testing.assert_allclose(
        result.model.predict_proba(Xva), expected.predict_proba(Xva)
    )
    assert result.tuning_score == pytest.approx(
        roc_auc_score(yva, result.model.predict_proba(Xva)[:, 1])
    )
    assert "training_cv_roc_auc" not in tuning_metrics(result, use_cv=False)


def test_cv_tuning_uses_k_folds_inside_train():
    X, y = data()
    Xtr, ytr, Xva, yva = X[:80], y[:80], X[80:], y[80:]
    base = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
    result = tune_and_fit(
        base, PARAMS, Xtr, ytr, Xva, yva, n_iter=3, use_cv=True, cv_folds=4
    )
    assert result.search.n_splits_ == 4
    assert result.method == "4-fold CV on train"
    # refit on train only: the scaler saw exactly the 80 training rows
    assert result.model[0].n_samples_seen_ == 80
    assert (
        tuning_metrics(result, use_cv=True)["training_cv_roc_auc"]
        == result.tuning_score
    )


def test_cross_validation_is_off_by_default():
    assert project_config.CONFIG.modeling.cross_validation is False


# ----------------------------------------------------------------------------
# Choosing the app's model
# ----------------------------------------------------------------------------
def test_app_model_setting_and_env_override(monkeypatch):
    monkeypatch.delenv(project_config.APP_MODEL_ENV_VAR, raising=False)
    assert project_config._app_model({}) == "logistic"
    assert project_config._app_model({"app": {"model": "SVM"}}) == "svm"
    monkeypatch.setenv(project_config.APP_MODEL_ENV_VAR, "best")
    assert project_config._app_model({"app": {"model": "svm"}}) == "best"
    monkeypatch.setenv(project_config.APP_MODEL_ENV_VAR, "random_forest")
    with pytest.raises(ValueError, match="Unknown app model"):
        project_config._app_model({})


def test_best_resolves_from_model_selection(tmp_path, monkeypatch):
    paths = dataclasses.replace(pipeline.PATHS, model_selection_dir=tmp_path)
    monkeypatch.setattr(pipeline, "PATHS", paths)
    with pytest.raises(FileNotFoundError, match="select_best_model"):
        pipeline.resolve_model_name("best")
    (tmp_path / "selected_model.json").write_text(
        json.dumps({"selected_model": "xgboost"})
    )
    assert pipeline.resolve_model_name("best") == "xgboost"
    assert pipeline.resolve_model_name("SVM") == "svm"
    with pytest.raises(ValueError):
        pipeline.resolve_model_name("forest")


def test_margin_model_keeps_its_decisions_on_a_0_1_scale(tmp_path):
    X, y = pixels()
    svm = make_pipeline(StandardScaler(), SVC(probability=False)).fit(X, y)
    raw_threshold = 0.25
    classifier = pipeline.Classifier(
        *save_model(tmp_path, svm, raw_threshold), name="svm"
    )
    assert classifier.score_kind == "margin"
    assert classifier.threshold == pytest.approx(expit(raw_threshold))
    results = classifier.predict_batch(X)
    margins = svm.decision_function(X)
    assert [r["predicted_class"] == "pneumonia" for r in results] == list(
        margins >= raw_threshold
    )
    scores = np.array([r["pneumonia_score"] for r in results])
    np.testing.assert_allclose(scores, expit(margins), rtol=1e-6)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_model_name_is_part_of_the_version(tmp_path):
    X, y = pixels()
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    files = save_model(tmp_path, model, 0.5)
    a = pipeline.Classifier(*files, name="logistic")
    b = pipeline.Classifier(*files, name="other")
    assert a.version != b.version


# ----------------------------------------------------------------------------
# Re-scoring the catalog after switching models
# ----------------------------------------------------------------------------
@pytest.fixture
def two_classifiers(tmp_path):
    X, y = pixels()
    logistic = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    xgb = XGBClassifier(n_estimators=5, max_depth=2, random_state=0).fit(X, y)
    return (
        pipeline.Classifier(
            *save_model(tmp_path, logistic, 0.5, "lr"), name="logistic"
        ),
        pipeline.Classifier(*save_model(tmp_path, xgb, 0.5, "xgb"), name="xgboost"),
    )


def test_rescore_switches_models_without_reading_files(tmp_path, two_classifiers):
    logistic, xgb = two_classifiers
    con = pipeline.connect(tmp_path / "catalog.sqlite3")
    for i, color in enumerate([10, 90, 200]):
        pipeline.ingest_bytes(con, png(color), f"{i}.png", f"src{i}", logistic)
    pipeline.ingest_bytes(con, png(90), "dup.png", "dup", logistic)  # same bytes
    pipeline.ingest_bytes(con, b"broken", "bad.png", "bad", logistic)
    assert pipeline.stale_prediction_count(con, xgb) == 3

    assert pipeline.rescore_catalog(con, xgb) == {"rescored": 3, "model_error": 0}
    assert pipeline.stale_prediction_count(con, xgb) == 0
    assert pipeline.rescore_catalog(con, xgb) == {"rescored": 0, "model_error": 0}

    rows = con.execute(
        "SELECT features, pneumonia_score, model_version FROM images "
        "WHERE processing_status='complete'"
    ).fetchall()
    assert {r["model_version"] for r in rows} == {xgb.version}
    for r in rows:
        expected = xgb.predict(np.frombuffer(r["features"], dtype=np.float32))
        assert r["pneumonia_score"] == pytest.approx(expected["pneumonia_score"])
    failed = con.execute("SELECT processing_status FROM images WHERE features IS NULL")
    assert {r[0] for r in failed} == {"failed"}
    con.close()


def test_shap_follows_the_active_model(tmp_path, two_classifiers):
    logistic, xgb = two_classifiers
    con = pipeline.connect(tmp_path / "catalog.sqlite3")
    pipeline.ingest_bytes(con, png(120), "a.png", "a", xgb)
    row = con.execute("SELECT * FROM images").fetchone()
    X, explanation = linear_shap.explain_catalog_image(
        xgb, row["features"], row["model_version"], row["pneumonia_score"]
    )
    assert explanation.values.shape == (1, 784)
    assert abs(explanation.residual).max() < 1e-3  # base + sum(phi) == margin
    with pytest.raises(ValueError, match="different model"):
        linear_shap.explain_catalog_image(
            logistic, row["features"], row["model_version"], row["pneumonia_score"]
        )

    Xs, ys = pixels()
    svm = make_pipeline(StandardScaler(), SVC()).fit(Xs, ys)
    svm_classifier = pipeline.Classifier(
        *save_model(tmp_path, svm, 0.0, "svm"), name="svm"
    )
    pipeline.rescore_catalog(con, svm_classifier)
    row = con.execute("SELECT * FROM images").fetchone()
    with pytest.raises(ValueError, match="not available for the svm"):
        linear_shap.explain_catalog_image(
            svm_classifier,
            row["features"],
            row["model_version"],
            row["pneumonia_score"],
        )
    con.close()


def test_app_flags_and_rescores_predictions_from_another_model(
    tmp_path, monkeypatch, two_classifiers
):
    logistic, xgb = two_classifiers
    database = tmp_path / "catalog.sqlite3"
    con = pipeline.connect(database)
    pipeline.ingest_bytes(con, png(30), "a.png", "a", logistic)
    pipeline.ingest_bytes(con, png(220), "b.png", "b", logistic)
    con.close()
    monkeypatch.setenv("IMAGE_CATALOG_DB", str(database))
    monkeypatch.setattr(pipeline, "default_classifier", lambda: xgb)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    assert not app.exception
    assert any("Classifying with **xgboost**" in c.value for c in app.caption)
    assert any(
        "2 images were classified by a different model" in w.value for w in app.warning
    )
    next(b for b in app.button if b.label == "Re-score with xgboost").click().run()
    assert not app.exception
    assert not any("different model" in w.value for w in app.warning)
    assert any("Re-scored 2 images with xgboost" in s.value for s in app.success)
    # SHAP works for the newly active tree model
    assert not any("SHAP unavailable" in c.value for c in app.caption)
