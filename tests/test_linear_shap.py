from itertools import combinations
from math import factorial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from streamlit.testing.v1 import AppTest

import image_pipeline
from linear_shap import explain_linear, explain_catalog_image, explanation_png
from test_pipeline import png


def fit(n_features=4):
    rng = np.random.default_rng(87)
    train = rng.normal(100, 30, (150, n_features)).astype(np.float32)
    train[:, -1] = 77  # Constant feature must have zero contribution.
    labels = (train[:, 0] + train[:, 1] > 200).astype(int)
    model = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression())]
    ).fit(train, labels)
    return model, train


def test_exact_shap_against_exhaustive_coalitions():
    model, train = fit()
    # Use float64 to eliminate cast rounding in the independent coalition oracle.
    background = train[:7].astype(np.float64)
    X = train[10:11].astype(np.float64)
    explanation = explain_linear(model, X, background)
    d = X.shape[1]

    def value(subset):
        masked = background.copy()
        masked[:, list(subset)] = X[0, list(subset)]
        return model.decision_function(masked).mean()

    oracle = np.zeros(d)
    for j in range(d):
        others = [i for i in range(d) if i != j]
        for size in range(d):
            for subset in combinations(others, size):
                weight = factorial(size) * factorial(d - size - 1) / factorial(d)
                oracle[j] += weight * (value((*subset, j)) - value(subset))
    np.testing.assert_allclose(explanation.values[0], oracle, atol=1e-12)
    assert explanation.base_value == pytest.approx(value(()))
    assert explanation.values[0, -1] == 0


def test_matches_shap_library_with_explicit_background():
    import shap

    model, train = fit()
    background = train[:120]
    result = explain_linear(model, train[120:125], background)
    scaler, classifier = model.named_steps.values()
    transformed_background = scaler.transform(background).astype(np.float64)
    masker = shap.maskers.Independent(
        transformed_background, max_samples=len(background)
    )
    reference = shap.LinearExplainer(classifier, masker)(
        scaler.transform(train[120:125]).astype(np.float64)
    )
    np.testing.assert_allclose(result.values, reference.values, atol=1e-12)
    np.testing.assert_allclose(result.base_value, reference.base_values, atol=1e-12)


def test_training_mean_reference_and_additivity():
    model, train = fit(784)
    result = explain_linear(model, train[:3])
    assert (
        abs(result.base_value - model.named_steps["classifier"].intercept_[0]) < 1e-12
    )
    np.testing.assert_allclose(
        result.base_value + result.values.sum(1),
        model.decision_function(train[:3]),
        atol=1e-10,
    )
    np.testing.assert_allclose(result.probability, model.predict_proba(train[:3])[:, 1])
    # Independent library oracle at the exact fitted mean; no validation/test data.
    import shap

    reference = shap.LinearExplainer(
        model.named_steps["classifier"], np.zeros((1, 784))
    )(model.named_steps["scaler"].transform(train[:3]).astype(np.float64))
    np.testing.assert_allclose(result.values, reference.values, atol=1e-12)
    image = explanation_png(train[0], result)
    import io

    assert Image.open(io.BytesIO(image)).size[0] > 1000
    with pytest.raises(ValueError, match="finite"):
        explain_linear(model, np.full((1, 784), np.nan))


def test_viewer_button_and_stale_model_guard(tmp_path, monkeypatch):
    model, train = fit(784)
    model_file = tmp_path / "model.joblib"
    metrics_file = tmp_path / "metrics.csv"
    joblib.dump(model, model_file)
    pd.DataFrame([{"threshold": 0.5}]).to_csv(metrics_file, index=False)
    classifier = image_pipeline.Classifier(model_file, metrics_file)
    database = tmp_path / "catalog.sqlite3"
    con = image_pipeline.connect(database)
    image_pipeline.ingest_bytes(con, png(), "sample.png", "sample", classifier)
    row = con.execute("SELECT * FROM images").fetchone()
    with pytest.raises(ValueError, match="different model"):
        explain_catalog_image(
            classifier, row["features"], "stale", row["pneumonia_score"]
        )
    with pytest.raises(ValueError, match="differs"):
        explain_catalog_image(classifier, row["features"], classifier.version, -1)
    con.close()
    monkeypatch.setenv("IMAGE_CATALOG_DB", str(database))
    monkeypatch.setattr(image_pipeline, "default_classifier", lambda: classifier)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
    ).run()
    assert not app.exception
    # SHAP is computed automatically when an image is selected; the viewer
    # exposes it through the "Show overlay" toggle rather than a button.
    assert not app.error
    assert not any("SHAP unavailable" in c.value for c in app.caption)
    toggle = next(t for t in app.toggle if t.label == "Show overlay")
    assert toggle.value is True
    toggle.set_value(False).run()
    assert not app.exception
    assert next(s for s in app.slider if s.label == "Opacity").disabled