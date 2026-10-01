"""Exact interventional SHAP for the project's binary linear baseline.

For standardized pixels z, phi_j = beta_j * (z_j - E[z_j]).
The explained output is class-1 log-odds, NOT probability. Pixel dependence
is ignored (interventional SHAP); no correlation-dependent claim is made.
"""

from dataclasses import dataclass
import io

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure

EXPLANATION_VERSION = "3.1"


@dataclass(frozen=True)
class LinearExplanation:
    values: np.ndarray
    base_value: float
    log_odds: np.ndarray
    probability: np.ndarray
    residual: np.ndarray
    background: str


def explain_linear(model, X, background=None):
    """Use the fitted training mean, or an explicit raw-pixel background.

    The default reference is the entire scaler fitting population (including
    train-only augmentation if used). StandardScaler.mean_ stores its per-pixel
    empirical mean; a linear interventional explanation needs only that mean.
    Explicit backgrounds retain the standalone exporter's sampled reference.
    """
    if not isinstance(model, Pipeline) or len(model.steps) != 2:
        raise ValueError(
            "Expected exactly StandardScaler followed by LogisticRegression"
        )
    scaler, classifier = (step[1] for step in model.steps)
    if not isinstance(scaler, StandardScaler) or not isinstance(
        classifier, LogisticRegression
    ):
        raise ValueError("SHAP viewer supports only the logistic baseline")
    if (
        not scaler.with_mean
        or not scaler.with_std
        or list(classifier.classes_) != [0, 1]
    ):
        raise ValueError("Expected centered/scaled binary classes [0, 1]")
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != model.n_features_in_ or not np.isfinite(X).all():
        raise ValueError("Expected a finite (N, features) matrix")
    # Preserve input dtype through the fitted transform, matching predict_proba.
    scaled = scaler.transform(X).astype(np.float64)
    if background is None:
        mean = scaler.transform(scaler.mean_.reshape(1, -1))[0]
        reference = "Full training population mean stored in fitted StandardScaler"
    else:
        background = np.asarray(background)
        if (
            background.ndim != 2
            or not len(background)
            or background.shape[1] != X.shape[1]
            or not np.isfinite(background).all()
        ):
            raise ValueError(
                "Expected a nonempty finite background with matching features"
            )
        mean = scaler.transform(background).astype(np.float64).mean(axis=0)
        reference = f"Explicit background ({len(background)} images)"
    # Older sklearn binary multinomial fits use softmax([-margin, margin]),
    # whose positive-class log-odds is 2 * margin. Probe the fitted classifier
    # independently of the selected image (which may have saturated scores).
    weights = classifier.coef_[0].astype(np.float64)
    norm2 = float(weights @ weights)
    probes = np.zeros((3, len(weights)), dtype=classifier.coef_.dtype)
    if norm2 > 0:
        probes[1] = weights / norm2 * (1 - float(classifier.intercept_[0]))
        probes[2] = weights / norm2 * (-1 - float(classifier.intercept_[0]))
    probe_margins = classifier.decision_function(probes).astype(np.float64)
    probe_probabilities = classifier.predict_proba(probes)[:, 1]
    tolerance = max(1e-9, 8 * np.finfo(probe_probabilities.dtype).eps)
    if np.allclose(expit(probe_margins), probe_probabilities, rtol=0, atol=tolerance):
        factor = 1.0
    elif np.allclose(
        expit(2 * probe_margins), probe_probabilities, rtol=0, atol=tolerance
    ):
        factor = 2.0
    else:
        raise ValueError(
            "Unsupported probability mapping: neither binary sigmoid nor binary multinomial softmax. "
            f"SHAP engine {EXPLANATION_VERSION}; solver={classifier.solver}."
        )
    weights = weights * factor
    values = (scaled - mean) * weights
    base = float(factor * classifier.intercept_[0] + mean @ weights)
    margins = np.asarray(model.decision_function(X)) * factor
    reconstructed = base + values.sum(axis=1)
    probabilities = model.predict_proba(X)[:, 1]
    # Float32 inference rounds the dot product and sigmoid; attribution sums
    # use float64. Check both stages separately so sigmoid rounding does not
    # look like a SHAP error. Tolerances follow the actual inference dtype.
    margin_eps = np.finfo(margins.dtype).eps
    margin_atol = np.maximum(
        1e-8,
        8
        * margin_eps
        * (
            np.abs(scaled * weights).sum(axis=1)
            + abs(float(classifier.intercept_[0]))
            + 1
        ),
    )
    residual = np.abs(reconstructed - margins)
    if not np.isfinite(residual).all() or np.any(residual > margin_atol):
        raise ValueError(
            f"SHAP additivity failed: maximum log-odds error={residual.max():.3g}; "
            f"maximum tolerance={margin_atol.max():.3g}"
        )
    probability_atol = max(1e-9, 8 * np.finfo(probabilities.dtype).eps)
    probability_error = np.abs(expit(margins.astype(np.float64)) - probabilities)
    if not np.isfinite(probability_error).all() or np.any(
        probability_error > probability_atol
    ):
        raise ValueError(
            "SHAP reconstruction does not match pneumonia probability: "
            f"maximum error={probability_error.max():.3g}, tolerance={probability_atol:.3g}, "
            f"dtype={probabilities.dtype}, engine={EXPLANATION_VERSION}. Expected binary sigmoid probabilities; "
            "check the fitted model's probability convention."
        )
    # Sigmoid has derivative <= 1/4, bounding the effect of margin rounding.
    if np.any(
        np.abs(expit(reconstructed) - probabilities)
        > probability_atol + margin_atol / 4
    ):
        raise ValueError(
            "SHAP probability reconstruction exceeds the numerical error bound"
        )
    return LinearExplanation(
        values, base, margins, probabilities, reconstructed - margins, reference
    )


def explanation_png(features, explanation):
    """Render at model resolution so pixels and attribution cells align exactly."""
    pixels = np.asarray(features).reshape(28, 28)
    values = explanation.values[0].reshape(28, 28)
    limit = float(np.abs(values).max()) or 1.0
    fig = Figure(figsize=(10, 3.7), layout="constrained")
    axes = fig.subplots(1, 3)
    axes[0].imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    heat = axes[1].imshow(
        values, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest"
    )
    axes[2].imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[2].imshow(
        values,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        alpha=0.65 * np.abs(values) / limit,
        interpolation="nearest",
    )
    for ax, title in zip(
        axes, ["Model input (28×28)", "Signed SHAP values", "SHAP overlay"]
    ):
        ax.set_title(title)
        ax.axis("off")
    fig.colorbar(
        heat,
        ax=list(axes),
        orientation="horizontal",
        shrink=0.7,
        label="Contribution to pneumonia log-odds · blue ↓ / red ↑",
    )
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160)
    return buffer.getvalue()


def explain_catalog_image(classifier, features, stored_version, stored_score):
    """Explain a cataloged image with the model that scored it.

    Logistic regression -> exact linear SHAP; XGBoost -> exact TreeSHAP.
    The SVM has no exact, fast explainer, so the overlay is unavailable for it.
    Stale predictions are rejected rather than overlaid with a different
    model's explanation.
    """
    if stored_version != classifier.version:
        raise ValueError(
            "Stored prediction uses a different model. Re-ingest this image first."
        )
    X = np.frombuffer(features, dtype=np.float32).reshape(1, -1)
    model = classifier.model
    if isinstance(model, Pipeline) and isinstance(
        model.steps[-1][1], LogisticRegression
    ):
        explanation = explain_linear(model, X)
    else:
        from tree_shap import explain_tree, is_tree_model

        if not is_tree_model(model):
            raise ValueError(
                f"SHAP overlay is not available for the {getattr(classifier, 'name', 'selected')} "
                "model (no exact, fast explainer). Switch to logistic or xgboost to see it."
            )
        explanation = explain_tree(model, X)
    fresh = (
        classifier.scores(X)[0]
        if hasattr(classifier, "scores")
        else explanation.probability[0]
    )
    if not np.isfinite(stored_score) or not np.isclose(
        fresh, stored_score, rtol=1e-7, atol=1e-8
    ):
        raise ValueError(
            "Stored prediction differs from current inference. Re-ingest this image first."
        )
    return X, explanation
