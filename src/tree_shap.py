"""TreeSHAP for the project's XGBoost model, in the same shape as linear SHAP.

TreeSHAP is exact for tree ensembles. Like the linear explainer, values are
contributions to class-1 log-odds (XGBoost's raw margin), not probability,
and base value + sum(values) reproduces the model's margin for each image.
"""

import numpy as np
from scipy.special import expit

from linear_shap import LinearExplanation


def is_tree_model(model) -> bool:
    return type(model).__name__ == "XGBClassifier"


def tree_shap_values(model, X):
    """Per-feature TreeSHAP values and base value from XGBoost itself.

    Uses XGBoost's built-in TreeSHAP (``pred_contribs=True``), the same
    path-dependent algorithm as ``shap.TreeExplainer``. The pinned shap 0.49
    cannot parse the ``base_score`` that XGBoost 3.x writes into saved models
    (``could not convert string to float: '[5.19E-1]'``), so going through
    XGBoost avoids that incompatibility. The last column is the bias term.
    """
    import xgboost

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != model.n_features_in_ or not np.isfinite(X).all():
        raise ValueError("Expected a finite (N, features) matrix")
    contributions = (
        model.get_booster()
        .predict(xgboost.DMatrix(X), pred_contribs=True)
        .astype(np.float64)
    )
    return contributions[:, :-1], contributions[:, -1]


def explain_tree(model, X):
    X = np.asarray(X, dtype=np.float32)
    values, bias = tree_shap_values(model, X)
    probabilities = model.predict_proba(X)[:, 1]
    margins = np.asarray(model.predict(X, output_margin=True), dtype=np.float64)
    reconstructed = bias + values.sum(axis=1)
    residual = reconstructed - margins
    # Trees are evaluated in float32, so allow float32-sized rounding.
    if not np.isfinite(residual).all() or np.abs(residual).max() > 1e-3:
        raise ValueError(
            f"TreeSHAP additivity failed: maximum log-odds error={np.abs(residual).max():.3g}"
        )
    if np.abs(expit(margins) - probabilities).max() > 1e-4:
        raise ValueError("XGBoost margin does not match its pneumonia probability")
    return LinearExplanation(
        values,
        float(bias[0]),
        margins,
        probabilities,
        residual,
        "TreeSHAP (path-dependent)",
    )
