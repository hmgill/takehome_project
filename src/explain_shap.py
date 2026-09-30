from runtime import configure_runtime

configure_runtime()

import argparse
import json
import uuid
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from logging_utils import configure_logging
from project_config import CONFIG, PATHS, ensure_output_directories
from model_utils import (
    get_mlflow_run_provenance,
    load_model_provenance,
)
from data_splits import load_dataset_splits
from utils import flatten_images, get_image_from_npz

MODEL_FILES = {
    "logistic": PATHS.model_file("logistic"),
    "xgboost": PATHS.model_file("xgboost"),
}

METRIC_FILES = {
    "logistic": PATHS.model_metrics_file("logistic"),
    "xgboost": PATHS.model_metrics_file("xgboost"),
}

LABEL_NAMES = {
    0: "Normal",
    1: "Pneumonia",
}


# ---------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------


def _metadata_from_row(row) -> dict:
    return {
        "image_id": str(row[0]),
        "split": str(row[1]),
        "split_index": int(row[2]),
        "label": int(row[3]),
        "mean_intensity": float(row[4]),
        "std_intensity": float(row[5]),
    }


def resolve_image_selection(
    database_path: Path,
    image_selector: str,
) -> list[dict]:
    """
    Resolve one UUID, a comma-delimited UUID list, or 'all'.

    Explicit UUID lists preserve the order supplied by the user.
    Duplicate UUIDs in the argument are collapsed to one image.
    """

    if not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    selector = image_selector.strip()

    if not selector:
        raise ValueError("--image-id cannot be empty")

    with duckdb.connect(
        str(database_path),
        read_only=True,
        config={"threads": 1},
    ) as connection:

        if selector.lower() == "all":
            rows = connection.execute("""
                SELECT
                    image_id,
                    split,
                    split_index,
                    label,
                    mean_intensity,
                    std_intensity
                FROM main.image_metadata
                ORDER BY
                    CASE split
                        WHEN 'train' THEN 1
                        WHEN 'val' THEN 2
                        WHEN 'test' THEN 3
                    END,
                    split_index
                """).fetchall()

            if not rows:
                raise ValueError("main.image_metadata contains no images")

            return [_metadata_from_row(row) for row in rows]

        requested_ids = []
        seen = set()

        for raw_id in selector.split(","):
            raw_id = raw_id.strip()

            if not raw_id:
                continue

            try:
                normalized_id = str(uuid.UUID(raw_id))
            except ValueError as exc:
                raise ValueError(f"Invalid image UUID: {raw_id}") from exc

            if normalized_id not in seen:
                requested_ids.append(normalized_id)
                seen.add(normalized_id)

        if not requested_ids:
            raise ValueError("No valid image IDs were supplied")

        placeholders = ",".join("?" for _ in requested_ids)

        rows = connection.execute(
            f"""
            SELECT
                image_id,
                split,
                split_index,
                label,
                mean_intensity,
                std_intensity
            FROM main.image_metadata
            WHERE CAST(image_id AS VARCHAR) IN ({placeholders})
            """,
            requested_ids,
        ).fetchall()

    metadata_by_id = {str(row[0]): _metadata_from_row(row) for row in rows}

    missing = [image_id for image_id in requested_ids if image_id not in metadata_by_id]

    if missing:
        raise ValueError(
            "Image ID(s) not found in main.image_metadata: " + ", ".join(missing)
        )

    return [metadata_by_id[image_id] for image_id in requested_ids]


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


def load_local_model(
    model_path: Path,
):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Run the corresponding training script first."
        )

    logger.info(
        "Loading local model: {}",
        model_path,
    )

    return joblib.load(model_path)


def load_mlflow_model(
    model_type: str,
    run_id: str,
    tracking_dir: Path,
):
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow model loading requested, " "but mlflow is not installed."
        ) from exc

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    model_uri = f"runs:/{run_id}/model"

    logger.info(
        "Loading model from MLflow run {}",
        run_id,
    )

    if model_type == "xgboost":
        import mlflow.xgboost

        return mlflow.xgboost.load_model(model_uri)

    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri)


def get_local_threshold(
    model_type: str,
) -> float:
    metrics_path = METRIC_FILES[model_type]

    if not metrics_path.exists():
        logger.warning("Metrics file not found; using threshold 0.5")
        return 0.5

    metrics = pd.read_csv(metrics_path)

    if metrics.empty or "threshold" not in metrics.columns:
        logger.warning("No threshold found in metrics file; using threshold 0.5")
        return 0.5

    return float(metrics.iloc[0]["threshold"])


def get_mlflow_threshold(
    run_id: str,
    tracking_dir: Path,
) -> float:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    run = MlflowClient().get_run(run_id)

    return float(
        run.data.metrics.get(
            "threshold",
            0.5,
        )
    )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------


def get_background_data(
    dataset_path: Path,
    background_size: int,
) -> np.ndarray:
    """Sample training images for the linear SHAP background."""

    rng = np.random.default_rng(42)

    # Background comes from the deduplicated training split.
    X_train = flatten_images(load_dataset_splits(dataset_path).train.images)

    sample_size = min(
        background_size,
        len(X_train),
    )

    indices = rng.choice(
        len(X_train),
        size=sample_size,
        replace=False,
    )

    return X_train[indices]


def explain_logistic_batch(
    model,
    X: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Compute per-pixel SHAP values for one or more logistic-regression
    inputs.
    """

    from linear_shap import explain_linear

    return explain_linear(model, X, background).values


def explain_xgboost_batch(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute TreeSHAP values for one or more XGBoost inputs.
    """

    import shap

    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)

    values = np.asarray(explanation.values)

    if values.ndim == 2:
        return values

    if values.ndim == 3 and values.shape[-1] == 2:
        return values[
            :,
            :,
            1,
        ]

    raise RuntimeError("Unexpected XGBoost SHAP shape: " f"{values.shape}")


# ---------------------------------------------------------------------
# Data loading and inference
# ---------------------------------------------------------------------


def load_selected_images(
    dataset_path: Path,
    metadata_rows: list[dict],
) -> np.ndarray:
    """
    Load the images corresponding to resolved metadata rows.
    """

    images = []

    with np.load(
        dataset_path,
        allow_pickle=False,
    ) as dataset:

        for metadata in metadata_rows:
            image = get_image_from_npz(
                dataset=dataset,
                split=metadata["split"],
                split_index=metadata["split_index"],
            )

            images.append(image)

    return np.stack(
        images,
        axis=0,
    )


def build_explanation_records(
    metadata_rows: list[dict],
    images: np.ndarray,
    model,
    model_type: str,
    threshold: float,
    dataset_path: Path,
    background_size: int,
) -> list[dict]:
    """
    Run prediction and SHAP for the selected images.
    """

    X = flatten_images(images)

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    logger.info(
        "Computing SHAP values for {} image(s)",
        len(metadata_rows),
    )

    if model_type == "logistic":
        background = get_background_data(
            dataset_path,
            background_size,
        )

        shap_values = explain_logistic_batch(
            model=model,
            X=X,
            background=background,
        )

    elif model_type == "xgboost":
        shap_values = explain_xgboost_batch(
            model=model,
            X=X,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if shap_values.shape != X.shape:
        raise RuntimeError(
            "SHAP output shape does not match feature matrix: "
            f"{shap_values.shape} vs {X.shape}"
        )

    records = []

    for index, metadata in enumerate(metadata_rows):
        image = images[index]

        shap_map = shap_values[index].reshape(image.shape)

        records.append(
            {
                "metadata": metadata,
                "image": image,
                "shap_map": shap_map,
                "probability": float(probabilities[index]),
                "prediction": int(predictions[index]),
                "threshold": float(threshold),
            }
        )

    return records


# ---------------------------------------------------------------------
# Static PNG
# ---------------------------------------------------------------------


def plot_explanation(
    record: dict,
    model_type: str,
    output_path: Path,
) -> None:
    """
    Create a static original / SHAP / overlay figure for one image.
    """

    metadata = record["metadata"]
    image = record["image"]
    shap_map = record["shap_map"]
    probability = record["probability"]
    prediction = record["prediction"]
    threshold = record["threshold"]

    max_abs = float(np.max(np.abs(shap_map)))

    if max_abs == 0:
        max_abs = 1.0

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(
            12,
            4.9,
        ),
    )

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[0].set_title(
        "Original",
        fontsize=12,
        pad=10,
    )
    axes[0].axis("off")

    heatmap = axes[1].imshow(
        shap_map,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    axes[1].set_title(
        "SHAP attribution",
        fontsize=12,
        pad=10,
    )
    axes[1].axis("off")

    axes[2].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[2].imshow(
        shap_map,
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
        alpha=0.38,
    )
    axes[2].set_title(
        "Attribution overlay",
        fontsize=12,
        pad=10,
    )
    axes[2].axis("off")

    figure.suptitle(
        f"{model_type.upper()} SHAP Explanation",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    subtitle = (
        f"Image ID: {metadata['image_id']}   |   "
        f"True: {LABEL_NAMES[metadata['label']]}   |   "
        f"Predicted: {LABEL_NAMES[prediction]}   |   "
        f"P(pneumonia): {probability:.3f}   |   "
        f"Threshold: {threshold:.3f}"
    )

    figure.text(
        0.5,
        0.915,
        subtitle,
        ha="center",
        va="center",
        fontsize=10.5,
    )

    provenance = record.get(
        "model_provenance",
        {},
    )

    if provenance.get("mlflow_run_id"):
        provenance_text = (
            f"MLflow experiment: "
            f"{provenance.get('mlflow_experiment_name')} "
            f"({provenance.get('mlflow_experiment_id')})   |   "
            f"Run: {provenance.get('mlflow_run_name')} "
            f"({provenance.get('mlflow_run_id')})"
        )

    else:
        provenance_text = "MLflow experiment/run: not recorded"

    figure.text(
        0.5,
        0.865,
        provenance_text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555555",
    )

    colorbar_axis = figure.add_axes(
        [
            0.20,
            0.12,
            0.60,
            0.025,
        ]
    )

    colorbar = figure.colorbar(
        heatmap,
        cax=colorbar_axis,
        orientation="horizontal",
    )

    colorbar.set_label(
        "SHAP value",
        fontsize=10,
        labelpad=5,
    )

    colorbar.ax.tick_params(
        labelsize=9,
    )

    figure.text(
        0.5,
        0.075,
        ("Blue → pushes toward normal    |    " "Red → pushes toward pneumonia"),
        ha="center",
        va="center",
        fontsize=9,
    )

    figure.text(
        0.5,
        0.025,
        (
            "SHAP reflects model attribution, not anatomical "
            "localization of pneumonia."
        ),
        ha="center",
        va="center",
        fontsize=8.5,
    )

    figure.subplots_adjust(
        top=0.77,
        bottom=0.25,
        left=0.03,
        right=0.97,
        wspace=0.16,
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Interactive HTML gallery
# ---------------------------------------------------------------------


def save_interactive_gallery(
    records: list[dict],
    model_type: str,
    output_path: Path,
) -> None:
    """
    Save one standalone interactive HTML gallery.

    A scrollable image list controls a single responsive Plotly viewer.
    The SHAP overlay can be toggled and its opacity adjusted.
    """

    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError(
            "Interactive HTML output requires Plotly. "
            "Install it with `pip install plotly`."
        ) from exc

    if len(records) > 500:
        logger.warning(
            "Embedding {} images and SHAP maps in one offline HTML "
            "file may produce a large file.",
            len(records),
        )

    items = []

    for record in records:
        metadata = record["metadata"]

        shap_map = record["shap_map"]

        max_abs = float(np.max(np.abs(shap_map)))

        if max_abs == 0:
            max_abs = 1.0

        items.append(
            {
                "image_id": metadata["image_id"],
                "split": metadata["split"],
                "split_index": metadata["split_index"],
                "true_label": LABEL_NAMES[metadata["label"]],
                "predicted_label": LABEL_NAMES[record["prediction"]],
                "probability": round(
                    record["probability"],
                    6,
                ),
                "threshold": round(
                    record["threshold"],
                    6,
                ),
                "mlflow_experiment_name": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_experiment_name"),
                "mlflow_experiment_id": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_experiment_id"),
                "mlflow_run_id": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_run_id"),
                "mlflow_run_name": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_run_name"),
                "image": (record["image"].astype(int).tolist()),
                "shap": (
                    np.round(
                        shap_map,
                        6,
                    ).tolist()
                ),
                "max_abs": round(
                    max_abs,
                    6,
                ),
            }
        )

    payload = json.dumps(
        items,
        separators=(
            ",",
            ":",
        ),
    )

    plotly_js = get_plotlyjs()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{model_type.upper()} SHAP Gallery</title>
<style>
    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        font-family: Arial, Helvetica, sans-serif;
        color: #1f2937;
        background: #f8fafc;
    }}

    header {{
        padding: 20px 24px 14px;
        background: white;
        border-bottom: 1px solid #e5e7eb;
    }}

    header h1 {{
        margin: 0 0 6px;
        font-size: 24px;
    }}

    header p {{
        margin: 0;
        color: #64748b;
        font-size: 14px;
    }}

    .layout {{
        display: grid;
        grid-template-columns: minmax(250px, 320px) minmax(0, 1fr);
        gap: 16px;
        padding: 16px;
    }}

    .sidebar,
    .viewer {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
    }}

    .sidebar {{
        padding: 12px;
    }}

    .search {{
        width: 100%;
        padding: 10px 12px;
        margin-bottom: 10px;
        border: 1px solid #cbd5e1;
        border-radius: 7px;
        font-size: 14px;
    }}

    .image-list {{
        max-height: 72vh;
        overflow-y: auto;
        display: grid;
        gap: 7px;
    }}

    .image-item {{
        width: 100%;
        text-align: left;
        padding: 10px;
        border: 1px solid #e2e8f0;
        border-radius: 7px;
        background: white;
        cursor: pointer;
    }}

    .image-item:hover {{
        background: #f8fafc;
    }}

    .image-item.active {{
        border-color: #475569;
        background: #f1f5f9;
    }}

    .image-item strong {{
        display: block;
        font-size: 13px;
        margin-bottom: 4px;
    }}

    .image-item span {{
        display: block;
        color: #64748b;
        font-size: 12px;
        line-height: 1.4;
    }}

    .viewer {{
        padding: 16px;
        min-width: 0;
    }}

    .meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px 18px;
        margin: 0 4px 8px;
        font-size: 14px;
    }}

    .meta strong {{
        font-weight: 600;
    }}

    .controls {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 16px;
        margin: 4px 4px 8px;
        font-size: 14px;
    }}

    .controls label {{
        display: flex;
        align-items: center;
        gap: 7px;
    }}

    .controls input[type="range"] {{
        width: 180px;
    }}

    .export-button {{
        padding: 8px 12px;
        border: 1px solid #94a3b8;
        border-radius: 7px;
        background: white;
        color: #1f2937;
        font-size: 13px;
        cursor: pointer;
    }}

    .export-button:hover {{
        background: #f8fafc;
    }}

    .export-button:disabled {{
        opacity: 0.6;
        cursor: wait;
    }}

    #plot {{
        width: 100%;
        min-height: 600px;
    }}

    .note {{
        margin: 8px 4px 0;
        color: #64748b;
        font-size: 12px;
    }}

    .empty {{
        padding: 18px;
        color: #64748b;
        text-align: center;
    }}

    @media (max-width: 820px) {{
        .layout {{
            grid-template-columns: 1fr;
        }}

        .image-list {{
            max-height: 260px;
        }}

        #plot {{
            min-height: 480px;
        }}
    }}
</style>
<script>{plotly_js}</script>
</head>
<body>
<header>
    <h1>{model_type.upper()} SHAP Gallery</h1>
    <p>
        {len(items)} image(s). Select an image to inspect its pixel-level
        SHAP attribution.
    </p>
</header>

<div class="layout">
    <aside class="sidebar">
        <input
            id="search"
            class="search"
            type="search"
            placeholder="Filter by image ID, split, label..."
            aria-label="Filter images"
        >
        <div id="image-list" class="image-list"></div>
    </aside>

    <main class="viewer">
        <div id="meta" class="meta"></div>

        <div class="controls">
            <label>
                <input id="overlay-toggle" type="checkbox" checked>
                SHAP overlay
            </label>

            <label>
                Overlay opacity
                <input
                    id="opacity"
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value="0.48"
                >
                <span id="opacity-value">0.48</span>
            </label>

            <button
                id="export-button"
                class="export-button"
                type="button"
            >
                Export current view
            </button>
        </div>

        <div id="plot"></div>

        <p class="note">
            Blue pushes the model toward normal; red pushes it toward
            pneumonia. SHAP reflects model attribution, not anatomical
            localization of pneumonia.
        </p>
    </main>
</div>

<script>
const items = {payload};

const listNode = document.getElementById("image-list");
const searchNode = document.getElementById("search");
const metaNode = document.getElementById("meta");
const plotNode = document.getElementById("plot");
const toggleNode = document.getElementById("overlay-toggle");
const opacityNode = document.getElementById("opacity");
const opacityValueNode = document.getElementById("opacity-value");
const exportButtonNode = document.getElementById("export-button");

let selectedIndex = 0;

function itemText(item) {{
    return [
        item.image_id,
        item.split,
        String(item.split_index),
        item.true_label,
        item.predicted_label,
        item.mlflow_experiment_name || "",
        item.mlflow_experiment_id || "",
        item.mlflow_run_name || "",
        item.mlflow_run_id || ""
    ].join(" ").toLowerCase();
}}

function wrapCanvasText(ctx, text, maxWidth) {{
    const words = text.split(" ");
    const lines = [];
    let line = "";

    words.forEach((word) => {{
        const testLine = line ? `${{line}} ${{word}}` : word;

        if (ctx.measureText(testLine).width > maxWidth && line) {{
            lines.push(line);
            line = word;
        }} else {{
            line = testLine;
        }}
    }});

    if (line) {{
        lines.push(line);
    }}

    return lines;
}}

async function exportCurrentView() {{
    const item = items[selectedIndex];
    const overlayVisible =
        toggleNode.checked && Number(opacityNode.value) > 0;
    const overlayOpacity = Number(opacityNode.value);

    exportButtonNode.disabled = true;
    exportButtonNode.textContent = "Exporting...";

    try {{
        const plotWidth = 1200;
        const plotHeight = 760;

        const plotUrl = await Plotly.toImage(
            plotNode,
            {{
                format: "png",
                width: plotWidth,
                height: plotHeight,
                scale: 2
            }}
        );

        const plotImage = new Image();

        await new Promise((resolve, reject) => {{
            plotImage.onload = resolve;
            plotImage.onerror = reject;
            plotImage.src = plotUrl;
        }});

        const canvasWidth = 1400;
        const sidePadding = 70;
        const headerHeight = 215;
        const footerHeight = 80;

        const renderedPlotWidth = canvasWidth - (2 * sidePadding);
        const renderedPlotHeight =
            renderedPlotWidth * (plotHeight / plotWidth);

        const canvasHeight =
            headerHeight + renderedPlotHeight + footerHeight;

        const canvas = document.createElement("canvas");
        canvas.width = canvasWidth * 2;
        canvas.height = Math.ceil(canvasHeight * 2);

        const ctx = canvas.getContext("2d");
        ctx.scale(2, 2);

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(
            0,
            0,
            canvasWidth,
            canvasHeight
        );

        ctx.fillStyle = "#111827";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";

        ctx.font = "bold 30px Arial, Helvetica, sans-serif";
        ctx.fillText(
            "{model_type.upper()} SHAP Explanation",
            canvasWidth / 2,
            28
        );

        const metadataText =
            `Image ID: ${{item.image_id}}  |  ` +
            `True: ${{item.true_label}}  |  ` +
            `Predicted: ${{item.predicted_label}}  |  ` +
            `P(pneumonia): ${{item.probability.toFixed(3)}}  |  ` +
            `Threshold: ${{item.threshold.toFixed(3)}}`;

        ctx.font = "18px Arial, Helvetica, sans-serif";

        const metadataLines = wrapCanvasText(
            ctx,
            metadataText,
            canvasWidth - 120
        );

        metadataLines.forEach((line, index) => {{
            ctx.fillText(
                line,
                canvasWidth / 2,
                82 + (index * 25)
            );
        }});

        const mlflowText = item.mlflow_run_id
            ? (
                `MLflow experiment: ${{item.mlflow_experiment_name}} ` +
                `(${{item.mlflow_experiment_id}}) | ` +
                `Run: ${{item.mlflow_run_name || "unnamed"}} ` +
                `(${{item.mlflow_run_id}})`
            )
            : "MLflow experiment/run: not recorded";

        ctx.fillStyle = "#475569";
        ctx.font = "14px Arial, Helvetica, sans-serif";

        const mlflowLines = wrapCanvasText(
            ctx,
            mlflowText,
            canvasWidth - 120
        );

        mlflowLines.forEach((line, index) => {{
            ctx.fillText(
                line,
                canvasWidth / 2,
                132 + (index * 20)
            );
        }});

        const overlayText = overlayVisible
            ? `SHAP overlay opacity: ${{overlayOpacity.toFixed(2)}}`
            : "SHAP overlay: off";

        ctx.fillStyle = "#475569";
        ctx.font = "16px Arial, Helvetica, sans-serif";
        ctx.fillText(
            overlayText,
            canvasWidth / 2,
            180
        );

        ctx.drawImage(
            plotImage,
            sidePadding,
            headerHeight,
            renderedPlotWidth,
            renderedPlotHeight
        );

        ctx.fillStyle = "#64748b";
        ctx.font = "14px Arial, Helvetica, sans-serif";

        const footerText = overlayVisible
            ? (
                "Blue pushes the model toward normal; red pushes it toward " +
                "pneumonia. SHAP reflects model attribution, not anatomical " +
                "localization of pneumonia."
            )
            : (
                "SHAP overlay is hidden in this export. The displayed image " +
                "is the source X-ray."
            );

        const footerLines = wrapCanvasText(
            ctx,
            footerText,
            canvasWidth - 120
        );

        footerLines.forEach((line, index) => {{
            ctx.fillText(
                line,
                canvasWidth / 2,
                headerHeight + renderedPlotHeight + 24 + (index * 20)
            );
        }});

        const downloadUrl = canvas.toDataURL(
            "image/png"
        );

        const link = document.createElement("a");
        const shortId = item.image_id.slice(0, 8);
        const opacitySuffix = overlayVisible
            ? `opacity_${{overlayOpacity.toFixed(2).replace(".", "_")}}`
            : "overlay_off";

        link.href = downloadUrl;
        link.download =
            `shap_{model_type}_${{shortId}}_${{opacitySuffix}}.png`;

        document.body.appendChild(link);
        link.click();
        link.remove();

    }} catch (error) {{
        console.error(
            "Failed to export current SHAP view:",
            error
        );

        alert(
            "Could not export the current view. " +
            "See the browser console for details."
        );

    }} finally {{
        exportButtonNode.disabled = false;
        exportButtonNode.textContent = "Export current view";
    }}
}}

function renderList(filterText = "") {{
    const query = filterText.trim().toLowerCase();
    listNode.innerHTML = "";

    let visibleCount = 0;

    items.forEach((item, index) => {{
        if (query && !itemText(item).includes(query)) {{
            return;
        }}

        visibleCount += 1;

        const button = document.createElement("button");
        button.type = "button";
        button.className = "image-item";
        button.dataset.index = String(index);

        if (index === selectedIndex) {{
            button.classList.add("active");
        }}

        const heading = document.createElement("strong");
        heading.textContent = item.image_id;

        const details = document.createElement("span");
        details.textContent =
            `${{item.split}} #${{item.split_index}} · ` +
            `True: ${{item.true_label}} · ` +
            `Pred: ${{item.predicted_label}} · ` +
            `P=${{item.probability.toFixed(3)}}`;

        button.appendChild(heading);
        button.appendChild(details);

        button.addEventListener("click", () => {{
            selectedIndex = index;
            renderList(searchNode.value);
            renderViewer(index);
        }});

        listNode.appendChild(button);
    }});

    if (visibleCount === 0) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No matching images.";
        listNode.appendChild(empty);
    }}
}}

function renderMeta(item) {{
    metaNode.innerHTML = "";

    const fields = [
        ["Image ID", item.image_id],
        ["Source", `${{item.split}} #${{item.split_index}}`],
        ["True", item.true_label],
        ["Predicted", item.predicted_label],
        ["P(pneumonia)", item.probability.toFixed(3)],
        ["Threshold", item.threshold.toFixed(3)],
        [
            "MLflow experiment",
            item.mlflow_experiment_name
                ? `${{item.mlflow_experiment_name}} (${{item.mlflow_experiment_id}})`
                : "not recorded"
        ],
        [
            "MLflow run",
            item.mlflow_run_id
                ? `${{item.mlflow_run_name || "unnamed"}} (${{item.mlflow_run_id}})`
                : "not recorded"
        ]
    ];

    fields.forEach(([label, value]) => {{
        const entry = document.createElement("span");
        const strong = document.createElement("strong");
        strong.textContent = `${{label}}: `;
        entry.appendChild(strong);
        entry.appendChild(document.createTextNode(value));
        metaNode.appendChild(entry);
    }});
}}

function renderViewer(index) {{
    const item = items[index];
    renderMeta(item);

    const overlayVisible = toggleNode.checked;
    const overlayOpacity = Number(opacityNode.value);

    const baseTrace = {{
        type: "heatmap",
        z: item.image,
        colorscale: [
            [0.0, "#000000"],
            [1.0, "#ffffff"]
        ],
        zmin: 0,
        zmax: 255,
        showscale: false,
        name: "Image",
        hovertemplate:
            "column=%{{x}}<br>" +
            "row=%{{y}}<br>" +
            "intensity=%{{z:.0f}}" +
            "<extra>Image</extra>"
    }};

    const shapTrace = {{
        type: "heatmap",
        z: item.shap,
        customdata: item.image,
        colorscale: [
            [0.0, "#3b4cc0"],
            [0.5, "#f2f2f2"],
            [1.0, "#b40426"]
        ],
        zmin: -item.max_abs,
        zmax: item.max_abs,
        zmid: 0,
        opacity: overlayOpacity,
        visible: overlayVisible && overlayOpacity > 0,
        name: "SHAP overlay",
        hovertemplate:
            "column=%{{x}}<br>" +
            "row=%{{y}}<br>" +
            "intensity=%{{customdata:.0f}}<br>" +
            "SHAP=%{{z:.4f}}" +
            "<extra>Attribution</extra>",
        colorbar: {{
            orientation: "h",
            x: 0.5,
            xanchor: "center",
            y: -0.16,
            yanchor: "top",
            len: 0.72,
            thickness: 18,
            title: {{
                text: "SHAP value",
                side: "top"
            }}
        }}
    }};

    const layout = {{
        autosize: true,
        margin: {{
            l: 25,
            r: 25,
            t: 30,
            b: 125
        }},
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        hovermode: "closest",
        xaxis: {{
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            constrain: "domain"
        }},
        yaxis: {{
            showgrid: false,
            zeroline: false,
            showticklabels: false,
            autorange: "reversed",
            scaleanchor: "x",
            scaleratio: 1
        }}
    }};

    const config = {{
        responsive: true,
        displaylogo: false,
        scrollZoom: true
    }};

    Plotly.react(
        plotNode,
        [baseTrace, shapTrace],
        layout,
        config
    );
}}

searchNode.addEventListener("input", () => {{
    renderList(searchNode.value);
}});

toggleNode.addEventListener("change", () => {{
    const opacity = Number(opacityNode.value);

    Plotly.restyle(
        plotNode,
        {{
            visible: toggleNode.checked && opacity > 0
        }},
        [1]
    );
}});

opacityNode.addEventListener("input", () => {{
    const value = Number(opacityNode.value);
    opacityValueNode.textContent = value.toFixed(2);

    Plotly.restyle(
        plotNode,
        {{
            opacity: value,
            visible: toggleNode.checked && value > 0
        }},
        [1]
    );
}});

exportButtonNode.addEventListener(
    "click",
    exportCurrentView
);

renderList();
renderViewer(0);
</script>
</body>
</html>
"""

    output_path.write_text(
        html,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------


def save_summary_csv(
    records: list[dict],
    model_type: str,
    output_path: Path,
) -> None:
    rows = []

    for record in records:
        metadata = record["metadata"]

        rows.append(
            {
                **metadata,
                "model": model_type,
                "pneumonia_probability": record["probability"],
                "decision_threshold": record["threshold"],
                "prediction": record["prediction"],
                "predicted_label": LABEL_NAMES[record["prediction"]],
                "mlflow_experiment_name": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_experiment_name"),
                "mlflow_experiment_id": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_experiment_id"),
                "mlflow_run_id": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_run_id"),
                "mlflow_run_name": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_run_name"),
                "mlflow_tracking_uri": record.get(
                    "model_provenance",
                    {},
                ).get("mlflow_tracking_uri"),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def save_pixel_csv(
    records: list[dict],
    output_path: Path,
) -> None:
    """
    Save pixel-level attribution rows incrementally.

    This avoids building one very large DataFrame in memory.
    """

    first = True

    for record in records:
        metadata = record["metadata"]
        image = record["image"]
        shap_map = record["shap_map"]

        rows = []

        for row in range(image.shape[0]):
            for column in range(image.shape[1]):
                rows.append(
                    {
                        "image_id": metadata["image_id"],
                        "row": row,
                        "column": column,
                        "pixel_intensity": float(
                            image[
                                row,
                                column,
                            ]
                        ),
                        "shap_value": float(
                            shap_map[
                                row,
                                column,
                            ]
                        ),
                    }
                )

        pd.DataFrame(rows).to_csv(
            output_path,
            mode=("w" if first else "a"),
            header=first,
            index=False,
        )

        first = False


# ---------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------


def log_explanation_to_mlflow(
    run_id: str,
    tracking_dir: Path,
    artifacts: list[Path],
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())

    with mlflow.start_run(run_id=run_id):
        for artifact in artifacts:
            mlflow.log_artifact(
                str(artifact),
                artifact_path="shap",
            )

    logger.info(
        "Logged SHAP artifacts to MLflow run {}",
        run_id,
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_explanation(
    image_selector: str,
    model_type: str,
    model_path: Path,
    database_path: Path,
    dataset_path: Path,
    output_dir: Path,
    background_size: int,
    output_format: str,
    save_pixel_data: bool,
    mlflow_run_id: str | None,
    mlflow_dir: Path,
    log_to_mlflow: bool,
) -> None:
    """
    Explain one image, a comma-delimited set of images, or all images.
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_format == "image":
        output_format = "png"

    metadata_rows = resolve_image_selection(
        database_path=database_path,
        image_selector=image_selector,
    )

    logger.info(
        "Resolved {} image(s) for SHAP explanation",
        len(metadata_rows),
    )

    images = load_selected_images(
        dataset_path=dataset_path,
        metadata_rows=metadata_rows,
    )

    if mlflow_run_id:
        model = load_mlflow_model(
            model_type=model_type,
            run_id=mlflow_run_id,
            tracking_dir=mlflow_dir,
        )

        threshold = get_mlflow_threshold(
            run_id=mlflow_run_id,
            tracking_dir=mlflow_dir,
        )

    else:
        model = load_local_model(model_path)

        threshold = get_local_threshold(model_type)

    records = build_explanation_records(
        metadata_rows=metadata_rows,
        images=images,
        model=model,
        model_type=model_type,
        threshold=threshold,
        dataset_path=dataset_path,
        background_size=background_size,
    )

    if mlflow_run_id:
        model_provenance = get_mlflow_run_provenance(
            run_id=mlflow_run_id,
            tracking_dir=mlflow_dir,
        )

    else:
        model_provenance = load_model_provenance(model_path)

    for record in records:
        record["model_provenance"] = model_provenance

    if model_provenance.get("mlflow_run_id"):
        logger.info(
            "Using model from MLflow experiment='{}' ({}), run='{}' ({})",
            model_provenance.get("mlflow_experiment_name"),
            model_provenance.get("mlflow_experiment_id"),
            model_provenance.get("mlflow_run_name"),
            model_provenance.get("mlflow_run_id"),
        )

    else:
        logger.info("No MLflow experiment/run provenance recorded for this model")

    generated_artifacts = []

    # -------------------------------------------------------------
    # Static image output: one PNG per selected image.
    # -------------------------------------------------------------

    if output_format in {
        "png",
        "both",
    }:
        for record in records:
            short_id = record["metadata"]["image_id"][:8]

            png_path = output_dir / f"shap_{model_type}_{short_id}.png"

            plot_explanation(
                record=record,
                model_type=model_type,
                output_path=png_path,
            )

            generated_artifacts.append(png_path)

            logger.success(
                "Saved static SHAP figure to {}",
                png_path,
            )

    # -------------------------------------------------------------
    # HTML output: one gallery containing all selected images.
    # -------------------------------------------------------------

    if output_format in {
        "html",
        "both",
    }:
        if len(records) == 1:
            suffix = records[0]["metadata"]["image_id"][:8]
        else:
            suffix = f"gallery_{len(records)}"

        html_path = output_dir / f"shap_{model_type}_{suffix}.html"

        save_interactive_gallery(
            records=records,
            model_type=model_type,
            output_path=html_path,
        )

        generated_artifacts.append(html_path)

        logger.success(
            "Saved interactive SHAP gallery to {}",
            html_path,
        )

    # -------------------------------------------------------------
    # Summary and optional pixel-level CSV data.
    # -------------------------------------------------------------

    if len(records) == 1:
        suffix = records[0]["metadata"]["image_id"][:8]
    else:
        suffix = f"batch_{len(records)}"

    summary_path = output_dir / f"shap_{model_type}_{suffix}_summary.csv"

    save_summary_csv(
        records=records,
        model_type=model_type,
        output_path=summary_path,
    )

    generated_artifacts.append(summary_path)

    logger.success(
        "Saved SHAP summary to {}",
        summary_path,
    )

    # Preserve the old one-image behavior automatically. For batches,
    # pixel-level CSV output is opt-in because 'all' can create millions
    # of rows.
    if len(records) == 1 or save_pixel_data:
        pixel_path = output_dir / f"shap_{model_type}_{suffix}_pixels.csv"

        save_pixel_csv(
            records=records,
            output_path=pixel_path,
        )

        generated_artifacts.append(pixel_path)

        logger.success(
            "Saved SHAP pixel data to {}",
            pixel_path,
        )

    elif len(records) > 1:
        logger.info("Skipping batch pixel CSV. Use --save-pixel-data " "to export it.")

    if log_to_mlflow:
        if not mlflow_run_id:
            raise ValueError("--log-to-mlflow requires --mlflow-run-id")

        log_explanation_to_mlflow(
            run_id=mlflow_run_id,
            tracking_dir=mlflow_dir,
            artifacts=generated_artifacts,
        )

    logger.success(
        "SHAP explanation completed for {} image(s)",
        len(records),
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pixel-level SHAP explanations for one image, "
            "a comma-delimited image-ID list, or all images."
        )
    )

    parser.add_argument(
        "--image-id",
        required=True,
        help=("One UUID, a comma-delimited UUID list, or 'all'."),
    )

    parser.add_argument(
        "--model",
        choices=[
            "logistic",
            "xgboost",
        ],
        default="xgboost",
    )

    parser.add_argument(
        "--model-file",
        type=Path,
        default=None,
        help="Optional local model path",
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
        default=None,
    )

    parser.add_argument(
        "--background-size",
        type=int,
        default=CONFIG.shap.background_size,
    )

    parser.add_argument(
        "--output-format",
        choices=[
            "png",
            "image",
            "html",
            "both",
        ],
        default=CONFIG.shap.default_output_format,
        help=(
            "png/image: one static image per selected record; "
            "html: one interactive gallery; both: generate both."
        ),
    )

    parser.add_argument(
        "--save-pixel-data",
        action="store_true",
        help=(
            "For multi-image selections, also save one combined "
            "pixel-level SHAP CSV. Single-image runs always save it."
        ),
    )

    parser.add_argument(
        "--mlflow-run-id",
        default=None,
    )

    parser.add_argument(
        "--mlflow-dir",
        type=Path,
        default=PATHS.mlflow_dir,
    )

    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
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
        default=PATHS.log_file("explain_shap"),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ensure_output_directories(CONFIG)

    configure_logging(
        console_level=args.log_level,
        log_file=args.log_file,
    )

    model_path = args.model_file or MODEL_FILES[args.model]

    output_dir = args.output_dir or PATHS.shap_model_dir(args.model)

    try:
        run_explanation(
            image_selector=args.image_id,
            model_type=args.model,
            model_path=model_path,
            database_path=args.database,
            dataset_path=args.dataset,
            output_dir=output_dir,
            background_size=args.background_size,
            output_format=args.output_format,
            save_pixel_data=args.save_pixel_data,
            mlflow_run_id=args.mlflow_run_id,
            mlflow_dir=args.mlflow_dir,
            log_to_mlflow=args.log_to_mlflow,
        )

    except Exception:
        logger.exception("SHAP explanation failed")
        raise
