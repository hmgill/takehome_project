"""Shared, incremental image ingestion and inference for CLI and Streamlit."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.special import expit

from project_config import CONFIG, MODEL_NAMES, PATHS

APP_DATABASE = PATHS.data_dir / "image_catalog.sqlite3"
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 25_000_000
EXTENSIONS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}
PIPELINE_VERSION = "1"

# Catalog `split` values for images that are not in the model's
# train/val/test partition (see data_splits.py):
#   external - uploaded or folder-ingested images from outside the NPZ
#   excluded - NPZ images removed by deduplication (redundant duplicate
#              copies, or duplicate groups with conflicting labels)
EXTERNAL_SPLIT = "external"
EXCLUDED_SPLIT = "excluded"
MODEL_SPLITS = ("train", "val", "test")


def now():
    return datetime.now(timezone.utc).isoformat()


def connect(database=APP_DATABASE):
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(database, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY, file_size INTEGER NOT NULL,
            format TEXT, width INTEGER, height INTEGER, aspect_ratio REAL,
            color_mode TEXT, megapixels REAL, exif TEXT, thumbnail BLOB,
            features BLOB, processing_status TEXT NOT NULL, error TEXT,
            predicted_class TEXT, confidence REAL, pneumonia_score REAL,
            model_version TEXT, processed_at TEXT NOT NULL,
            attempts INTEGER NOT NULL, pipeline_version TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            path TEXT PRIMARY KEY, filename TEXT NOT NULL,
            image_id TEXT REFERENCES images(image_id), split TEXT, label INTEGER,
            source_status TEXT NOT NULL, source_error TEXT, seen_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS source_image ON sources(image_id);
    """)
    return con


class Classifier:
    """Load one locally trained model; never accept uploaded joblib files.

    Works with any binary model trained by this project. Models with
    ``predict_proba`` (logistic regression, XGBoost) score in probability
    space. The SVM is trained without probability estimates, so its
    decision-function margin and its validation threshold are both mapped
    through a sigmoid. That keeps every prediction on the same 0-1 scale for
    the app without changing any decision (the map is monotonic), but the
    SVM's score is an uncalibrated ranking score, not a probability.
    """

    def __init__(self, model_path, metrics_path, name="logistic"):
        model_path, metrics_path = Path(model_path), Path(metrics_path)
        self.name = str(name)
        self.model = joblib.load(model_path)
        threshold = float(pd.read_csv(metrics_path).iloc[0]["threshold"])
        if list(getattr(self.model, "classes_", [])) != [0, 1] or getattr(
            self.model, "n_features_in_", None
        ) != 784:
            raise ValueError(
                "Expected a binary PneumoniaMNIST model with 784 pixel features"
            )
        if hasattr(self.model, "predict_proba"):
            self.score_kind = "probability"
            if not 0 <= threshold <= 1:
                raise ValueError("Decision threshold must be a probability in [0, 1]")
        elif hasattr(self.model, "decision_function"):
            self.score_kind = "margin"
            if not np.isfinite(threshold):
                raise ValueError("Decision threshold must be finite")
        else:
            raise ValueError("Model exposes neither predict_proba nor decision_function")
        self.raw_threshold = threshold
        self.threshold = threshold if self.score_kind == "probability" else float(
            expit(threshold)
        )
        self.version = hashlib.sha256(
            self.name.encode() + model_path.read_bytes() + metrics_path.read_bytes()
        ).hexdigest()

    def raw_scores(self, X):
        X = np.asarray(X, dtype=np.float32).reshape(-1, 784)
        if self.score_kind == "probability":
            return np.asarray(self.model.predict_proba(X)[:, 1], dtype=np.float64)
        return np.asarray(self.model.decision_function(X), dtype=np.float64).reshape(-1)

    def scores(self, X):
        """Pneumonia scores on a 0-1 scale (see the class docstring)."""
        raw = self.raw_scores(X)
        return raw if self.score_kind == "probability" else expit(raw)

    def predict_batch(self, X):
        raw = self.raw_scores(X)
        scores = raw if self.score_kind == "probability" else expit(raw)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError("Model returned an invalid score")
        results = []
        for value, score in zip(raw, scores):
            positive = value >= self.raw_threshold
            results.append(
                {
                    "predicted_class": "pneumonia" if positive else "normal",
                    "confidence": float(score if positive else 1 - score),
                    "pneumonia_score": float(score),
                    "model_version": self.version,
                }
            )
        return results

    def predict(self, features):
        return self.predict_batch(np.asarray(features).reshape(1, -1))[0]


def resolve_model_name(name=None):
    """Explicit name, else IMAGE_EXPLORER_MODEL / config.toml [app].model.

    "best" resolves to the model recorded by src/select_best_model.py.
    """
    name = str(name or CONFIG.app.model).strip().lower()
    if name == "best":
        manifest = PATHS.model_selection_dir / "selected_model.json"
        if not manifest.exists():
            raise FileNotFoundError(
                "App model is 'best' but no model selection exists yet. "
                "Run src/select_best_model.py after training the candidates."
            )
        name = str(json.loads(manifest.read_text())["selected_model"]).lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
    return name


def load_classifier(name=None):
    """Load the configured model. Raises with a clear reason if it can't."""
    resolved = resolve_model_name(name)
    model = PATHS.model_file(resolved)
    if not model.exists():
        raise FileNotFoundError(
            f"No trained {resolved} model at {model}. Train it first."
        )
    return Classifier(model, PATHS.model_metrics_file(resolved), name=resolved)


def default_classifier(name=None):
    """The configured classifier, or None if that model hasn't been trained."""
    try:
        return load_classifier(name)
    except FileNotFoundError:
        return None


def extract(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as probe:
            if probe.width * probe.height > MAX_PIXELS:
                raise ValueError("Image exceeds 25 million pixels")
            if probe.format not in set(EXTENSIONS.values()):
                raise ValueError(f"Unsupported decoded format: {probe.format}")
            if getattr(probe, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are not supported")
            probe.verify()
        with Image.open(io.BytesIO(data)) as original:
            original.load()
            fmt, mode = original.format, original.mode
            # Keep EXIF values readable and JSON serializable; no original files in exports.
            exif = json.dumps(
                {str(k): str(v)[:2000] for k, v in original.getexif().items()}
            )
            oriented = ImageOps.exif_transpose(original)
            width, height = oriented.size
            features = np.asarray(
                oriented.convert("L").resize((28, 28), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            thumb = oriented.convert("RGB")
            thumb.thumbnail((256, 256))
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG")
    return {
        "format": fmt,
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "color_mode": mode,
        "megapixels": width * height / 1_000_000,
        "exif": exif,
        "thumbnail": buffer.getvalue(),
        "features": features.tobytes(),
    }


def ingest_bytes(
    con,
    data,
    filename,
    source,
    classifier=None,
    retry_failed=False,
    split=None,
    label=None,
):
    """One transaction per source. Byte-identical files share extraction and predictions."""
    if split is None:
        split = EXTERNAL_SPLIT
    digest = hashlib.sha256(data).hexdigest()
    extension = Path(filename).suffix.lower()
    source_status, source_error = "accepted", None
    if extension not in EXTENSIONS:
        source_status, source_error = "unsupported", "Unsupported extension"
    elif len(data) > MAX_FILE_BYTES:
        source_status, source_error = "unsupported", "File exceeds 25 MiB"
    old = con.execute("SELECT * FROM images WHERE image_id=?", (digest,)).fetchone()
    row = (
        dict(old)
        if old
        else {"image_id": digest, "file_size": len(data), "attempts": 0}
    )
    needs_extract = (
        not old
        or old["pipeline_version"] != PIPELINE_VERSION
        or (retry_failed and old["processing_status"] == "failed")
    )
    # An unsupported filename must not poison a valid copy of the same bytes.
    if source_status == "accepted" and (
        needs_extract or row.get("processing_status") == "unsupported"
    ):
        row.update(
            processing_status="awaiting_model",
            error=None,
            attempts=row["attempts"] + 1,
            processed_at=now(),
            pipeline_version=PIPELINE_VERSION,
            predicted_class=None,
            confidence=None,
            pneumonia_score=None,
            model_version=None,
        )
        try:
            row.update(extract(data))
        except Exception as exc:
            row.update(processing_status="failed", error=f"{type(exc).__name__}: {exc}")
    elif not old:
        row.update(
            processing_status="unsupported",
            error=source_error,
            processed_at=now(),
            pipeline_version=PIPELINE_VERSION,
        )
    if (
        source_status == "accepted"
        and row.get("format")
        and row["format"] != EXTENSIONS[extension]
    ):
        source_status, source_error = (
            "unsupported",
            "Extension does not match decoded format",
        )
    needs_prediction = (
        classifier is not None
        and source_status == "accepted"
        and row.get("features") is not None
        and row["processing_status"] != "failed"
        and (
            row.get("model_version") != classifier.version
            or row["processing_status"] == "awaiting_model"
            or (retry_failed and row["processing_status"] == "model_error")
        )
    )
    if needs_prediction:
        row.update(
            predicted_class=None,
            confidence=None,
            pneumonia_score=None,
            model_version=classifier.version,
            processed_at=now(),
        )
        try:
            row.update(
                classifier.predict(np.frombuffer(row["features"], dtype=np.float32))
            )
            row.update(processing_status="complete", error=None)
        except Exception as exc:
            row.update(
                processing_status="model_error", error=f"{type(exc).__name__}: {exc}"
            )
    columns = list(row)
    with con:
        con.execute(
            f"INSERT INTO images ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            "ON CONFLICT(image_id) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in columns if c != "image_id"),
            list(row.values()),
        )
        con.execute(
            """INSERT INTO sources VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                    filename=excluded.filename,image_id=excluded.image_id,split=excluded.split,label=excluded.label,
                    source_status=excluded.source_status,source_error=excluded.source_error,seen_at=excluded.seen_at""",
            (
                source,
                filename,
                digest,
                split,
                label,
                source_status,
                source_error,
                now(),
            ),
        )
    return source_status if source_status != "accepted" else row["processing_status"]


def ingest_path(con, path, classifier=None, retry_failed=False):
    path = Path(path).resolve()
    try:
        # Bound the read even if the file grows after discovery.
        with path.open("rb") as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File exceeds 25 MiB")
    except (OSError, ValueError) as exc:
        with con:
            con.execute(
                """INSERT INTO sources VALUES (?,?,NULL,?,NULL,'failed',?,?)
                        ON CONFLICT(path) DO UPDATE SET image_id=NULL,source_status='failed',
                        source_error=excluded.source_error,seen_at=excluded.seen_at""",
                (str(path), path.name, EXTERNAL_SPLIT, str(exc), now()),
            )
        return "failed"
    return ingest_bytes(con, data, path.name, str(path), classifier, retry_failed)


def ingest_directory(con, directory, classifier=None, retry_failed=False):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    # Includes unexpected extensions so they appear as audit records. Skip symlinks.
    results = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            results.append(ingest_path(con, path, classifier, retry_failed))
    return pd.Series(results, dtype="str").value_counts().to_dict()


def ingest_npz(con, dataset_path, classifier=None, retry_failed=False):
    """
    Bridge existing NPZ experiments to the same image catalog; labels are audit-only.

    Each image is cataloged under the split it was assigned by the shared
    deduplicated partition (``data_splits``), not its official NPZ split.
    Copies removed by deduplication are cataloged as ``excluded``.
    """
    from data_splits import load_dataset_splits

    partition = load_dataset_splits(Path(dataset_path))
    assigned = {
        (str(source_split), int(source_index)): split
        for split in MODEL_SPLITS
        for source_split, source_index in zip(
            partition.get(split).source_split,
            partition.get(split).source_index,
        )
    }
    results = []
    with np.load(dataset_path, allow_pickle=False) as dataset:
        for split in ("train", "val", "test"):
            images = dataset[f"{split}_images"]
            labels = dataset[f"{split}_labels"].reshape(-1)
            if len(images) != len(labels):
                raise ValueError(f"Mismatched {split} images/labels")
            if (
                images.dtype != np.uint8
                or images.ndim not in (3, 4)
                or images.shape[1:3] != (28, 28)
            ):
                raise ValueError("Expected uint8 28x28 PneumoniaMNIST images")
            for i, (array, label) in enumerate(zip(images, labels)):
                array = array.squeeze()
                if array.shape != (28, 28) or label not in (0, 1):
                    raise ValueError("Invalid grayscale image or binary label")
                buffer = io.BytesIO()
                Image.fromarray(array).save(buffer, format="PNG")
                filename = f"{split}_{i:06d}.png"
                results.append(
                    ingest_bytes(
                        con,
                        buffer.getvalue(),
                        filename,
                        f"{Path(dataset_path).resolve()}::{split}:{i}",
                        classifier,
                        retry_failed,
                        assigned.get((split, i), EXCLUDED_SPLIT),
                        int(label),
                    )
                )
    return pd.Series(results, dtype="str").value_counts().to_dict()


def stale_prediction_count(con, classifier):
    """Images whose stored prediction did not come from this classifier."""
    if classifier is None:
        return 0
    return con.execute(
        """SELECT COUNT(*) FROM images WHERE features IS NOT NULL
           AND processing_status IN ('complete','awaiting_model')
           AND (model_version IS NULL OR model_version != ?)""",
        (classifier.version,),
    ).fetchone()[0]


def rescore_catalog(con, classifier, retry_failed=False, batch_size=2048):
    """Re-predict stored features with the current model; no files are re-read.

    Only rows whose prediction came from a different model (or none yet) are
    touched, so this is cheap to run every time and safe to repeat. Rows that
    failed decoding have no features and are left alone.
    """
    statuses = ["complete", "awaiting_model"] + (["model_error"] if retry_failed else [])
    rows = con.execute(
        f"""SELECT image_id, features FROM images WHERE features IS NOT NULL
            AND processing_status IN ({','.join('?' for _ in statuses)})
            AND (model_version IS NULL OR model_version != ?
                 OR processing_status != 'complete')""",
        (*statuses, classifier.version),
    ).fetchall()
    counts = {"rescored": 0, "model_error": 0}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        X = np.stack([np.frombuffer(r["features"], dtype=np.float32) for r in chunk])
        try:
            results = classifier.predict_batch(X)
            errors = [None] * len(chunk)
        except Exception:
            # Isolate the bad row(s) instead of failing the whole batch.
            results, errors = [], []
            for x in X:
                try:
                    results.append(classifier.predict(x))
                    errors.append(None)
                except Exception as exc:
                    results.append(None)
                    errors.append(f"{type(exc).__name__}: {exc}")
        stamp = now()
        with con:
            for row, result, error in zip(chunk, results, errors):
                if error is None:
                    con.execute(
                        """UPDATE images SET predicted_class=?, confidence=?,
                           pneumonia_score=?, model_version=?, processing_status='complete',
                           error=NULL, processed_at=? WHERE image_id=?""",
                        (result["predicted_class"], result["confidence"],
                         result["pneumonia_score"], result["model_version"], stamp,
                         row["image_id"]),
                    )
                    counts["rescored"] += 1
                else:
                    con.execute(
                        """UPDATE images SET predicted_class=NULL, confidence=NULL,
                           pneumonia_score=NULL, model_version=?, processing_status='model_error',
                           error=?, processed_at=? WHERE image_id=?""",
                        (classifier.version, error, stamp, row["image_id"]),
                    )
                    counts["model_error"] += 1
    return counts


def catalog(con):
    return pd.read_sql_query(
        f"""SELECT s.path,s.filename,COALESCE(s.split,'{EXTERNAL_SPLIT}') split,s.label,i.image_id,i.file_size,
        i.format,i.width,i.height,i.aspect_ratio,i.color_mode,i.megapixels,i.exif,
        CASE WHEN s.source_status='accepted' THEN i.processing_status ELSE s.source_status END processing_status,
        COALESCE(s.source_error,i.error) error,
        CASE WHEN s.source_status='accepted' THEN i.predicted_class END predicted_class,
        CASE WHEN s.source_status='accepted' THEN i.confidence END confidence,
        CASE WHEN s.source_status='accepted' THEN i.pneumonia_score END pneumonia_score,
        i.model_version,i.processed_at,i.attempts,i.pipeline_version,
        (SELECT COUNT(*) FROM sources d WHERE d.image_id=s.image_id) copies
        FROM sources s LEFT JOIN images i ON i.image_id=s.image_id""",
        con,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--npz", type=Path)
    source.add_argument(
        "--rescore",
        action="store_true",
        help="re-predict every cataloged image with the current model",
    )
    parser.add_argument("--database", type=Path, default=APP_DATABASE)
    parser.add_argument(
        "--model",
        choices=[*MODEL_NAMES, "best"],
        default=None,
        help="model to classify with (default: IMAGE_EXPLORER_MODEL or config.toml [app].model)",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.rescore and args.metadata_only:
        parser.error("--rescore needs a model; drop --metadata-only")
    classifier = None if args.metadata_only else default_classifier(args.model)
    if classifier is None and not args.metadata_only:
        print(f"No trained {resolve_model_name(args.model)} model; cataloging without predictions")
    con = connect(args.database)
    try:
        if args.rescore:
            if classifier is None:
                raise SystemExit("Cannot rescore: the configured model is not trained")
            result = {"model": classifier.name,
                      **rescore_catalog(con, classifier, args.retry_failed)}
        else:
            runner = ingest_directory if args.directory else ingest_npz
            result = runner(con, args.directory or args.npz, classifier, args.retry_failed)
        print(json.dumps(result, indent=2))
    finally:
        con.close()


if __name__ == "__main__":
    main()
