import io
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import image_pipeline as pipeline


@pytest.fixture
def con(tmp_path):
    connection = pipeline.connect(tmp_path / "catalog.sqlite3")
    yield connection
    connection.close()


def png(color=100):
    buffer = io.BytesIO()
    Image.new("L", (28, 28), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_recursive_ingest_and_duplicate_cache(con, tmp_path):
    folder = tmp_path / "inputs"
    (folder / "nested").mkdir(parents=True)
    (folder / "first.PNG").write_bytes(png())
    (folder / "nested" / "copy.png").write_bytes(png())
    (folder / "bad.png").write_bytes(b"not an image")
    (folder / "notes.txt").write_text("not an image")
    counts = pipeline.ingest_directory(con, folder)
    assert counts == {"awaiting_model": 2, "failed": 1, "unsupported": 1}
    with patch.object(
        pipeline, "extract", side_effect=AssertionError("must reuse metadata")
    ):
        assert pipeline.ingest_directory(con, folder) == counts
    rows = pipeline.catalog(con)
    assert len(rows) == 4
    assert rows.loc[rows.filename == "first.PNG", "copies"].iloc[0] == 2
    assert con.execute("SELECT COUNT(*) FROM images").fetchone()[0] == 2


def test_retry_and_corrected_file(con, tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"broken")
    pipeline.ingest_path(con, path)
    pipeline.ingest_path(con, path)
    assert pipeline.catalog(con).iloc[0].attempts == 1
    pipeline.ingest_path(con, path, retry_failed=True)
    assert pipeline.catalog(con).iloc[0].attempts == 2
    path.write_bytes(png())
    assert pipeline.ingest_path(con, path) == "awaiting_model"
    assert len(pipeline.catalog(con)) == 1


def test_extension_mismatch_does_not_poison_valid_source(con):
    assert pipeline.ingest_bytes(con, png(), "x.jpg", "bad") == "unsupported"
    assert pipeline.ingest_bytes(con, png(), "x.png", "good") == "awaiting_model"
    assert (
        pipeline.catalog(con).set_index("path").loc["bad", "processing_status"]
        == "unsupported"
    )


def test_multiframe_and_limits(con, tmp_path):
    out = io.BytesIO()
    Image.new("L", (5, 5)).save(
        out, format="TIFF", save_all=True, append_images=[Image.new("L", (5, 5), 100)]
    )
    assert pipeline.ingest_bytes(con, out.getvalue(), "x.tiff", "frames") == "failed"
    with patch.object(pipeline, "MAX_PIXELS", 100):
        assert pipeline.ingest_bytes(con, png(), "x.png", "large") == "failed"
    missing = tmp_path / "missing.png"
    assert pipeline.ingest_path(con, missing) == "failed"
    assert pipeline.catalog(con).set_index("path").loc[str(missing), "image_id"] is None


def test_real_classifier_and_versioned_inference(con, tmp_path):
    rng = np.random.default_rng(42)
    X = rng.integers(0, 256, (20, 784)).astype(np.float32)
    y = np.array([0, 1] * 10)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    model_file, metrics = tmp_path / "model.joblib", tmp_path / "metrics.csv"
    joblib.dump(model, model_file)
    pd.DataFrame([{"threshold": 0.4}]).to_csv(metrics, index=False)
    classifier = pipeline.Classifier(model_file, metrics)
    pipeline.ingest_bytes(con, png(), "a.png", "a")
    with patch.object(pipeline, "extract", side_effect=AssertionError("cached")):
        assert pipeline.ingest_bytes(con, png(), "a.png", "a", classifier) == "complete"
    row = pipeline.catalog(con).iloc[0]
    score = model.predict_proba(np.full((1, 784), 100, dtype=np.float32))[0, 1]
    assert row.pneumonia_score == pytest.approx(score)
    assert row.predicted_class == ("pneumonia" if score >= 0.4 else "normal")
    with patch.object(classifier, "predict", side_effect=AssertionError("cached")):
        assert pipeline.ingest_bytes(con, png(), "b.png", "b", classifier) == "complete"
    pd.DataFrame([{"threshold": 0.6}]).to_csv(metrics, index=False)
    updated = pipeline.Classifier(model_file, metrics)
    assert updated.version != classifier.version
    pipeline.ingest_bytes(con, png(), "a.png", "a", updated)
    assert set(pipeline.catalog(con).model_version) == {updated.version}


def test_npz_bridge(con, tmp_path):
    path = tmp_path / "dataset.npz"
    arrays = {}
    for split in ("train", "val", "test"):
        arrays[f"{split}_images"] = np.full((2, 28, 28), 120, dtype=np.uint8)
        arrays[f"{split}_labels"] = np.array([[0], [1]])
    np.savez(path, **arrays)
    assert pipeline.ingest_npz(con, path) == {"awaiting_model": 6}
    rows = pipeline.catalog(con)
    # One image with conflicting labels: every copy is excluded by dedup.
    assert set(rows.split) == {pipeline.EXCLUDED_SPLIT}
    assert rows.image_id.nunique() == 1
    assert set(rows.label) == {0, 1}  # Preserve conflicting source labels for audit.


def test_exif_orientation_and_color_metadata(con):
    buffer = io.BytesIO()
    image = Image.new("RGB", (20, 10), "red")
    exif = image.getexif()
    exif[274] = 6
    image.save(buffer, format="JPEG", exif=exif)
    pipeline.ingest_bytes(con, buffer.getvalue(), "rotated.jpg", "rotation")
    row = pipeline.catalog(con).iloc[0]
    assert (row.width, row.height) == (10, 20)
    assert row.color_mode == "RGB"
    assert row.file_size == len(buffer.getvalue())
    assert row.aspect_ratio == 0.5
    assert "274" in row.exif


def test_inference_failure_and_explicit_retry(con):
    class BrokenClassifier:
        version = "model-v1"

        def predict(self, features):
            raise RuntimeError("temporary inference error")

    classifier = BrokenClassifier()
    assert pipeline.ingest_bytes(con, png(), "a.png", "a", classifier) == "model_error"
    with patch.object(classifier, "predict") as predict:
        pipeline.ingest_bytes(con, png(), "a.png", "a", classifier)
        predict.assert_not_called()
        predict.return_value = {
            "predicted_class": "normal",
            "confidence": 0.8,
            "pneumonia_score": 0.2,
            "model_version": classifier.version,
        }
        with patch.object(
            pipeline, "extract", side_effect=AssertionError("metadata should survive")
        ):
            assert (
                pipeline.ingest_bytes(
                    con, png(), "a.png", "a", classifier, retry_failed=True
                )
                == "complete"
            )
        predict.assert_called_once()


def test_catalog_categories_for_npz_and_external_images(con, tmp_path):
    rng = np.random.default_rng(0)
    train = rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)
    val = rng.integers(0, 256, (6, 28, 28), dtype=np.uint8)
    test = rng.integers(0, 256, (6, 28, 28), dtype=np.uint8)
    test[0] = train[0]  # cross-split duplicate: one copy must be excluded
    labels = lambda n: (np.arange(n) % 2).reshape(-1, 1)
    npz = tmp_path / "toy.npz"
    np.savez(
        npz,
        train_images=train,
        train_labels=labels(20),
        val_images=val,
        val_labels=labels(6),
        test_images=test,
        test_labels=np.vstack([[0], labels(6)[1:]]),
    )
    pipeline.ingest_npz(con, npz)
    pipeline.ingest_bytes(con, png(), "upload.png", "upload:1")

    rows = pipeline.catalog(con)
    counts = rows.split.value_counts().to_dict()

    assert counts[pipeline.EXCLUDED_SPLIT] == 1
    assert counts[pipeline.EXTERNAL_SPLIT] == 1
    assert sum(counts.get(s, 0) for s in pipeline.MODEL_SPLITS) == 31
    excluded = rows[rows.split == pipeline.EXCLUDED_SPLIT]
    assert excluded.filename.iloc[0] == "test_000000.png"


def test_legacy_null_split_reads_as_external(con):
    pipeline.ingest_bytes(con, png(), "old.png", "upload:old")
    with con:
        con.execute("UPDATE sources SET split=NULL")
    assert pipeline.catalog(con).split.iloc[0] == pipeline.EXTERNAL_SPLIT
