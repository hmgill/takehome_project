# Image Explorer — PneumoniaMNIST pipeline

A small local application that discovers or accepts images, validates and processes them, extracts metadata, applies a trained classifier, persists structured results, and exposes them through **Streamlit**.

The main application uses Streamlit widgets and native charts. Plotly is not required for the application. The original optional SHAP HTML exporter is retained for compatibility in the experiment dependencies; PNG remains its default.

## Quick start (Python 3.12)

Run commands from the project directory. Use a virtual environment:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` is the single, pinned dependency file for the application, optional experiments (MLflow, XGBoost, SHAP, Albumentations) and tests.

### Providing the dataset

**The dataset is never included in this repository or in the Docker image.** PneumoniaMNIST is public, but here it stands in for private data, so every data file and every artifact derived from it (DuckDB/SQLite databases, metadata CSVs, QC figures, fitted models, logs) is excluded by `.gitignore` and `.dockerignore`.

Download the **28×28 PneumoniaMNIST** archive from the [official MedMNIST project](https://medmnist.com/) and keep it anywhere outside the project. Point the pipeline at it with an environment variable:

```bash
# macOS/Linux
export PNEUMONIAMNIST_NPZ=/path/to/pneumoniamnist.npz
# Windows PowerShell
$env:PNEUMONIAMNIST_NPZ = "C:\path\to\pneumoniamnist.npz"
```

`PNEUMONIAMNIST_NPZ` overrides `[paths].dataset` in `config.toml` for every script. Each script also still accepts an explicit `--dataset`/`--input`/`--npz` flag. Do not substitute another resolution without updating the model contract.

### Steps 1–3: existing research workflow

```bash
# 1. Validate the NPZ, derive metadata, and build the research DuckDB database.
python src/build_database.py

# 2. Produce SQL-based dataset analysis and QC outputs.
python src/analyze.py

# 3. Fit the logistic-regression baseline; select threshold on validation,
#    then write test metrics and the fitted model.
python src/model.py
```

### Process images and launch the application

```bash
# Import all three NPZ splits and classify their images with the trained baseline.
python src/image_pipeline.py --npz "$PNEUMONIAMNIST_NPZ"

# Or recursively ingest an ordinary image directory (including audit failures).
python src/image_pipeline.py --directory /path/to/images

# Launch the interactive explorer.
python -m streamlit run app.py
```

To inspect uploads before training, launch the app directly, or use `--metadata-only` on the ingestion CLI. These images have status `awaiting_model`, not invented predictions. Once training finishes, repeat ingestion to classify cached features without extracting metadata again. Use `--retry-failed` to retry failed decoding or inference; modified files are detected automatically.

The UI includes upload processing, paginated thumbnails, filename/path search, class/status/format/split filters, width and height limits, file-size limits, score thresholds, sorting, duplicate-only views, class/status distributions, image details, EXIF, and CSV export. Unclassified images remain visible by default and can be excluded. The minimum score refers to the assigned class, not necessarily pneumonia.

Server-side directory and configured-NPZ buttons are hidden by default. For trusted local use, set `IMAGE_EXPLORER_LOCAL_IMPORT=1` before launch. `IMAGE_CATALOG_DB` selects another catalog for the app; the CLI equivalent is `--database`. Never put the catalog inside the input directory, which deliberately audits every discovered file.

## Running with Docker

The image contains code and dependencies only. The NPZ is bind-mounted read-only at run time, and generated databases/models live in named volumes, so no data is baked into the image.

```powershell
docker build -t pneumoniamnist-explorer .

# Tests use synthetic data and need no dataset
docker run --rm pneumoniamnist-explorer python -m pytest -q

# Full workflow with Compose
$env:NPZ_PATH = "C:\path\to\pneumoniamnist.npz"
docker compose run --rm explorer python src/build_database.py
docker compose run --rm explorer python src/analyze.py
docker compose run --rm explorer python src/model.py
docker compose run --rm explorer python src/image_pipeline.py --npz /input/pneumoniamnist.npz
docker compose up
```

Then open <http://localhost:8501>. `docker compose down -v` removes the generated volumes.

Before pushing, run `python tools/check_no_private_data.py --git` to confirm nothing data-derived is tracked.

## Design and requirement coverage

| Requirement | Implementation |
|---|---|
| Recursive discovery and upload | `src/image_pipeline.py` shared by CLI and `app.py`; uploads never become filesystem paths |
| Supported types | PNG, JPEG, BMP, single-frame TIFF, WebP; case-insensitive extensions and decoded-format matching |
| Validation | Pillow verify **and** full decode; corrupt files, unexpected extensions, format mismatches and multiframe files recorded separately; limits of 25 MiB and 25 million pixels |
| Metadata | SHA-256 content ID, source filename/path, encoded byte size, decoded format, orientation-adjusted dimensions/aspect ratio, original color mode, megapixels, EXIF JSON, UTC time, status, error, attempts and pipeline version |
| Processing | EXIF orientation correction, RGB thumbnail, grayscale 28×28 bilinear resize and float32 pixel features |
| Classification | Existing trained StandardScaler + logistic regression; validation-derived threshold; class score, pneumonia score and model fingerprint |
| Structured storage | SQLite application catalog with indexed, related content and source tables; existing research DuckDB/CSV outputs retained |
| Exploration | Streamlit filters, thumbnails, details, metrics, native charts, uploads and CSV export |
| Incremental processing | Content-addressed metadata/features, idempotent source upserts, explicit failure retries and model-version-aware inference |

`images` has one row per SHA-256 of encoded file bytes; `sources` has one row per source path or upload identity. Identical bytes at two paths retain two source records but share processing. Same-name uploads with different contents are distinct. Repeat uploads with identical name and bytes are idempotent. A modified filesystem path points to its latest content; unreferenced historical content remains in `images`. No automatic deletion or historical attempt ledger is implemented.

Failed reads have source records without a hash. Files that decode unsuccessfully have a hashed content record with the exception and extraction attempt count. Extension rejection belongs to the source, so a wrongly named copy cannot invalidate a correctly named copy. Statuses are `complete`, `awaiting_model`, `failed`, `model_error`, and `unsupported`. Successful metadata is reused after a classifier failure. SQLite writes each content/source update atomically and uses WAL with a busy timeout for local concurrent sessions.

Exact duplication means identical **encoded bytes**, not perceptual similarity. Different encodings of the same image will be separate catalog contents. The original NPZ QC separately hashes decoded arrays to identify exact pixel duplicates across splits. NPZ import creates deterministic PNG representations; their format/size describe those generated PNGs, **not original hospital files**. Original filenames, acquisition details and EXIF cannot be recovered from the NPZ. Labels and splits stay on source records and never feed inference.

SQLite suits small incremental writes and is included with Python. DuckDB remains the analytical store for the original assessment, avoiding changes to its SQL schema. The Streamlit app reads SQLite only; importing the NPZ is the explicit bridge between the two workflows. Thumbnails and inference features are stored as BLOBs; original uploads are not retained. Detail views show a thumbnail, not a full-resolution diagnostic image.

## Model architecture, tradeoffs and limitations

The default is a **custom linear baseline**, not a pretrained neural network. It standardizes each of 784 grayscale pixels using training statistics, then learns a binary logistic-regression decision boundary. The validation split chooses a Youden-J threshold; test data is reserved for evaluation. The persisted scaler is reused for inference, with raw pixel values on the same 0–255 scale as training. Predictions identify the model and threshold files by a combined SHA-256 fingerprint; changing either refreshes inference on subsequent ingestion.

This is cheap to train and run on CPU, easy to reproduce and straightforward to explain. Flattened pixels discard explicit spatial structure and are sensitive to acquisition, crop, orientation and intensity differences. Resizing arbitrary uploads does not make them in-domain. The model has no modality detector or out-of-distribution rejection: a photograph can receive a chest-X-ray label. This is an exploratory engineering demo, not a clinical tool. Confidence is the uncalibrated score of the threshold-selected class; with a non-0.5 threshold, that score can be below 0.5. It is not a validated probability of disease.

A missing model produces an explicit pending state. An incompatible model or absent/invalid threshold surfaces an error. Only load trusted locally generated joblib files (joblib uses pickle); users cannot upload model artifacts through the app. EXIF can contain sensitive information: the local details panel exposes it, while the default CSV omits EXIF and source paths. A hosted multi-user product would need authentication, isolation and retention controls.

## Research/data engineering judgment (Part 4)

**Data leakage.** Worry about the same patient, repeated studies, adjacent slices, exact/near duplicates, or augmented derivatives crossing splits; preprocessing fitted before splitting; site/device/text markers correlated with labels; label-derived features; temporal leakage; and tuning repeatedly against test results. The NPZ permits pixel-hash overlap and label-conflict checks, but lacks patient/study identifiers, acquisition dates, hospital/device provenance and original headers. It cannot rule these risks out. Split by patient/study (and evaluate site/time holdouts where appropriate), fit transforms only on training data, and keep test results out of selection. The existing optional candidate selector ranks validation ROC AUC. Candidate scripts still report test metrics, so repeatedly comparing those reports would compromise the final evaluation.

**Scaling — first three changes.**

1. Replace full rescans with a durable arrival manifest/queue and stable source IDs/object versions; keep original objects in durable storage and acknowledge arrivals only after recording processing state.
2. Separate bounded decoding and batched inference workers, with resource limits, retry/backoff, dead-letter handling and processing latency/failure monitoring.
3. Replace local SQLite writes and whole-catalog UI reads with an indexed shared metadata store, idempotent transactional upserts, server-side pagination/aggregation, and separately stored thumbnails/features. Retain analytical exports for batch queries.

**Incremental processing.** For tomorrow's 10,000 arrivals, hash the new arrivals and look up `(content hash, pipeline version)`; reuse metadata/features for known contents and compute missing `(content hash, model version)` predictions. A new source ID identifies a new arrival, an existing byte hash identifies an exact duplicate, and a persisted failure state identifies a retry candidate. This implementation follows that pattern locally. Re-running a directory still reads bytes to hash every file, and NPZ import still enumerates/encodes every array; it avoids repeated extraction/inference, not discovery or hashing. At a million images, the arrival manifest is the first priority. Metadata version changes invalidate extraction, while model/threshold changes invalidate only inference. Production retry history should record attempt timestamps, error categories and backoff, beyond the current extraction count/latest error.

## Project layout and optional experiments

- `app.py`: Streamlit explorer.
- `src/image_pipeline.py`: application ingestion, inference and SQLite catalog.
- `src/build_database.py`, `src/analyze.py`, `sql/analysis.sql`: original NPZ research/QC path.
- `src/model.py`, `src/model_utils.py`: baseline training, metrics and provenance.
- `src/model_svm.py`, `src/model_xgboost.py`, `src/select_best_model.py`, `src/create_predictions.py`, `src/explain_shap.py`: optional model comparison, batch predictions and SHAP artifacts.
- `config.toml`, `src/project_config.py`: shared dataset and experiment paths.
- `tests/`: ingestion, inference, NPZ bridge and Streamlit smoke tests.
- `tools/check_no_private_data.py`: fails if dataset files or derived artifacts are tracked by git (`--git`) or present in a directory (used by the Docker build).
- `Dockerfile`, `compose.yaml`: container build and run configuration.

The explorer deliberately uses the logistic baseline; it does not silently adopt an optional experiment winner. See `CONFIG_AND_OUTPUTS.md` for experiment outputs. All optional experiment dependencies are included in `requirements.txt`. Existing scripts expose their arguments with `--help`.

## Verification

```bash
python -m pytest -q
```

Tests cover nested discovery, unsupported/corrupt files, exact duplicates, cached processing, retries, replacement at a path, format mismatch, frame/pixel limits, real scikit-learn inference, model-version invalidation, NPZ split/label preservation, and Streamlit empty/populated/search states. Tests use synthetic inputs to verify mechanics, not medical model performance. No measured real-dataset accuracy is claimed for this delivery.

Streamlit API reference: [file uploads](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader), [app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

Delivery verification: the original metadata → SQL analysis → baseline training → catalog inference workflow completed on 64 synthetic images. Real-dataset evaluation was not run because the provided archive did not include the NPZ or trained weights.


## SHAP overlays in Streamlit

Select an image in **Images → Image details**, open **Explain prediction · SHAP**,
and click **Explain prediction**. The viewer displays the exact 28×28 grayscale
model input, signed attributions and an aligned overlay. A PNG download is
provided. This works with existing fitted logistic baselines; no retraining or
NPZ is required for viewer explanations. Invalid/unclassified images cannot be
explained. If the model/threshold fingerprint or stored score is stale, re-ingest
that image before explaining it.

### Correctness and interpretation

The baseline is a StandardScaler followed by binary LogisticRegression. For a
standardized pixel `z_j`, the exact **interventional SHAP value** is
`phi_j = coefficient_j * (z_j - background_mean_j)`. These values explain
**class 1 (pneumonia) log-odds**, not changes in probability and not the predicted
class's score. Red increases pneumonia log-odds; blue decreases them, including
when the predicted class is normal. The validation-selected classification
threshold does not change these contributions.

The viewer uses the full training population mean retained by the fitted
StandardScaler. That mean becomes zero after centering; the baseline log-odds is
therefore the learned intercept. If train-only augmentation was used, the
reference includes the augmented training population. No validation, test or
uploaded images enter this background. For this linear interventional game,
only the background feature means are needed, so the closed-form implementation
does not need a SHAP runtime dependency. Its formula follows the
[SHAP LinearExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html).

Every explanation checks `baseline + sum(phi) == decision_function(input)` and
`sigmoid(baseline + sum(phi)) == predict_proba(input)[1]` within numerical
tolerances. The viewer additionally verifies the stored model fingerprint and
stored pneumonia score. It shows the baseline, contribution sum, reconstructed
log-odds, pneumonia score and additivity residual. The sigmoid of baseline
log-odds is not necessarily the mean training probability.

`src/linear_shap.py` is shared by the viewer and the existing logistic SHAP
exporter. The exporter still uses its explicitly sampled training background;
its attributions can consequently differ from the viewer's full-training-mean
reference. This is an intentional reference choice, not a probability/log-odds
conversion. The exporter now also checks numerical reconstruction. The optional
XGBoost exporter is unchanged and is not supported by the viewer.

The overlay uses model-resolution pixels with no interpolation. The color range
is symmetric about zero and scaled per image; overlay opacity reflects absolute
contribution so zero-valued pixels remain transparent. Colors should not be
compared quantitatively across images without checking their color bars. Pixel
dependence is ignored: these are not correlation-dependent SHAP values, causal
claims, or anatomical lesion masks.

### Additional verification

The tests independently compare contributions with an exhaustive coalition-based
Shapley calculation on a small model and with the SHAP library's LinearExplainer.
They also cover a constant feature, full-training and explicit sampled
backgrounds, model-output reconstruction, invalid input, stale model/score
rejection, PNG generation, and the Streamlit explanation button. The SHAP
library is a development/test dependency only for these reference checks.
