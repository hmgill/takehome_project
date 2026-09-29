# Image Explorer: PneumoniaMNIST pipeline

A local application that ingests images, validates them, extracts metadata, classifies them with a trained baseline, stores the results in a structured catalog, and lets you explore everything in a **Streamlit** app with per-pixel SHAP explanations.

> This is an exploratory engineering demo, not a clinical tool.

**Contents:** [Walkthrough video](#walkthrough-video) · [Quick start](#quick-start) · [Data handling](#data-handling) · [Pipeline](#pipeline) · [Using the app](#using-the-app) · [Configuration](#configuration) · [Design](#design) · [Limitations](#limitations) · [Research and data-engineering notes](#research-and-data-engineering-notes) · [Publishing the image](#publishing-the-image) · [Project layout](#project-layout) · [Testing](#testing)

---

## Walkthrough video

<!-- TODO: add the walkthrough video. -->
*Coming soon.* A short walkthrough of the pipeline, the Docker setup and the Streamlit app will be linked here.

---

## Quick start

The dataset is **not** included. Download the 28×28 PneumoniaMNIST archive (`pneumoniamnist.npz`) from the [MedMNIST project](https://medmnist.com/) and keep it anywhere outside this folder. See [Data handling](#data-handling) for why.

### Option A: Docker (recommended)

Requires Docker Desktop (or Docker Engine with Compose).

**1. Build the image and run the tests.** The tests use synthetic images and need no dataset.

```bash
docker build -t pneumoniamnist-explorer .
docker run --rm pneumoniamnist-explorer python -m pytest -q
```

**2. Tell Compose where the NPZ is.** Create a `.env` file next to `compose.yaml` (it is git-ignored):

```
NPZ_PATH=C:\path\to\pneumoniamnist.npz
```

Or set it for the current shell instead:

| Shell | Command |
|---|---|
| Command Prompt | `set "NPZ_PATH=C:\path\to\pneumoniamnist.npz"` |
| PowerShell | `$env:NPZ_PATH = "C:\path\to\pneumoniamnist.npz"` |
| macOS/Linux | `export NPZ_PATH=/path/to/pneumoniamnist.npz` |

**3. Run the pipeline and start the app.**

```bash
docker compose run --rm explorer python src/build_database.py
docker compose run --rm explorer python src/analyze.py
docker compose run --rm explorer python src/model.py
docker compose run --rm explorer python src/image_pipeline.py --npz /input/pneumoniamnist.npz
docker compose up
```

**4. Open the app** at <http://localhost:8501>.

Results (QC tables, figures, model metrics, logs) are written to `./output` on your machine. The catalog databases live in a Docker volume. `docker compose down -v` removes that volume; delete `./output` yourself when you are finished.

> **Host vs container paths.** `NPZ_PATH` is a path on your machine. Anything passed to a script runs *inside* the container, which only sees the mounted copy at `/input/pneumoniamnist.npz`. A path like `D:\...` will not be found there.

### Option B: Published image from Docker Hub

The image is published as [`hmgill/pneumoniamnist-explorer`](https://hub.docker.com/r/hmgill/pneumoniamnist-explorer). Like a local build, it contains code and dependencies only, with no dataset and no trained model. It still needs your own NPZ.

**With a clone of this repo**, add the image name to `.env` alongside `NPZ_PATH`:

```
NPZ_PATH=C:\path\to\pneumoniamnist.npz
IMAGE=hmgill/pneumoniamnist-explorer:1.0.0
```

Then run `docker compose pull` instead of building, and continue from step 3 of Option A. Use `docker compose up --no-build` to start the app.

**Without the repo**, use plain `docker run`. This example is for Command Prompt; use `` ` `` for line breaks in PowerShell or `\` on macOS/Linux. First create a named volume for the catalog databases:

```cmd
docker volume create pneumonia-data
```

Then run each step with the same mounts:

```cmd
docker run --rm ^
  -v "C:\path\to\pneumoniamnist.npz:/input/pneumoniamnist.npz:ro" ^
  -v pneumonia-data:/app/data ^
  -v "%cd%\output:/app/output" ^
  hmgill/pneumoniamnist-explorer:1.0.0 python src/build_database.py
```

Repeat that command with `python src/analyze.py`, `python src/model.py` and `python src/image_pipeline.py --npz /input/pneumoniamnist.npz` in place of `python src/build_database.py`. Finally start the app with the same mounts, plus `-p 8501:8501` and `-e IMAGE_EXPLORER_LOCAL_IMPORT=1`, and no command at the end.

### Option C: Local Python (3.12)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

export PNEUMONIAMNIST_NPZ=/path/to/pneumoniamnist.npz   # Windows: see Configuration

python src/build_database.py
python src/analyze.py
python src/model.py
python src/image_pipeline.py --npz "$PNEUMONIAMNIST_NPZ"
python -m streamlit run app.py
```

`requirements.txt` is the single, pinned dependency file. It covers the app, the optional experiments (MLflow, XGBoost, SHAP, Albumentations) and the tests.

---

## Data handling

PneumoniaMNIST is public, but in this project it **stands in for private data**. The rule is that no dataset file, and nothing derived from it, leaves the machine it is processed on.

- **Git.** `.gitignore` excludes the NPZ, all databases, fitted models, and everything under `data/` and `output/`. That covers metadata CSVs with per-image hashes, QC figures that render real images, models whose scaler stores the mean training image, and logs containing local paths.
- **Docker.** `.dockerignore` is an allowlist, so only source code enters the build context. The NPZ is mounted read-only at run time and is never copied into the image.
- **Guard script.** `tools/check_no_private_data.py` detects dataset archives by content as well as by name, so a renamed copy is still caught. The Docker build runs it and fails on a match. Run it before every push:

  ```bash
  python tools/check_no_private_data.py --git
  ```

---

## Pipeline

| Step | Command | What it does | Main outputs |
|---|---|---|---|
| 1. Build | `src/build_database.py` | Validates the NPZ (arrays, labels, 28×28 dimensions, missing values), derives per-image metadata, and flags low-variance images and cross-split duplicates | `data/pneumoniamnist.duckdb`, metadata CSV |
| 2. Analyze | `src/analyze.py` | Runs the SQL analysis views in `sql/analysis.sql` for class balance, image characteristics, duplicates and unusual images | `output/analysis/` |
| 3. Train | `src/model.py` | Fits the StandardScaler + logistic-regression baseline, selects a Youden-J threshold on validation, and evaluates on test | `output/models/logistic/` |
| 4. Ingest | `src/image_pipeline.py` | Loads the NPZ splits or an image folder into the app catalog and classifies each image | `data/image_catalog.sqlite3` |
| 5. Explore | `streamlit run app.py` | Interactive explorer | — |

Useful ingestion flags:

- `--directory PATH` ingests a folder of ordinary images recursively instead of the NPZ.
- `--metadata-only` catalogs images before a model exists. They get status `awaiting_model`, not invented predictions.
- `--retry-failed` retries decoding or inference failures. Modified files are detected automatically.

Every script documents its options with `--help`.

---

## Using the app

- **Add images:** upload PNG, JPEG, BMP, TIFF or WebP files from the sidebar.
- **Filter and sort:**
  - Search by filename or path.
  - Filter by class, status, split, outcome (correct, false positive or false negative), format, minimum score, dimensions and file size.
  - Show duplicates only.
- **Browse:** a paginated thumbnail gallery, a table view, class and status distributions, and per-split model metrics.
- **Inspect an image:**
  - Prediction, score and outcome versus the ground-truth label.
  - Full metadata and EXIF.
  - A **SHAP overlay** with a Show overlay toggle, an Opacity slider and hover values.
  - **Save snapshot** downloads the current view as a PNG.
- **Export:** download the filtered catalog as CSV. EXIF and source paths are omitted by default.

Server-side folder and NPZ import buttons are hidden unless `IMAGE_EXPLORER_LOCAL_IMPORT=1` is set. Compose enables them, and inside a container they only see the container's filesystem.

---

## Configuration

Shared paths and settings live in `config.toml`. The following environment variables override them:

| Variable | Purpose | Default |
|---|---|---|
| `PNEUMONIAMNIST_NPZ` | Dataset location for all scripts (overrides `[paths].dataset`) | `data/pneumoniamnist.npz` |
| `NPZ_PATH` | *Compose only:* host path of the NPZ to mount | required |
| `OUTPUT_DIR` | *Compose only:* host folder for results | `./output` |
| `IMAGE` | *Compose only:* image to run, e.g. the Docker Hub image | `pneumoniamnist-explorer:latest` |
| `IMAGE_CATALOG_DB` | SQLite catalog used by the app (the CLI uses `--database`) | `data/image_catalog.sqlite3` |
| `IMAGE_EXPLORER_LOCAL_IMPORT` | Set to `1` to show the server-side import buttons | off |

Windows shells set these variables differently; see the table in [Option A](#option-a-docker-recommended). See `CONFIG_AND_OUTPUTS.md` for the full output layout.

---

## Design

### Catalog storage

The app uses a **SQLite** catalog with two related tables:

- `images` has one row per SHA-256 hash of the encoded file bytes. It holds metadata, the thumbnail, the inference features and predictions.
- `sources` has one row per path or upload identity. It holds the filename, split and label.

This design has several consequences:

- Identical bytes at two paths get two source records but are processed once.
- Re-uploading the same file is idempotent.
- A modified file at an existing path points to its new content.
- Labels and splits live on source records and never feed inference.
- Statuses are `complete`, `awaiting_model`, `failed`, `model_error` and `unsupported`.
- Writes are atomic per image. WAL mode with a busy timeout supports concurrent local sessions.

**Validation** runs Pillow's verify step *and* a full decode. Corrupt files, extension or format mismatches and multi-frame files are recorded separately. Images are limited to 25 MiB and 25 million pixels. **Processing** corrects EXIF orientation, stores an RGB thumbnail, and resizes to 28×28 grayscale float32 features.

**DuckDB** remains the analytical store for the research workflow (steps 1–3). SQLite suits small incremental writes and ships with Python. NPZ ingestion (step 4) is the explicit bridge between the two stores.

"Duplicate" means identical encoded bytes, not perceptual similarity. The research QC separately hashes decoded pixel arrays to find exact duplicates across splits. NPZ images are stored as deterministic PNGs, so their format and size describe those PNGs, not original hospital files.

### Model

The model is a **linear baseline**: a StandardScaler over the 784 pixels followed by binary logistic regression.

- The decision threshold is chosen on the validation split (Youden's J). The test split is used only for evaluation.
- Predictions record a SHA-256 fingerprint of the model and threshold files. Changing either triggers re-inference on the next ingestion.
- The baseline is cheap to train on CPU, reproducible and easy to explain.
- The app always uses this baseline. It does not silently switch to a winner from the optional experiments (`model_svm.py`, `model_xgboost.py`, `select_best_model.py`).

### SHAP explanations

For a linear model, exact interventional SHAP values have a closed form:

`phi_j = coef_j × (z_j − background_mean_j)`

Here `z_j` is the standardized pixel value. The background is the full training mean stored by the scaler. That mean is zero after standardization, so the baseline value is the model's intercept.

- **What the values mean.** They explain **pneumonia log-odds** (class 1). Red pixels raise the pneumonia score and blue pixels lower it, whatever the predicted class. The decision threshold does not affect them.
- **Built-in checks.** Every explanation verifies that `intercept + Σphi` reproduces `decision_function`, and that its sigmoid reproduces `predict_proba`. The viewer also rejects explanations whose model fingerprint or stored score is stale.
- **Display.** The color scale is symmetric and scaled per image, so colors are not comparable across images. The 28×28 attribution grid is upsampled for display; hover values are per model pixel.
- **Assumptions.** The values treat pixels as independent. They are not causal claims or lesion masks.
- **Exporter differences.** `src/linear_shap.py` is shared with the optional batch exporter (`explain_shap.py`). The exporter uses a sampled training background, so its values can differ slightly from the viewer's. The XGBoost exporter is not supported in the viewer.

The closed form follows the [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html). The tests check it against both an exhaustive Shapley calculation and the SHAP library.

---

## Limitations

- **Spatial structure.** Flattened pixels discard spatial structure and are sensitive to crop, orientation, acquisition and intensity differences.
- **Out-of-domain inputs.** Resizing an arbitrary upload does not make it in-domain. There is no modality or out-of-distribution check, so a photograph can receive a chest-X-ray label.
- **Scores are not probabilities.** The displayed score is the uncalibrated score of the threshold-selected class. With a non-0.5 threshold it can be below 0.5. It is not a validated probability of disease.
- **Trusted model files only.** Model files are joblib, which uses pickle, so only load trusted, locally generated models. The app does not accept model uploads.
- **EXIF privacy.** EXIF can contain sensitive information. It is shown in the local details panel but excluded from the default CSV.
- **Local use only.** A hosted, multi-user deployment would need authentication, isolation and retention controls.

---

## Research and data-engineering notes

**Data leakage.** The main risks are:

- The same patient, repeated studies, near-duplicates or augmented copies crossing splits.
- Preprocessing fitted before the split.
- Site, device or text markers that correlate with labels.
- Temporal leakage.
- Repeated tuning against test results.

The NPZ supports pixel-hash overlap and label-conflict checks. It has no patient, study, date, site or device information, so these risks cannot be ruled out. The remedies are:

- Split by patient or study, and evaluate site and time holdouts where appropriate.
- Fit transforms on training data only.
- Keep test results out of model selection. The optional selector ranks on validation ROC AUC, but the candidate scripts still report test metrics, so repeatedly comparing those reports would compromise the final evaluation.

**Scaling: the first three changes.**

1. Replace full rescans with a durable arrival manifest or queue and stable source IDs. Keep originals in durable storage, and acknowledge arrivals only after their processing state is recorded.
2. Separate bounded decoding workers from batched inference workers. Add resource limits, retry with backoff, dead-letter handling, and latency and failure monitoring.
3. Move from local SQLite and whole-catalog UI reads to a shared, indexed metadata store. Use transactional upserts, server-side pagination and aggregation, and separate storage for thumbnails and features.

**Incremental processing.** For a batch of new arrivals:

- Hash each file and look up `(content hash, pipeline version)` to reuse existing metadata and features.
- Compute only the missing `(content hash, model version)` predictions.
- A new source ID marks a new arrival, an existing hash marks a duplicate, and a stored failure state marks a retry candidate.
- Metadata-version changes invalidate extraction, while model or threshold changes invalidate only inference.

The current implementation follows this pattern locally. It still re-reads and hashes every file on a rescan, so at a million images the arrival manifest becomes the first priority. Production retry history should also record timestamps, error categories and backoff.

---

## Publishing the image

This section is for maintainers. The image is safe to publish because it is built only from the allowlisted source files, and the build runs the data-leak guard. Before pushing a new version, confirm three things.

**1. The guard passes inside the built image:**

```bash
docker run --rm pneumoniamnist-explorer python tools/check_no_private_data.py /app
```

**2. No dataset-named files exist anywhere in the image.** This should print nothing:

```bash
docker run --rm pneumoniamnist-explorer find / -xdev -iname "*pneumonia*" -not -path "/proc/*"
```

**3. The layer history shows no unexpected content.** Every step should come from the Dockerfile, and the single `COPY` layer should be small:

```bash
docker history --no-trunc pneumoniamnist-explorer
```

Then tag and push:

```bash
docker login
docker tag pneumoniamnist-explorer hmgill/pneumoniamnist-explorer:1.0.0
docker tag pneumoniamnist-explorer hmgill/pneumoniamnist-explorer:latest
docker push hmgill/pneumoniamnist-explorer:1.0.0
docker push hmgill/pneumoniamnist-explorer:latest
```

As an extra precaution, build from a clean clone rather than a working folder that has data in it. `.dockerignore` and the guard keep data out of the image either way.

---

## Project layout

```
app.py                      Streamlit explorer
config.toml                 Shared paths and settings
requirements.txt            Pinned dependencies (app, experiments, tests)
Dockerfile, compose.yaml    Container build and run configuration
sql/analysis.sql            Research analysis views
src/
  build_database.py         Step 1: NPZ validation and metadata
  analyze.py                Step 2: SQL analysis and QC outputs
  model.py, model_utils.py  Step 3: baseline training, metrics, provenance
  image_pipeline.py         Step 4: ingestion, inference, SQLite catalog
  linear_shap.py            Closed-form linear SHAP (shared)
  model_svm.py, model_xgboost.py, select_best_model.py,
  create_predictions.py, explain_shap.py     Optional experiments
  project_config.py, runtime.py, utils.py, ...  Shared helpers
tests/                      Pipeline, SHAP and Streamlit tests
tools/check_no_private_data.py   Data-leak guard (git and Docker)
data/, output/              Generated locally; never committed
```

---

## Testing

```bash
python -m pytest -q                                            # local
docker run --rm pneumoniamnist-explorer python -m pytest -q    # Docker
```

All 13 tests pass on Python 3.12, both locally and in the Docker image. They use synthetic inputs and check the pipeline's mechanics, not medical performance:

- Nested discovery, unsupported and corrupt files, format mismatch, and frame and pixel limits.
- Exact duplicates, cached processing, retries, and replacement at a path.
- Real scikit-learn inference, model-version invalidation, and NPZ split and label preservation.
- SHAP correctness against exhaustive Shapley values and the SHAP library, including stale-model rejection.
- Streamlit empty, populated and search states, and the overlay controls.

The full workflow (all 5,030 images through build → analyze → train → ingest → app) has also been run end to end in the container. Real-dataset metrics are written to `output/models/logistic/` and are intentionally not committed.