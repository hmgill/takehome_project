# PneumoniaMNIST Take Home Project

<p align="center">
  <img src="media/pmnist_logo.jpg" alt="PneumoniaMNIST Image Explorer" width="350">
</p>

**Contents:**  [Overview](#overview) · [Dataset](#dataset) · [Run from Docker Hub](#run-from-docker-hub) · [Build and run with Docker](#build-and-run-with-docker) · [Run locally](#run-locally) · [Scripts](#scripts) · [Extra things to try](#extra-things-to-try) · [Configuration](#configuration) · [Tests](#tests)

## Overview
A small end-to-end pipeline for working with the PneumoniaMNIST chest X-ray dataset. It validates the source data, builds a metadata database, runs a few QC and analysis steps, trains a simple classifier, and makes the results browsable in a Streamlit app.

---

## Dataset

**Important:** the PneumoniaMNIST dataset itself is deliberately excluded and must be downloaded separately.

Download the 28×28 PneumoniaMNIST archive (`pneumoniamnist.npz`) from [MedMNIST](https://medmnist.com/) and keep it outside the repository. The dataset location can be set through an environment variable or mounted into the Docker container.

---

## Run from Docker Hub

If you just want to run the finished project, the Docker Hub image is the quickest option.

Windows Command Prompt:

```cmd
docker run --rm -p 8501:8501 ^
  -v "C:\path\to\pneumoniamnist.npz:/input/pneumoniamnist.npz:ro" ^
  -v "%cd%\output:/app/output" ^
  hmgill/pneumoniamnist-explorer:1.4.0
```

For PowerShell, use `` ` `` for line continuation. On macOS/Linux, use `\`.

Once the log shows `Starting Streamlit`, open <http://localhost:8501>. Press Ctrl+C when you want to stop it.

A few useful variations:

| If you want to... | Do this |
|---|---|
| Leave no generated output on the host | Remove the `output` mount |
| Force every pipeline step to rerun | Append `python src/run_all.py --force` |
| Run the test suite | `docker run --rm hmgill/pneumoniamnist-explorer:1.4.0 python -m pytest -q` |

---

## Build and run with Docker

To build the image yourself:

```bash
docker build -t pneumoniamnist-explorer .
docker run --rm pneumoniamnist-explorer python -m pytest -q
```

For Docker Compose, create a `.env` file next to `compose.yaml`:

```text
NPZ_PATH=C:\path\to\pneumoniamnist.npz
```

Then run:

```bash
docker compose up --build
```

Open <http://localhost:8501>. Generated results are written to `./output`.

| Task | Command |
|---|---|
| Stop the stack | `docker compose down` |
| Stop and delete the database volume | `docker compose down -v` |
| Run one script | `docker compose run --rm explorer python src/model.py` |
| Use the Docker Hub image instead of building | Add `IMAGE=hmgill/pneumoniamnist-explorer:1.4.0` to `.env`, then run `docker compose pull` and `docker compose up --no-build` |

One Docker-specific detail: paths passed to scripts inside the container should use the container path `/input/pneumoniamnist.npz`, not the path on your host machine.

---

## Run locally

The project is set up for Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Tell the project where the dataset lives:

| Shell | Command |
|---|---|
| Command Prompt | `set "PNEUMONIAMNIST_NPZ=C:\path\to\pneumoniamnist.npz"` |
| PowerShell | `$env:PNEUMONIAMNIST_NPZ = "C:\path\to\pneumoniamnist.npz"` |
| macOS/Linux | `export PNEUMONIAMNIST_NPZ=/path/to/pneumoniamnist.npz` |

To run the main pipeline and launch the app:

```bash
python src/run_all.py
```

You can also run each step on its own:

```bash
python src/build_database.py
python src/analyze.py
python src/model.py
python src/image_pipeline.py --npz "$PNEUMONIAMNIST_NPZ"
python -m streamlit run app.py
```

---

## Scripts

Every script supports `--help`. Most also accept `--log-level` and `--log-file`.

### Main pipeline

| Script | What it does | Useful options |
|---|---|---|
| `src/run_all.py` | Runs the pipeline (also training the model set in `[app].model` if it isn't the baseline), skips outputs that already exist, re-scores the catalog if the model changed, then starts the app | `--force`, `--no-app` |
| `src/build_database.py` | Validates the NPZ and builds the metadata database | `--input`, `--output`, `--csv-output` |
| `src/analyze.py` | Runs the SQL analysis and writes QC outputs | `--database`, `--dataset`, `--sql`, `--output-dir` |
| `src/model.py` | Tunes, trains and evaluates the logistic-regression baseline | `--cv`, `--cv-folds`, `--n-iter`, `--augment`, `--mlflow`, `--mlflow-dir` |
| `src/image_pipeline.py` | Ingests images into the app catalog and classifies them, or re-scores the catalog with another model | `--npz PATH`, `--directory PATH` or `--rescore`; `--model`, `--database`, `--retry-failed`, `--metadata-only` |
| `app.py` | Starts the Streamlit explorer with `python -m streamlit run app.py` | — |

### Optional modeling and explanation scripts

| Script | What it does |
|---|---|
| `src/model_svm.py` | Runs an RBF-SVM experiment with randomized hyperparameter search |
| `src/model_xgboost.py` | Runs a CPU-only XGBoost experiment with randomized hyperparameter search |
| `src/select_best_model.py` | Compares fitted candidates using held-out validation ROC AUC |
| `src/create_predictions.py` | Exports per-image predictions for one or more fitted models |
| `src/explain_shap.py` | Generates SHAP explanations for individual images or batches |

Outputs go to `data/` and `output/`. See `CONFIG_AND_OUTPUTS.md` for the full layout.

---

## Extra things to try

The default path through the project keeps the modeling intentionally simple, but there are a few extra pieces in the repo if you want to dig further.

### Track runs with MLflow

MLflow tracking is on by default (`[mlflow].enabled = true` in `config.toml`), so every baseline, SVM, and XGBoost training run is recorded, including the Docker pipeline's baseline run. To skip tracking for one run, pass `--no-mlflow`:

```bash
python src/model.py --no-mlflow
```

Runs are stored locally under `output/mlflow/`. To browse them:

```bash
mlflow ui --backend-store-uri ./output/mlflow
```

Then open <http://127.0.0.1:5000>.

The tracked runs include model parameters, evaluation metrics, saved artifacts, model provenance, and the data partition used (split sizes, deduplication counts, a split fingerprint, and the split manifest under `splits/`). This makes it easier to compare experiments without having to keep separate notes or manually match output folders to a particular run.

### Compare classifier models

The required baseline is logistic regression, but the repo also includes RBF-SVM and XGBoost experiments. All three are tuned the same way: the same seeded randomized search (`[modeling].search_iterations` settings), scored by ROC AUC, then refit on the training split only. The validation split then picks the decision threshold, and the test split is scored once.

```bash
python src/model.py
python src/model_svm.py
python src/model_xgboost.py
python src/select_best_model.py
```

`select_best_model.py` compares whichever fitted candidates are available and writes a model comparison table plus a selected-model manifest under `output/model_selection/`.

Model selection is based on held-out validation ROC AUC. The test split is not used to choose the model.

#### Holdout tuning (default) or k-fold cross-validation

By default each candidate setting is fit on the training split and scored on the validation split, a plain train/val/test holdout. With about 5,800 images, that is enough for a stable estimate and keeps runs fast.

To score each setting by stratified k-fold CV inside the training split instead, set `[modeling].cross_validation = true` in `config.toml` (folds from `cv_folds`), or pass `--cv` to any of the three scripts:

```bash
python src/model.py --cv --cv-folds 5
python src/model_svm.py --cv
python src/model_xgboost.py --cv
```

Use the same mode for every candidate before running `select_best_model.py` so they're compared like for like. Each model's metrics CSV records `tuning_roc_auc` (plus `training_cv_roc_auc` in CV mode), and MLflow records the `tuning_method`.

This is also a useful place to look at whether a more flexible classifier is actually buying much over the simple baseline, rather than assuming that the more complicated model will be better.

### Choose the model the app uses

The image catalog and Streamlit app classify with logistic regression by default. To use another trained model, set `[app].model` in `config.toml` or the `IMAGE_EXPLORER_MODEL` environment variable:

| Value | Model |
|---|---|
| `logistic` | Logistic-regression baseline (default) |
| `svm` | RBF SVM |
| `xgboost` | XGBoost |
| `best` | Whichever model `select_best_model.py` selected |

```bash
export IMAGE_EXPLORER_MODEL=xgboost
python src/run_all.py           # trains it if needed, re-scores the catalog, starts the app
```

Or, step by step:

```bash
python src/model_xgboost.py
python src/image_pipeline.py --rescore --model xgboost
IMAGE_EXPLORER_MODEL=xgboost python -m streamlit run app.py
```

`--rescore` re-predicts stored features with the selected model without re-reading any image, and only touches predictions that came from a different model. The app shows which model is active and offers a **Re-score** button if any stored predictions came from another model. The Model metrics tab only counts predictions from the active model.

Notes:

- The SVM is trained without probability estimates. Its margin and threshold are mapped through a sigmoid so its scores sit on the same 0–1 scale as the others, with identical decisions. Treat its score as a ranking, not a calibrated probability.
- The SHAP overlay is exact for logistic regression (linear SHAP) and XGBoost (TreeSHAP). It isn't available for the SVM, which has no exact, fast explainer.

With Docker, pass the variable through: `docker run -e IMAGE_EXPLORER_MODEL=xgboost ...`, or add `IMAGE_EXPLORER_MODEL=xgboost` to `.env` for Compose.

### Export per-image predictions

Once models have been trained, you can generate prediction tables for a closer look at false positives, false negatives, and disagreements between models:

```bash
python src/create_predictions.py --split test --models all
```

You can also pass a subset, for example:

```bash
python src/create_predictions.py --split test --models logistic,xgboost
```

The exported CSVs make it easier to inspect individual mistakes and compare how different classifiers behave on the same images.

### Try light training augmentation

The training scripts support a small set of image augmentations applied only to the training split:

```bash
python src/model.py --augment
python src/model_xgboost.py --augment
```

The augmentation settings live in `config.toml`.

This is mainly useful as an experiment rather than something assumed to improve the result. With small 28×28 medical images, it is worth checking whether augmentation actually helps validation performance rather than applying it automatically.

### Generate SHAP explanations

For supported models, SHAP can be used to inspect pixel-level contributions for one image or a batch of images:

```bash
python src/explain_shap.py --image-id <IMAGE_ID> --model xgboost --output-format both
```

Use `--image-id all` for a batch run.

For large batches, pixel-level CSV export is opt-in because it can produce a fairly large amount of data.

---

## Configuration

Most settings live in `config.toml`. These environment variables can override the common paths and Docker settings:

| Variable | Purpose | Default |
|---|---|---|
| `PNEUMONIAMNIST_NPZ` | Dataset path used by the scripts | `data/pneumoniamnist.npz` |
| `IMAGE_CATALOG_DB` | App catalog database | `data/image_catalog.sqlite3` |
| `IMAGE_EXPLORER_MODEL` | Model the catalog and app classify with: `logistic`, `svm`, `xgboost` or `best` | `[app].model` (`logistic`) |
| `IMAGE_EXPLORER_LOCAL_IMPORT` | Set to `1` to show server-side import controls in the app | off locally, on in Docker |
| `NPZ_PATH` | Compose: host path to the NPZ | required |
| `OUTPUT_DIR` | Compose: host folder for generated results | `./output` |
| `IMAGE` | Compose: image to run | `pneumoniamnist-explorer:latest` |

---

## Tests

Run the test suite locally with:

```bash
python -m pytest -q
```

Or against the Docker image:

```bash
docker run --rm pneumoniamnist-explorer python -m pytest -q
```
