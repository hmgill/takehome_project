# PneumoniaMNIST Take Home Project

<p align="center">
  <img src="media/pmnist_logo.jpg" alt="PneumoniaMNIST Image Explorer" width="350">
</p>
A small end-to-end pipeline for working with the PneumoniaMNIST chest X-ray dataset. It validates the source data, builds a metadata database, runs a few QC and analysis steps, trains a simple classifier, and makes the results browsable in a Streamlit app.


**Important:** the PneumoniaMNIST dataset itself is deliberately excluded and must be downloaded separately.

**Contents:**  [Dataset](#dataset) · [Run from Docker Hub](#run-from-docker-hub) · [Build and run with Docker](#build-and-run-with-docker) · [Run locally](#run-locally) · [Scripts](#scripts) · [Extra things to try](#extra-things-to-try) · [Configuration](#configuration) · [Tests](#tests)


---

## Dataset

Download the 28×28 PneumoniaMNIST archive (`pneumoniamnist.npz`) from [MedMNIST](https://medmnist.com/) and keep it outside the repository. You can pass its location in through an environment variable or mount it into the Docker container.

---

## Run from Docker Hub

If you just want to run the finished project, the Docker Hub image is the quickest option.

Windows Command Prompt:

```cmd
docker run --rm -p 8501:8501 ^
  -v "C:\path\to\pneumoniamnist.npz:/input/pneumoniamnist.npz:ro" ^
  -v "%cd%\output:/app/output" ^
  hmgill/pneumoniamnist-explorer:1.1.0
```

For PowerShell, use `` ` `` for line continuation. On macOS/Linux, use `\`.

Once the log shows `Starting Streamlit`, open <http://localhost:8501>. Press Ctrl+C when you want to stop it.

A few useful variations:

| If you want to... | Do this |
|---|---|
| Leave no generated output on the host | Remove the `output` mount |
| Force every pipeline step to rerun | Append `python src/run_all.py --force` |
| Run the test suite | `docker run --rm hmgill/pneumoniamnist-explorer:1.1.0 python -m pytest -q` |

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
| Use the Docker Hub image instead of building | Add `IMAGE=hmgill/pneumoniamnist-explorer:1.1.0` to `.env`, then run `docker compose pull` and `docker compose up --no-build` |

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
| `src/run_all.py` | Runs steps 1–4, skips outputs that already exist, then starts the app | `--force`, `--no-app` |
| `src/build_database.py` | Validates the NPZ and builds the metadata database | `--input`, `--output`, `--csv-output` |
| `src/analyze.py` | Runs the SQL analysis and writes QC outputs | `--database`, `--dataset`, `--sql`, `--output-dir` |
| `src/model.py` | Trains and evaluates the logistic-regression baseline | `--dataset`, `--output-dir`, `--augment`, `--mlflow`, `--mlflow-dir` |
| `src/image_pipeline.py` | Ingests images into the app catalog and classifies them | `--npz PATH` or `--directory PATH`, `--database`, `--retry-failed`, `--metadata-only` |
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

Add `--mlflow` to the baseline, SVM, or XGBoost training command:

```bash
python src/model.py --mlflow
python src/model_svm.py --mlflow
python src/model_xgboost.py --mlflow
```

Runs are stored locally under `output/mlflow/`. To browse them:

```bash
mlflow ui --backend-store-uri ./output/mlflow
```

Then open <http://127.0.0.1:5000>.

The tracked runs include model parameters, evaluation metrics, saved artifacts, and model provenance. This makes it easier to compare experiments without having to keep separate notes or manually match output folders to a particular run.

### Compare classifier models

The required baseline is logistic regression, but the repo also includes RBF-SVM and XGBoost experiments:

```bash
python src/model.py
python src/model_svm.py
python src/model_xgboost.py
python src/select_best_model.py
```

`select_best_model.py` compares whichever fitted candidates are available and writes a model comparison table plus a selected-model manifest under `output/model_selection/`.

Model selection is based on held-out validation ROC AUC. The test split is not used to choose the model.

This is also a useful place to look at whether a more flexible classifier is actually buying much over the simple baseline, rather than assuming that the more complicated model will be better.

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
