# Image Explorer: PneumoniaMNIST

Ingest, validate, classify and explore chest X-ray images with a Streamlit app.

**Contents:** [Walkthrough video](#walkthrough-video) · [Dataset](#dataset) · [Run from Docker Hub](#run-from-docker-hub) · [Build and run with Docker](#build-and-run-with-docker) · [Run locally](#run-locally) · [Scripts](#scripts) · [Configuration](#configuration) · [Tests](#tests)

---

## Walkthrough video

<!-- TODO: add the walkthrough video. -->
*Coming soon.*

---

## Dataset

The dataset is not included. Download the 28×28 PneumoniaMNIST archive (`pneumoniamnist.npz`) from [MedMNIST](https://medmnist.com/) and keep it outside this folder. You supply its path when running.

---

## Run from Docker Hub

Command Prompt (use `` ` `` for line breaks in PowerShell, `\` on macOS/Linux):

```cmd
docker run --rm -p 8501:8501 ^
  -v "C:\path\to\pneumoniamnist.npz:/input/pneumoniamnist.npz:ro" ^
  -v "%cd%\output:/app/output" ^
  hmgill/pneumoniamnist-explorer:1.1.0
```

Open <http://localhost:8501> once the log shows `Starting Streamlit`. Press Ctrl+C to stop.

| To | Change |
|---|---|
| Keep nothing on your machine | Remove the `output` mount |
| Rerun every step | Append `python src/run_all.py --force` |
| Run the tests | `docker run --rm hmgill/pneumoniamnist-explorer:1.1.0 python -m pytest -q` |

---

## Build and run with Docker

**Build and test:**

```bash
docker build -t pneumoniamnist-explorer .
docker run --rm pneumoniamnist-explorer python -m pytest -q
```

**Run with Compose.** Create a `.env` file next to `compose.yaml`:

```
NPZ_PATH=C:\path\to\pneumoniamnist.npz
```

Then:

```bash
docker compose up --build
```

Open <http://localhost:8501>. Results are written to `./output`.

| Task | Command |
|---|---|
| Stop | `docker compose down` |
| Stop and delete the database volume | `docker compose down -v` |
| Run one script | `docker compose run --rm explorer python src/model.py` |
| Use the Docker Hub image | Add `IMAGE=hmgill/pneumoniamnist-explorer:1.1.0` to `.env`, then `docker compose pull` and `docker compose up --no-build` |

Paths passed to scripts inside the container must use the container path `/input/pneumoniamnist.npz`, not a host path.

---

## Run locally

Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Set the dataset path:

| Shell | Command |
|---|---|
| Command Prompt | `set "PNEUMONIAMNIST_NPZ=C:\path\to\pneumoniamnist.npz"` |
| PowerShell | `$env:PNEUMONIAMNIST_NPZ = "C:\path\to\pneumoniamnist.npz"` |
| macOS/Linux | `export PNEUMONIAMNIST_NPZ=/path/to/pneumoniamnist.npz` |

Run everything and start the app:

```bash
python src/run_all.py
```

Or run the steps individually:

```bash
python src/build_database.py
python src/analyze.py
python src/model.py
python src/image_pipeline.py --npz "$PNEUMONIAMNIST_NPZ"
python -m streamlit run app.py
```

---

## Scripts

Every script accepts `--help`. Most also accept `--log-level` and `--log-file`.

### Pipeline

| Script | Purpose | Options |
|---|---|---|
| `src/run_all.py` | Run steps 1–4, skipping steps whose outputs exist, then start the app | `--force`, `--no-app` |
| `src/build_database.py` | 1. Validate the NPZ and build the metadata database | `--input`, `--output`, `--csv-output` |
| `src/analyze.py` | 2. Run the SQL analysis and QC outputs | `--database`, `--dataset`, `--sql`, `--output-dir` |
| `src/model.py` | 3. Train and evaluate the logistic-regression baseline | `--dataset`, `--output-dir`, `--augment`, `--mlflow`, `--mlflow-dir` |
| `src/image_pipeline.py` | 4. Ingest images into the app catalog and classify them | `--npz PATH` or `--directory PATH`, `--database`, `--retry-failed`, `--metadata-only` |
| `app.py` | 5. Streamlit explorer (`python -m streamlit run app.py`) | — |

### Optional experiments

| Script | Purpose | Options |
|---|---|---|
| `src/model_svm.py` | RBF-SVM experiment | `--n-iter`, `--cv-folds`, `--augment`, `--mlflow` |
| `src/model_xgboost.py` | XGBoost experiment | `--n-iter`, `--cv-folds`, `--augment`, `--mlflow` |
| `src/select_best_model.py` | Pick the best candidate by validation ROC AUC | `--dataset`, `--output-dir` |
| `src/create_predictions.py` | Per-image prediction CSVs | `--split {train,val,test}`, `--models` |
| `src/explain_shap.py` | SHAP explanations for images | `--image-id ID\|all`, `--model`, `--output-format` |

### Utilities

| Script | Purpose | Usage |
|---|---|---|
| `tools/check_no_private_data.py` | Check that no dataset files or derived outputs are included | `--git` (tracked files) or a folder path |

Outputs are written to `data/` and `output/`. See `CONFIG_AND_OUTPUTS.md` for the layout.

---

## Configuration

Settings live in `config.toml`. The following environment variables override them:

| Variable | Purpose | Default |
|---|---|---|
| `PNEUMONIAMNIST_NPZ` | Dataset path for all scripts | `data/pneumoniamnist.npz` |
| `IMAGE_CATALOG_DB` | App catalog database | `data/image_catalog.sqlite3` |
| `IMAGE_EXPLORER_LOCAL_IMPORT` | `1` shows server-side import buttons in the app | off locally, on in Docker |
| `NPZ_PATH` | Compose: host path of the NPZ | required |
| `OUTPUT_DIR` | Compose: host folder for results | `./output` |
| `IMAGE` | Compose: image to run | `pneumoniamnist-explorer:latest` |

---

## Tests

```bash
python -m pytest -q                                            # local
docker run --rm pneumoniamnist-explorer python -m pytest -q    # Docker
```
