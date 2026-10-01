"""Run the full pipeline if needed, then launch the Streamlit app.

This is the Docker image's default command, so one ``docker run`` does
everything:

1. build the research DuckDB database from the NPZ,
2. run the SQL analysis,
3. train the logistic baseline (plus SVM/XGBoost when config.toml
   [app].model asks for them, and model selection for "best"),
4. ingest the NPZ into the app catalog,
5. re-score the catalog if the configured model changed,
6. start Streamlit.

Steps whose outputs already exist are skipped, so restarting a container that
keeps its data (for example with Compose volumes) goes straight to the app.
Use ``--force`` to rerun every step, or ``--no-app`` to stop before the app.

If no dataset is mounted, the pipeline is skipped and the app starts with an
empty catalog.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from project_config import CONFIG, DATASET_ENV_VAR, PATHS  # noqa: E402

CATALOG = PATHS.data_dir / "image_catalog.sqlite3"


def log(message: str) -> None:
    print(f"[run_all] {message}", flush=True)


def run_step(label: str, *args: str) -> None:
    log(f"{label} ...")
    started = time.monotonic()
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)
    log(f"{label} finished in {time.monotonic() - started:.1f}s")


def run_pipeline(dataset: Path, force: bool) -> None:
    app_model = CONFIG.app.model
    steps = [
        ("build database", PATHS.database, ["src/build_database.py"]),
        ("SQL analysis", PATHS.analysis_dir / "unusual_images.png", ["src/analyze.py"]),
        ("train logistic", PATHS.model_file("logistic"), ["src/model.py"]),
    ]
    extra = {"svm": ["svm"], "xgboost": ["xgboost"], "best": ["svm", "xgboost"]}
    for name in extra.get(app_model, []):
        steps.append((f"train {name}", PATHS.model_file(name), [f"src/model_{name}.py"]))
    if app_model == "best":
        steps.append(
            (
                "select best model",
                PATHS.model_selection_dir / "selected_model.json",
                ["src/select_best_model.py"],
            )
        )
    steps.append(
        ("ingest images", CATALOG, ["src/image_pipeline.py", "--npz", str(dataset)])
    )
    total = len(steps) + 1
    for number, (label, output, args) in enumerate(steps, start=1):
        label = f"Step {number}/{total}: {label}"
        if output.exists() and not force:
            log(f"{label}: skipped (found {output.relative_to(PROJECT_ROOT)})")
            continue
        run_step(label, *args)
    # Cheap and idempotent: only predictions from a different model are redone.
    run_step(f"Step {total}/{total}: re-score catalog ({app_model})",
             "src/image_pipeline.py", "--rescore")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true", help="rerun every step even if outputs exist"
    )
    parser.add_argument(
        "--no-app",
        action="store_true",
        help="run the pipeline only; don't start the app",
    )
    args = parser.parse_args()

    dataset = PATHS.dataset
    if dataset.is_file():
        log(f"Dataset found at {dataset}")
        try:
            run_pipeline(dataset, args.force)
        except subprocess.CalledProcessError as exc:
            log(f"Pipeline step failed with exit code {exc.returncode}")
            return exc.returncode
    else:
        log(
            f"No dataset at {dataset}. Skipping the pipeline. Mount your NPZ "
            f"there, or set {DATASET_ENV_VAR}, to process it."
        )

    if args.no_app:
        return 0

    log("Starting Streamlit at http://localhost:8501")
    os.chdir(PROJECT_ROOT)
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", "app.py"])
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main())