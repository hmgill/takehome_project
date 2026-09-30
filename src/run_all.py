"""Run the full pipeline if needed, then launch the Streamlit app.

This is the Docker image's default command, so one ``docker run`` does
everything:

1. build the research DuckDB database from the NPZ,
2. run the SQL analysis,
3. train the logistic baseline,
4. ingest the NPZ into the app catalog,
5. start Streamlit.

Steps whose outputs already exist are skipped, so restarting a container that
keeps its data (for example with Compose volumes) goes straight to the app.
Use ``--force`` to rerun every step, or ``--no-app`` to stop after step 4.

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

from project_config import DATASET_ENV_VAR, PATHS  # noqa: E402

CATALOG = PATHS.data_dir / "image_catalog.sqlite3"


def log(message: str) -> None:
    print(f"[run_all] {message}", flush=True)


def run_step(label: str, *args: str) -> None:
    log(f"{label} ...")
    started = time.monotonic()
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)
    log(f"{label} finished in {time.monotonic() - started:.1f}s")


def run_pipeline(dataset: Path, force: bool) -> None:
    steps = [
        (
            "Step 1/4: build database",
            PATHS.database,
            ["src/build_database.py"],
        ),
        (
            "Step 2/4: SQL analysis",
            PATHS.analysis_dir / "unusual_images.png",
            ["src/analyze.py"],
        ),
        (
            "Step 3/4: train baseline",
            PATHS.model_file("logistic"),
            ["src/model.py"],
        ),
        (
            "Step 4/4: ingest images",
            CATALOG,
            ["src/image_pipeline.py", "--npz", str(dataset)],
        ),
    ]
    for label, output, args in steps:
        if output.exists() and not force:
            log(f"{label}: skipped (found {output.relative_to(PROJECT_ROOT)})")
            continue
        run_step(label, *args)


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