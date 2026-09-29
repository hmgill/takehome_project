import os

from project_config import CONFIG, PATHS


def configure_runtime() -> None:
    """
    Apply process-level resource limits before numerical libraries load.

    This keeps the small assessment pipeline robust on constrained HPC
    nodes and centralizes the thread settings used by multiple scripts.
    """

    thread_count = str(CONFIG.project.threads)

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(
            variable,
            thread_count,
        )

    PATHS.temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ.setdefault(
        "JOBLIB_TEMP_FOLDER",
        str(PATHS.temp_dir),
    )
