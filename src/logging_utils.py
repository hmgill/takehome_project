import sys
from pathlib import Path

from loguru import logger


def configure_logging(
    log_file: Path,
    console_level: str = "SUCCESS",
) -> None:
    """
    Configure concise console logging and a detailed per-run log file.
    """

    logger.remove()

    logger.add(
        sys.stderr,
        level=console_level.upper(),
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.add(
        log_file,
        level="INFO",
        mode="w",
        encoding="utf-8",
        format=("{time:YYYY-MM-DD HH:mm:ss} | " "{level: <8} | " "{message}"),
    )

    logger.info("Starting new run")

    logger.info(
        "Log file: {}",
        log_file,
    )
