# syntax=docker/dockerfile:1
#
# PneumoniaMNIST Image Explorer
#
# The dataset is NOT part of this image. Mount your own copy at runtime:
#   docker run -p 8501:8501 \
#     -v "C:\path\to\pneumoniamnist.npz:/input/pneumoniamnist.npz:ro" \
#     pneumoniamnist-explorer

FROM python:3.12-slim

LABEL org.opencontainers.image.title="PneumoniaMNIST Image Explorer" \
      org.opencontainers.image.description="Image ingestion, classification and Streamlit explorer. Contains no dataset; mount your own NPZ at /input/pneumoniamnist.npz." \
      org.opencontainers.image.source="https://github.com/hmgill/takehome_project"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Unprivileged runtime user
RUN useradd --create-home --uid 1000 appuser

# Dependencies first so code edits don't trigger a full reinstall
COPY requirements.txt .
RUN pip install -r requirements.txt

# Source code only (.dockerignore is an allowlist)
COPY --chown=appuser:appuser . .

# Fail the build if any dataset file or derived artifact slipped in
RUN python tools/check_no_private_data.py /app

# Writable locations for generated databases, models and outputs.
# Created here with the right owner so named volumes inherit it.
RUN mkdir -p /app/data /app/output /input \
    && chown -R appuser:appuser /app/data /app/output

USER appuser

# Where the user-supplied NPZ is expected (bind-mounted read-only)
ENV PNEUMONIAMNIST_NPZ=/input/pneumoniamnist.npz \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTEST_ADDOPTS="-p no:cacheprovider"

VOLUME ["/app/data", "/app/output"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["python", "-m", "streamlit", "run", "app.py"]