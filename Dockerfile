# ─────────────────────────────────────────────────────────────
# EDA OpenEnv Agent — Dockerfile
# ─────────────────────────────────────────────────────────────
# Build:
#   docker build -t eda-openenv .
#
# Run Streamlit app:
#   docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... eda-openenv
#
# Run LLM baseline (all 3 tasks):
#   docker run \
#     -e OPENAI_API_KEY=sk-... \
#     -v /your/local/data:/app/data \
#     eda-openenv \
#     python baseline_agent.py --csv data/sample.csv
#
# Run baseline with specific model and save results:
#   docker run \
#     -e OPENAI_API_KEY=sk-... \
#     -v /your/local/data:/app/data \
#     -v /your/local/results:/app/results \
#     eda-openenv \
#     python baseline_agent.py --csv data/sample.csv --model gpt-4o --episodes 3
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="EDA OpenEnv"
LABEL description="RL environment for EDA pipeline agents — OpenEnv Hackathon"

# Prevents Python from buffering stdout/stderr (critical for readable logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OPENAI_API_KEY must be passed at runtime — never bake it into the image
# docker run -e OPENAI_API_KEY=sk-... eda-openenv
ENV OPENAI_API_KEY=""

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project source
COPY app.py             .
COPY inference.py  .
COPY pipeline.py        .
COPY env/               ./env/
COPY tools/             ./tools/

# Directory for mounting CSV data at runtime
RUN mkdir -p /app/data /app/results

# Streamlit config — headless, no telemetry, no browser pop-up
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /root/.streamlit/config.toml

# Expose Streamlit port
EXPOSE 8501

# Default: run Streamlit app
# Override for baseline: docker run eda-openenv python baseline_agent.py --csv data/sample.csv
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]