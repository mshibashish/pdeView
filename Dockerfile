# ---- 1. build the React frontend -------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. Python runtime (CPU-only PyTorch) ------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PDEVIEW_DEVICE=cpu

# Hugging Face Spaces run containers as uid 1000
RUN useradd --create-home --uid 1000 user
WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY pyproject.toml README.md LICENSE ./
COPY backend ./backend
RUN pip install -e .

COPY reports ./reports
COPY scripts ./scripts
COPY --from=web /web/dist ./frontend/dist
RUN mkdir -p data checkpoints && chown -R user:user /app

USER user
EXPOSE 7860
# First start downloads the checkpoint (11 MB, MD5-checked) and the 110-sample data subset
# (~730 MB via HTTP range reads, ~2 min); later starts reuse the mounted volume.
CMD ["pdeview-serve"]
