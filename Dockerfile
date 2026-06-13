# Serving image for the PhonemeZA G2P API. CPU-only: no CUDA wheels.
FROM python:3.12-slim

# espeak-ng provides the optional "reference pronunciation" audio. The single
# apt package bundles all voice data (including Afrikaans 'af'), so the runtime
# probe in api/main.py flips afr to speakable with no code change.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install CPU-only torch FIRST from the PyTorch CPU index so the big CUDA
# wheels are never pulled. Then the remaining deps: torch is already satisfied,
# so `-r requirements.txt` does not re-resolve it against PyPI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    # Strip artifacts unused by pure-Python eager inference, in THIS layer so
    # the bytes never land in the image: C++ headers, the bundled test suite,
    # CLI tools, and compiled bytecode caches.
    && SP="$(python -c 'import site; print(site.getsitepackages()[0])')" \
    && rm -rf "$SP/torch/include" "$SP/torch/test" \
    # torch/bin/torch_shm_manager is loaded at `import torch`; keep only it.
    && find "$SP/torch/bin" -type f ! -name 'torch_shm_manager' -delete \
    && find "$SP" -depth -type d -name '__pycache__' -exec rm -rf {} + \
    && find "$SP" -depth -type d -name 'tests' -exec rm -rf {} +

# Application code and trained model bundles. models/baselines/ (the afr_none
# bottleneck baseline, never served) is excluded via .dockerignore.
COPY g2p/ ./g2p/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Drop root.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
