"""FastAPI app serving the trained G2P models and the frontend.

Run locally:
  uvicorn api.main:app --reload
"""

import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from g2p.infer import G2PPredictor

logger = logging.getLogger("g2p.api")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

LANG_NAMES = {"zul": "isiZulu", "xho": "isiXhosa", "afr": "Afrikaans"}

# In-process rate limit for /api/* : sliding window per client IP.
RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW = 60.0  # seconds

predictors: dict[str, G2PPredictor] = {}
_request_times: dict[str, deque] = defaultdict(deque)


def load_predictors():
    """Scan the TOP LEVEL of models/ for *.pt bundles (non-recursive, so
    models/baselines/ is excluded) and key them by language code."""
    predictors.clear()
    for path in sorted(MODELS_DIR.glob("*.pt")):
        predictor = G2PPredictor(path, device="cpu")
        code = predictor.lang
        if code in predictors:
            logger.error(
                "duplicate language code %r: %s ignored (already loaded "
                "from another bundle)", code, path.name)
            continue
        predictors[code] = predictor
        logger.info("loaded %s -> lang=%s", path.name, code)
    return predictors


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictors()
    logger.info("serving languages: %s", sorted(predictors))
    yield
    predictors.clear()


app = FastAPI(title="phonemeza G2P API", lifespan=lifespan)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        times = _request_times[ip]
        while times and now - times[0] > RATE_LIMIT_WINDOW:
            times.popleft()
        if len(times) >= RATE_LIMIT_MAX:
            return JSONResponse(
                status_code=429,
                content={"detail": "rate limit exceeded; try again shortly"})
        times.append(now)
    return await call_next(request)


@app.get("/api/languages")
def languages():
    out = []
    for code in sorted(predictors):
        metrics = predictors[code].metadata["metrics"]
        out.append({
            "code": code,
            "name": LANG_NAMES.get(code, code),
            "test_per": metrics["test_per"],
            "test_word_acc": metrics["test_word_acc"],
        })
    return out


@app.get("/api/phonemize")
def phonemize(word: str = Query(...), lang: str = Query(...)):
    predictor = predictors.get(lang)
    if predictor is None:
        raise HTTPException(status_code=404,
                            detail=f"unknown language {lang!r}; "
                                   f"available: {sorted(predictors)}")
    word = word.strip()
    if any(ch.isspace() for ch in word):
        raise HTTPException(
            status_code=400,
            detail="the model is word-level; send one word at a time "
                   "(no spaces)")
    try:
        result = predictor.predict(word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"lang": lang, **result}


@app.get("/api/health")
def health():
    return {"status": "ok", "models": sorted(predictors)}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True),
              name="frontend")
