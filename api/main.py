"""FastAPI app serving the trained G2P models and the frontend.

Run locally:
  uvicorn api.main:app --reload
"""

import logging
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from g2p.infer import G2PPredictor

logger = logging.getLogger("g2p.api")

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FRONTEND_DIR = ROOT / "frontend"

LANG_NAMES = {"zul": "isiZulu", "xho": "isiXhosa", "afr": "Afrikaans"}
# espeak-ng voice codes for our language codes. espeak-ng synthesizes from
# SPELLING with its own rules — it is an independent reference pronunciation,
# never a rendering of the model's predicted phonemes.
ESPEAK_VOICES = {"afr": "af", "zul": "zu", "xho": "xh"}

# In-process rate limit for /api/* : sliding window per client IP.
RATE_LIMIT_MAX = 30
RATE_LIMIT_WINDOW = 60.0  # seconds

predictors: dict[str, G2PPredictor] = {}
speakable: dict[str, str] = {}  # lang code -> espeak voice, probed at startup
_request_times: dict[str, deque] = defaultdict(deque)


def probe_espeak():
    """Probe espeak-ng at runtime; return {lang: voice} for the languages
    whose voice is actually installed. Empty dict if espeak-ng is absent."""
    exe = shutil.which("espeak-ng")
    if exe is None:
        logger.info("espeak-ng not found; audio disabled")
        return {}
    try:
        out = subprocess.run([exe, "--voices"], capture_output=True,
                             text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("espeak-ng --voices failed (%s); audio disabled", e)
        return {}
    # Column 2 of each voice line is the language code (e.g. "af", "zu").
    voice_codes = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            voice_codes.add(parts[1])
    found = {lang: voice for lang, voice in ESPEAK_VOICES.items()
             if voice in voice_codes}
    logger.info("espeak-ng voices available for: %s", sorted(found) or "none")
    return found


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
    speakable.clear()
    speakable.update(probe_espeak())
    logger.info("serving languages: %s", sorted(predictors))
    yield
    predictors.clear()
    speakable.clear()


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
            "speakable": code in speakable,
        })
    return out


def _check_lang(lang):
    if lang not in predictors:
        raise HTTPException(status_code=404,
                            detail=f"unknown language {lang!r}; "
                                   f"available: {sorted(predictors)}")


def _clean_word(word):
    word = word.strip()
    if any(ch.isspace() for ch in word):
        raise HTTPException(
            status_code=400,
            detail="the model is word-level; send one word at a time "
                   "(no spaces)")
    return word


@app.get("/api/phonemize")
def phonemize(word: str = Query(...), lang: str = Query(...)):
    _check_lang(lang)
    word = _clean_word(word)
    try:
        result = predictors[lang].predict(word)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"lang": lang, **result}


@app.get("/api/speak")
def speak(word: str = Query(...), lang: str = Query(...)):
    """Reference pronunciation synthesized by espeak-ng from the word's
    SPELLING using espeak's own rules — independent of the model's predicted
    phonemes, and only for languages whose espeak voice is installed."""
    _check_lang(lang)
    voice = speakable.get(lang)
    if voice is None:
        raise HTTPException(
            status_code=501,
            detail=f"no espeak-ng voice available for {lang!r}")
    word = _clean_word(word)
    if not word or len(word) > 40:
        raise HTTPException(status_code=400,
                            detail="word must be 1-40 characters")
    exe = shutil.which("espeak-ng")
    if exe is None:  # disappeared since startup
        raise HTTPException(status_code=501, detail="espeak-ng not available")
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "out.wav"
        try:
            subprocess.run(
                [exe, "-v", voice, "-w", str(wav_path), "--", word],
                capture_output=True, timeout=15, check=True)
            wav = wav_path.read_bytes()
        except (OSError, subprocess.SubprocessError) as e:
            logger.error("espeak-ng failed for %r (%s): %s", word, lang, e)
            raise HTTPException(status_code=502,
                                detail="speech synthesis failed")
    return Response(content=wav, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {"status": "ok", "models": sorted(predictors)}


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True),
              name="frontend")
