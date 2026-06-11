"""API tests against the real bundles in models/ (skipped if absent)."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLES = sorted((ROOT / "models").glob("*.pt"))

pytestmark = pytest.mark.skipif(
    not BUNDLES, reason="no trained bundles in models/")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"] == ["afr", "xho", "zul"]


def test_languages(client):
    r = client.get("/api/languages")
    assert r.status_code == 200
    langs = {row["code"]: row for row in r.json()}
    assert set(langs) == {"zul", "xho", "afr"}
    assert langs["zul"]["name"] == "isiZulu"
    assert langs["xho"]["name"] == "isiXhosa"
    assert langs["afr"]["name"] == "Afrikaans"
    for row in langs.values():
        assert 0.0 <= row["test_per"] < 1.0
        assert 0.0 < row["test_word_acc"] <= 1.0


@pytest.mark.parametrize("lang,word", [
    ("zul", "umuntu"),
    ("xho", "abantu"),
    ("afr", "aanbeveling"),
])
def test_phonemize_success(client, lang, word):
    r = client.get("/api/phonemize", params={"word": word, "lang": lang})
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == lang
    assert body["word"] == word
    assert body["chars"] == list(word)
    assert len(body["phonemes"]) > 0
    assert all(isinstance(p, str) and p for p in body["phonemes"])
    # All served models are attention models: one row per output phoneme,
    # one column per input character.
    attn = body["attention"]
    assert attn is not None
    assert len(attn) == len(body["phonemes"])
    assert all(len(row) == len(word) for row in attn)
    assert all(abs(sum(row) - 1.0) < 1e-4 for row in attn)


def test_unknown_lang(client):
    r = client.get("/api/phonemize", params={"word": "umuntu", "lang": "eng"})
    assert r.status_code == 404


def test_empty_word(client):
    r = client.get("/api/phonemize", params={"word": "   ", "lang": "zul"})
    assert r.status_code == 400


def test_multi_word_input(client):
    r = client.get("/api/phonemize",
                   params={"word": "umuntu ngumuntu", "lang": "zul"})
    assert r.status_code == 400
    assert "one word" in r.json()["detail"]


def test_too_long_word(client):
    r = client.get("/api/phonemize", params={"word": "a" * 41, "lang": "zul"})
    assert r.status_code == 400
