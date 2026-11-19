import json

from fastapi.testclient import TestClient

from app import main
from dc.codemix import Baseline, words_of
from dc.llm import CachedLLM


def offline_llm(tmp_path):
    return CachedLLM(None, cache_path=tmp_path / "empty.json")


class FakeCodemix:
    """Stands in for the 180 MB model, so the test runs anywhere."""

    def predict(self, text):
        words = words_of(text)
        if not words:
            return None
        return {
            "words": words,
            "model": {"intent": "alarm_set", "confidence": 0.9, "slots": [{"type": "time", "text": "7 baje"}]},
            "baseline": {"intent": "cooking_recipe", "confidence": 0.1},
        }


def test_words_of_cleans_punctuation():
    assert words_of("Kal subah 7 baje alarm laga do!") == ["kal", "subah", "7", "baje", "alarm", "laga", "do"]


def test_baseline_picks_the_intent_with_matching_pieces(tmp_path):
    # Two intents; the letter piece " a" (start of a word beginning with "a") only helps "alarm_set".
    export = {
        "intents": ["alarm_set", "play_music"],
        "vocab": {" a": 0, "ga": 1},
        "idf": [1.0, 1.0],
        "coef": [[3.0, 0.0], [0.0, 3.0]],
        "intercept": [0.0, 0.0],
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    result = Baseline(path).predict(["alarm"])
    assert result["intent"] == "alarm_set"
    assert 0.5 < result["confidence"] <= 1.0


def test_codemix_page_without_model_says_unavailable(tmp_path):
    app = main.create_app(llm=offline_llm(tmp_path))
    with TestClient(app) as client:
        assert "Hinglish" in client.get("/codemix").text
        if client.get("/stats").json()["codemix_loaded"] is False:
            assert client.post("/codemix/predict", json={"text": "alarm laga do"}).status_code == 503


def test_codemix_predict_returns_both_models(tmp_path):
    with TestClient(main.create_app(llm=offline_llm(tmp_path), codemix=FakeCodemix())) as client:
        body = client.post("/codemix/predict", json={"text": "kal subah 7 baje alarm laga do"}).json()
        assert body["model"]["intent"] == "alarm_set"
        assert body["baseline"]["intent"] == "cooking_recipe"
        assert body["latency_ms"] >= 0
        assert client.get("/stats").json()["codemix_sentences"] == 1
