from fastapi.testclient import TestClient

from app import main
from dc.llm import CachedLLM

OUTCOMES = {"auto_resolve", "escape_hatch", "escalate"}


def fake_llm(tmp_path, verdict="SUPPORTED"):
    def backend(system, user):
        return verdict if system.startswith("You check") else "Refunds take 5 to 7 business days."

    return CachedLLM(backend, cache_path=tmp_path / "cache.json")


def test_health_and_home_page(tmp_path):
    with TestClient(main.create_app(llm=fake_llm(tmp_path))) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "QuickKart" in client.get("/").text


def test_ask_returns_a_decision(tmp_path):
    with TestClient(main.create_app(llm=fake_llm(tmp_path))) as client:
        body = client.post("/ask", json={"query": "refund kab tak aayega?"}).json()
        assert body["outcome"] in OUTCOMES
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["sources"]
        assert client.get("/stats").json()["questions"] == 1
        assert "dc_questions_total" in client.get("/metrics/").text


def test_model_outage_hands_off_to_a_human(tmp_path):
    offline = CachedLLM(None, cache_path=tmp_path / "empty.json")
    with TestClient(main.create_app(llm=offline)) as client:
        body = client.post("/ask", json={"query": "refund kab tak aayega?"}).json()
        assert body["degraded"] is True
        assert body["outcome"] == "escalate"
        assert body["answer"] == main.HANDOFF_MESSAGE


def test_visitor_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "QUESTIONS_PER_VISITOR_PER_DAY", 2)
    with TestClient(main.create_app(llm=fake_llm(tmp_path))) as client:
        for _ in range(2):
            assert client.post("/ask", json={"query": "refund kab?"}).status_code == 200
        assert client.post("/ask", json={"query": "refund kab?"}).status_code == 429


def test_overlong_question_is_rejected(tmp_path):
    with TestClient(main.create_app(llm=fake_llm(tmp_path))) as client:
        assert client.post("/ask", json={"query": "x" * 301}).status_code == 422
