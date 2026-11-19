import time
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from dc.agent import Agent
from dc.codemix import load_codemix
from dc.corpus import load_corpus
from dc.llm import default_llm
from dc.policy import decide, features, load_calibration
from dc.retrieve import HybridRetriever

PAGE = Path(__file__).parent / "index.html"
CODEMIX_PAGE = Path(__file__).parent / "codemix.html"

# Per-visitor ceiling, enforced before any model call, so a stranger can't spend the whole free Groq quota.
QUESTIONS_PER_VISITOR_PER_DAY = 30
QUESTIONS_PER_SITE_PER_DAY = 400  # hard ceiling for everyone together, in case the per-visitor check is dodged
MAX_QUESTION_CHARS = 300
CODEMIX_PER_VISITOR_PER_DAY = 300  # no paid model behind /codemix, only our CPU
LATENCIES_KEPT = 1000  # /stats is worked out from the most recent 1000 questions

HANDOFF_MESSAGE = "I'm not confident enough to answer this one, so I'm passing it to a human teammate."

# Prometheus metrics, served at /metrics.
ASKED = Counter("dc_questions_total", "Questions answered, by outcome", ["outcome"])
LATENCY = Histogram("dc_answer_seconds", "Time to answer one question",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20))


class Question(BaseModel):
    # The body of POST /ask. FastAPI rejects an empty or overlong question before our code runs.
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class Sentence(BaseModel):
    # The body of POST /codemix/predict.
    text: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class DailyUsage:
    """Counts today's questions per visitor. Starts from zero again when the date changes."""

    def __init__(self):
        self.day = date.today()
        self.counts = {}

    def reset_if_new_day(self):
        if self.day != date.today():
            self.day = date.today()
            self.counts = {}

    def get(self, who):
        return self.counts.get(who, 0)

    def add(self, who):
        self.counts[who] = self.get(who) + 1


def peak_memory_mb():
    try:
        import resource  # Linux only, which is where the app runs
    except ImportError:
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)  # Linux reports kilobytes


def load_policy():
    # The calibrator and threshold fitted in CI. If they're missing, hand everything to a human.
    try:
        return load_calibration()
    except FileNotFoundError:
        return None, None


def visitor_of(request):
    # The hosting proxy appends the real address last; earlier entries can be faked by the visitor.
    visitor = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    return visitor.split(",")[-1].strip()


def keep_recent(latencies, elapsed):
    latencies.append(elapsed)
    if len(latencies) > LATENCIES_KEPT:
        latencies.pop(0)  # drop the oldest


def percentiles_ms(latencies):
    times = np.array(latencies) * 1000
    return round(float(np.percentile(times, 50))), round(float(np.percentile(times, 95)))


def create_app(llm=None, codemix=None):
    # Builds the web app. Tests pass in a fake llm; the live site passes nothing and gets Groq.
    if llm is None:
        llm = default_llm(tries=1, timeout=20, cooldown=60)
    agent = Agent(HybridRetriever(load_corpus()), llm)
    calibrator, threshold = load_policy()
    if codemix is None:
        codemix = load_codemix()  # None when the model files were not downloaded (e.g. in CI)

    latencies = []  # seconds taken by recent questions
    codemix_latencies = []  # seconds taken by recent /codemix sentences
    usage = DailyUsage()
    codemix_usage = DailyUsage()

    app = FastAPI(title="deferral-curve")
    app.mount("/metrics", make_asgi_app())

    def check_visitor_budget(request):
        usage.reset_if_new_day()
        visitor = visitor_of(request)

        # "__all__" counts everyone together.
        if usage.get("__all__") >= QUESTIONS_PER_SITE_PER_DAY:
            raise HTTPException(429, "The demo has used up today's free model quota. Please try again tomorrow.")
        if usage.get(visitor) >= QUESTIONS_PER_VISITOR_PER_DAY:
            raise HTTPException(429, f"Demo limit: {QUESTIONS_PER_VISITOR_PER_DAY} questions per day.")
        usage.add(visitor)
        usage.add("__all__")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "policy_loaded": calibrator is not None}

    @app.get("/stats")
    def stats():
        # Response times of recent questions (ms) and peak memory, for the 512 MB resource-budget write-up.
        out = {"questions": len(latencies), "peak_memory_mb": peak_memory_mb()}
        if latencies:
            out["p50_ms"], out["p95_ms"] = percentiles_ms(latencies)
        out["codemix_loaded"] = codemix is not None
        out["codemix_sentences"] = len(codemix_latencies)
        if codemix_latencies:
            out["codemix_p50_ms"], out["codemix_p95_ms"] = percentiles_ms(codemix_latencies)
        return out

    @app.get("/codemix", response_class=HTMLResponse)
    def codemix_home():
        return CODEMIX_PAGE.read_text(encoding="utf-8")

    @app.post("/codemix/predict")
    def codemix_predict(sentence: Sentence, request: Request):
        if codemix is None:
            raise HTTPException(503, "The Hinglish model is not loaded on this server.")
        codemix_usage.reset_if_new_day()
        visitor = visitor_of(request)
        if codemix_usage.get(visitor) >= CODEMIX_PER_VISITOR_PER_DAY:
            raise HTTPException(429, f"Demo limit: {CODEMIX_PER_VISITOR_PER_DAY} sentences per day.")
        codemix_usage.add(visitor)
        start = time.perf_counter()
        result = codemix.predict(sentence.text)
        elapsed = time.perf_counter() - start
        if result is None:
            raise HTTPException(422, "Type a sentence with at least one word.")
        keep_recent(codemix_latencies, elapsed)
        result["latency_ms"] = round(elapsed * 1000)
        return result

    @app.post("/ask")
    def ask(question: Question, request: Request):
        check_visitor_budget(request)
        start = time.perf_counter()
        result = agent.answer(question.query)

        if result.degraded or calibrator is None:
            # The model was unreachable, or there is no fitted policy: always hand off.
            confidence = None
            outcome = "escalate"
        else:
            row = {"verified": result.verified, "answer": result.answer,
                   "tool_used": result.tool_used, "features": result.features}
            confidence = float(calibrator.predict([features(row)])[0])
            outcome = decide(confidence, threshold)

        elapsed = time.perf_counter() - start
        keep_recent(latencies, elapsed)
        LATENCY.observe(elapsed)
        ASKED.labels(outcome).inc()
        return {
            "outcome": outcome,
            "answer": HANDOFF_MESSAGE if outcome == "escalate" else result.answer,
            "bot_draft": result.answer,  # shown on the demo page so visitors can see what was held back
            "confidence": confidence,
            "sources": result.doc_ids,
            "tool_used": result.tool_used,
            "degraded": result.degraded,
            "latency_ms": round(elapsed * 1000),
        }

    return app


app = create_app()
