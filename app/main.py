import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from dc.agent import Agent
from dc.corpus import load_corpus
from dc.llm import default_llm
from dc.policy import decide, features, load_calibration
from dc.retrieve import HybridRetriever

PAGE = Path(__file__).parent / "index.html"

# Per-visitor ceiling, enforced before any model call, so a stranger can't spend the whole free Groq quota.
QUESTIONS_PER_VISITOR_PER_DAY = 30
QUESTIONS_PER_SITE_PER_DAY = 400  # hard ceiling for everyone together, in case the per-visitor check is dodged
MAX_QUESTION_CHARS = 300

HANDOFF_MESSAGE = "I'm not confident enough to answer this one, so I'm passing it to a human teammate."

ASKED = Counter("dc_questions_total", "Questions answered, by outcome", ["outcome"])
LATENCY = Histogram("dc_answer_seconds", "Time to answer one question",
                    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 20))


class Question(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


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


def create_app(llm=None):
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        live_llm = default_llm(tries=1, timeout=20, cooldown=60) if llm is None else llm
        state["agent"] = Agent(HybridRetriever(load_corpus()), live_llm)
        state["calibrator"], state["threshold"] = load_policy()
        state["latencies"] = deque(maxlen=1000)
        state["usage"] = defaultdict(int)
        state["usage_day"] = date.today()
        yield

    app = FastAPI(title="deferral-curve", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    def check_visitor_budget(request):
        if state["usage_day"] != date.today():
            state["usage"].clear()
            state["usage_day"] = date.today()
        # The hosting proxy appends the real address last; earlier entries can be faked by the visitor.
        visitor = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
        visitor = visitor.split(",")[-1].strip()
        if state["usage"]["__all__"] >= QUESTIONS_PER_SITE_PER_DAY:
            raise HTTPException(429, "The demo has used up today's free model quota. Please try again tomorrow.")
        if state["usage"][visitor] >= QUESTIONS_PER_VISITOR_PER_DAY:
            raise HTTPException(429, f"Demo limit: {QUESTIONS_PER_VISITOR_PER_DAY} questions per day.")
        state["usage"][visitor] += 1
        state["usage"]["__all__"] += 1

    @app.get("/", response_class=HTMLResponse)
    def home():
        return PAGE.read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", "policy_loaded": state.get("calibrator") is not None}

    @app.get("/stats")
    def stats():
        # Response times of the last 1000 questions (ms) and peak memory, for the 512 MB resource-budget write-up.
        out = {"questions": len(state["latencies"]), "peak_memory_mb": peak_memory_mb()}
        if state["latencies"]:
            times = np.array(state["latencies"]) * 1000
            out["p50_ms"] = round(float(np.percentile(times, 50)))
            out["p95_ms"] = round(float(np.percentile(times, 95)))
        return out

    @app.post("/ask")
    def ask(question: Question, request: Request):
        check_visitor_budget(request)
        start = time.perf_counter()
        result = state["agent"].answer(question.query)

        if result.degraded or state["calibrator"] is None:
            confidence, outcome = None, "escalate"
        else:
            row = {"verified": result.verified, "answer": result.answer,
                   "tool_used": result.tool_used, "features": result.features}
            confidence = float(state["calibrator"].predict([features(row)])[0])
            outcome = decide(confidence, state["threshold"])

        elapsed = time.perf_counter() - start
        state["latencies"].append(elapsed)
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
