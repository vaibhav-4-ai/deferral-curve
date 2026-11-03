import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

CACHE = Path(__file__).parent.parent / "evals" / "llm_cache.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


def groq_backend(api_key, model, *, pace=0.0, tries=4, timeout=60, cooldown=float("inf")):
    # Returns a function (system, user) -> reply text that calls Groq's free API.
    # Batch eval: pace=2.5 between calls, 4 tries, and once the quota is hit stop for the whole run.
    # Live site: no pacing, 1 try, short timeout, and retry Groq again after `cooldown` seconds.
    state = {"blocked_until": 0.0}
    payload = {"model": model, "temperature": 0}
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"  # fewer hidden reasoning tokens, so the free daily quota lasts

    def call(system, user):
        if time.monotonic() < state["blocked_until"]:
            raise LookupError("Groq rate limit hit recently; answering from retrieval only for now")
        for _ in range(tries):
            response = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    **payload,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=timeout,
            )
            if response.status_code == 429:
                if tries == 1:
                    break
                # Free tier rate limit: wait as long as Groq asks (at most a minute), then try again.
                time.sleep(min(float(response.headers.get("retry-after", 10)), 60))
                continue
            response.raise_for_status()
            time.sleep(pace)  # batch runs stay under the free tier's requests-per-minute limit
            return response.json()["choices"][0]["message"]["content"]
        state["blocked_until"] = time.monotonic() + cooldown
        raise LookupError(f"Groq rate limit still hit after {tries} tries; rerun later to continue from the cache")

    return call


class CachedLLM:
    """Asks the model, but remembers every answer on disk so the same question is never paid for twice.

    backend=None means replay-only: cached answers work, anything new raises an error.
    """

    def __init__(self, backend, model=DEFAULT_MODEL, cache_path=CACHE):
        self.backend = backend
        self.model = model
        self.cache_path = Path(cache_path)
        self.cache = {}
        self.lock = threading.Lock()  # the live site answers several visitors at once
        if self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, system, user):
        raw = json.dumps([self.model, system, user], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def complete(self, system, user):
        key = self._key(system, user)
        if key in self.cache:
            return self.cache[key]
        if self.backend is None:
            raise LookupError("no cached reply for this prompt and no GROQ_API_KEY set")
        text = self.backend(system, user)
        with self.lock:
            self.cache[key] = text
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        return text


def default_llm(**backend_options):
    # Live if a Groq key is present in the environment, otherwise replay from the cache.
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    backend = groq_backend(key, model, **backend_options) if key else None
    return CachedLLM(backend, model=model)
