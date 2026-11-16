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
ONE_DAY = 24 * 60 * 60  # in seconds


class GroqClient:
    """Sends one prompt to Groq's free API and returns the reply text.

    Batch eval: a 2.5 second pause between calls, 4 tries, and once the quota is hit stop for the whole run.
    Live site: no pause, 1 try, a short timeout, and try Groq again 60 seconds after a rate limit.
    """

    def __init__(self, api_key, model, pace=0.0, tries=4, timeout=60, cooldown=ONE_DAY):
        self.api_key = api_key
        self.model = model
        self.pace = pace  # seconds to wait after each call
        self.tries = tries  # how many times to try when Groq says "too many requests"
        self.timeout = timeout  # seconds to wait for one reply
        self.cooldown = cooldown  # seconds to stay away from Groq after giving up
        self.blocked_until = 0.0  # the moment we're allowed to call Groq again

    def ask(self, system, user):
        if time.monotonic() < self.blocked_until:
            raise LookupError("Groq rate limit hit recently; answering from retrieval only for now")

        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.model.startswith("openai/gpt-oss"):
            payload["reasoning_effort"] = "low"  # fewer hidden reasoning tokens, so the free daily quota lasts

        for _ in range(self.tries):
            response = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 429:  # 429 means "too many requests"
                if self.tries == 1:
                    break
                # Free tier rate limit: wait as long as Groq asks (at most a minute), then try again.
                wait = float(response.headers.get("retry-after", 10))
                time.sleep(min(wait, 60))
                continue
            response.raise_for_status()  # any other error (wrong key, model gone) is raised, not hidden
            time.sleep(self.pace)  # batch runs stay under the free tier's requests-per-minute limit
            return response.json()["choices"][0]["message"]["content"]

        # Every try was rate-limited. Stop calling Groq for a while instead of hammering it.
        self.blocked_until = time.monotonic() + self.cooldown
        raise LookupError(f"Groq rate limit still hit after {self.tries} tries; rerun later to continue from the cache")


class CachedLLM:
    """Asks the model, but remembers every answer on disk so the same question is never paid for twice.

    backend is any function that takes (system, user) and returns the reply text.
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
        # A fingerprint of the model name and both prompts. Same three things -> same key -> saved reply.
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


def default_llm(pace=0.0, tries=4, timeout=60, cooldown=ONE_DAY):
    # Live if a Groq key is present in the environment, otherwise replay from the cache.
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    if not key:
        return CachedLLM(None, model=model)
    client = GroqClient(key, model, pace=pace, tries=tries, timeout=timeout, cooldown=cooldown)
    return CachedLLM(client.ask, model=model)
