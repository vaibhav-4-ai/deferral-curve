import hashlib
import json
import os
import time
from pathlib import Path

import httpx

CACHE = Path(__file__).parent.parent / "evals" / "llm_cache.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def groq_backend(api_key, model):
    # Returns a function (system, user) -> reply text that calls Groq's free API.
    def call(system, user):
        for _ in range(4):
            response = httpx.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=60,
            )
            if response.status_code == 429:
                # Free tier rate limit: wait as long as Groq asks, then try again.
                time.sleep(float(response.headers.get("retry-after", 10)))
                continue
            response.raise_for_status()
            time.sleep(2.5)  # stay under the free tier's requests-per-minute limit
            return response.json()["choices"][0]["message"]["content"]
        response.raise_for_status()

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
        self.cache[key] = text
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return text


def default_llm():
    # Live if a Groq key is present in the environment, otherwise replay from the cache.
    key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    backend = groq_backend(key, model) if key else None
    return CachedLLM(backend, model=model)
