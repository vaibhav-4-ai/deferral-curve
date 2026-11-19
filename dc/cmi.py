"""CMI (code-mixing index): how mixed Hindi and English a question is. 0 = one language, 50 = half/half.

The same meter as the sister project codemix-nlu (cmx/features.py), copied here so this project
needs nothing but its word lists, which codemix-nlu publishes on Hugging Face:
    every word -> 'hi', 'en' or 'other' (numbers) -> CMI = 100 * (1 - biggest language / tagged words)
    "refund kab tak aayega"  ->  en hi hi hi  ->  100 * (1 - 3/4) = 25
"""

import json
import urllib.request
from pathlib import Path

WORDLISTS = Path(__file__).parent.parent / "artifacts" / "wordlists.json"
URL = "https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/charts/eda/wordlists.json"


def fetch():
    WORDLISTS.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as response:
        WORDLISTS.write_bytes(response.read())


def clean(word):
    kept = ""
    for ch in word.lower():
        if ch.isalnum():
            kept += ch
    return kept


def is_devanagari(word):
    for ch in word:
        if "ऀ" <= ch <= "ॿ":
            return True
    return False


class CmiMeter:
    def __init__(self, path=WORDLISTS):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.english = data["english"]
        self.hindi = data["hindi"]
        self.short = set(data["hindi_short_words"])
        self.english_total = max(1, sum(self.english.values()))
        self.hindi_total = max(1, sum(self.hindi.values()))

    def language(self, word):
        if is_devanagari(word):
            return "hi"
        word = clean(word)
        if word == "" or word.isdigit():
            return "other"
        if word in self.short:
            return "hi"
        english_share = self.english.get(word, 0) / self.english_total
        hindi_share = self.hindi.get(word, 0) / self.hindi_total
        if english_share == 0 and hindi_share == 0:
            return "hi"  # unknown: Hindi spellings vary far more than English ones
        if english_share > hindi_share:
            return "en"
        return "hi"

    def cmi(self, text):
        n_hi = 0
        n_en = 0
        for word in text.split():
            tag = self.language(word)
            if tag == "hi":
                n_hi += 1
            if tag == "en":
                n_en += 1
        if n_hi + n_en == 0:
            return 0.0
        return 100 * (1 - max(n_hi, n_en) / (n_hi + n_en))
