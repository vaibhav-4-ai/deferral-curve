"""Hinglish intent + slots, served from the sister project codemix-nlu.

Two models side by side, so a visitor sees the difference:
    baseline  TF-IDF + logistic regression trained on Devanagari Hindi only (4 MB, pure Python maths)
    model     hing-mbert fine-tuned on Devanagari + romanized + English, shrunk to int8 ONNX (~180 MB)

Both files are downloaded once when the Docker image is built (`python -m dc fetch-codemix`),
never at request time. Light libraries only: onnxruntime + tokenizers, no torch, no transformers,
so the whole service still fits in Render's 512 MB.
"""

import json
import math
import urllib.request
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).parent.parent / "artifacts" / "codemix"
HF = "https://huggingface.co/vaibhav-4-ai/codemix-nlu/resolve/main/serving/"
FILES = {
    "model.int8.onnx": HF + "model.int8.onnx",
    "labels.json": HF + "labels.json",
    "tokenizer.json": HF + "tokenizer/tokenizer.json",
    "baseline.json": "https://vaibhav-4-ai.github.io/codemix-nlu/model_plain.json",
}
MAX_PIECES = 64


def fetch():
    """Download the four files into artifacts/codemix/ (run once, at image build time)."""
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        with urllib.request.urlopen(url) as response:
            (FOLDER / name).write_bytes(response.read())
        print(f"{name}: {(FOLDER / name).stat().st_size / 1e6:.1f} MB")


def words_of(text):
    """'Kal subah 7 baje alarm laga do!' -> ['kal', 'subah', '7', 'baje', 'alarm', 'laga', 'do']"""
    words = []
    for word in text.lower().split():
        word_clean = word.strip(".,?!'\"")
        if word_clean:
            words.append(word_clean)
    return words


def softmax(scores):
    biggest = max(scores)
    exps = []
    for score in scores:
        exps.append(math.exp(score - biggest))
    total = sum(exps)
    probs = []
    for value in exps:
        probs.append(value / total)
    return probs


class Baseline:
    """TF-IDF letter pieces + logistic regression, redone by hand from the exported numbers."""

    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.intents = data["intents"]
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.coef = data["coef"]
        self.intercept = data["intercept"]

    def predict(self, words):
        # 1. cut each word (padded with spaces) into 2-4 letter pieces and count the known ones
        counts = {}
        for word in words:
            padded = f" {word} "
            for n in range(2, 5):
                last = max(0, len(padded) - n)
                for start in range(last + 1):
                    index = self.vocab.get(padded[start : start + n])
                    if index is not None:
                        counts[index] = counts.get(index, 0) + 1
                if last == 0:  # word shorter than n: counted once, stop
                    break
        # 2. weight each piece: (1 + log(count)) * idf, then scale the list to length 1
        weights = {}
        for index, count in counts.items():
            weights[index] = (1 + math.log(count)) * self.idf[index]
        norm = math.sqrt(sum(v * v for v in weights.values())) or 1.0
        # 3. one score per intent; softmax turns scores into probabilities
        scores = []
        for k, row in enumerate(self.coef):
            total = self.intercept[k]
            for index, value in weights.items():
                total += value / norm * row[index]
            scores.append(total)
        probs = softmax(scores)
        best = probs.index(max(probs))
        return {"intent": self.intents[best], "confidence": round(probs[best], 3)}


class CodemixModel:
    """The fine-tuned model: intent with confidence, plus a slot tag for every word."""

    def __init__(self, folder=FOLDER):
        import onnxruntime  # imported here so the rest of the app never pays for it
        from tokenizers import Tokenizer

        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        self.intent_names = labels["intent_names"]
        self.tag_names = labels["tag_names"]
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_PIECES)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1  # Render's free plan has a tenth of one CPU
        # Memory over speed: no extra re-arranged copy of the weights, no big reserved memory pool.
        options.add_session_config_entry("session.disable_prepacking", "1")
        options.enable_cpu_mem_arena = False
        self.session = onnxruntime.InferenceSession(str(folder / "model.int8.onnx"), options)
        self.baseline = Baseline(folder / "baseline.json")

    def predict(self, text):
        words = words_of(text)
        if not words:
            return None
        enc = self.tokenizer.encode(words, is_pretokenized=True)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.ones_like(ids)
        intent_logits, slot_logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})
        probs = softmax(list(intent_logits[0]))
        best = probs.index(max(probs))
        tags = ["O"] * len(words)
        previous = None
        for position, word_id in enumerate(enc.word_ids):
            if word_id is not None and word_id != previous:
                tags[word_id] = self.tag_names[int(slot_logits[0][position].argmax())]
            previous = word_id
        # B-time I-time -> one slot "time: 7 baje". An I- tag only continues the slot right before it.
        slots = []
        for k, word in enumerate(words):
            if tags[k] == "O":
                continue
            kind = tags[k][2:]
            continues = (tags[k].startswith("I-") and slots and slots[-1]["type"] == kind
                         and slots[-1]["last"] == k - 1)
            if continues:
                slots[-1]["text"] += " " + word
                slots[-1]["last"] = k
            else:
                slots.append({"type": kind, "text": word, "last": k})
        return {
            "words": words,
            "model": {"intent": self.intent_names[best], "confidence": round(probs[best], 3), "slots": slots},
            "baseline": self.baseline.predict(words),
        }


def load_codemix():
    """The model if its files were downloaded, else None (the page then says it is unavailable)."""
    if not (FOLDER / "model.int8.onnx").exists():
        return None
    return CodemixModel()
