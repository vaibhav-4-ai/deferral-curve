import json
from pathlib import Path

import numpy as np

from dc.grade import normalize

CALIBRATION = Path(__file__).parent.parent / "artifacts" / "calibration.json"

# The control: what most production systems do. One similarity score, one guessed cut-off.
GUESSED_THRESHOLD = 0.5

# How far below the auto-resolve threshold an answer is still shown, with a visible "talk to a human" button.
ESCAPE_HATCH_MARGIN = 0.15


def features(row):
    # row: a dict with "verified", "answer", "tool_used" and the retriever's raw "features".
    f = row["features"]
    return [
        1.0 if row["verified"] else 0.0,
        1.0 if "i don't know" in normalize(row["answer"]) else 0.0,
        1.0 if row["tool_used"] else 0.0,
        f["top_meaning"],
        f["meaning_gap"],
        float(np.log1p(f["top_word"])),
    ]


def raw_confidence(row):
    # The uncalibrated signal: cosine similarity of the best page, clipped to 0..1.
    return float(np.clip(row["features"]["top_meaning"], 0.0, 1.0))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Calibrator:
    """Turns the agent's signals into a probability that its answer is right (logistic regression)."""

    def __init__(self):
        # Empty until fit() learns these from data, or load_calibration() reads them from the saved file.
        self.mean = None
        self.std = None
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y, l2=1.0, steps=3000, lr=0.1):
        # X: one row of signals per question. y: 1 if that answer was right, 0 if wrong.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Standardize: put every signal on the same scale (mean 0, spread 1) so none dominates by its units.
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        Z = (X - self.mean) / self.std

        # Gradient descent: start from zero and nudge the weights towards better predictions, 3000 times.
        w = np.zeros(Z.shape[1])
        b = 0.0
        for _ in range(steps):
            p = sigmoid(Z @ w + b)  # current predictions
            w -= lr * (Z.T @ (p - y) + l2 * w) / len(y)  # l2 keeps the weights from growing wild
            b -= lr * float(np.mean(p - y))
        self.weights = w
        self.bias = b
        return self

    def predict(self, X):
        # Weighted sum of the signals, squashed into 0..1.
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return sigmoid(Z @ self.weights + self.bias)

    def to_dict(self):
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }


def decide(probability, threshold):
    # The three outcomes. threshold=None means no threshold met the error budget: escalate everything.
    if threshold is None:
        return "escalate"
    if probability >= threshold:
        return "auto_resolve"
    if probability >= threshold - ESCAPE_HATCH_MARGIN:
        return "escape_hatch"
    return "escalate"


def load_calibration(path=CALIBRATION):
    # Reads the calibrator and the threshold that CI fitted. Returns both.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    saved = data["model"]
    calibrator = Calibrator()
    calibrator.mean = np.array(saved["mean"])
    calibrator.std = np.array(saved["std"])
    calibrator.weights = np.array(saved["weights"])
    calibrator.bias = saved["bias"]
    return calibrator, data["threshold"]
