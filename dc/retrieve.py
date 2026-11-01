import numpy as np

from dc import bm25, embed


def _scale(scores):
    # Squash scores into 0..1 so word scores and meaning scores can be averaged fairly.
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros(len(scores))
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """Finds the pages most relevant to a question, using word match, meaning match, or both."""

    def __init__(self, pages):
        self.pages = pages
        self.word_index = bm25.build_index(pages)
        self.model = embed.load_model()
        self.vectors = embed.embed_pages(self.model, pages)

    def _full_scores(self, query):
        # Score every page (not just the top k) with each librarian.
        n = len(self.pages)
        word = np.zeros(n)
        positions, scores = bm25.search(self.word_index, query, k=n)
        word[positions] = scores
        meaning = np.zeros(n)
        positions, scores = embed.search(self.model, self.vectors, query, k=n)
        meaning[positions] = scores
        return word, meaning

    def raw_features(self, query):
        # Absolute (not rescaled) signals of how well the best page matches. Used later as confidence inputs.
        word, meaning = self._full_scores(query)
        meaning_sorted = np.sort(meaning)[::-1]
        word_sorted = np.sort(word)[::-1]
        return {
            "top_meaning": float(meaning_sorted[0]),
            "meaning_gap": float(meaning_sorted[0] - meaning_sorted[1]),
            "top_word": float(word_sorted[0]),
            "word_gap": float(word_sorted[0] - word_sorted[1]),
        }

    def search(self, query, k=3, mode="hybrid"):
        # Returns [(doc_id, score), ...] best first. mode: "hybrid", "word" or "meaning".
        word, meaning = self._full_scores(query)
        if mode == "word":
            combined = _scale(word)
        elif mode == "meaning":
            combined = _scale(meaning)
        else:
            combined = (_scale(word) + _scale(meaning)) / 2
        order = np.argsort(-combined)[:k]
        return [(self.pages[i][0], float(combined[i])) for i in order]
