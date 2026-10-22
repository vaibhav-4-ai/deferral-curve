from pathlib import Path

CORPUS=Path(__file__).parent.parent/"corpus"

def load_corpus():
    files = sorted(CORPUS.glob("*.md"))
    pairs = []
    for f in files:
        pairs.append((f.stem, f.read_text(encoding="utf-8")))
    return pairs
