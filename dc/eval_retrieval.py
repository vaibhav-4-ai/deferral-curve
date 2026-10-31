import json
from pathlib import Path

from dc.corpus import load_corpus
from dc.retrieve import HybridRetriever

GOLDEN = Path(__file__).parent.parent / "evals" / "golden.jsonl"


def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retriever, golden, k, mode):
    # Of the questions that have a right page, how often is it among the top k pages found?
    hits = 0
    total = 0
    for row in golden:
        needed = row["supporting_doc_ids"]
        if not needed:
            continue
        found = [doc_id for doc_id, _ in retriever.search(row["query"], k=k, mode=mode)]
        total += 1
        if any(doc_id in found for doc_id in needed):
            hits += 1
    return hits / total


def report():
    retriever = HybridRetriever(load_corpus())
    golden = load_golden()
    print("recall@k over the golden set (questions that have a right page)")
    for mode in ("word", "meaning", "hybrid"):
        parts = [f"@{k}={recall_at_k(retriever, golden, k, mode):.2f}" for k in (1, 3, 5)]
        print(f"  {mode:8s} " + "  ".join(parts))
