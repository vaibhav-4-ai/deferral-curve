from dc.corpus import load_corpus
from dc.eval_retrieval import load_golden, recall_at_k
from dc.retrieve import HybridRetriever


def test_meaning_search_understands_paraphrase():
    retriever = HybridRetriever(load_corpus())
    top = retriever.search("when will I get my money back", k=3, mode="meaning")
    assert "refund-timeline" in [doc_id for doc_id, _ in top]


def test_hybrid_finds_the_right_page():
    retriever = HybridRetriever(load_corpus())
    top = retriever.search("coupon expiry date", k=1)
    assert top[0][0] == "coupon-codes"


def test_recall_is_not_terrible():
    # A floor, not a target: if hybrid recall@3 falls below this, something is broken.
    retriever = HybridRetriever(load_corpus())
    assert recall_at_k(retriever, load_golden(), k=3, mode="hybrid") >= 0.5
