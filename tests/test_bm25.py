from dc.bm25 import build_index, search
from dc.corpus import load_corpus


def test_search_finds_the_right_page():
    pages = load_corpus()
    retriever = build_index(pages)
    doc_numbers, scores = search(retriever, "coupon expiry date", k=1)
    best = pages[doc_numbers[0]][0]
    assert best == "coupon-codes"
