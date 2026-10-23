from dc.corpus import load_corpus
from dc.embed import embed_pages, load_model


def test_one_row_per_page():
    pages = load_corpus()
    model = load_model()
    vectors = embed_pages(model, pages)
    assert vectors.shape[0] == len(pages)
