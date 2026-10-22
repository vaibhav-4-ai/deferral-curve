from dc.corpus import load_corpus


def test_every_page_has_text():
    pages = load_corpus()
    assert len(pages) > 0
    for doc_id, text in pages:
        assert text.strip() != ""
