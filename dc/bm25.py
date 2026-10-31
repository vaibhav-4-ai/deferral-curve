import bm25s


def build_index(pages):
    texts = []
    for doc_id, text in pages:
        texts.append(text)
    tokens = bm25s.tokenize(texts, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    return retriever

def search(retriever, query, k=3):
    query_tokens = bm25s.tokenize(query, show_progress=False)
    doc_numbers, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)
    return doc_numbers[0], scores[0]
