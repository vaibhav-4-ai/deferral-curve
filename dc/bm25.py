import bm25s


def build_index(pages):
    texts = []
    for doc_id, text in pages:
        texts.append(text)
    tokens = bm25s.tokenize(texts)
    retriever = bm25s.BM25()
    retriever.index(tokens)
    return retriever

def search(retriever, query, k=3):
    query_tokens = bm25s.tokenize(query)
    doc_numbers, scores = retriever.retrieve(query_tokens, k=k)
    return doc_numbers[0], scores[0]
