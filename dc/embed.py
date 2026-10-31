import numpy as np
from model2vec import StaticModel

MODEL_NAME = "minishlab/potion-base-8M"


def load_model():
    return StaticModel.from_pretrained(MODEL_NAME)


def embed_pages(model, pages):
    texts = []
    for doc_id, text in pages:
        texts.append(text)
    return model.encode(texts)


def search(model, vectors, query, k=3):
    # Meaning search: score every page by cosine similarity to the query, return the top k.
    q = model.encode([query])[0]
    lengths = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q)
    scores = vectors @ q / lengths
    order = np.argsort(-scores)[:k]
    return order, scores[order]
