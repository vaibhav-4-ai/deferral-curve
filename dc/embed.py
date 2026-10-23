from model2vec import StaticModel

MODEL_NAME = "minishlab/potion-base-8M"


def load_model():
    return StaticModel.from_pretrained(MODEL_NAME)


def embed_pages(model, pages):
    texts = []
    for doc_id, text in pages:
        texts.append(text)
    return model.encode(texts)
