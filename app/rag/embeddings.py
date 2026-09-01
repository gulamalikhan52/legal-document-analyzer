from langchain_huggingface import HuggingFaceEmbeddings

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        print("Loading embedding model...")

        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    return _embedding_model