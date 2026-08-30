from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    embeddings = get_embedding_model()

    text = "What are the principles governing bail?"

    vector = embeddings.embed_query(text)

    print(f"Embedding dimensions: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")