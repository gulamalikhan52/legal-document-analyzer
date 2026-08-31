from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import CHROMA_DIR, COLLECTION_NAME

from langchain_chroma import Chroma


def get_vectorstore():
    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def search(query: str, k: int = 5):
    vectorstore = get_vectorstore()

    # Retrieve extra results so duplicates can be removed
    results = vectorstore.similarity_search_with_score(
        query,
        k=k * 2,
    )

    unique_results = []
    seen = set()

    for document, score in results:
        key = (
            document.metadata.get("source_file"),
            document.metadata.get("page"),
            document.page_content.strip(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_results.append((document, score))

        if len(unique_results) >= k:
            break

    return unique_results


if __name__ == "__main__":
    query = "What are the principles governing bail?"

    results = search(query)

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for i, (document, score) in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print("-" * 70)

        print("Source:", document.metadata.get("source_file"))
        print("Document type:", document.metadata.get("document_type"))
        print("Page:", document.metadata.get("page"))
        print("Score:", score)

        print("\nContent:")
        print(document.page_content[:800])