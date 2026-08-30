from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.rag.chunking import chunk_documents
from app.rag.ingestion import load_documents


class BM25Retriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents

        # Tokenize every chunk
        corpus = [
            document.page_content.lower().split()
            for document in documents
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> list[Document]:
        query_tokens = query.lower().split()

        results = self.bm25.get_top_n(
            query_tokens,
            self.documents,
            n=k,
        )

        return results


def build_bm25_retriever():
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Building BM25 index from {len(chunks)} chunks...")

    return BM25Retriever(chunks)


if __name__ == "__main__":
    retriever = build_bm25_retriever()

    query = "What are the principles governing bail?"

    results = retriever.search(query, k=5)

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    for i, document in enumerate(results, start=1):
        print(f"\nRESULT {i}")
        print("-" * 70)

        print("Source:", document.metadata.get("source_file"))
        print("Document type:", document.metadata.get("document_type"))
        print("Page:", document.metadata.get("page"))

        print("\nContent:")
        print(document.page_content[:800])