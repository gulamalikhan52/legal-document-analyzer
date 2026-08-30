from sentence_transformers import CrossEncoder

from app.retrieval.hybrid import hybrid_search


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LegalReranker:

    def __init__(self):
        print("Loading reranker model...")
        self.model = CrossEncoder(MODEL_NAME)

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return ranked_results[:top_k]


if __name__ == "__main__":

    query = "What are the principles governing bail?"

    print("\nRetrieving hybrid candidates...")

    candidates = hybrid_search(
        query,
        vector_k=15,
        bm25_k=15,
        candidate_k=20,
    )

    print(
        f"Hybrid candidates retrieved: {len(candidates)}"
    )

    reranker = LegalReranker()

    results = reranker.rerank(
        query,
        candidates,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("RERANKED RESULTS")
    print("=" * 70)

    for i, (document, score) in enumerate(
        results,
        start=1,
    ):

        print(f"\nRESULT {i}")
        print("-" * 70)

        print(
            "Source:",
            document.metadata.get("source_file"),
        )

        print(
            "Document type:",
            document.metadata.get("document_type"),
        )

        print(
            "Page:",
            document.metadata.get("page"),
        )

        print(
            "Reranker score:",
            float(score),
        )

        print("\nContent:")
        print(document.page_content[:800])