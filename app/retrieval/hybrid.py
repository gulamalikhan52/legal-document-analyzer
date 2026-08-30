from rank_bm25 import BM25Okapi

from app.rag.chunking import chunk_documents
from app.rag.ingestion import load_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import get_vectorstore


def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):
    scores = {}
    documents = {}

    for rank, doc in enumerate(vector_docs, start=1):
        doc_id = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content,
        )

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        documents[doc_id] = doc

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content,
        )

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        documents[doc_id] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [documents[doc_id] for doc_id, _ in ranked]


def hybrid_search(query, vector_k=15, bm25_k=15, candidate_k=20):
    print("Loading documents...")

    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    # --------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------

    print("Loading vector store...")

    vectorstore = get_vectorstore()

    vector_results = vectorstore.similarity_search(
        query,
        k=vector_k,
    )

    # --------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------

    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:bm25_k]

    bm25_results = [
        chunks[i]
        for i in top_bm25_indices
    ]

    # --------------------------------------------------
    # HYBRID / RRF
    # --------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
    )

    return hybrid_results[:candidate_k]


if __name__ == "__main__":

    query = "What are the principles governing bail?"

    results = hybrid_search(
        query,
        vector_k=15,
        bm25_k=15,
        candidate_k=20,
    )

    print("\n" + "=" * 70)
    print("HYBRID SEARCH RESULTS")
    print("=" * 70)

    for i, document in enumerate(results, start=1):

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

        print("\nContent:")
        print(document.page_content[:600])