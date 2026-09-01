from rank_bm25 import BM25Okapi

from app.rag.chunking import chunk_documents
from app.rag.ingestion import load_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vectorstore import get_vectorstore


# ============================================================
# CACHE RAG RESOURCES
# ============================================================

_documents = None
_chunks = None
_bm25 = None
_vectorstore = None


def initialize_retrieval():
    global _documents
    global _chunks
    global _bm25
    global _vectorstore

    if (
        _documents is not None
        and _chunks is not None
        and _bm25 is not None
        and _vectorstore is not None
    ):
        return

    print("\n" + "=" * 70)
    print("INITIALIZING RAG RETRIEVAL")
    print("=" * 70)

    # --------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------

    print("\nLoading documents...")

    _documents = load_documents()

    # --------------------------------------------------
    # CHUNK DOCUMENTS
    # --------------------------------------------------

    print("\nChunking documents...")

    _chunks = chunk_documents(_documents)

    print(f"Total chunks: {len(_chunks)}")

    # --------------------------------------------------
    # VECTOR STORE
    # --------------------------------------------------

    print("\nLoading vector store...")

    _vectorstore = get_vectorstore()

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print("\nBuilding BM25 index...")

    tokenized_chunks = [
        chunk.page_content.lower().split()
        for chunk in _chunks
    ]

    _bm25 = BM25Okapi(tokenized_chunks)

    print("BM25 index ready.")

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL READY")
    print("=" * 70)


def reciprocal_rank_fusion(vector_docs, bm25_docs, k=60):

    scores = {}
    documents = {}

    for rank, doc in enumerate(vector_docs, start=1):

        doc_id = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content,
        )

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        documents[doc_id] = doc

    for rank, doc in enumerate(bm25_docs, start=1):

        doc_id = (
            doc.metadata.get("source_file"),
            doc.metadata.get("page"),
            doc.page_content,
        )

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        documents[doc_id] = doc

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        documents[doc_id]
        for doc_id, _ in ranked
    ]


def hybrid_search(
    query,
    vector_k=15,
    bm25_k=15,
    candidate_k=20,
):

    # Make sure resources are loaded only once.
    initialize_retrieval()

    # --------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------

    print("\nRunning vector search...")

    vector_results = _vectorstore.similarity_search(
        query,
        k=vector_k,
    )

    # --------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------

    print("Running BM25 search...")

    query_tokens = query.lower().split()

    bm25_scores = _bm25.get_scores(query_tokens)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:bm25_k]

    bm25_results = [
        _chunks[i]
        for i in top_bm25_indices
    ]

    # --------------------------------------------------
    # HYBRID / RRF
    # --------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
    )

    print(
        f"Hybrid candidates: "
        f"{len(hybrid_results[:candidate_k])}"
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