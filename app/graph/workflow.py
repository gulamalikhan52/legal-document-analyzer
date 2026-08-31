from langgraph.graph import StateGraph, START, END

from app.graph.state import GraphState
from app.retrieval.hybrid import hybrid_search
from app.retrieval.reranker import LegalReranker
from app.generation.answer import generate_answer


_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        print("\nLoading Legal Reranker...")
        _reranker = LegalReranker()

    return _reranker


def retrieve_node(state: GraphState):

    query = state["query"]

    print("\nRetrieving documents...")

    documents = hybrid_search(
        query,
        vector_k=15,
        bm25_k=15,
        candidate_k=20,
    )

    print(f"Hybrid candidates: {len(documents)}")

    return {
        "documents": documents
    }


def rerank_node(state: GraphState):

    query = state["query"]
    documents = state["documents"]

    print("\nReranking documents...")

    reranker = get_reranker()

    reranked = reranker.rerank(
        query,
        documents,
        top_k=5,
    )

    reranked_documents = [
        document
        for document, score in reranked
    ]

    print(
        f"Documents after reranking: "
        f"{len(reranked_documents)}"
    )

    return {
        "reranked_documents": reranked_documents
    }


def check_relevance_node(state: GraphState):

    documents = state["reranked_documents"]

    print("\nChecking retrieved context...")

    relevant = bool(documents)

    print(f"Relevant context: {relevant}")

    return {
        "relevant": relevant
    }


def route_after_relevance(state: GraphState):

    return "generate"


def generate_node(state: GraphState):

    query = state["query"]
    documents = state["reranked_documents"]

    print("\nGenerating answer...")

    answer = generate_answer(
        query,
        documents,
    )

    return {
        "answer": answer
    }


def build_workflow():

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "rerank",
        rerank_node,
    )

    workflow.add_node(
        "check_relevance",
        check_relevance_node,
    )

    workflow.add_node(
        "generate",
        generate_node,
    )

    workflow.add_edge(
        START,
        "retrieve",
    )

    workflow.add_edge(
        "retrieve",
        "rerank",
    )

    workflow.add_edge(
        "rerank",
        "check_relevance",
    )

    workflow.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate": "generate",
        },
    )

    workflow.add_edge(
        "generate",
        END,
    )

    return workflow.compile()


if __name__ == "__main__":

    query = "What are the principles governing bail?"

    graph = build_workflow()

    result = graph.invoke(
        {
            "query": query,
            "documents": [],
            "reranked_documents": [],
            "relevant": False,
            "answer": "",
        }
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(result["answer"])