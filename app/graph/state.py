from typing import TypedDict
from langchain_core.documents import Document


class GraphState(TypedDict):
    query: str
    documents: list[Document]
    reranked_documents: list[Document]
    relevant: bool
    answer: str