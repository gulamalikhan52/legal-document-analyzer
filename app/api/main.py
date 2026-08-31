from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.workflow import build_workflow


app = FastAPI(
    title="Legal Document Analyzer",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "Legal Document Analyzer API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    graph = build_workflow()

    result = graph.invoke(
        {
            "query": request.query,
            "documents": [],
            "reranked_documents": [],
            "relevant": False,
            "answer": "",
        }
    )

    return {
        "answer": result["answer"]
    }