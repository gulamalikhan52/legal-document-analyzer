from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.workflow import build_workflow


app = FastAPI(
    title="Legal Document Analyzer",
    version="1.0.0",
)


graph = build_workflow()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "message": "Legal Document Analyzer API is running"
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

    result = graph.invoke(
        {
            "query": request.query,
            "documents": [],
            "reranked_documents": [],
            "answer": "",
        }
    )

    return {
        "answer": result["answer"]
    }