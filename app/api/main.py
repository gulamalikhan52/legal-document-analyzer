from fastapi import FastAPI
from pydantic import BaseModel

from app.graph.workflow import build_workflow
from app.retrieval.hybrid import initialize_retrieval


app = FastAPI(
    title="LegalAI",
    version="1.0.0",
)


graph = build_workflow()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    print("\n" + "=" * 70)
    print("STARTING LEGALAI BACKEND")
    print("=" * 70)

    initialize_retrieval()

    print("\nLEGALAI BACKEND READY")


@app.get("/")
def root():
    return {
        "message": "LegalAI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):

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