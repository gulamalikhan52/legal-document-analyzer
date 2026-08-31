from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.workflow import build_workflow


app = FastAPI(
    title="LegalAI",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


# Build workflow once when the application starts
graph = build_workflow()


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

    try:
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

    except Exception as e:
        print(f"ERROR in /ask: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {str(e)}"
        )