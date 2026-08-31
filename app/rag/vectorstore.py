from pathlib import Path

from langchain_chroma import Chroma

from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embedding_model
from app.rag.ingestion import load_documents


BASE_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "legal_documents"


def create_vectorstore():
    print("Loading documents...")

    documents = load_documents()

    print("Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Creating Chroma vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print("Vector store created successfully!")
    print(f"Location: {CHROMA_DIR}")

    return vectorstore


def get_vectorstore():
    """
    Load the existing Chroma vector store.

    The vector store must already exist at CHROMA_DIR.
    """

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma vector store not found at: {CHROMA_DIR}. "
            "Create the vector store before starting the application."
        )

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


if __name__ == "__main__":
    create_vectorstore()