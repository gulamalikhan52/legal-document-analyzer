from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


def load_pdf(pdf_path: Path) -> list[Document]:
    reader = PdfReader(str(pdf_path))

    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        document = Document(
            page_content=text,
            metadata={
                "source_file": pdf_path.name,
                "document_type": pdf_path.parent.name,
                "page": page_number + 1,
            },
        )

        documents.append(document)

    return documents


def load_documents() -> list[Document]:
    documents = []

    pdf_files = list(DATA_DIR.rglob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.\n")

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        docs = load_pdf(pdf_path)

        print(f"  Pages extracted: {len(docs)}")

        documents.extend(docs)

    return documents


if __name__ == "__main__":
    documents = load_documents()

    print("\n" + "=" * 60)
    print(f"Total pages loaded: {len(documents)}")
    print("=" * 60)

    if documents:
        print("\nSample document:")
        print("-" * 60)
        print(documents[0].page_content[:1000])

        print("\nMetadata:")
        print(documents[0].metadata)