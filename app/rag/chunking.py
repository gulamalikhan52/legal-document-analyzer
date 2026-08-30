import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.ingestion import load_documents


def is_english_text(text: str) -> bool:
    """
    Check whether the text is predominantly English/ASCII text.

    Legal documents can contain symbols, numbers and punctuation,
    so we don't require every character to be English.
    """

    if not text or len(text.strip()) < 100:
        return False

    # English alphabet characters
    english_chars = len(re.findall(r"[A-Za-z]", text))

    # Non-ASCII alphabetic characters
    non_english_chars = len(
        re.findall(r"[^\x00-\x7F]", text)
    )

    total_letters = english_chars + non_english_chars

    if total_letters == 0:
        return False

    english_ratio = english_chars / total_letters

    return english_ratio >= 0.70


def chunk_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    # First create chunks
    chunks = text_splitter.split_documents(documents)

    print(f"Chunks before filtering: {len(chunks)}")

    # Keep predominantly English chunks
    filtered_chunks = [
        chunk
        for chunk in chunks
        if is_english_text(chunk.page_content)
    ]

    print(f"Chunks after English filtering: {len(filtered_chunks)}")
    print(f"Chunks removed: {len(chunks) - len(filtered_chunks)}")

    return filtered_chunks


if __name__ == "__main__":

    documents = load_documents()

    print(f"\nTotal pages loaded: {len(documents)}")

    chunks = chunk_documents(documents)

    print(f"\nTotal usable chunks: {len(chunks)}")

    print("\n" + "=" * 60)
    print("SAMPLE CHUNK")
    print("=" * 60)

    print(chunks[0].page_content)

    print("\n" + "=" * 60)
    print("CHUNK METADATA")
    print("=" * 60)

    print(chunks[0].metadata)