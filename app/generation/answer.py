from app.generation.llm import get_llm


def build_context(documents):
    context = []

    for document in documents:
        source = document.metadata.get("source_file")
        page = document.metadata.get("page")

        context.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context)


def generate_answer(query, documents):

    context = build_context(documents)

    llm = get_llm()

    prompt = f"""
Answer the following question using only the provided legal documents.

Question:
{query}

Legal Documents:
{context}

Rules:
- Use only information present in the documents.
- Do not invent or assume legal information.
- If the documents do not contain enough information, clearly say that the available documents do not provide enough information.
- Give a clear and concise answer.
- Mention the source document and page where relevant.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    print("answer.py is ready for LangGraph.")