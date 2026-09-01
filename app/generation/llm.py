from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set.")

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_tokens=1500,
        api_key=api_key,
        timeout=60,
        max_retries=2,
    )

    return llm


if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke(
        "Explain bail in one sentence."
    )

    print(response.content)