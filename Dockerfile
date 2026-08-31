FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv
RUN uv sync --frozen

COPY app ./app
COPY data ./data
COPY chroma_db ./chroma_db

EXPOSE 10000

CMD ["sh", "-c", "uv run uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT}"]