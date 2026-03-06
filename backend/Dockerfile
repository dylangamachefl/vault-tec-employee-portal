FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .

# Install dependencies (no dev extras)
RUN uv pip install --system --no-cache \
    fastapi uvicorn[standard] pyjwt sqlalchemy psycopg2-binary \
    pydantic-settings langchain langchain-text-splitters \
    sentence-transformers google-generativeai \
    qdrant-client fastembed \
    pypdf python-docx tiktoken

# Copy source and data files
COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
