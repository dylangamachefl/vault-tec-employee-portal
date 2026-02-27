# Task 5D-a — Developer Specification
## Migrate Vector Store from ChromaDB to Qdrant

---

**Task ID:** Task 5D-a — Qdrant Migration  
**Phase:** Phase 1: Base RAG & Ingestion (Pre-Gate Migration)  
**Depends On:** Task 5C (complete) — documents chunked and embedded, 34-query eval harness functional  
**Blocks:** Task 5D-b (Hybrid Search), Task 6A (Streamlit UI), all Phase 2 RBAC work  
**Assigned To:** AI Developer  
**Prepared By:** Lead AI Solutions Architect / TPM  

---

## 1. Why We Are Doing This Now

ChromaDB was an appropriate choice for local prototyping. It is being replaced before Phase 2 for three concrete reasons:

**Reason 1 — Native Hybrid Search.** Qdrant supports sparse+dense hybrid retrieval with built-in RRF natively. The `rank-bm25` manual implementation in Task 5D-b would require managing parallel list alignment and custom fusion logic that Qdrant handles internally. This is strictly less code and strictly fewer failure modes.

**Reason 2 — Production-Grade RBAC Filtering.** Phase 2 requires metadata filtering by role (General, HR, Marketing, Admin) at query time. Qdrant's filtering model — payload filters applied during the ANN search itself — is significantly more robust than ChromaDB's post-retrieval metadata filtering. Migrating after the FastAPI backend is built around ChromaDB patterns doubles the migration cost.

**Reason 3 — Portfolio Signal.** Qdrant is the vector database that appears most frequently in production RAG architecture discussions in 2025–2026. It is a stronger portfolio signal than ChromaDB.

**The tradeoff:** this is not a dependency swap. It is a full migration — new Docker service, updated `docker-compose.yml`, re-ingestion of all documents, updated ingest and retrieval pipeline code, and all unit tests that mock ChromaDB's response format need updating. Scope is real. Doing it now is still cheaper than doing it after Phase 2.

---

## 2. Scope

### In Scope
- Replace ChromaDB container with Qdrant in `docker-compose.yml`
- Update `src/pipelines/persist.py` to write to Qdrant
- Update `src/retrieval/retriever.py` to read from Qdrant (dense only — sparse/hybrid comes in 5D-b)
- Update `src/config.py` with Qdrant connection settings
- Re-ingest all documents into a new Qdrant collection: `vault_documents_256`
- Update all unit tests that mock or reference ChromaDB client/response format
- Update `docker-compose.yml` to add Qdrant service and remove ChromaDB service
- Update `Dev set up guide` / README with new stack

### Out of Scope
- Sparse vector indexing (Task 5D-b)
- Hybrid search / RRF fusion (Task 5D-b)
- RBAC payload filtering (Phase 2)
- Any frontend changes

---

## 3. New Dependency

```bash
uv add qdrant-client
```

Remove ChromaDB:

```bash
uv remove chromadb
```

---

## 4. Docker Compose Changes

### 4.1 Remove ChromaDB Service

Remove the `chromadb` service block entirely from `docker-compose.yml`.

### 4.2 Add Qdrant Service

```yaml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_storage:/qdrant/storage
  environment:
    - QDRANT__SERVICE__HTTP_PORT=6333
    - QDRANT__SERVICE__GRPC_PORT=6334

volumes:
  qdrant_storage:
```

Qdrant exposes two ports: `6333` for REST/HTTP and `6334` for gRPC. The Python client defaults to REST. Both should be mapped.

### 4.3 Update Service Dependencies

Any service in `docker-compose.yml` that previously declared `depends_on: chromadb` should be updated to `depends_on: qdrant`.

---

## 5. Environment Variables

### 5.1 Update `.env.example`

Remove:
```env
CHROMA_HOST=localhost
CHROMA_PORT=8000
```

Add:
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=vault_documents_256
```

### 5.2 Update `src/config.py`

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    google_api_key: str = ""
    llm_model: str = "gemma-3-27b-it"
    database_url: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "vault_documents_256"
```

---

## 6. Persist Pipeline Changes (`src/pipelines/persist.py`)

### 6.1 Client Initialisation

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
```

### 6.2 Collection Creation

Create the collection once on first run. Check for existence before creating to make the operation idempotent.

```python
def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
```

`vector_size` must match the embedding model's output dimension. For `sentence-transformers/all-MiniLM-L6-v2` this is **384**. For `text-embedding-004` (Google) this is **768**. Use whatever model is currently configured — do not hardcode 384 unless that is what the project uses.

### 6.3 Upsert Points

Qdrant uses `PointStruct` objects. Each point requires a numeric or UUID id, a vector, and a payload (the metadata).

```python
from uuid import uuid4

def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: list[ProcessedChunk]  # existing Pydantic model
) -> None:
    points = [
        PointStruct(
            id=str(uuid4()),
            vector=chunk.embedding,        # list[float]
            payload={
                "chunk_id":       chunk.chunk_id,
                "text":           chunk.text,
                "source":         chunk.metadata.source,
                "access_level":   chunk.metadata.access_level,
                "department":     chunk.metadata.department,
                "doc_type":       chunk.metadata.doc_type,
                "version_date":   chunk.metadata.version_date,
                "section_header": chunk.metadata.section_header,
                "status":         chunk.metadata.status,
            }
        )
        for chunk in chunks
    ]
    client.upsert(collection_name=collection_name, points=points)
```

> ⚠️ **Critical: Payload field naming.** All metadata fields that Phase 2 RBAC will filter on — specifically `access_level` and `department` — must be present in the payload on every point. A missing field at ingest time means a broken filter at query time. Do not use a catch-all dict; map every field explicitly as shown above.

### 6.4 Idempotent Re-ingestion

The ingest pipeline should delete and recreate the collection if it already exists, to support clean re-ingestion runs.

```python
def reset_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        client.delete_collection(collection_name)
    ensure_collection(client, collection_name, vector_size)
```

---

## 7. Retrieval Pipeline Changes (`src/retrieval/retriever.py`)

This task implements **dense-only retrieval** in Qdrant. Sparse/hybrid comes in 5D-b.

### 7.1 Dense Query

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

def dense_retrieve(
    client: QdrantClient,
    collection_name: str,
    query_embedding: list[float],
    k: int = 5,
    access_filter: dict | None = None   # placeholder for Phase 2 RBAC
) -> list[dict]:

    query_filter = None
    if access_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=access_filter['field'],
                    match=MatchValue(value=access_filter['value'])
                )
            ]
        )

    results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k,
        with_payload=True,
        query_filter=query_filter
    )

    return [
        {
            'id':       str(result.id),
            'text':     result.payload['text'],
            'metadata': {k: v for k, v in result.payload.items() if k != 'text'},
            'score':    result.score
        }
        for result in results
    ]
```

Note the `access_filter` parameter: it accepts `None` now and will be populated by the FastAPI role middleware in Phase 2. Scaffold it now so Phase 2 only needs to pass the argument — it does not need to modify this function.

### 7.2 Output Format

The output format is identical to what Task 5D-b's `hybrid_retrieve()` will return:

```python
[
    { 'id': str, 'text': str, 'metadata': dict, 'score': float },
    ...
]
```

The LLM chain reads from this format. No chain changes are required.

---

## 8. Re-ingestion

After the pipeline changes are complete, re-ingest all documents into Qdrant:

```bash
uv run python src/pipelines/ingest.py --reset
```

The `--reset` flag should trigger `reset_collection()` before ingestion. After completion, verify:

```bash
uv run python -c "
from qdrant_client import QdrantClient
from src.config import settings
client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
info = client.get_collection(settings.qdrant_collection_name)
print(f'Points in collection: {info.points_count}')
"
```

The point count must be non-zero and should match the total chunk count from the Task 5C ingest run.

---

## 9. Unit Test Updates

All tests that currently mock `chromadb.Client`, `chromadb.Collection`, or parse ChromaDB's nested response format (`result['ids'][0]`, `result['documents'][0]`) must be updated to mock `QdrantClient` and parse the flat `ScoredPoint` response format.

### 9.1 Mock Pattern for QdrantClient

```python
from unittest.mock import MagicMock, patch
from qdrant_client.models import ScoredPoint

mock_point = ScoredPoint(
    id="test-uuid-001",
    version=0,
    score=0.92,
    payload={
        "text": "The Tier 2 treatment is Rad-X.",
        "source": "Doc02_Radiation_Sickness_Symptom_Guide.docx",
        "access_level": "general",
        "department": "medical",
        "doc_type": "sop",
        "version_date": "2076-08",
        "section_header": "Section 3: Approved Treatment",
        "status": "active",
    },
    vector=None
)

with patch('src.retrieval.retriever.QdrantClient') as mock_client:
    mock_client.return_value.search.return_value = [mock_point]
    # run test
```

### 9.2 Tests That Must Be Updated

Review and update every test in `tests/unit/` that references:
- `chromadb`
- `collection.query()`
- `result['ids'][0]`
- `result['documents'][0]`
- `result['metadatas'][0]`

All 40 existing tests must still pass after the update.

---

## 10. Execution Order

> ⚠️ **Do not run ingest before the persist pipeline is tested. Do not run tests before Docker is up.**

1. Update `docker-compose.yml` (remove ChromaDB, add Qdrant)
2. Run `docker compose up qdrant -d` and verify Qdrant is accessible at `http://localhost:6333`
3. Update dependencies (`uv add qdrant-client`, `uv remove chromadb`)
4. Update `src/config.py` and `.env.example`
5. Update `src/pipelines/persist.py`
6. Update `src/retrieval/retriever.py`
7. Update all affected unit tests
8. Run full unit test suite — all 40 must pass before proceeding
9. Run `uv run python src/pipelines/ingest.py --reset`
10. Verify point count in collection (Section 8)
11. Run a single manual spot check query through the retrieval chain end-to-end

---

## 11. Acceptance Criteria

| AC | Criterion |
|---|---|
| AC1 | Qdrant container starts via `docker compose up` without error |
| AC2 | `uv remove chromadb` produces no import errors anywhere in the codebase |
| AC3 | All 40 existing unit tests pass with updated Qdrant mocks |
| AC4 | `vault_documents_256` collection exists in Qdrant with non-zero point count matching Task 5C chunk count |
| AC5 | Every point in the collection has `access_level` and `department` fields in its payload |
| AC6 | A manual dense retrieval query returns 5 results with the correct `id`, `text`, `metadata`, `score` structure |
| AC7 | No ChromaDB references remain anywhere in `src/` |

---

## 12. Deliverables

| File | Change |
|---|---|
| `docker-compose.yml` | Qdrant service added, ChromaDB service removed |
| `.env.example` | Qdrant env vars added, ChromaDB vars removed |
| `src/config.py` | Qdrant settings added |
| `src/pipelines/persist.py` | Rewritten for Qdrant upsert |
| `src/retrieval/retriever.py` | Rewritten for Qdrant dense search with `access_filter` scaffold |
| `tests/unit/` (all affected files) | ChromaDB mocks replaced with QdrantClient mocks |

---

*Vault-Tec Corporation — Engineering Division | Task 5D-a | Phase 1 Pre-Gate Migration*

*"ChromaDB served us faithfully. Its contributions to this project will be noted in the logs, then the logs will be purged. Such is the way of infrastructure migrations."*
