# VAULT-TEC EMPLOYEE PORTAL
## Project Overview — Developer Reference
**Issued by:** Lead AI Solutions Architect
**Date:** 2026-02-23
**Keep this document handy. It is your north star.**

---

## WHAT WE ARE BUILDING

A unified, AI-powered internal knowledge portal for a fictional company (Vault-Tec Industries). Employees log in, ask questions about company policy, and get back accurate, cited answers — but only for documents they are authorized to see. Admins get a separate dashboard showing usage patterns and knowledge base health.

This is a **portfolio capstone** combining three distinct AI engineering roles into one cohesive system. When complete, it demonstrates end-to-end ownership of an AI product: from raw documents to a secured, monitored, production-style application.

---

## THE THREE PHASES (Build in Order — No Exceptions)

### Phase 1 — Base RAG Pipeline
*"Can the system answer questions accurately?"*

Ingest 16 corporate documents (PDFs, DOCX, Markdown), clean and chunk the text, tag every chunk with rich metadata, embed using `sentence-transformers`, and store in ChromaDB. Build a LangChain retrieval chain that answers employee questions with grounded source citations. Evaluate accuracy using a 30+ item golden Q&A test suite via RAGAS.

**You are here. This phase must pass evaluation before Phase 2 starts.**

---

### Phase 2 — RBAC Security Gateway
*"Does the system answer questions only for the right people?"*

Build a FastAPI backend with JWT authentication. Define four user roles: `admin`, `hr`, `marketing`, `general`. When a user queries the chatbot, their role is resolved and passed as a metadata filter to ChromaDB — so an HR question about payroll policy is invisible to a Marketing employee. Every query is written to an audit log (PostgreSQL). Prove data isolation with automated cross-role leakage tests.

**Phase 2 cannot start until Phase 1 evaluation results are acceptable.**

---

### Phase 3 — Automated Quality Monitor
*"Does the system know when its knowledge base is going stale or broken?"*

Scheduled background jobs continuously scan the vector database. A staleness scorer flags documents whose `doc_date` metadata is old or whose text references outdated time periods. A cosine-similarity duplicate detector flags redundant chunks. An LLM-powered contradiction detector retrieves chunks on the same topic and checks for conflicting statements (e.g., two policy documents giving different instructions). All findings surface in an admin Streamlit dashboard alongside audit log metrics.

---

## THE ARCHITECTURE AT A GLANCE

```
[Streamlit UI]  ←→  [FastAPI Backend]  ←→  [ChromaDB]
  Employee chat         JWT auth               Vector embeddings
  Admin dashboard       Role resolution        Metadata filters
                        Audit logging
                              ↕
                        [PostgreSQL]
                         Users, roles
                         Audit trail
```

---

## TECH STACK (Fixed — Do Not Substitute)

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11/3.12 via `pyenv-win` | |
| Package manager | `uv` | Replaces pip/venv |
| AI orchestration | LangChain | Retrieval chain, text splitting |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, fast, no API cost |
| Generation | `gemma-3-27b-it` via Google AI Studio | The LLM for answers + quality checks |
| Vector DB | ChromaDB | Metadata filtering for RBAC |
| Relational DB | PostgreSQL | Users, roles, audit logs |
| Backend | FastAPI + PyJWT | Secure API gateway |
| Frontend | Streamlit | Employee chat + admin dashboard |
| Evaluation | RAGAS | Measures faithfulness, precision, recall |
| Containers | Docker Compose | Runs all services locally |

**One important distinction:** `sentence-transformers` handles embeddings. The Google AI / `gemma` model handles text generation and quality analysis. They are intentionally separate — swapping one should never require touching the other.

---

## THE DATASET

16 fictional "Vault-Tec Corporation" internal documents spanning four access levels. The dark humor in the documents is intentional — they are the synthetic corporate dataset that makes this portfolio project memorable. Treat them as real enterprise documents for all engineering purposes.

| Access Level | Example Documents |
|---|---|
| `general` | Code of Conduct, Radiation Symptom Guide, Evacuation Procedures |
| `hr` | Overseer Compensation, GOAT Exam Administration, Resident Reassignment |
| `marketing` | Tricentennial Promotional Strategy, GECK Advertising Guidelines |
| `admin` | ZAX Mainframe Root Access, Vault Door Override Protocol, Pip-Boy SOPs |

---

## THE GOLDEN RULE

**Pipeline before polish.** The retrieval chain must be built and evaluated before the API is built. The API must be secured and tested before the UI is built. The quality monitor is last.

If you are ever uncertain about build order, ask the Architect.

---

## KEY FILES TO KNOW

| File | Purpose |
|---|---|
| `src/config.py` | All environment variables — import `settings` from here, never use `os.environ` directly |
| `src/pipelines/models.py` | `ChunkMetadata` Pydantic model — the data contract for the entire system |
| `src/pipelines/ingest.py` | The ingestion pipeline entrypoint |
| `tests/eval/golden_qa_pairs.json` | The evaluation ground truth — do not modify without Architect approval |
| `docker-compose.yml` | Starts all four services: api, frontend, postgres, chromadb |
| `scripts/dev.ps1` | PowerShell dev commands — `setup`, `test`, `docker-up`, `docker-down`, `lint` |

> ⚠️ **Windows path rule:** All file paths in Python must use `pathlib.Path`. Never hardcode `/` separators. The pipeline runs locally on Windows but inside Linux Docker containers — `pathlib` handles both transparently.

---

*"The data thanks you for your contributions." — Vault-Tec IT Division*