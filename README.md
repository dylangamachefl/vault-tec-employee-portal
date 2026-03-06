# Vault-Tec Employee Portal

> *"The data thanks you for your contributions." — Vault-Tec IT Division*

An AI-powered internal knowledge portal for Vault-Tec Industries. Employees ask questions about company policy and receive accurate, cited answers — restricted by role. Admins get a dashboard showing usage patterns and knowledge base health.

This is a **portfolio capstone** demonstrating end-to-end ownership of an AI product: from raw documents to a secured, monitored, production-style application.

---

## Architecture

```
[React Frontend]  ←→  [FastAPI Backend]  ←→  [ChromaDB]
  Employee chat         JWT auth               Vector embeddings
  Admin dashboard       Role resolution        Metadata filters
  Document Viewer                              Client-side embeddings
                        Audit logging
                              ↕
                        [PostgreSQL]
                         Users, roles
                         Audit trail
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ via `pyenv-win` |
| Package Manager | `uv` |
| AI Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Generation | `gemma-3-27b-it` via Google AI Studio |
| Vector DB | ChromaDB |
| Relational DB | PostgreSQL 15 |
| Backend | FastAPI + PyJWT |
| Frontend | React + Vite + TailwindCSS |
| Evaluation | RAGAS |
| Containers | Docker Compose |

---

## Getting Started

### Prerequisites

- Python 3.11+ installed via [`pyenv-win`](https://github.com/pyenv-win/pyenv-win)
- [`uv`](https://docs.astral.sh/uv/) — `pip install uv`
- Docker Desktop (with Compose v2)
- Git for Windows (includes Git Bash — required for pre-commit hooks)

> ✅ All development runs natively in **Windows PowerShell**. WSL2 is not required.

#### Install Python via pyenv-win

```powershell
# If needed, allow scripts first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"
./install-pyenv-win.ps1

# Reopen PowerShell, then:
pyenv install 3.11.9
pyenv global 3.11.9
python --version   # Should output: Python 3.11.9
```

#### Install uv

```powershell
pip install uv
uv --version
```

### Setup

```powershell
# Install dependencies and pre-commit hooks
./scripts/dev.ps1 setup

# Copy and fill in your secrets
Copy-Item .env.example .env
# Edit .env and add your GOOGLE_API_KEY and JWT_SECRET_KEY
```

### Running Locally

```powershell
# Start database and backend services
./scripts/dev.ps1 docker-up

# Verify services are up
docker compose ps
```

| Service | URL |
|---|---|
| FastAPI backend | http://localhost:8000 |
| React frontend | http://localhost:3000 |
| ChromaDB | http://localhost:8002 |
| PostgreSQL | localhost:5432 |

```powershell
# Start the frontend dev server (in a separate terminal)
cd frontend
npm install
npm run dev
```

### Development Commands

```powershell
./scripts/dev.ps1 lint         # Run ruff linter with auto-fix
./scripts/dev.ps1 test         # Run pytest
./scripts/dev.ps1 docker-down  # Stop all containers
```

> ⚠️ If PowerShell blocks the script, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry.

---

## Project Structure

```
employee-portal/
├── backend/          # FastAPI API, pipelines, and data (Deployed to server)
│   ├── src/          # Python source code
│   ├── tests/        # Unit & eval tests
│   ├── data/         # Source docs & chunked JSON
│   ├── Dockerfile
│   └── pyproject.toml
├── frontend/         # React + Vite frontend (Deployed to Vercel)
├── docker/
│   └── Dockerfile.frontend
├── scripts/
│   └── dev.ps1       # PowerShell dev commands
└── docker-compose.yml
```

> ⚠️ **Windows path rule:** All file path construction in `backend/src/` must use `pathlib.Path` — never hardcoded forward slashes. Use `Path("data") / "raw"`, not `"data/raw"`.

---

## Three-Phase Build Plan

| Phase | Goal | Gate |
|---|---|---|
| **Phase 1** | Base RAG Pipeline | RAGAS evaluation passes |
| **Phase 2** | RBAC Security Gateway | Cross-role leakage tests pass |
| **Phase 3** | Automated Quality Monitor | Admin dashboard live |

---

## Evaluation

`tests/eval/golden_qa_pairs.json` contains **34 ground-truth Q&A pairs** covering all four access levels (`general`, `hr`, `marketing`, `admin`), grounded in the 15 Vault-Tec source documents. This file is the objective measure of RAG pipeline quality and must not be modified without Architect approval.

---

*VT-SPEC-ENV-001 | Lead Architect | 2026-02-23*
