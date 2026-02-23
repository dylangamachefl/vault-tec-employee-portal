"""FastAPI application entrypoint — scaffold for Phase 2."""

from fastapi import FastAPI

app = FastAPI(title="Vault-Tec Employee Portal API", version="0.1.0")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "message": "Vault-Tec systems nominal."}
