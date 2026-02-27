from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str = ""
    llm_model: str = "gemma-3-27b-it"
    database_url: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "vault_documents"
    jwt_secret_key: str = ""
    retrieval_dedup_threshold: float = 0.95

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64


settings = Settings()
