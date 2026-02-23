from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    google_api_key: str = ""
    llm_model: str = "gemma-3-27b-it"
    database_url: str = ""
    chroma_host: str = "localhost"
    chroma_port: int = 8002
    jwt_secret_key: str = ""


settings = Settings()
