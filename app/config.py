from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./essaylense.db"
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    # Grading (LLM). If no key is set, a heuristic fallback grader is used.
    anthropic_api_key: str | None = None
    grading_model: str = "claude-sonnet-5-5"

    # Essay text encryption at rest (Fernet key, base64). Auto-derived from jwt_secret if empty.
    data_key: str | None = None

    # HTR
    htr_weights_path: str = "weights/crnn_htr.pt"
    htr_min_confidence: float = 0.45
    upload_dir: str = "uploads"
    max_upload_mb: int = 15

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg://", 1)
        return value


settings = Settings()
