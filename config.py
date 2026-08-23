from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    MISTRAL_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    TECHXBYTES_API_KEY: Optional[str] = None

    # Google Cloud (Vertex AI) - Prioritized to consume $300 GCP Credits first
    USE_VERTEX_AI: bool = True
    GCP_PROJECT_ID: Optional[str] = "project-8e0caa5b-233c-4e40-823"
    GCP_LOCATION: str = "us-central1"

    # Configuration
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    APP_ENV: str = "development"
    PORT: int = 8080

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

settings = Settings()
