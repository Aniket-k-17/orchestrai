import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings parsed from environment variables or .env file.
    Uses Pydantic BaseSettings for strong typing and validation.
    """
    # General Project Settings
    PROJECT_NAME: str = "ML Experiment Orchestrator"
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Backend API Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    API_V1_STR: str = "/api/v1"

    # Frontend Configuration
    FRONTEND_PORT: int = 8501
    BACKEND_URL: str = "http://localhost:8000"

    # LLM & Agent API Keys
    GROQ_API_KEY: Optional[str] = None

    # Load from .env file if it exists
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def is_dev(self) -> bool:
        return self.ENV.lower() in ("dev", "development")

    @property
    def is_prod(self) -> bool:
        return self.ENV.lower() in ("prod", "production")


# Instantiate settings singleton
settings = Settings()
