import os
from typing import List, Union
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "DermAI: Skin Disease Detector"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dermai-deep-learning-infosys-springboard-secret-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Defaults to SQLite async for self-contained execution, accepts PostgreSQL via DATABASE_URL
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./dermai.db"
    )
    
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    ORIGINALS_DIR: str = os.getenv("ORIGINALS_DIR", "./uploads/originals")
    PREPROCESSED_DIR: str = os.getenv("PREPROCESSED_DIR", "./uploads/preprocessed")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "./saved_models")
    
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

    class Config:
        env_file = ".env"
        extra = "allow"
        protected_namespaces = ()


settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.ORIGINALS_DIR, exist_ok=True)
os.makedirs(settings.PREPROCESSED_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
