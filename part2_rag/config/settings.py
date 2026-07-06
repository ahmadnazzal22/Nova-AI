import os
from pydantic_settings import BaseSettings
from typing import ClassVar


class Settings(BaseSettings):
    app_name: str = "RAG Platform API"
    app_version: str = "3.1.0"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"
    allowed_hosts: str = "*"

    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rag.db')}")
    db_pool_size: int = 10
    db_max_overflow: int = 20

    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 300
    redis_semantic_threshold: float = 0.92

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "rag_documents"
    qdrant_vector_size: int = 384
    qdrant_namespace_per_user: bool = True

    s3_endpoint: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_bucket: str = "rag-documents"
    s3_region: str = "us-east-1"

    jwt_secret: str = "change-me-in-production-rag-system-2024"
    jwt_access_expire: int = 3600
    jwt_refresh_expire: int = 2592000

    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 60
    groq_api_key: str = ""
    preferred_models: list[str] = ["phi3:mini", "phi4", "llama3.2", "gemma4", "gemma3", "llama3.2:1b", "tinyllama", "mistral"]
    weak_models: set[str] = {"llama3.2:1b", "tinyllama"}
    max_model_size_gb: float = 5.0
    max_context_chars: int = 6000
    retrieval_k: int = 3
    live_chunk_size: int = 300
    live_chunk_overlap: int = 60

    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "langchain")

    rate_limit_requests: int = 60
    rate_limit_window: int = 60

    otlp_endpoint: str = ""
    sentry_dsn: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()
