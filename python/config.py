from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"

    # Vertex AI
    google_cloud_project: str = "bobi-rag-489509"
    google_cloud_location: str = "us-central1"       # Embedding 用
    google_cloud_llm_location: str = "us-central1"   # LLM 用（gemini-3-flash-preview 需改為 global）
    google_application_credentials: str = "/app/credentials/py-rag-sa.json"

    # Embedding
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimensions: int = 3072

    # Sparse Embedding（Hybrid Search 用，本地執行不需 API）
    sparse_model: str = "Qdrant/bm25"

    # LLM
    llm_model: str = "gemini-3-flash-preview"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096     # phase1: 2048 → 4096（原始值：8192）

    # Retrieval（原始值：similarity_top_k=60, rerank_top_n=25, num_queries=3）
    similarity_top_k: int = 25     # phase1: 10 → 25
    rerank_top_n: int = 15         # phase1: 5 → 10 → 15
    num_queries: int = 3           # phase1: 1 → 3（multi-query 重新啟用）
    hybrid_alpha: float = 0.6      # phase1: 0.4 → 0.6（偏向語意搜尋）

    # Re-ranking
    cohere_api_key: str | None = None
    cohere_rerank_model: str = "rerank-v3.5"

    # Langfuse
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # Slack
    slack_bot_token: str | None = None
    slack_signing_secret: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
