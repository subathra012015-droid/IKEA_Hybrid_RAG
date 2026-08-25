from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from the local .env file.
    Secrets must never be hard-coded in source code.
    """

    # =========================================================
    # OPENAI
    # =========================================================

    openai_api_key: str = ""

    openai_chat_model: str = ""

    openai_embedding_model: str = "text-embedding-3-small"

    openai_rerank_model: str = ""

    # =========================================================
    # LANGSMITH
    # =========================================================

    langsmith_api_key: str = ""

    langsmith_tracing: bool = True

    langsmith_project: str = "IKEA_Hybrid_RAG"

    # =========================================================
    # CHROMADB
    # =========================================================

    chroma_collection_name: str = "ikea_faq"

    chroma_persist_directory: str = "./chroma_db"

    # =========================================================
    # CHUNKING
    # =========================================================

    chunk_size: int = 1000

    chunk_overlap: int = 150

    # =========================================================
    # RETRIEVAL
    # =========================================================

    semantic_top_k: int = 8

    bm25_top_k: int = 8

    fusion_top_k: int = 12

    rerank_top_k: int = 5

    rrf_k: int = 60

    # =========================================================
    # IKEA
    # =========================================================

    ikea_allowed_domain: str = "ikea.com"

    ikea_faq_urls: str = "https://www.ikea.com/" "us/en/customer-service/faq/"

    user_agent: str = "IKEA_Hybrid_RAG/1.0"

    # =========================================================
    # PYDANTIC SETTINGS
    # =========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
