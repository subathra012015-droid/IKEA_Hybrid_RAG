from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-5-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "IKEA_Hybrid_RAG"

    # ChromaDB
    chroma_collection_name: str = "ikea_faq"
    chroma_persist_directory: str = "./chroma_db"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_k: int = 4

    # IKEA
    ikea_allowed_domain: str = "ikea.com"
    ikea_faq_urls: str = "https://www.ikea.com/us/en/customer-service/faq/"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
