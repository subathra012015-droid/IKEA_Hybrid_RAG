"""
# This script serves as a configuration test for the IKEA Hybrid RAG project. It prints out the current settings for various components, including OpenAI models, LangSmith tracing, Chroma collection details, and IKEA-specific configurations.
# It is useful for verifying that the environment variables and settings are correctly loaded and accessible within the application.

from backend.a01_config.settings import settings


def main():
    print("IKEA Hybrid RAG - Configuration Test")
    print("------------------------------------")

    print(f"Chat model: {settings.openai_chat_model}")
    print(f"Embedding model: {settings.openai_embedding_model}")

    print(f"LangSmith tracing: {settings.langsmith_tracing}")
    print(f"LangSmith project: {settings.langsmith_project}")

    print(f"Chroma collection: {settings.chroma_collection_name}")
    print(f"Chroma directory: {settings.chroma_persist_directory}")

    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"Retrieval K: {settings.retrieval_k}")

    print(f"IKEA allowed domain: {settings.ikea_allowed_domain}")
    print(f"IKEA FAQ URL: {settings.ikea_faq_urls}")


if __name__ == "__main__":
    main()

"""

# This script serves as a configuration test for the IKEA Hybrid RAG project.
# It prints out the current settings for various components, including OpenAI models, LangSmith tracing, Chroma collection details, and IKEA-specific configurations.
# It is useful for verifying that the environment variables and settings are correctly loaded and accessible within the application.

from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)


def main() -> None:
    """
    Main entry point for the IKEA Hybrid RAG project.
    """

    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Functionality 2 - Website Ingestion")
    print("=" * 70)

    run_ingestion()


if __name__ == "__main__":
    main()
