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
