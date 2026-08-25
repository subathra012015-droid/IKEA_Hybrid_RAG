from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from backend.a01_config.a01_1_settings import settings


def create_embedding_model() -> OpenAIEmbeddings:
    """
    Create the OpenAI embedding model used by
    semantic retrieval.

    Configuration is read from .env through
    the central Settings object.

    Returns:
        OpenAIEmbeddings:
            Configured LangChain embedding model.
    """

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing. " "Add it to the local .env file.")

    if not settings.openai_embedding_model:
        raise ValueError("OPENAI_EMBEDDING_MODEL is missing.")

    embedding_model = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )

    return embedding_model


def embed_text(
    text: str,
) -> list[float]:
    """
    Convert one text string into an embedding vector.

    This function is intended mainly for testing
    the embedding configuration.

    Args:
        text:
            Text to embed.

    Returns:
        list[float]:
            Numerical embedding vector.
    """

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("Cannot embed empty text.")

    embedding_model = create_embedding_model()

    vector = embedding_model.embed_query(cleaned_text)

    return vector


def embedding_test() -> None:
    """
    Run a small embedding test.

    This makes one OpenAI embedding API call
    and prints only diagnostic information.

    The full vector is intentionally not printed
    because it is large and not useful for
    normal terminal debugging.
    """

    test_text = "What kitchen services are available?"

    print("=" * 70)

    print("OPENAI EMBEDDING TEST")

    print("=" * 70)

    print(f"Embedding model: " f"{settings.openai_embedding_model}")

    print(f"Test text: " f"{test_text}")

    vector = embed_text(test_text)

    print()
    print("Embedding created successfully.")

    print(f"Vector dimensions: " f"{len(vector)}")

    print("First 10 vector values:")

    print(vector[:10])
