from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class GroundedContext:
    """
    Retrieval context prepared for answer generation.
    """

    context_text: str
    documents: list[Document]
    source_urls: list[str]


def normalize_content(
    text: str,
) -> str:
    """
    Normalize text for duplicate detection.
    """

    return " ".join(text.strip().split()).lower()


def deduplicate_context_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Remove exact duplicate document content.

    Metadata differences alone do not make identical
    text useful twice for generation.
    """

    unique_documents = []
    seen_content = set()

    for document in documents:

        normalized = normalize_content(document.page_content)

        if not normalized:
            continue

        if normalized in seen_content:
            continue

        seen_content.add(normalized)

        unique_documents.append(document)

    return unique_documents


def collect_source_urls(
    documents: list[Document],
) -> list[str]:
    """
    Collect unique source URLs while preserving order.
    """

    sources = []

    seen_sources = set()

    for document in documents:

        source = str(
            document.metadata.get(
                "source",
                "",
            )
        ).strip()

        if not source:
            continue

        if source in seen_sources:
            continue

        seen_sources.add(source)

        sources.append(source)

    return sources


def build_context_block(
    document: Document,
    index: int,
) -> str:
    """
    Format one retrieved document as a grounded
    evidence block for the answer model.
    """

    question = str(
        document.metadata.get(
            "question",
            "",
        )
    ).strip()

    category = str(
        document.metadata.get(
            "category",
            "",
        )
    ).strip()

    source = str(
        document.metadata.get(
            "source",
            "",
        )
    ).strip()

    document_type = str(
        document.metadata.get(
            "document_type",
            "",
        )
    ).strip()

    chunk_index = document.metadata.get("chunk_index")

    lines = [
        f"[DOCUMENT {index}]",
        f"Type: {document_type}",
        f"Category: {category}",
        f"Question: {question}",
    ]

    if chunk_index is not None:

        lines.append(f"Chunk index: {chunk_index}")

    lines.extend(
        [
            f"Source: {source}",
            "",
            document.page_content.strip(),
        ]
    )

    return "\n".join(lines)


def build_grounded_context(
    documents: list[Document],
) -> GroundedContext:
    """
    Prepare selected retrieval Documents for
    grounded answer generation.
    """

    if not documents:

        return GroundedContext(
            context_text="",
            documents=[],
            source_urls=[],
        )

    unique_documents = deduplicate_context_documents(documents)

    context_blocks = []

    for index, document in enumerate(
        unique_documents,
        start=1,
    ):

        context_blocks.append(
            build_context_block(
                document=document,
                index=index,
            )
        )

    context_text = "\n\n".join(context_blocks)

    source_urls = collect_source_urls(unique_documents)

    return GroundedContext(
        context_text=context_text,
        documents=unique_documents,
        source_urls=source_urls,
    )
