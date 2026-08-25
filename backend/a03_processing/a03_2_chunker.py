from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.a01_config.a01_1_settings import settings

ANSWER_SEPARATOR = "\n\nAnswer: "


def create_answer_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the fallback answer splitter.

    This is currently a structure-friendly recursive splitter.

    Later, when embeddings are introduced, oversized sections
    can additionally use semantic chunking.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


def extract_answer(
    document: Document,
) -> str:
    """
    Extract only the answer from a parent FAQ Document.

    Expected format:

        Question: ...

        Answer: ...
    """

    if ANSWER_SEPARATOR not in document.page_content:
        return ""

    _, answer = document.page_content.split(
        ANSWER_SEPARATOR,
        1,
    )

    return answer.strip()


def chunk_faq_document(
    document: Document,
) -> list[Document]:
    """
    Create child chunks from one parent FAQ.

    Important:

    - The parent retains the complete Question + Answer.
    - Child chunks contain answer content.
    - Children reference the parent using parent_faq_id.
    - The complete question is preserved in metadata rather
      than duplicated into every child page_content.
    """

    faq_id = str(
        document.metadata.get(
            "faq_id",
            "",
        )
    ).strip()

    question = str(
        document.metadata.get(
            "question",
            "",
        )
    ).strip()

    answer = extract_answer(document)

    if not faq_id:
        raise ValueError("Cannot create child chunks: faq_id is missing.")

    if not answer:
        raise ValueError(
            f"Cannot create child chunks for FAQ {faq_id}: " "answer is missing."
        )

    # =========================================================
    # SMALL FAQ
    # =========================================================

    if len(answer) <= settings.chunk_size:

        metadata = document.metadata.copy()

        metadata["parent_faq_id"] = faq_id
        metadata["document_type"] = "child"
        metadata["chunk_index"] = 0
        metadata["chunk_count"] = 1

        return [
            Document(
                page_content=answer,
                metadata=metadata,
            )
        ]

    # =========================================================
    # LARGE FAQ
    # =========================================================

    splitter = create_answer_splitter()

    answer_chunks = splitter.split_text(answer)

    chunk_count = len(answer_chunks)

    child_documents = []

    for index, answer_chunk in enumerate(answer_chunks):

        metadata = document.metadata.copy()

        metadata["parent_faq_id"] = faq_id
        metadata["document_type"] = "child"
        metadata["chunk_index"] = index
        metadata["chunk_count"] = chunk_count

        child_documents.append(
            Document(
                page_content=answer_chunk,
                metadata=metadata,
            )
        )

    return child_documents


def chunk_faq_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Create child chunks for all parent FAQ Documents.
    """

    child_documents = []

    for document in documents:

        chunks = chunk_faq_document(document)

        child_documents.extend(chunks)

    return child_documents
