from __future__ import annotations

from langchain_core.documents import Document


def build_faq_document(
    faq: dict,
) -> Document:
    """
    Convert one validated IKEA FAQ dictionary
    into a LangChain Document.

    The FAQ question and answer are stored together
    in page_content.

    Metadata is preserved separately for:
        - retrieval
        - filtering
        - citations
        - future change detection
    """

    faq_id = str(
        faq.get(
            "faq_id",
            "",
        )
    ).strip()

    question = str(
        faq.get(
            "question",
            "",
        )
    ).strip()

    answer = str(
        faq.get(
            "answer",
            "",
        )
    ).strip()

    category = str(
        faq.get(
            "category",
            "",
        )
    ).strip()

    source = str(
        faq.get(
            "source",
            "",
        )
    ).strip()

    page_content = f"Question: {question}\n\n" f"Answer: {answer}"

    metadata = {
        "faq_id": faq_id,
        "question": question,
        "category": category,
        "source": source,
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )


def build_faq_documents(
    faqs: list[dict],
) -> list[Document]:
    """
    Convert all validated FAQ dictionaries
    into LangChain Documents.
    """

    documents = []

    for faq in faqs:

        document = build_faq_document(faq)

        documents.append(document)

    return documents
