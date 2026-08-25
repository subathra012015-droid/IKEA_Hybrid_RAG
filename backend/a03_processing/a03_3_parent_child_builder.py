from __future__ import annotations

from langchain_core.documents import Document


def prepare_parent_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Prepare complete FAQ Documents as parent records.

    Parent Documents preserve the entire IKEA FAQ:
        Question + Answer

    They are not chopped into small retrieval pieces.
    """

    parent_documents = []

    for document in documents:

        metadata = document.metadata.copy()

        faq_id = str(
            metadata.get(
                "faq_id",
                "",
            )
        ).strip()

        if not faq_id:
            raise ValueError("Parent FAQ cannot be created without faq_id.")

        metadata["parent_faq_id"] = faq_id
        metadata["document_type"] = "parent"

        parent_document = Document(
            page_content=document.page_content,
            metadata=metadata,
        )

        parent_documents.append(parent_document)

    return parent_documents


def build_parent_child_structure(
    parent_documents: list[Document],
    child_documents: list[Document],
) -> dict:
    """
    Build an in-memory parent-child representation.

    Returns:

        {
            "parents": [...],
            "children": [...],
            "children_by_parent": {...}
        }
    """

    children_by_parent = {}

    for child in child_documents:

        parent_faq_id = child.metadata.get("parent_faq_id")

        if parent_faq_id not in children_by_parent:

            children_by_parent[parent_faq_id] = []

        children_by_parent[parent_faq_id].append(child)

    return {
        "parents": parent_documents,
        "children": child_documents,
        "children_by_parent": children_by_parent,
    }
