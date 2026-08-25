from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.a01_config.a01_1_settings import settings
from backend.a04_retrieval.a04_1_embeddings import (
    create_embedding_model,
)

# ============================================================
# COLLECTION NAMES
# ============================================================

PARENT_COLLECTION_SUFFIX = "_parents"
CHILD_COLLECTION_SUFFIX = "_children"


# ============================================================
# CHROMA STORE CREATION
# ============================================================


def create_chroma_store(
    collection_name: str,
) -> Chroma:
    """
    Create or open one persistent ChromaDB collection.

    Args:
        collection_name:
            Name of the Chroma collection.

    Returns:
        Chroma:
            Persistent vector store.
    """

    persist_directory = Path(settings.chroma_persist_directory)

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_model = create_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(persist_directory),
    )


def create_parent_store() -> Chroma:
    """
    Create/open the parent FAQ semantic collection.
    """

    collection_name = f"{settings.chroma_collection_name}" f"{PARENT_COLLECTION_SUFFIX}"

    return create_chroma_store(collection_name)


def create_child_store() -> Chroma:
    """
    Create/open the child FAQ semantic collection.
    """

    collection_name = f"{settings.chroma_collection_name}" f"{CHILD_COLLECTION_SUFFIX}"

    return create_chroma_store(collection_name)


# ============================================================
# IDS
# ============================================================


def build_parent_id(
    document: Document,
) -> str:
    """
    Build deterministic parent ID.
    """

    faq_id = str(
        document.metadata.get(
            "faq_id",
            "",
        )
    ).strip()

    if not faq_id:
        raise ValueError("Cannot build parent ID: faq_id is missing.")

    return f"{faq_id}_parent"


def build_child_id(
    document: Document,
) -> str:
    """
    Build deterministic child chunk ID.
    """

    parent_faq_id = str(
        document.metadata.get(
            "parent_faq_id",
            "",
        )
    ).strip()

    chunk_index = int(
        document.metadata.get(
            "chunk_index",
            0,
        )
    )

    if not parent_faq_id:
        raise ValueError("Cannot build child ID: " "parent_faq_id is missing.")

    return f"{parent_faq_id}" f"_chunk_{chunk_index}"


# ============================================================
# HASHING
# ============================================================


def build_content_hash(
    document: Document,
) -> str:
    """
    Generate SHA-256 hash for document content.

    Used to avoid re-embedding unchanged text.
    """

    normalized_content = document.page_content.strip()

    return sha256(normalized_content.encode("utf-8")).hexdigest()


def prepare_document(
    document: Document,
) -> Document:
    """
    Copy a Document and attach its content hash.
    """

    metadata = document.metadata.copy()

    metadata["content_hash"] = build_content_hash(document)

    return Document(
        page_content=document.page_content,
        metadata=metadata,
    )


# ============================================================
# EXISTING RECORD LOOKUP
# ============================================================


def get_existing_records(
    vector_store: Chroma,
    ids: list[str],
) -> dict:
    """
    Retrieve metadata for existing Chroma records.

    This does not call OpenAI.
    """

    if not ids:
        return {}

    result = vector_store._collection.get(
        ids=ids,
        include=[
            "metadatas",
        ],
    )

    existing_records = {}

    returned_ids = result.get(
        "ids",
        [],
    )

    returned_metadata = result.get(
        "metadatas",
        [],
    )

    for record_id, metadata in zip(
        returned_ids,
        returned_metadata or [],
    ):
        existing_records[record_id] = metadata or {}

    return existing_records


# ============================================================
# GENERIC INCREMENTAL INDEXER
# ============================================================


def incremental_index(
    vector_store: Chroma,
    documents: list[Document],
    ids: list[str],
) -> dict:
    """
    Incrementally index Documents.

    New:
        embed + add

    Changed:
        remove old vector
        embed new content

    Unchanged:
        skip embedding
    """

    if len(documents) != len(ids):
        raise ValueError("Document count and ID count do not match.")

    prepared_documents = [prepare_document(document) for document in documents]

    existing_records = get_existing_records(
        vector_store=vector_store,
        ids=ids,
    )

    new_documents = []
    new_ids = []

    changed_documents = []
    changed_ids = []

    unchanged_count = 0

    for document, document_id in zip(
        prepared_documents,
        ids,
    ):
        current_hash = document.metadata["content_hash"]

        existing_metadata = existing_records.get(document_id)

        # ----------------------------------------------------
        # NEW
        # ----------------------------------------------------

        if existing_metadata is None:

            new_documents.append(document)

            new_ids.append(document_id)

            continue

        # ----------------------------------------------------
        # EXISTING
        # ----------------------------------------------------

        existing_hash = existing_metadata.get("content_hash")

        if existing_hash == current_hash:

            unchanged_count += 1

            continue

        # ----------------------------------------------------
        # CHANGED
        # ----------------------------------------------------

        changed_documents.append(document)

        changed_ids.append(document_id)

    # Remove old versions of changed documents.
    if changed_ids:

        vector_store.delete(ids=changed_ids)

    documents_to_index = new_documents + changed_documents

    ids_to_index = new_ids + changed_ids

    if documents_to_index:

        vector_store.add_documents(
            documents=documents_to_index,
            ids=ids_to_index,
        )

    return {
        "documents_received": len(documents),
        "new_documents": len(new_documents),
        "changed_documents": len(changed_documents),
        "unchanged_documents": (unchanged_count),
        "documents_embedded": len(documents_to_index),
    }


# ============================================================
# PARENT INDEXING
# ============================================================


def index_parent_documents(
    parent_documents: list[Document],
) -> dict:
    """
    Index complete FAQ parent Documents.

    Parent vectors are useful for:
        - broad questions
        - FAQ intent matching
        - enumeration queries
    """

    if not parent_documents:
        raise ValueError("No parent documents provided.")

    vector_store = create_parent_store()

    ids = [build_parent_id(document) for document in parent_documents]

    result = incremental_index(
        vector_store=vector_store,
        documents=parent_documents,
        ids=ids,
    )

    result["collection_type"] = "parent"

    return result


# ============================================================
# CHILD INDEXING
# ============================================================


def index_child_documents(
    child_documents: list[Document],
) -> dict:
    """
    Index child answer chunks.

    Child vectors are useful for:
        - precise factual retrieval
        - details inside large FAQ answers
    """

    if not child_documents:
        raise ValueError("No child documents provided.")

    vector_store = create_child_store()

    ids = [build_child_id(document) for document in child_documents]

    result = incremental_index(
        vector_store=vector_store,
        documents=child_documents,
        ids=ids,
    )

    result["collection_type"] = "child"

    return result


# ============================================================
# PARENT SEMANTIC SEARCH
# ============================================================


def parent_semantic_search(
    query: str,
    top_k: int | None = None,
) -> list[
    tuple[
        Document,
        float,
    ]
]:
    """
    Semantic search against complete parent FAQs.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    if top_k is None:
        top_k = settings.semantic_top_k

    vector_store = create_parent_store()

    return vector_store.similarity_search_with_score(
        query=cleaned_query,
        k=top_k,
    )


# ============================================================
# CHILD SEMANTIC SEARCH
# ============================================================


def child_semantic_search(
    query: str,
    top_k: int | None = None,
) -> list[
    tuple[
        Document,
        float,
    ]
]:
    """
    Semantic search against detailed child chunks.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    if top_k is None:
        top_k = settings.semantic_top_k

    vector_store = create_child_store()

    return vector_store.similarity_search_with_score(
        query=cleaned_query,
        k=top_k,
    )


# ============================================================
# CHILD LOOKUP BY PARENT
# ============================================================


def get_children_for_parent(
    parent_faq_id: str,
) -> list[Document]:
    """
    Retrieve every child belonging to one parent FAQ.

    This performs metadata filtering rather than
    semantic Top-K retrieval.

    Useful for enumeration queries where we want
    all available information under a matched parent.
    """

    cleaned_parent_id = parent_faq_id.strip()

    if not cleaned_parent_id:
        raise ValueError("parent_faq_id cannot be empty.")

    vector_store = create_child_store()

    result = vector_store._collection.get(
        where={"parent_faq_id": (cleaned_parent_id)},
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = []

    raw_documents = result.get("documents") or []

    raw_metadatas = result.get("metadatas") or []

    for page_content, metadata in zip(
        raw_documents,
        raw_metadatas,
    ):
        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata or {},
            )
        )

    documents.sort(
        key=lambda document: (
            document.metadata.get(
                "chunk_index",
                0,
            )
        )
    )

    return documents
