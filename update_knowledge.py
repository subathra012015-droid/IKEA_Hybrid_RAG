from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_2_chroma_store import (
    index_parent_documents,
    index_child_documents,
)


def print_index_result(
    title: str,
    result: dict,
) -> None:
    """
    Print incremental indexing statistics.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(f"Documents received: " f"{result['documents_received']}")

    print(f"New documents: " f"{result['new_documents']}")

    print(f"Changed documents: " f"{result['changed_documents']}")

    print(f"Unchanged documents: " f"{result['unchanged_documents']}")

    print(f"Documents embedded: " f"{result['documents_embedded']}")


def main() -> None:
    """
    Refresh IKEA FAQ knowledge from the live website.

    Run this separately from the chatbot.

    Flow:

        IKEA website
            ↓
        extract
            ↓
        validate
            ↓
        save snapshot
            ↓
        create parent/child Documents
            ↓
        update ChromaDB incrementally
    """

    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Knowledge Update")
    print("=" * 70)

    # =========================================================
    # LIVE WEBSITE INGESTION
    # =========================================================

    valid_faqs = run_ingestion()

    # =========================================================
    # PROCESS FAQ DATA
    # =========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # UPDATE PARENT VECTORS
    # =========================================================

    parent_result = index_parent_documents(parents)

    print_index_result(
        title="PARENT VECTOR UPDATE",
        result=parent_result,
    )

    # =========================================================
    # UPDATE CHILD VECTORS
    # =========================================================

    child_result = index_child_documents(children)

    print_index_result(
        title="CHILD VECTOR UPDATE",
        result=child_result,
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("KNOWLEDGE UPDATE SUMMARY")
    print("=" * 70)

    print(f"Validated FAQs: " f"{len(valid_faqs)}")

    print(f"Parents: " f"{len(parents)}")

    print(f"Children: " f"{len(children)}")

    print("Parent embeddings created: " f"{parent_result['documents_embedded']}")

    print("Child embeddings created: " f"{child_result['documents_embedded']}")


if __name__ == "__main__":
    main()
