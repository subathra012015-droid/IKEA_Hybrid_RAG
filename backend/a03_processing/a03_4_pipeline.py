from __future__ import annotations

from langchain_core.documents import Document

from backend.a03_processing.a03_1_document_builder import (
    build_faq_documents,
)

from backend.a03_processing.a03_2_chunker import (
    chunk_faq_documents,
)

from backend.a03_processing.a03_3_parent_child_builder import (
    prepare_parent_documents,
    build_parent_child_structure,
)


def run_processing(
    faqs: list[dict],
) -> dict:
    """
    Run IKEA FAQ processing.

    Execution:

        1. Build complete FAQ Documents.
        2. Prepare those Documents as parents.
        3. Create child answer chunks.
        4. Build parent-child mapping.
        5. Display diagnostics.

    Returns:
        dict:
            parents
            children
            children_by_parent
    """

    # =========================================================
    # STEP 5 - BUILD COMPLETE FAQ DOCUMENTS
    # =========================================================

    print()
    print("STEP 5 - BUILD LANGCHAIN DOCUMENTS")
    print("-" * 70)

    documents = build_faq_documents(faqs)

    print()
    print("Document creation completed.")

    print(f"FAQ records received: " f"{len(faqs)}")

    print(f"Complete FAQ documents created: " f"{len(documents)}")

    # =========================================================
    # STEP 6 - PREPARE PARENT DOCUMENTS
    # =========================================================

    print()
    print("STEP 6 - PREPARE PARENT FAQ DOCUMENTS")
    print("-" * 70)

    parent_documents = prepare_parent_documents(documents)

    print()
    print(f"Parent documents created: " f"{len(parent_documents)}")

    # =========================================================
    # STEP 7 - CREATE CHILD DOCUMENTS
    # =========================================================

    print()
    print("STEP 7 - CREATE CHILD CHUNKS")
    print("-" * 70)

    child_documents = chunk_faq_documents(documents)

    split_parent_ids = {
        child.metadata.get("parent_faq_id")
        for child in child_documents
        if child.metadata.get(
            "chunk_count",
            1,
        )
        > 1
    }

    print()
    print(f"Child documents created: " f"{len(child_documents)}")

    print(f"Parent FAQs requiring splitting: " f"{len(split_parent_ids)}")

    # =========================================================
    # STEP 8 - BUILD PARENT-CHILD MAP
    # =========================================================

    print()
    print("STEP 8 - BUILD PARENT-CHILD MAP")
    print("-" * 70)

    knowledge_structure = build_parent_child_structure(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )

    print()
    print(f"Parents mapped: " f"{len(knowledge_structure['children_by_parent'])}")

    # =========================================================
    # CHILD PREVIEW
    # =========================================================

    print()
    print("=" * 70)

    print("PARENT-CHILD PREVIEW")

    print("=" * 70)

    for parent in parent_documents[:3]:

        parent_id = parent.metadata["parent_faq_id"]

        children = knowledge_structure["children_by_parent"].get(
            parent_id,
            [],
        )

        print()
        print(f"Parent FAQ ID: " f"{parent_id}")

        print(f"Question: " f"{parent.metadata.get('question')}")

        print(f"Child count: " f"{len(children)}")

        for child in children:

            print()
            print(
                f"  Child "
                f"{child.metadata.get('chunk_index') + 1}"
                f"/"
                f"{child.metadata.get('chunk_count')}"
            )

            print(f"  Length: " f"{len(child.page_content)}")

            print(f"  Text: " f"{child.page_content[:300]}")

    return knowledge_structure
