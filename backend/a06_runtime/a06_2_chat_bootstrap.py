from __future__ import annotations

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_7_pipeline import (
    HybridRetrievalPipeline,
)

from backend.a05_rag.a05_9_pipeline import (
    ConversationalIKEAAssistant,
)

from backend.a06_runtime.a06_1_snapshot_loader import (
    load_faq_snapshot,
    print_snapshot_summary,
)


def build_chat_assistant() -> ConversationalIKEAAssistant:
    """
    Build the IKEA chatbot from locally persisted data.

    IMPORTANT:

    This startup path does NOT:
        - fetch ikea.com
        - scrape HTML
        - extract FAQs
        - validate live website data
        - rewrite the snapshot

    It DOES:
        - load the saved snapshot
        - recreate parent/child Documents
        - open persistent ChromaDB
        - skip unchanged embeddings
        - build local BM25 indexes
        - prepare the conversational assistant
    """

    # =========================================================
    # STEP 1 - LOAD LOCAL SNAPSHOT
    # =========================================================

    faq_records = load_faq_snapshot()

    print_snapshot_summary(faq_records)

    # =========================================================
    # STEP 2 - PROCESS LOCAL FAQ DATA
    # =========================================================

    knowledge_structure = run_processing(faq_records)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # STEP 3 - INITIALIZE HYBRID RETRIEVAL
    # =========================================================

    retrieval_pipeline = HybridRetrievalPipeline(
        parent_documents=parents,
        child_documents=children,
    )

    retrieval_pipeline.initialize()

    # =========================================================
    # STEP 4 - CREATE CONVERSATIONAL ASSISTANT
    # =========================================================

    assistant = ConversationalIKEAAssistant(retrieval_pipeline=(retrieval_pipeline))

    return assistant
