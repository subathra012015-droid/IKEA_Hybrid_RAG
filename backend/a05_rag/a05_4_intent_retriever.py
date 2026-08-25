from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from backend.a04_retrieval.a04_7_pipeline import (
    HybridRetrievalPipeline,
)

from backend.a04_retrieval.a04_5_reranker import (
    RerankedSearchResult,
)

from backend.a04_retrieval.a04_4_candidate_fusion import (
    FusedSearchResult,
)

from backend.a05_rag.a05_1_query_intent import (
    QueryIntent,
    QueryIntentResult,
    classify_query_intent,
)

from backend.a05_rag.a05_2_query_router import (
    QueryRoute,
    RetrievalStrategy,
    route_query,
)

from backend.a05_rag.a05_3_contextual_query import (
    ContextualQueryResult,
    prepare_contextual_query,
)


@dataclass
class IntentRetrievalResult:
    """
    Final retrieval result after applying:

        intent classification
        conversation handling
        query routing
        hybrid retrieval
        intent-specific selection
    """

    original_query: str

    retrieval_query: str

    intent_result: QueryIntentResult

    route: QueryRoute

    selected_documents: list[Document]

    best_parent: Document | None

    related_children: list[Document]

    parent_reranked: list[RerankedSearchResult]

    child_candidates: list[FusedSearchResult]

    explanation: str


def deduplicate_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Remove duplicate Documents.

    Deduplication uses:
        parent_faq_id
        chunk_index

    where possible.
    """

    unique_documents = []

    seen_ids = set()

    for document in documents:

        parent_faq_id = str(
            document.metadata.get(
                "parent_faq_id",
                "",
            )
        ).strip()

        faq_id = str(
            document.metadata.get(
                "faq_id",
                "",
            )
        ).strip()

        chunk_index = int(
            document.metadata.get(
                "chunk_index",
                0,
            )
        )

        document_type = str(
            document.metadata.get(
                "document_type",
                "",
            )
        ).strip()

        if document_type == "child":

            document_id = f"{parent_faq_id}" f"_chunk_{chunk_index}"

        else:

            document_id = f"{faq_id}_parent"

        if document_id in seen_ids:
            continue

        seen_ids.add(document_id)

        unique_documents.append(document)

    return unique_documents


def get_top_reranked_parent_documents(
    results: list[RerankedSearchResult],
    top_n: int = 3,
) -> list[Document]:
    """
    Extract the strongest parent Documents
    from reranked parent results.
    """

    documents = []

    for result in results[:top_n]:

        documents.append(result.result.document)

    return documents


def get_top_child_documents(
    results: list[FusedSearchResult],
    top_n: int = 5,
) -> list[Document]:
    """
    Extract strongest child Documents
    from child fusion results.
    """

    documents = []

    for result in results[:top_n]:

        documents.append(result.document)

    return documents


def select_focused_documents(
    parent_reranked: list[RerankedSearchResult],
    child_candidates: list[FusedSearchResult],
) -> list[Document]:
    """
    Select focused evidence for factual,
    procedural, policy, definition,
    and troubleshooting queries.

    Strategy:
        strongest parent
        +
        strongest child evidence
    """

    documents = []

    parent_documents = get_top_reranked_parent_documents(
        parent_reranked,
        top_n=2,
    )

    child_documents = get_top_child_documents(
        child_candidates,
        top_n=5,
    )

    documents.extend(parent_documents)

    documents.extend(child_documents)

    return deduplicate_documents(documents)


def select_enumeration_documents(
    best_parent: Document | None,
    related_children: list[Document],
) -> list[Document]:
    """
    Select documents for enumeration queries.

    Example:
        "What kitchen services are available?"

    Behavior:
        Best parent FAQ
        +
        ALL child chunks belonging to that parent.

    This prevents Top-K from hiding valid options.
    """

    documents = []

    if best_parent is not None:

        documents.append(best_parent)

    documents.extend(related_children)

    return deduplicate_documents(documents)


def select_multi_target_documents(
    parent_reranked: list[RerankedSearchResult],
    child_candidates: list[FusedSearchResult],
) -> list[Document]:
    """
    Select broader evidence for comparison
    and recommendation queries.

    Multiple parent and child candidates are
    intentionally preserved.
    """

    documents = []

    documents.extend(
        get_top_reranked_parent_documents(
            parent_reranked,
            top_n=4,
        )
    )

    documents.extend(
        get_top_child_documents(
            child_candidates,
            top_n=8,
        )
    )

    return deduplicate_documents(documents)


def select_documents_for_route(
    route: QueryRoute,
    best_parent: Document | None,
    related_children: list[Document],
    parent_reranked: list[RerankedSearchResult],
    child_candidates: list[FusedSearchResult],
) -> tuple[
    list[Document],
    str,
]:
    """
    Apply the retrieval strategy selected
    by the query router.
    """

    # =========================================================
    # OUT OF SCOPE
    # =========================================================

    if route.strategy == RetrievalStrategy.OUT_OF_SCOPE:

        return (
            [],
            (
                "No IKEA FAQ retrieval was performed "
                "because the query was classified "
                "as out of scope."
            ),
        )

    # =========================================================
    # ENUMERATION
    # =========================================================

    if route.strategy == RetrievalStrategy.EXPAND_PARENT:

        documents = select_enumeration_documents(
            best_parent=best_parent,
            related_children=related_children,
        )

        return (
            documents,
            (
                "Enumeration strategy selected the "
                "best parent FAQ and all of its "
                "related child chunks."
            ),
        )

    # =========================================================
    # COMPARISON
    # =========================================================

    if route.strategy == RetrievalStrategy.MULTI_TARGET:

        documents = select_multi_target_documents(
            parent_reranked=parent_reranked,
            child_candidates=child_candidates,
        )

        return (
            documents,
            ("Comparison strategy preserved " "multiple parent and child targets."),
        )

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    if route.strategy == RetrievalStrategy.RECOMMENDATION_MULTI_TARGET:

        documents = select_multi_target_documents(
            parent_reranked=parent_reranked,
            child_candidates=child_candidates,
        )

        if best_parent is not None:

            documents.extend(related_children)

        documents = deduplicate_documents(documents)

        return (
            documents,
            (
                "Recommendation strategy collected "
                "multiple options and supporting "
                "parent-child evidence."
            ),
        )

    # =========================================================
    # ALL FOCUSED STRATEGIES
    # =========================================================

    documents = select_focused_documents(
        parent_reranked=parent_reranked,
        child_candidates=child_candidates,
    )

    return (
        documents,
        ("Focused strategy selected the strongest " "parent and child evidence."),
    )


def retrieve_for_intent(
    query: str,
    retrieval_pipeline: HybridRetrievalPipeline,
    conversation_history: list[dict] | None = None,
) -> IntentRetrievalResult:
    """
    Run the complete intent-aware retrieval flow.

    Execution:

        1. Classify query intent.
        2. Determine retrieval route.
        3. Rewrite contextual query when needed.
        4. Run hybrid retrieval.
        5. Select documents according to intent.

    Args:
        query:
            Current user query.

        retrieval_pipeline:
            Initialized HybridRetrievalPipeline.

        conversation_history:
            Recent user/assistant conversation.

    Returns:
        IntentRetrievalResult
    """

    if conversation_history is None:
        conversation_history = []

    # =========================================================
    # INTENT
    # =========================================================

    intent_result = classify_query_intent(query)

    # =========================================================
    # ROUTE
    # =========================================================

    route = route_query(intent_result)

    # =========================================================
    # CONTEXTUAL QUERY
    # =========================================================

    contextual_result: ContextualQueryResult = prepare_contextual_query(
        intent_result=(intent_result),
        conversation_history=(conversation_history),
    )

    retrieval_query = contextual_result.retrieval_query

    # =========================================================
    # OUT OF SCOPE
    # =========================================================

    if route.strategy == RetrievalStrategy.OUT_OF_SCOPE:

        return IntentRetrievalResult(
            original_query=query,
            retrieval_query=retrieval_query,
            intent_result=intent_result,
            route=route,
            selected_documents=[],
            best_parent=None,
            related_children=[],
            parent_reranked=[],
            child_candidates=[],
            explanation=(
                "Query is outside the configured " "IKEA FAQ retrieval scope."
            ),
        )

    # =========================================================
    # HYBRID RETRIEVAL
    # =========================================================

    hybrid_result = retrieval_pipeline.retrieve(query=retrieval_query)

    # =========================================================
    # INTENT-SPECIFIC DOCUMENT SELECTION
    # =========================================================

    selected_documents, explanation = select_documents_for_route(
        route=route,
        best_parent=(hybrid_result.best_parent),
        related_children=(hybrid_result.related_children),
        parent_reranked=(hybrid_result.parent_reranked),
        child_candidates=(hybrid_result.child_candidates),
    )

    return IntentRetrievalResult(
        original_query=query,
        retrieval_query=retrieval_query,
        intent_result=intent_result,
        route=route,
        selected_documents=(selected_documents),
        best_parent=(hybrid_result.best_parent),
        related_children=(hybrid_result.related_children),
        parent_reranked=(hybrid_result.parent_reranked),
        child_candidates=(hybrid_result.child_candidates),
        explanation=explanation,
    )


def print_intent_retrieval_result(
    result: IntentRetrievalResult,
) -> None:
    """
    Print intent-aware retrieval diagnostics.
    """

    print()
    print("=" * 70)
    print("INTENT-AWARE RETRIEVAL")
    print("=" * 70)

    print(f"Original query: " f"{result.original_query}")

    print(f"Retrieval query: " f"{result.retrieval_query}")

    print(f"Intent: " f"{result.intent_result.intent.value}")

    print(f"Conversation type: " f"{result.intent_result.conversation_type.value}")

    print(f"Strategy: " f"{result.route.strategy.value}")

    print(f"Selected documents: " f"{len(result.selected_documents)}")

    print(f"Explanation: " f"{result.explanation}")

    # =========================================================
    # BEST PARENT
    # =========================================================

    if result.best_parent:

        print()
        print("-" * 70)
        print("BEST PARENT")
        print("-" * 70)

        print("Question: " f"{result.best_parent.metadata.get('question')}")

        print("Category: " f"{result.best_parent.metadata.get('category')}")

    # =========================================================
    # SELECTED DOCUMENTS
    # =========================================================

    print()
    print("-" * 70)
    print("SELECTED DOCUMENTS")
    print("-" * 70)

    for index, document in enumerate(
        result.selected_documents,
        start=1,
    ):

        print()
        print(f"Document {index}")

        print(f"Type: " f"{document.metadata.get('document_type')}")

        print(f"Question: " f"{document.metadata.get('question')}")

        print(f"Category: " f"{document.metadata.get('category')}")

        print(f"Chunk index: " f"{document.metadata.get('chunk_index')}")

        print()

        print(document.page_content[:800])
