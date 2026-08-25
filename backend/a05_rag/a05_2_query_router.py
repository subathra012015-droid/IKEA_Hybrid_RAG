from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from backend.a05_rag.a05_1_query_intent import (
    QueryIntent,
    QueryIntentResult,
)


class RetrievalStrategy(str, Enum):
    """
    Supported retrieval strategies.
    """

    FOCUSED = "focused"

    EXPAND_PARENT = "expand_parent"

    MULTI_TARGET = "multi_target"

    POLICY_FOCUSED = "policy_focused"

    TROUBLESHOOTING_FOCUSED = "troubleshooting_focused"

    DEFINITION_FOCUSED = "definition_focused"

    RECOMMENDATION_MULTI_TARGET = "recommendation_multi_target"

    CONFIRMATION_CONTEXTUAL = "confirmation_contextual"

    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class QueryRoute:
    """
    Retrieval-routing decision.

    Attributes:
        strategy:
            Retrieval strategy to execute.

        use_semantic:
            Whether semantic retrieval is enabled.

        use_bm25:
            Whether BM25 retrieval is enabled.

        use_rrf:
            Whether RRF fusion is enabled.

        use_reranker:
            Whether reranking is enabled.

        expand_parent:
            Whether all children of the best parent
            should be retrieved.

        retrieve_multiple_targets:
            Whether multiple entities/options need
            separate coverage.

        use_conversation_context:
            Whether query history should first
            be considered.

        reason:
            Human-readable routing explanation.
    """

    strategy: RetrievalStrategy

    use_semantic: bool

    use_bm25: bool

    use_rrf: bool

    use_reranker: bool

    expand_parent: bool

    retrieve_multiple_targets: bool

    use_conversation_context: bool

    reason: str


def route_query(
    intent_result: QueryIntentResult,
) -> QueryRoute:
    """
    Convert intent classification into
    retrieval behavior.
    """

    intent = intent_result.intent

    use_conversation_context = intent_result.requires_conversation_context

    # ========================================================
    # OUT OF SCOPE
    # ========================================================

    if intent == QueryIntent.OUT_OF_SCOPE:

        return QueryRoute(
            strategy=(RetrievalStrategy.OUT_OF_SCOPE),
            use_semantic=False,
            use_bm25=False,
            use_rrf=False,
            use_reranker=False,
            expand_parent=False,
            retrieve_multiple_targets=False,
            use_conversation_context=False,
            reason=(
                "Out-of-scope queries should not " "force retrieval from IKEA FAQ data."
            ),
        )

    # ========================================================
    # ENUMERATION
    # ========================================================

    if intent == QueryIntent.ENUMERATION:

        return QueryRoute(
            strategy=(RetrievalStrategy.EXPAND_PARENT),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=True,
            retrieve_multiple_targets=False,
            use_conversation_context=(use_conversation_context),
            reason=(
                "Enumeration query should identify "
                "the strongest parent FAQ and then "
                "retrieve all related children."
            ),
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    if intent == QueryIntent.COMPARISON:

        return QueryRoute(
            strategy=(RetrievalStrategy.MULTI_TARGET),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=False,
            retrieve_multiple_targets=True,
            use_conversation_context=(use_conversation_context),
            reason=(
                "Comparison queries require balanced " "retrieval for multiple targets."
            ),
        )

    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if intent == QueryIntent.RECOMMENDATION:

        return QueryRoute(
            strategy=(RetrievalStrategy.RECOMMENDATION_MULTI_TARGET),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=True,
            retrieve_multiple_targets=True,
            use_conversation_context=(use_conversation_context),
            reason=(
                "Recommendations need the available "
                "options plus comparison evidence."
            ),
        )

    # ========================================================
    # POLICY
    # ========================================================

    if intent == QueryIntent.POLICY:

        return QueryRoute(
            strategy=(RetrievalStrategy.POLICY_FOCUSED),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=False,
            retrieve_multiple_targets=False,
            use_conversation_context=(use_conversation_context),
            reason=(
                "Policy questions benefit from exact "
                "lexical matches plus semantic retrieval."
            ),
        )

    # ========================================================
    # TROUBLESHOOTING
    # ========================================================

    if intent == QueryIntent.TROUBLESHOOTING:

        return QueryRoute(
            strategy=(RetrievalStrategy.TROUBLESHOOTING_FOCUSED),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=False,
            retrieve_multiple_targets=False,
            use_conversation_context=(use_conversation_context),
            reason=(
                "Troubleshooting queries often use "
                "different wording than the source FAQ."
            ),
        )

    # ========================================================
    # DEFINITION
    # ========================================================

    if intent == QueryIntent.DEFINITION:

        return QueryRoute(
            strategy=(RetrievalStrategy.DEFINITION_FOCUSED),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=False,
            retrieve_multiple_targets=False,
            use_conversation_context=(use_conversation_context),
            reason=("Definitions benefit from exact terms " "and semantic similarity."),
        )

    # ========================================================
    # CONFIRMATION
    # ========================================================

    if intent == QueryIntent.CONFIRMATION:

        return QueryRoute(
            strategy=(RetrievalStrategy.CONFIRMATION_CONTEXTUAL),
            use_semantic=True,
            use_bm25=True,
            use_rrf=True,
            use_reranker=True,
            expand_parent=False,
            retrieve_multiple_targets=False,
            use_conversation_context=True,
            reason=(
                "Confirmation requires conversation " "context plus grounded retrieval."
            ),
        )

    # ========================================================
    # PROCEDURAL + FACTUAL
    # ========================================================

    return QueryRoute(
        strategy=(RetrievalStrategy.FOCUSED),
        use_semantic=True,
        use_bm25=True,
        use_rrf=True,
        use_reranker=True,
        expand_parent=False,
        retrieve_multiple_targets=False,
        use_conversation_context=(use_conversation_context),
        reason=(
            "Focused retrieval is appropriate for "
            "specific factual or procedural queries."
        ),
    )


def print_query_route(
    route: QueryRoute,
) -> None:
    """
    Print routing diagnostics.
    """

    print()
    print("=" * 70)
    print("QUERY ROUTE")
    print("=" * 70)

    print(f"Strategy: " f"{route.strategy.value}")

    print(f"Semantic: " f"{route.use_semantic}")

    print(f"BM25: " f"{route.use_bm25}")

    print(f"RRF: " f"{route.use_rrf}")

    print(f"Reranker: " f"{route.use_reranker}")

    print(f"Expand parent: " f"{route.expand_parent}")

    print(f"Multiple targets: " f"{route.retrieve_multiple_targets}")

    print(f"Conversation context: " f"{route.use_conversation_context}")

    print(f"Reason: " f"{route.reason}")
