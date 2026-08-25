from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from backend.a04_retrieval.a04_2_chroma_store import (
    parent_semantic_search,
    child_semantic_search,
    get_children_for_parent,
)

from backend.a04_retrieval.a04_3_bm25_index import (
    BM25Index,
)

from backend.a04_retrieval.a04_4_candidate_fusion import (
    FusedSearchResult,
    fuse_results,
)

from backend.a04_retrieval.a04_5_reranker import (
    RerankedSearchResult,
    rerank_candidates,
)


@dataclass
class HybridRetrievalResult:
    """
    Final output from the hybrid retriever.

    Contains:
        query
        parent results
        child results
        best parent
        related children
    """

    query: str

    parent_candidates: list[FusedSearchResult]

    parent_reranked: list[RerankedSearchResult]

    child_candidates: list[FusedSearchResult]

    best_parent: Document | None

    related_children: list[Document]


def retrieve_parent_candidates(
    query: str,
    parent_bm25: BM25Index,
) -> list[FusedSearchResult]:
    """
    Retrieve parent FAQ candidates using:

        Semantic
        +
        BM25
        ↓
        RRF fusion
    """

    semantic_results = parent_semantic_search(query=query)

    bm25_results = parent_bm25.search(query=query)

    fused_results = fuse_results(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
    )

    return fused_results


def retrieve_child_candidates(
    query: str,
    child_bm25: BM25Index,
) -> list[FusedSearchResult]:
    """
    Retrieve child FAQ candidates using:

        Semantic
        +
        BM25
        ↓
        RRF fusion

    Child candidates are currently not passed
    through the LLM reranker.

    They provide detailed supporting evidence.
    """

    semantic_results = child_semantic_search(query=query)

    bm25_results = child_bm25.search(query=query)

    fused_results = fuse_results(
        semantic_results=semantic_results,
        bm25_results=bm25_results,
    )

    return fused_results


def get_best_parent(
    reranked_results: list[RerankedSearchResult],
) -> Document | None:
    """
    Return the highest-ranked parent Document.
    """

    if not reranked_results:
        return None

    return reranked_results[0].result.document


def expand_best_parent(
    best_parent: Document | None,
) -> list[Document]:
    """
    Retrieve every child belonging to the
    highest-ranked parent FAQ.

    This is important for broad queries such as:

        "What kitchen services are available?"

    Instead of returning only Top-K chunks,
    we can retrieve all child content for the
    relevant parent FAQ.
    """

    if best_parent is None:
        return []

    parent_faq_id = str(
        best_parent.metadata.get(
            "parent_faq_id",
            "",
        )
    ).strip()

    if not parent_faq_id:
        return []

    return get_children_for_parent(parent_faq_id)


def hybrid_retrieve(
    query: str,
    parent_bm25: BM25Index,
    child_bm25: BM25Index,
) -> HybridRetrievalResult:
    """
    Run the complete current hybrid retrieval process.

    Flow:

        User query
            ↓
        Parent semantic search
            +
        Parent BM25
            ↓
        Parent RRF
            ↓
        Parent reranking
            ↓
        Best parent
            ↓
        Parent-child expansion

    In parallel:

        Child semantic search
            +
        Child BM25
            ↓
        Child RRF

    Args:
        query:
            User query.

        parent_bm25:
            Parent BM25 index.

        child_bm25:
            Child BM25 index.

    Returns:
        HybridRetrievalResult
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Hybrid retrieval query " "cannot be empty.")

    # =========================================================
    # PARENT RETRIEVAL
    # =========================================================

    parent_candidates = retrieve_parent_candidates(
        query=cleaned_query,
        parent_bm25=parent_bm25,
    )

    # =========================================================
    # PARENT RERANKING
    # =========================================================

    parent_reranked = rerank_candidates(
        query=cleaned_query,
        candidates=parent_candidates,
    )

    # =========================================================
    # BEST PARENT
    # =========================================================

    best_parent = get_best_parent(parent_reranked)

    # =========================================================
    # PARENT EXPANSION
    # =========================================================

    related_children = expand_best_parent(best_parent)

    # =========================================================
    # CHILD RETRIEVAL
    # =========================================================

    child_candidates = retrieve_child_candidates(
        query=cleaned_query,
        child_bm25=child_bm25,
    )

    return HybridRetrievalResult(
        query=cleaned_query,
        parent_candidates=parent_candidates,
        parent_reranked=parent_reranked,
        child_candidates=child_candidates,
        best_parent=best_parent,
        related_children=related_children,
    )


def print_hybrid_result(
    result: HybridRetrievalResult,
) -> None:
    """
    Print hybrid retrieval diagnostics.
    """

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL RESULT")
    print("=" * 70)

    print()
    print(f"Query: {result.query}")

    print(f"Parent fused candidates: " f"{len(result.parent_candidates)}")

    print(f"Parent reranked results: " f"{len(result.parent_reranked)}")

    print(f"Child fused candidates: " f"{len(result.child_candidates)}")

    # =========================================================
    # BEST PARENT
    # =========================================================

    if result.best_parent:

        print()
        print("-" * 70)
        print("BEST PARENT")
        print("-" * 70)

        print("FAQ ID: " f"{result.best_parent.metadata.get('faq_id')}")

        print("Question: " f"{result.best_parent.metadata.get('question')}")

        print("Category: " f"{result.best_parent.metadata.get('category')}")

        print()
        print(result.best_parent.page_content[:1000])

    else:

        print()
        print("No suitable parent was found.")

    # =========================================================
    # EXPANDED CHILDREN
    # =========================================================

    print()
    print("-" * 70)
    print("RELATED CHILDREN")
    print("-" * 70)

    print(f"Children found: " f"{len(result.related_children)}")

    for index, child in enumerate(
        result.related_children,
        start=1,
    ):

        print()
        print(f"Child {index}")

        print(f"Chunk index: " f"{child.metadata.get('chunk_index')}")

        print()

        print(child.page_content[:1000])
