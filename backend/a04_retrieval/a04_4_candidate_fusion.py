from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from backend.a01_config.a01_1_settings import settings
from backend.a04_retrieval.a04_3_bm25_index import (
    BM25SearchResult,
)


@dataclass
class FusedSearchResult:
    """
    One result after Semantic + BM25 fusion.

    Attributes:
        document:
            Retrieved LangChain Document.

        document_id:
            Stable identifier used for deduplication.

        fusion_score:
            Combined Reciprocal Rank Fusion score.

        semantic_rank:
            Rank from semantic retrieval.
            None when semantic retrieval did not return it.

        semantic_score:
            Raw semantic distance score.
            None when not returned semantically.

        bm25_rank:
            Rank from BM25 retrieval.
            None when BM25 did not return it.

        bm25_score:
            Raw BM25 relevance score.
            None when not returned by BM25.

        sources:
            Which retrieval systems found the result.

        final_rank:
            Position after RRF fusion.
    """

    document: Document

    document_id: str

    fusion_score: float

    semantic_rank: int | None

    semantic_score: float | None

    bm25_rank: int | None

    bm25_score: float | None

    sources: list[str]

    final_rank: int


def build_document_id(
    document: Document,
) -> str:
    """
    Create a stable identifier for fusion.

    Parent:
        <faq_id>_parent

    Child:
        <parent_faq_id>_chunk_<chunk_index>

    This allows the semantic and BM25 result lists
    to recognize when they returned the same record.
    """

    document_type = str(
        document.metadata.get(
            "document_type",
            "",
        )
    ).strip()

    faq_id = str(
        document.metadata.get(
            "faq_id",
            "",
        )
    ).strip()

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

    # =========================================================
    # PARENT DOCUMENT
    # =========================================================

    if document_type == "parent":

        if not faq_id:
            raise ValueError("Parent document is missing faq_id.")

        return f"{faq_id}_parent"

    # =========================================================
    # CHILD DOCUMENT
    # =========================================================

    if document_type == "child":

        if not parent_faq_id:
            raise ValueError("Child document is missing " "parent_faq_id.")

        return f"{parent_faq_id}" f"_chunk_{chunk_index}"

    # =========================================================
    # FALLBACK
    # =========================================================

    if parent_faq_id:

        return f"{parent_faq_id}" f"_chunk_{chunk_index}"

    if faq_id:

        return f"{faq_id}_parent"

    raise ValueError("Unable to build a stable document ID.")


def calculate_rrf_score(
    rank: int,
) -> float:
    """
    Calculate one Reciprocal Rank Fusion contribution.

    Formula:

        1 / (RRF_K + rank)

    Example:

        RRF_K = 60
        rank  = 1

        score = 1 / 61
    """

    if rank <= 0:

        raise ValueError("RRF rank must be greater than zero.")

    return 1.0 / (settings.rrf_k + rank)


def fuse_results(
    semantic_results: list[
        tuple[
            Document,
            float,
        ]
    ],
    bm25_results: list[BM25SearchResult],
    top_k: int | None = None,
) -> list[FusedSearchResult]:
    """
    Fuse semantic and BM25 rankings using RRF.

    Important:
        Raw semantic and BM25 scores are NOT added
        together because they use different scales.

    Instead:
        semantic rank
        +
        BM25 rank
        ↓
        Reciprocal Rank Fusion

    Args:
        semantic_results:
            Chroma semantic search results.

        bm25_results:
            BM25 lexical results.

        top_k:
            Number of fused candidates to keep.
            Defaults to FUSION_TOP_K.

    Returns:
        Ranked FusedSearchResult objects.
    """

    if top_k is None:

        top_k = settings.fusion_top_k

    combined = {}

    # =========================================================
    # SEMANTIC RESULTS
    # =========================================================

    for semantic_rank, (
        document,
        semantic_score,
    ) in enumerate(
        semantic_results,
        start=1,
    ):

        document_id = build_document_id(document)

        rrf_score = calculate_rrf_score(semantic_rank)

        combined[document_id] = {
            "document": document,
            "document_id": document_id,
            "fusion_score": rrf_score,
            "semantic_rank": semantic_rank,
            "semantic_score": float(semantic_score),
            "bm25_rank": None,
            "bm25_score": None,
            "sources": ["semantic"],
        }

    # =========================================================
    # BM25 RESULTS
    # =========================================================

    for bm25_result in bm25_results:

        document = bm25_result.document

        document_id = build_document_id(document)

        rrf_score = calculate_rrf_score(bm25_result.rank)

        existing = combined.get(document_id)

        # -----------------------------------------------------
        # DOCUMENT ALREADY FOUND SEMANTICALLY
        # -----------------------------------------------------

        if existing is not None:

            existing["fusion_score"] += rrf_score

            existing["bm25_rank"] = bm25_result.rank

            existing["bm25_score"] = bm25_result.score

            if "bm25" not in existing["sources"]:

                existing["sources"].append("bm25")

        # -----------------------------------------------------
        # BM25-ONLY DOCUMENT
        # -----------------------------------------------------

        else:

            combined[document_id] = {
                "document": document,
                "document_id": document_id,
                "fusion_score": rrf_score,
                "semantic_rank": None,
                "semantic_score": None,
                "bm25_rank": (bm25_result.rank),
                "bm25_score": (bm25_result.score),
                "sources": ["bm25"],
            }

    # =========================================================
    # SORT BY FUSION SCORE
    # =========================================================

    ranked_records = sorted(
        combined.values(),
        key=lambda item: (item["fusion_score"]),
        reverse=True,
    )

    # =========================================================
    # BUILD FINAL RESULT OBJECTS
    # =========================================================

    fused_results = []

    for final_rank, record in enumerate(
        ranked_records[:top_k],
        start=1,
    ):

        fused_results.append(
            FusedSearchResult(
                document=record["document"],
                document_id=record["document_id"],
                fusion_score=record["fusion_score"],
                semantic_rank=record["semantic_rank"],
                semantic_score=record["semantic_score"],
                bm25_rank=record["bm25_rank"],
                bm25_score=record["bm25_score"],
                sources=record["sources"],
                final_rank=final_rank,
            )
        )

    return fused_results


def print_fused_results(
    title: str,
    results: list[FusedSearchResult],
) -> None:
    """
    Print RRF fusion diagnostics.
    """

    print()
    print("=" * 70)

    print(title)

    print("=" * 70)

    if not results:

        print()
        print("No fused results found.")

        return

    for result in results:

        document = result.document

        print()
        print(f"Result " f"{result.final_rank}")

        print("-" * 70)

        print(f"Document ID: " f"{result.document_id}")

        print(f"Fusion score: " f"{result.fusion_score:.6f}")

        print(f"Sources: " f"{', '.join(result.sources)}")

        print(f"Semantic rank: " f"{result.semantic_rank}")

        print(f"BM25 rank: " f"{result.bm25_rank}")

        print(f"Semantic score: " f"{result.semantic_score}")

        print(f"BM25 score: " f"{result.bm25_score}")

        print(f"Question: " f"{document.metadata.get('question')}")

        print(f"Category: " f"{document.metadata.get('category')}")

        print()

        print(document.page_content[:800])
