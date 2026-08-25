from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.a01_config.a01_1_settings import settings
from backend.a04_retrieval.a04_4_candidate_fusion import (
    FusedSearchResult,
)


@dataclass
class RerankedSearchResult:
    """
    One candidate after LLM reranking.

    Attributes:
        result:
            Original RRF fused candidate.

        rerank_score:
            Relevance score assigned by the reranker.

        reason:
            Short explanation of why the candidate
            is relevant.

        rerank_rank:
            Final rank after reranking.
    """

    result: FusedSearchResult
    rerank_score: float
    reason: str
    rerank_rank: int


def create_rerank_model() -> ChatOpenAI:
    """
    Create the OpenAI model used for reranking.

    The model is used only after Semantic + BM25 + RRF
    have reduced the candidate set.

    Returns:
        ChatOpenAI:
            Configured reranking model.
    """

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    if not settings.openai_rerank_model:
        raise ValueError("OPENAI_RERANK_MODEL is missing.")

    return ChatOpenAI(
        model=settings.openai_rerank_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=0,
    )


def build_candidate_text(
    candidate: FusedSearchResult,
    candidate_number: int,
) -> str:
    """
    Convert one fused result into compact text
    for the reranking model.

    We include:
        - candidate number
        - question
        - category
        - content
        - retrieval evidence
    """

    document = candidate.document

    question = str(
        document.metadata.get(
            "question",
            "",
        )
    ).strip()

    category = str(
        document.metadata.get(
            "category",
            "",
        )
    ).strip()

    content = document.page_content.strip()

    return (
        f"CANDIDATE {candidate_number}\n"
        f"Question: {question}\n"
        f"Category: {category}\n"
        f"Content: {content}\n"
        f"Semantic Rank: "
        f"{candidate.semantic_rank}\n"
        f"BM25 Rank: "
        f"{candidate.bm25_rank}\n"
        f"RRF Score: "
        f"{candidate.fusion_score:.6f}"
    )


def build_rerank_prompt(
    query: str,
    candidates: list[FusedSearchResult],
) -> str:
    """
    Build the reranking instruction.

    The model is not asked to answer the customer.

    Its only task is to rank retrieval candidates
    by relevance to the user query.
    """

    candidate_blocks = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate_blocks.append(
            build_candidate_text(
                candidate=candidate,
                candidate_number=index,
            )
        )

    candidates_text = "\n\n".join(candidate_blocks)

    return f"""
You are a retrieval reranking system for an IKEA FAQ assistant.

Your task is ONLY to rank the supplied retrieval candidates
according to how directly and completely they help answer
the user's query.

Do not answer the user's question.
Do not add outside IKEA knowledge.
Use only the supplied candidates.

Consider:
1. Direct relevance to the user's query.
2. Whether the candidate addresses the user's actual intent.
3. Whether the content contains useful factual evidence.
4. Prefer directly relevant IKEA FAQ content over loosely related content.
5. Do not automatically prefer a candidate just because it had a high
   Semantic, BM25, or RRF rank.
6. Broad questions may favor a broad parent FAQ.
7. Narrow factual questions may favor a specific child or focused FAQ.

USER QUERY:
{query}

CANDIDATES:
{candidates_text}

Return ONLY valid JSON in this exact structure:

{{
  "rankings": [
    {{
      "candidate_number": 1,
      "score": 0.95,
      "reason": "Short relevance explanation"
    }}
  ]
}}

Requirements:
- Include every supplied candidate exactly once.
- candidate_number must correspond to the supplied candidate number.
- score must be between 0.0 and 1.0.
- Sort rankings from highest relevance to lowest relevance.
- Keep each reason brief.
""".strip()


def parse_rerank_response(
    response_text: str,
    candidates: list[FusedSearchResult],
    top_k: int,
) -> list[RerankedSearchResult]:
    """
    Parse and validate the JSON returned by the reranking model.
    """

    try:

        parsed = json.loads(response_text)

    except json.JSONDecodeError as error:

        raise ValueError("Reranker returned invalid JSON.") from error

    rankings = parsed.get("rankings")

    if not isinstance(
        rankings,
        list,
    ):
        raise ValueError("Reranker response is missing " "a valid rankings list.")

    candidate_lookup = {
        index: candidate
        for index, candidate in enumerate(
            candidates,
            start=1,
        )
    }

    seen_candidate_numbers = set()

    reranked_results = []

    for ranking in rankings:

        candidate_number = int(
            ranking.get(
                "candidate_number",
                0,
            )
        )

        if candidate_number not in candidate_lookup:
            continue

        if candidate_number in seen_candidate_numbers:
            continue

        seen_candidate_numbers.add(candidate_number)

        raw_score = float(
            ranking.get(
                "score",
                0.0,
            )
        )

        score = max(
            0.0,
            min(
                1.0,
                raw_score,
            ),
        )

        reason = str(
            ranking.get(
                "reason",
                "",
            )
        ).strip()

        reranked_results.append(
            RerankedSearchResult(
                result=candidate_lookup[candidate_number],
                rerank_score=score,
                reason=reason,
                rerank_rank=0,
            )
        )

    # If the model accidentally omitted candidates,
    # append them using their existing RRF order.
    for candidate_number, candidate in candidate_lookup.items():

        if candidate_number in seen_candidate_numbers:
            continue

        reranked_results.append(
            RerankedSearchResult(
                result=candidate,
                rerank_score=0.0,
                reason=("Candidate was not explicitly " "ranked by the reranker."),
                rerank_rank=0,
            )
        )

    reranked_results.sort(
        key=lambda item: (item.rerank_score),
        reverse=True,
    )

    final_results = reranked_results[:top_k]

    for index, result in enumerate(
        final_results,
        start=1,
    ):
        result.rerank_rank = index

    return final_results


def rerank_candidates(
    query: str,
    candidates: list[FusedSearchResult],
    top_k: int | None = None,
) -> list[RerankedSearchResult]:
    """
    Rerank fused retrieval candidates using OpenAI.

    Only the fused candidate pool is sent to the model.

    Args:
        query:
            User query.

        candidates:
            RRF fused candidates.

        top_k:
            Number of candidates to retain.
            Defaults to RERANK_TOP_K.

    Returns:
        list[RerankedSearchResult]
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Reranking query cannot be empty.")

    if not candidates:
        return []

    if top_k is None:
        top_k = settings.rerank_top_k

    model = create_rerank_model()

    prompt = build_rerank_prompt(
        query=cleaned_query,
        candidates=candidates,
    )

    response = model.invoke(prompt)

    response_text = str(response.content).strip()

    return parse_rerank_response(
        response_text=response_text,
        candidates=candidates,
        top_k=top_k,
    )


def print_reranked_results(
    title: str,
    results: list[RerankedSearchResult],
) -> None:
    """
    Print reranking diagnostics.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not results:

        print()
        print("No reranked results.")

        return

    for result in results:

        candidate = result.result
        document = candidate.document

        print()
        print(f"Result {result.rerank_rank}")

        print("-" * 70)

        print(f"Rerank score: " f"{result.rerank_score:.4f}")

        print(f"RRF rank: " f"{candidate.final_rank}")

        print(f"RRF score: " f"{candidate.fusion_score:.6f}")

        print(f"Sources: " f"{', '.join(candidate.sources)}")

        print(f"Question: " f"{document.metadata.get('question')}")

        print(f"Category: " f"{document.metadata.get('category')}")

        print(f"Reason: " f"{result.reason}")

        print()

        print(document.page_content[:800])
