from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.a01_config.a01_1_settings import settings


@dataclass
class BM25SearchResult:
    """
    One BM25 retrieval result.

    Attributes:
        document:
            Matching LangChain Document.

        score:
            Raw BM25 relevance score.

        rank:
            Ranking position starting from 1.

        document_type:
            parent or child
    """

    document: Document
    score: float
    rank: int
    document_type: str


def tokenize_text(
    text: str,
) -> list[str]:
    """
    Convert text into normalized lexical tokens.

    BM25 does not understand semantic meaning.

    Therefore we normalize:
        - lowercase
        - punctuation
        - repeated whitespace

    Example:

        "What Kitchen Services are available?"

    becomes:

        [
            "what",
            "kitchen",
            "services",
            "are",
            "available"
        ]
    """

    normalized_text = text.lower()

    tokens = re.findall(
        r"[a-z0-9]+(?:['-][a-z0-9]+)*",
        normalized_text,
    )

    return tokens


def build_parent_search_text(
    document: Document,
) -> str:
    """
    Build searchable lexical text for a parent FAQ.

    Parent documents already contain the complete
    Question + Answer.

    We additionally include category metadata because
    category terms can be useful lexical signals.
    """

    category = str(
        document.metadata.get(
            "category",
            "",
        )
    ).strip()

    return (f"Category: {category}\n" f"{document.page_content}").strip()


def build_child_search_text(
    document: Document,
) -> str:
    """
    Build lexical search text for a child document.

    Important:
        Child page_content intentionally contains only
        the answer section.

    For BM25 indexing we can safely include the parent
    question dynamically in the searchable representation.

    This does NOT modify the actual stored child Document.
    """

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

    return (
        f"Question: {question}\n"
        f"Category: {category}\n"
        f"Answer: {document.page_content}"
    ).strip()


class BM25Index:
    """
    Local BM25 lexical index.

    Supports:
        - parent FAQ search
        - child FAQ search

    No OpenAI API calls are made.
    """

    def __init__(
        self,
        documents: list[Document],
        document_type: str,
    ) -> None:
        """
        Build an in-memory BM25 index.

        Args:
            documents:
                Parent or child Documents.

            document_type:
                Must be:
                    parent
                    child
        """

        if document_type not in {
            "parent",
            "child",
        }:
            raise ValueError("document_type must be " "'parent' or 'child'.")

        if not documents:
            raise ValueError("Cannot build BM25 index " "without documents.")

        self.documents = documents
        self.document_type = document_type

        self.search_texts = []

        for document in documents:

            if document_type == "parent":

                search_text = build_parent_search_text(document)

            else:

                search_text = build_child_search_text(document)

            self.search_texts.append(search_text)

        self.tokenized_corpus = [
            tokenize_text(search_text) for search_text in self.search_texts
        ]

        self.index = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[BM25SearchResult]:
        """
        Search the BM25 index.

        Args:
            query:
                User query.

            top_k:
                Maximum results.
                Defaults to BM25_TOP_K.

        Returns:
            Ranked BM25SearchResult objects.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError("BM25 query cannot be empty.")

        if top_k is None:

            top_k = settings.bm25_top_k

        query_tokens = tokenize_text(cleaned_query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for document_index in ranked_indexes[:top_k]:

            score = float(scores[document_index])

            # Do not return completely irrelevant
            # zero-score lexical matches.
            if score <= 0:
                continue

            document = self.documents[document_index]

            results.append(
                BM25SearchResult(
                    document=document,
                    score=score,
                    rank=len(results) + 1,
                    document_type=(self.document_type),
                )
            )

        return results


def create_parent_bm25_index(
    parent_documents: list[Document],
) -> BM25Index:
    """
    Create BM25 index for complete parent FAQs.
    """

    return BM25Index(
        documents=parent_documents,
        document_type="parent",
    )


def create_child_bm25_index(
    child_documents: list[Document],
) -> BM25Index:
    """
    Create BM25 index for detailed child chunks.
    """

    return BM25Index(
        documents=child_documents,
        document_type="child",
    )


def print_bm25_results(
    title: str,
    results: list[BM25SearchResult],
) -> None:
    """
    Print BM25 retrieval diagnostics.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not results:

        print()
        print("No BM25 results found.")

        return

    for result in results:

        document = result.document

        print()
        print(f"Result {result.rank}")

        print("-" * 70)

        print(f"BM25 score: " f"{result.score}")

        print(f"Document type: " f"{result.document_type}")

        print(f"FAQ ID: " f"{document.metadata.get('faq_id')}")

        print(f"Parent FAQ ID: " f"{document.metadata.get('parent_faq_id')}")

        print(f"Question: " f"{document.metadata.get('question')}")

        print(f"Category: " f"{document.metadata.get('category')}")

        print()

        print(document.page_content[:800])
