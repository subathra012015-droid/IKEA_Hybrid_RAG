from __future__ import annotations

from langchain_core.documents import Document

from backend.a04_retrieval.a04_2_chroma_store import (
    index_parent_documents,
    index_child_documents,
)

from backend.a04_retrieval.a04_3_bm25_index import (
    create_parent_bm25_index,
    create_child_bm25_index,
)

from backend.a04_retrieval.a04_6_hybrid_retriever import (
    HybridRetrievalResult,
    hybrid_retrieve,
    print_hybrid_result,
)


class HybridRetrievalPipeline:
    """
    Final retrieval pipeline for the IKEA Hybrid RAG project.

    Responsibilities:

        1. Incrementally index parent Documents in ChromaDB.
        2. Incrementally index child Documents in ChromaDB.
        3. Build local parent BM25 index.
        4. Build local child BM25 index.
        5. Run Semantic + BM25 + RRF + reranking retrieval.

    The pipeline keeps BM25 indexes in memory so they do
    not need to be rebuilt for every query during one app run.
    """

    def __init__(
        self,
        parent_documents: list[Document],
        child_documents: list[Document],
    ) -> None:
        """
        Initialize the hybrid retrieval pipeline.

        Args:
            parent_documents:
                Complete IKEA FAQ parent Documents.

            child_documents:
                Smaller child answer Documents.
        """

        if not parent_documents:
            raise ValueError("parent_documents cannot be empty.")

        if not child_documents:
            raise ValueError("child_documents cannot be empty.")

        self.parent_documents = parent_documents

        self.child_documents = child_documents

        self.parent_bm25 = None
        self.child_bm25 = None

        self.parent_index_result = None
        self.child_index_result = None

        self.is_ready = False

    def initialize(self) -> dict:
        """
        Initialize all retrieval indexes.

        ChromaDB:
            Incremental and persistent.

        BM25:
            Built locally in memory.

        Returns:
            dict:
                Initialization statistics.
        """

        print()
        print("=" * 70)
        print("INITIALIZE HYBRID RETRIEVAL")
        print("=" * 70)

        # =====================================================
        # STEP 1 - INDEX PARENTS IN CHROMA
        # =====================================================

        print()
        print("STEP 1 - INDEX PARENT DOCUMENTS")

        print("-" * 70)

        self.parent_index_result = index_parent_documents(self.parent_documents)

        print(
            f"Parent documents received: "
            f"{self.parent_index_result['documents_received']}"
        )

        print(f"Parent new: " f"{self.parent_index_result['new_documents']}")

        print(f"Parent changed: " f"{self.parent_index_result['changed_documents']}")

        print(
            f"Parent unchanged: " f"{self.parent_index_result['unchanged_documents']}"
        )

        print(f"Parent embedded: " f"{self.parent_index_result['documents_embedded']}")

        # =====================================================
        # STEP 2 - INDEX CHILDREN IN CHROMA
        # =====================================================

        print()
        print("STEP 2 - INDEX CHILD DOCUMENTS")

        print("-" * 70)

        self.child_index_result = index_child_documents(self.child_documents)

        print(
            f"Child documents received: "
            f"{self.child_index_result['documents_received']}"
        )

        print(f"Child new: " f"{self.child_index_result['new_documents']}")

        print(f"Child changed: " f"{self.child_index_result['changed_documents']}")

        print(f"Child unchanged: " f"{self.child_index_result['unchanged_documents']}")

        print(f"Child embedded: " f"{self.child_index_result['documents_embedded']}")

        # =====================================================
        # STEP 3 - BUILD PARENT BM25
        # =====================================================

        print()
        print("STEP 3 - BUILD PARENT BM25")

        print("-" * 70)

        self.parent_bm25 = create_parent_bm25_index(self.parent_documents)

        print(f"Parent BM25 documents: " f"{len(self.parent_documents)}")

        # =====================================================
        # STEP 4 - BUILD CHILD BM25
        # =====================================================

        print()
        print("STEP 4 - BUILD CHILD BM25")

        print("-" * 70)

        self.child_bm25 = create_child_bm25_index(self.child_documents)

        print(f"Child BM25 documents: " f"{len(self.child_documents)}")

        self.is_ready = True

        print()
        print("Hybrid retrieval initialization completed.")

        return {
            "parent_documents": len(self.parent_documents),
            "child_documents": len(self.child_documents),
            "parent_documents_embedded": (
                self.parent_index_result["documents_embedded"]
            ),
            "child_documents_embedded": (self.child_index_result["documents_embedded"]),
            "ready": self.is_ready,
        }

    def retrieve(
        self,
        query: str,
    ) -> HybridRetrievalResult:
        """
        Run the complete hybrid retrieval stack.

        Query flow:

            Semantic Parent
                  +
            BM25 Parent
                  ↓
            RRF Parent
                  ↓
            Parent Reranker
                  ↓
            Best Parent
                  ↓
            Parent Expansion

        In parallel:

            Semantic Child
                  +
            BM25 Child
                  ↓
            RRF Child

        Args:
            query:
                User question.

        Returns:
            HybridRetrievalResult
        """

        if not self.is_ready:

            raise RuntimeError("Hybrid retrieval pipeline " "has not been initialized.")

        if self.parent_bm25 is None:
            raise RuntimeError("Parent BM25 index is missing.")

        if self.child_bm25 is None:
            raise RuntimeError("Child BM25 index is missing.")

        return hybrid_retrieve(
            query=query,
            parent_bm25=self.parent_bm25,
            child_bm25=self.child_bm25,
        )


def run_retrieval_test(
    parent_documents: list[Document],
    child_documents: list[Document],
    query: str,
) -> HybridRetrievalResult:
    """
    Convenience function for terminal testing.

    This creates the retrieval pipeline,
    initializes it,
    runs one query,
    and prints the result.
    """

    pipeline = HybridRetrievalPipeline(
        parent_documents=parent_documents,
        child_documents=child_documents,
    )

    pipeline.initialize()

    result = pipeline.retrieve(query=query)

    print_hybrid_result(result)

    return result
