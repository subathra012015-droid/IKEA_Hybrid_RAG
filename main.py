"""
# This script serves as a configuration test for the IKEA Hybrid RAG project. It prints out the current settings for various components, including OpenAI models, LangSmith tracing, Chroma collection details, and IKEA-specific configurations.
# It is useful for verifying that the environment variables and settings are correctly loaded and accessible within the application.

from backend.a01_config.settings import settings


def main():
    print("IKEA Hybrid RAG - Configuration Test")
    print("------------------------------------")

    print(f"Chat model: {settings.openai_chat_model}")
    print(f"Embedding model: {settings.openai_embedding_model}")

    print(f"LangSmith tracing: {settings.langsmith_tracing}")
    print(f"LangSmith project: {settings.langsmith_project}")

    print(f"Chroma collection: {settings.chroma_collection_name}")
    print(f"Chroma directory: {settings.chroma_persist_directory}")

    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"Retrieval K: {settings.retrieval_k}")

    print(f"IKEA allowed domain: {settings.ikea_allowed_domain}")
    print(f"IKEA FAQ URL: {settings.ikea_faq_urls}")


if __name__ == "__main__":
    main()



# This script serves as a configuration test for the IKEA Hybrid RAG project.
# It prints out the current settings for various components, including OpenAI models, LangSmith tracing, Chroma collection details, and IKEA-specific configurations.
# It is useful for verifying that the environment variables and settings are correctly loaded and accessible within the application.

from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)


def main() -> None:

 #   Main entry point for the IKEA Hybrid RAG project.


    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Functionality 2 - Website Ingestion")
    print("=" * 70)

    run_ingestion()


if __name__ == "__main__":
    main()



from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)


def main() -> None:

    # Main entry point for the IKEA Hybrid RAG application.


    print("=" * 70)

    print("IKEA HYBRID RAG")

    print("Functionality 2 + 3")

    print("Ingestion + Parent-Child FAQ Processing")

    print("=" * 70)

    # =========================================================
    # FUNCTIONALITY 2
    # =========================================================

    valid_faqs = run_ingestion()

    # =========================================================
    # FUNCTIONALITY 3
    # =========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)

    print("PIPELINE SUMMARY")

    print("=" * 70)

    print(f"Valid FAQs: " f"{len(valid_faqs)}")

    print(f"Parent FAQs: " f"{len(parents)}")

    print(f"Child documents: " f"{len(children)}")


if __name__ == "__main__":
    main()



from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_2_chroma_store import (
    index_parent_documents,
    index_child_documents,
    parent_semantic_search,
    child_semantic_search,
    get_children_for_parent,
)


def print_indexing_result(
    title: str,
    result: dict,
) -> None:

    Print incremental indexing statistics.


    print()
    print(title)
    print("-" * 70)

    print(f"Documents received: " f"{result['documents_received']}")

    print(f"New documents: " f"{result['new_documents']}")

    print(f"Changed documents: " f"{result['changed_documents']}")

    print(f"Unchanged documents: " f"{result['unchanged_documents']}")

    print(f"Documents embedded: " f"{result['documents_embedded']}")


def print_search_results(
    title: str,
    results: list,
) -> None:

    Print semantic search diagnostics.


    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        document, score = result

        print()
        print(f"Result {rank}")

        print("-" * 70)

        print(f"Distance score: " f"{score}")

        print(f"FAQ ID: " f"{document.metadata.get('faq_id')}")

        print(f"Parent FAQ ID: " f"{document.metadata.get('parent_faq_id')}")

        print(f"Question: " f"{document.metadata.get('question')}")

        print(f"Category: " f"{document.metadata.get('category')}")

        print()

        print(document.page_content[:800])


def main() -> None:

    IKEA Hybrid RAG development pipeline.

    Current functionality:

        1. Website ingestion
        2. FAQ extraction
        3. Validation
        4. Parent-child processing
        5. Parent semantic indexing
        6. Child semantic indexing
        7. Parent semantic search
        8. Child semantic search
        9. Parent-child expansion


    print("=" * 70)

    print("IKEA HYBRID RAG")

    print("Semantic Parent + Child Retrieval")

    print("=" * 70)

    # ========================================================
    # INGESTION
    # ========================================================

    valid_faqs = run_ingestion()

    # ========================================================
    # PROCESSING
    # ========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # ========================================================
    # INDEX PARENTS
    # ========================================================

    parent_index_result = index_parent_documents(parents)

    print_indexing_result(
        title=("STEP 9 - INDEX " "PARENT FAQ DOCUMENTS"),
        result=parent_index_result,
    )

    # ========================================================
    # INDEX CHILDREN
    # ========================================================

    child_index_result = index_child_documents(children)

    print_indexing_result(
        title=("STEP 10 - INDEX " "CHILD FAQ DOCUMENTS"),
        result=child_index_result,
    )

    # ========================================================
    # TEST QUERY
    # ========================================================

    query = "What kitchen services are available?"

    print()
    print(f"Test query: {query}")

    # ========================================================
    # PARENT SEMANTIC SEARCH
    # ========================================================

    parent_results = parent_semantic_search(query=query)

    print_search_results(
        title=("PARENT SEMANTIC RESULTS"),
        results=parent_results,
    )

    # ========================================================
    # CHILD SEMANTIC SEARCH
    # ========================================================

    child_results = child_semantic_search(query=query)

    print_search_results(
        title=("CHILD SEMANTIC RESULTS"),
        results=child_results,
    )

    # ========================================================
    # PARENT EXPANSION TEST
    # ========================================================

    if parent_results:

        best_parent_document = parent_results[0][0]

        best_parent_id = str(
            best_parent_document.metadata.get(
                "parent_faq_id",
                "",
            )
        )

        related_children = get_children_for_parent(best_parent_id)

        print()
        print("=" * 70)

        print("BEST PARENT EXPANSION")

        print("=" * 70)

        print(f"Parent ID: " f"{best_parent_id}")

        print(f"Question: " f"{best_parent_document.metadata.get('question')}")

        print(f"Children found: " f"{len(related_children)}")

        for index, child in enumerate(
            related_children,
            start=1,
        ):
            print()
            print(f"Child {index}")

            print("-" * 70)

            print(child.page_content[:1000])

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)

    print("PIPELINE SUMMARY")

    print("=" * 70)

    print(f"Valid FAQs: " f"{len(valid_faqs)}")

    print(f"Parents: " f"{len(parents)}")

    print(f"Children: " f"{len(children)}")

    print(
        "Parent documents embedded this run: "
        f"{parent_index_result['documents_embedded']}"
    )

    print(
        "Child documents embedded this run: "
        f"{child_index_result['documents_embedded']}"
    )


if __name__ == "__main__":
    main()



from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_2_chroma_store import (
    index_parent_documents,
    index_child_documents,
    parent_semantic_search,
    child_semantic_search,
)

from backend.a04_retrieval.a04_3_bm25_index import (
    create_parent_bm25_index,
    create_child_bm25_index,
)

from backend.a04_retrieval.a04_4_candidate_fusion import (
    fuse_results,
    print_fused_results,
)

from backend.a04_retrieval.a04_5_reranker import (
    rerank_candidates,
    print_reranked_results,
)


def print_indexing_result(
    title: str,
    result: dict,
) -> None:

#    Print incremental Chroma indexing statistics.


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

    # IKEA Hybrid RAG retrieval development pipeline.

    # Current retrieval architecture:

    #     Semantic
    #         +
    #     BM25
    #         ↓
    #     RRF Fusion
    #         ↓
    #     LLM Reranking


    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Semantic + BM25 + RRF + Reranking")
    print("=" * 70)

    # =========================================================
    # STEP 1 - INGEST
    # =========================================================

    valid_faqs = run_ingestion()

    # =========================================================
    # STEP 2 - PROCESS
    # =========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # STEP 3 - INDEX PARENTS
    # =========================================================

    parent_index_result = index_parent_documents(parents)

    print_indexing_result(
        title="PARENT CHROMA INDEX",
        result=parent_index_result,
    )

    # =========================================================
    # STEP 4 - INDEX CHILDREN
    # =========================================================

    child_index_result = index_child_documents(children)

    print_indexing_result(
        title="CHILD CHROMA INDEX",
        result=child_index_result,
    )

    # =========================================================
    # STEP 5 - BUILD BM25
    # =========================================================

    parent_bm25 = create_parent_bm25_index(parents)

    child_bm25 = create_child_bm25_index(children)

    # =========================================================
    # TEST QUERY
    # =========================================================

    query = "What kitchen services are available?"

    print()
    print("=" * 70)
    print("TEST QUERY")
    print("=" * 70)
    print(query)

    # =========================================================
    # PARENT SEMANTIC
    # =========================================================

    parent_semantic_results = parent_semantic_search(query=query)

    # =========================================================
    # PARENT BM25
    # =========================================================

    parent_bm25_results = parent_bm25.search(query=query)

    # =========================================================
    # PARENT RRF
    # =========================================================

    parent_fused_results = fuse_results(
        semantic_results=(parent_semantic_results),
        bm25_results=(parent_bm25_results),
    )

    print_fused_results(
        title=("PARENT RRF FUSION RESULTS"),
        results=(parent_fused_results),
    )

    # =========================================================
    # PARENT RERANKING
    # =========================================================

    parent_reranked_results = rerank_candidates(
        query=query,
        candidates=(parent_fused_results),
    )

    print_reranked_results(
        title=("PARENT RERANKED RESULTS"),
        results=(parent_reranked_results),
    )

    # =========================================================
    # CHILD SEMANTIC
    # =========================================================

    child_semantic_results = child_semantic_search(query=query)

    # =========================================================
    # CHILD BM25
    # =========================================================

    child_bm25_results = child_bm25.search(query=query)

    # =========================================================
    # CHILD RRF
    # =========================================================

    child_fused_results = fuse_results(
        semantic_results=(child_semantic_results),
        bm25_results=(child_bm25_results),
    )

    print_fused_results(
        title=("CHILD RRF FUSION RESULTS"),
        results=(child_fused_results),
    )

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("RETRIEVAL SUMMARY")
    print("=" * 70)

    print(f"Parents: " f"{len(parents)}")

    print(f"Children: " f"{len(children)}")

    print(f"Parent semantic candidates: " f"{len(parent_semantic_results)}")

    print(f"Parent BM25 candidates: " f"{len(parent_bm25_results)}")

    print(f"Parent fused candidates: " f"{len(parent_fused_results)}")

    print(f"Parent reranked results: " f"{len(parent_reranked_results)}")

    print(f"Child semantic candidates: " f"{len(child_semantic_results)}")

    print(f"Child BM25 candidates: " f"{len(child_bm25_results)}")

    print(f"Child fused candidates: " f"{len(child_fused_results)}")

    print(
        "Parent documents embedded this run: "
        f"{parent_index_result['documents_embedded']}"
    )

    print(
        "Child documents embedded this run: "
        f"{child_index_result['documents_embedded']}"
    )


if __name__ == "__main__":
    main()


from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_7_pipeline import (
    HybridRetrievalPipeline,
)

from backend.a04_retrieval.a04_6_hybrid_retriever import (
    print_hybrid_result,
)


def main() -> None:
    #
    # Main entry point for the IKEA Hybrid RAG project.

    # Current functionality:

    #     Functionality 2:
    #         IKEA website ingestion

    #     Functionality 3:
    #         Parent-child FAQ processing

    #     Functionality 4:
    #         Semantic + BM25 + RRF + reranking retrieval


    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Hybrid Retrieval")
    print("=" * 70)

    # =========================================================
    # FUNCTIONALITY 2 - INGESTION
    # =========================================================

    valid_faqs = run_ingestion()

    # =========================================================
    # FUNCTIONALITY 3 - PROCESSING
    # =========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # FUNCTIONALITY 4 - RETRIEVAL INITIALIZATION
    # =========================================================

    retrieval_pipeline = HybridRetrievalPipeline(
        parent_documents=parents,
        child_documents=children,
    )

    retrieval_pipeline.initialize()

    # =========================================================
    # TEST QUERY
    # =========================================================

    query = "What kitchen services are available?"

    print()
    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    print(f"Query: {query}")

    result = retrieval_pipeline.retrieve(query=query)

    print_hybrid_result(result)

    # =========================================================
    # SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("APPLICATION SUMMARY")
    print("=" * 70)

    print(f"Valid FAQs: " f"{len(valid_faqs)}")

    print(f"Parent documents: " f"{len(parents)}")

    print(f"Child documents: " f"{len(children)}")

    print("Retrieval pipeline ready: " f"{retrieval_pipeline.is_ready}")


if __name__ == "__main__":
    main()



from backend.a02_ingestion.a02_5_pipeline import (
    run_ingestion,
)

from backend.a03_processing.a03_4_pipeline import (
    run_processing,
)

from backend.a04_retrieval.a04_7_pipeline import (
    HybridRetrievalPipeline,
)

from backend.a05_rag.a05_5_pipeline import (
    IntentAwareRAGPipeline,
    print_rag_pipeline_result,
)


def main() -> None:

    # Main entry point for the IKEA Hybrid RAG project.

    # Current functionality:

    #     Functionality 2:
    #         IKEA FAQ ingestion

    #     Functionality 3:
    #         Parent-child processing

    #     Functionality 4:
    #         Semantic + BM25 + RRF + reranking

    #     Functionality 5:
    #         Intent-aware retrieval
    #

    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Intent-Aware Hybrid Retrieval")
    print("=" * 70)

    # =========================================================
    # FUNCTIONALITY 2 - INGESTION
    # =========================================================

    valid_faqs = run_ingestion()

    # =========================================================
    # FUNCTIONALITY 3 - PROCESSING
    # =========================================================

    knowledge_structure = run_processing(valid_faqs)

    parents = knowledge_structure["parents"]

    children = knowledge_structure["children"]

    # =========================================================
    # FUNCTIONALITY 4 - HYBRID RETRIEVAL INITIALIZATION
    # =========================================================

    retrieval_pipeline = HybridRetrievalPipeline(
        parent_documents=parents,
        child_documents=children,
    )

    retrieval_pipeline.initialize()

    # =========================================================
    # FUNCTIONALITY 5 - INTENT-AWARE RAG
    # =========================================================

    rag_pipeline = IntentAwareRAGPipeline(retrieval_pipeline=(retrieval_pipeline))

    # =========================================================
    # TEST QUERY
    # =========================================================

    query = "What kitchen services are available?"

    result = rag_pipeline.retrieve(
        query=query,
        conversation_history=[],
    )

    print_rag_pipeline_result(result)

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print()
    print("=" * 70)
    print("APPLICATION SUMMARY")
    print("=" * 70)

    print(f"Valid FAQs: " f"{len(valid_faqs)}")

    print(f"Parent documents: " f"{len(parents)}")

    print(f"Child documents: " f"{len(children)}")

    print(f"Intent: " f"{result.retrieval_result.intent_result.intent.value}")

    print(f"Strategy: " f"{result.retrieval_result.route.strategy.value}")

    print("Selected documents: " f"{len(result.retrieval_result.selected_documents)}")


if __name__ == "__main__":
    main()

"""

from backend.a05_rag.a05_9_pipeline import (
    run_terminal_chat,
)

from backend.a06_runtime.a06_2_chat_bootstrap import (
    build_chat_assistant,
)


def main() -> None:
    """
    Start the IKEA customer assistant.

    Normal startup uses the locally saved FAQ snapshot.
    It does not fetch ikea.com.
    """

    print("=" * 70)
    print("IKEA HYBRID RAG")
    print("Fast Chat Startup")
    print("=" * 70)

    assistant = build_chat_assistant()

    run_terminal_chat(assistant)


if __name__ == "__main__":
    main()
