from __future__ import annotations

from dataclasses import dataclass

from backend.a04_retrieval.a04_7_pipeline import (
    HybridRetrievalPipeline,
)

from backend.a05_rag.a05_4_intent_retriever import (
    IntentRetrievalResult,
    retrieve_for_intent,
)

from backend.a05_rag.a05_6_generator import (
    GeneratedAnswer,
    generate_grounded_answer,
)

from backend.a05_rag.a05_7_conversation_manager import (
    ConversationState,
    build_assistant_history_text,
)

from backend.a05_rag.a05_8_conversation_resolver import (
    ConversationResolutionResult,
    resolve_conversation_query,
)


@dataclass
class RAGPipelineResult:
    """
    Complete result of one conversational interaction.
    """

    original_query: str

    resolved_query: str

    resolution_result: ConversationResolutionResult

    retrieval_result: IntentRetrievalResult | None

    generated_answer: GeneratedAnswer


class ConversationalIKEAAssistant:
    """
    Complete conversational IKEA Hybrid RAG assistant.

    Flow:

        Customer message
              ↓
        Conversation resolver
              ↓
        standalone retrieval query
              ↓
        Intent classification
              ↓
        Semantic + BM25
              ↓
        RRF
              ↓
        reranking
              ↓
        intent-aware evidence
              ↓
        grounded answer
              ↓
        conversation memory
    """

    def __init__(
        self,
        retrieval_pipeline: HybridRetrievalPipeline,
        max_history_messages: int = 12,
    ) -> None:

        if not retrieval_pipeline.is_ready:

            raise RuntimeError("HybridRetrievalPipeline " "must be initialized first.")

        self.retrieval_pipeline = retrieval_pipeline

        self.conversation = ConversationState(max_messages=(max_history_messages))

    def invoke(
        self,
        query: str,
    ) -> RAGPipelineResult:
        """
        Process one customer message.
        """

        cleaned_query = query.strip()

        if not cleaned_query:

            raise ValueError("User query cannot be empty.")

        # =====================================================
        # HISTORY BEFORE CURRENT MESSAGE
        # =====================================================

        conversation_history = self.conversation.as_dict_list()

        # =====================================================
        # CONVERSATION RESOLUTION
        # =====================================================

        resolution_result = resolve_conversation_query(
            query=cleaned_query,
            conversation_history=(conversation_history),
        )

        # =====================================================
        # SAVE USER MESSAGE
        # =====================================================

        self.conversation.add_user_message(cleaned_query)

        # =====================================================
        # NO RETRIEVAL REQUIRED
        # =====================================================

        if not resolution_result.needs_retrieval:

            direct_answer = resolution_result.direct_response

            generated_answer = GeneratedAnswer(
                answer=direct_answer,
                options=[],
                follow_up_question="",
                sources=[],
                grounded=True,
            )

            self.conversation.add_assistant_message(direct_answer)

            return RAGPipelineResult(
                original_query=cleaned_query,
                resolved_query="",
                resolution_result=(resolution_result),
                retrieval_result=None,
                generated_answer=(generated_answer),
            )

        # =====================================================
        # RETRIEVAL QUERY
        # =====================================================

        retrieval_query = resolution_result.resolved_query.strip()

        if not retrieval_query:

            retrieval_query = cleaned_query

        # =====================================================
        # INTENT-AWARE HYBRID RETRIEVAL
        # =====================================================

        retrieval_result = retrieve_for_intent(
            query=retrieval_query,
            retrieval_pipeline=(self.retrieval_pipeline),
            conversation_history=(conversation_history),
        )

        # Preserve what the customer actually typed
        # for final answer generation.
        retrieval_result.original_query = cleaned_query

        # =====================================================
        # GENERATION
        # =====================================================

        generated_answer = generate_grounded_answer(retrieval_result)

        # =====================================================
        # STORE ASSISTANT TURN
        # =====================================================

        assistant_history_text = build_assistant_history_text(
            answer=(generated_answer.answer),
            options=(generated_answer.options),
            follow_up_question=(generated_answer.follow_up_question),
        )

        self.conversation.add_assistant_message(assistant_history_text)

        return RAGPipelineResult(
            original_query=cleaned_query,
            resolved_query=retrieval_query,
            resolution_result=(resolution_result),
            retrieval_result=(retrieval_result),
            generated_answer=(generated_answer),
        )

    def clear_conversation(
        self,
    ) -> None:
        """
        Start a new session.
        """

        self.conversation.clear()

    def get_history(
        self,
    ) -> list[dict]:
        """
        Return conversation history.
        """

        return self.conversation.as_dict_list()


def print_customer_response(
    result: RAGPipelineResult,
) -> None:
    """
    Print only customer-facing content.
    """

    answer = result.generated_answer

    print()
    print(answer.answer)

    if answer.options:

        print()

        for index, option in enumerate(
            answer.options,
            start=1,
        ):

            print(f"{index}. {option}")

    if answer.follow_up_question:

        print()

        print(answer.follow_up_question)

    if answer.sources:

        print()
        print("Source:")

        for source in answer.sources:

            print(source)


def run_terminal_chat(
    assistant: ConversationalIKEAAssistant,
) -> None:
    """
    Run terminal-based IKEA chat.
    """

    print()
    print("=" * 70)
    print("IKEA CUSTOMER ASSISTANT")
    print("=" * 70)

    print()
    print("Ask an IKEA FAQ question.")

    print("Type 'clear' to start a new conversation.")

    print("Type 'exit' to stop.")

    while True:

        print()

        user_query = input("You: ").strip()

        if not user_query:

            continue

        command = user_query.lower()

        # =====================================================
        # EXIT
        # =====================================================

        if command in {
            "exit",
            "quit",
        }:

            print()
            print("Assistant: Goodbye.")

            break

        # =====================================================
        # CLEAR
        # =====================================================

        if command == "clear":

            assistant.clear_conversation()

            print()
            print("Assistant: " "Conversation cleared.")

            continue

        # =====================================================
        # QUERY
        # =====================================================

        try:

            result = assistant.invoke(user_query)

            print()
            print(
                "Assistant:",
                end="",
            )

            print_customer_response(result)

        except Exception as error:

            print()
            print("Assistant: " "An error occurred while " "processing the request.")

            print(f"Development error: {error}")
