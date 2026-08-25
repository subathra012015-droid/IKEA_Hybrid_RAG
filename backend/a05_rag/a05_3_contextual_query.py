from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.a01_config.a01_1_settings import settings
from backend.a05_rag.a05_1_query_intent import (
    ConversationType,
    QueryIntentResult,
)


@dataclass
class ContextualQueryResult:
    """
    Result produced after contextual query handling.

    Attributes:
        original_query:
            Query exactly as supplied by the user.

        retrieval_query:
            Standalone query that should be sent
            to the retrieval pipeline.

        was_rewritten:
            Whether conversation context was needed.

        reason:
            Explanation useful for debugging.
    """

    original_query: str
    retrieval_query: str
    was_rewritten: bool
    reason: str


def create_query_rewrite_model() -> ChatOpenAI:
    """
    Create the model used only for contextual
    query rewriting.

    This model does NOT answer the customer.
    It only converts an incomplete conversational
    query into a standalone retrieval query.
    """

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is missing.")

    if not settings.openai_chat_model:
        raise ValueError("OPENAI_CHAT_MODEL is missing.")

    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=SecretStr(settings.openai_api_key),
        temperature=0,
    )


def format_conversation_history(
    conversation_history: list[dict],
    max_messages: int = 6,
) -> str:
    """
    Convert recent conversation messages into
    compact text for the query rewriter.

    Expected message format:

        {
            "role": "user",
            "content": "..."
        }

        {
            "role": "assistant",
            "content": "..."
        }

    Only recent messages are included so the
    rewrite prompt remains small and inexpensive.
    """

    if not conversation_history:
        return ""

    recent_messages = conversation_history[-max_messages:]

    formatted_messages = []

    for message in recent_messages:

        role = str(
            message.get(
                "role",
                "",
            )
        ).strip()

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not role or not content:
            continue

        formatted_messages.append(f"{role.upper()}: {content}")

    return "\n".join(formatted_messages)


def build_rewrite_prompt(
    query: str,
    conversation_history: list[dict],
) -> str:
    """
    Build the contextual rewrite prompt.

    The model must:
        - preserve user meaning
        - use conversation context only when needed
        - produce one standalone retrieval query
        - not answer the question
        - not add unsupported IKEA facts
    """

    history_text = format_conversation_history(conversation_history)

    return f"""
You are a query-rewriting component for an IKEA FAQ retrieval system.

Your task is NOT to answer the customer.

Convert the CURRENT USER QUERY into one clear,
standalone retrieval query using the conversation history.

Rules:
1. Preserve the user's actual intent.
2. Resolve references such as:
   - it
   - that
   - this
   - those
   - installation
   - that option
   using the conversation history when possible.
3. Do not invent IKEA facts.
4. Do not answer the query.
5. Do not add information that the user did not imply.
6. Keep the rewritten query concise.
7. Return ONLY the rewritten query as plain text.

CONVERSATION HISTORY:
{history_text}

CURRENT USER QUERY:
{query}
""".strip()


def rewrite_query_with_context(
    query: str,
    conversation_history: list[dict],
) -> str:
    """
    Use conversation history to create
    a standalone retrieval query.
    """

    cleaned_query = query.strip()

    if not cleaned_query:
        raise ValueError("Query cannot be empty.")

    if not conversation_history:
        return cleaned_query

    model = create_query_rewrite_model()

    prompt = build_rewrite_prompt(
        query=cleaned_query,
        conversation_history=conversation_history,
    )

    response = model.invoke(prompt)

    rewritten_query = str(response.content).strip()

    if not rewritten_query:
        return cleaned_query

    return rewritten_query


def prepare_contextual_query(
    intent_result: QueryIntentResult,
    conversation_history: list[dict] | None = None,
) -> ContextualQueryResult:
    """
    Decide whether the query needs rewriting.

    Standalone queries:
        use original query directly.

    Follow-up / clarification / confirmation:
        use recent conversation context to create
        a standalone retrieval query.

    Args:
        intent_result:
            Result from a05_1_query_intent.py.

        conversation_history:
            Previous user/assistant messages.

    Returns:
        ContextualQueryResult
    """

    if conversation_history is None:
        conversation_history = []

    original_query = intent_result.query.strip()

    needs_context = intent_result.requires_conversation_context

    # =========================================================
    # STANDALONE QUERY
    # =========================================================

    if not needs_context:

        return ContextualQueryResult(
            original_query=original_query,
            retrieval_query=original_query,
            was_rewritten=False,
            reason=(
                "Query is standalone and does not " "require conversation history."
            ),
        )

    # =========================================================
    # CONTEXT REQUIRED BUT HISTORY MISSING
    # =========================================================

    if not conversation_history:

        return ContextualQueryResult(
            original_query=original_query,
            retrieval_query=original_query,
            was_rewritten=False,
            reason=(
                "Query appears contextual, but no "
                "conversation history was available."
            ),
        )

    # =========================================================
    # CONTEXTUAL REWRITE
    # =========================================================

    rewritten_query = rewrite_query_with_context(
        query=original_query,
        conversation_history=(conversation_history),
    )

    return ContextualQueryResult(
        original_query=original_query,
        retrieval_query=rewritten_query,
        was_rewritten=(rewritten_query != original_query),
        reason=(
            "Conversation context was used to " "prepare a standalone retrieval query."
        ),
    )


def print_contextual_query_result(
    result: ContextualQueryResult,
) -> None:
    """
    Print contextual-query diagnostics.
    """

    print()
    print("=" * 70)
    print("CONTEXTUAL QUERY")
    print("=" * 70)

    print(f"Original query: " f"{result.original_query}")

    print(f"Retrieval query: " f"{result.retrieval_query}")

    print(f"Was rewritten: " f"{result.was_rewritten}")

    print(f"Reason: " f"{result.reason}")


def contextual_query_test() -> None:
    """
    Demonstrate follow-up rewriting.

    This test DOES call the configured OpenAI
    chat model once.
    """

    from backend.a05_rag.a05_1_query_intent import (
        classify_query_intent,
    )

    conversation_history = [
        {
            "role": "user",
            "content": ("What kitchen services are available?"),
        },
        {
            "role": "assistant",
            "content": (
                "IKEA offers measuring service, "
                "in-store planning, online planning, "
                "kitchen validation, and installation."
            ),
        },
    ]

    query = "What about installation?"

    intent_result = classify_query_intent(query)

    result = prepare_contextual_query(
        intent_result=intent_result,
        conversation_history=conversation_history,
    )

    print_contextual_query_result(result)
