from __future__ import annotations

import json

from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from backend.a01_config.a01_1_settings import (
    settings,
)

from backend.a05_rag.a05_1_query_intent import (
    QueryIntent,
)

from backend.a05_rag.a05_4_intent_retriever import (
    IntentRetrievalResult,
)

from backend.a05_rag.a05_5_context_builder import (
    GroundedContext,
    build_grounded_context,
)


@dataclass
class GeneratedAnswer:
    """
    Final grounded answer returned by
    the answer-generation component.
    """

    answer: str

    options: list[str]

    follow_up_question: str

    sources: list[str]

    grounded: bool


def create_answer_model() -> ChatOpenAI:
    """
    Create the OpenAI chat model used for final
    grounded answer generation.
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


def build_generation_prompt(
    query: str,
    intent: QueryIntent,
    context: GroundedContext,
) -> str:
    """
    Build the grounded generation prompt.
    """

    return f"""
You are an IKEA FAQ customer-support assistant.

Your answer MUST be based ONLY on the supplied IKEA FAQ context.

Do not use outside knowledge.
Do not invent policies, services, prices, timings, availability,
or other IKEA information.

If the supplied context does not support an answer,
say exactly:

I don't know based on the IKEA FAQ information I have.

USER QUERY:
{query}

QUERY INTENT:
{intent.value}

IKEA FAQ CONTEXT:
{context.context_text}

INSTRUCTIONS BY INTENT:

If intent is "enumeration":
- Find every distinct option supported by the context.
- Do not arbitrarily stop at 3 or 5 options.
- Remove duplicates.
- Present the options clearly.
- After presenting them, ask the customer which option
  they would like to explore further.

If intent is "comparison":
- Cover every comparison target supported by the context.
- Do not omit one side merely because another ranked higher.
- Compare only facts explicitly supported by the context.

If intent is "recommendation":
- Explain the supported alternatives first.
- Do not invent personal recommendations unsupported by the FAQ.
- Ask for clarification if a choice depends on information
  that the FAQ does not provide.

If intent is "procedural":
- Present supported steps in a clear order.

If intent is "policy":
- Preserve important limitations, conditions and exceptions.

If intent is "troubleshooting":
- Present the supported resolution steps clearly.

If intent is "definition":
- Give a concise explanation grounded in the context.

If intent is "confirmation":
- Confirm only when the context supports the claim.

If intent is "factual":
- Answer directly and concisely.

Return ONLY valid JSON in this exact structure:

{{
  "answer": "customer-facing answer",
  "options": [
    "option 1",
    "option 2"
  ],
  "follow_up_question": "question to continue the conversation",
  "grounded": true
}}

CAPABILITY RULES:
- For non-enumeration queries, options may be [].
- Do not place source URLs inside the answer.
- Do not invent information.
- Do not include markdown fences.

- You can explain information found in IKEA FAQ context.
- You cannot directly book, schedule, modify, cancel, or place IKEA orders.
- Do not say "I can book", "I can schedule", or "I can help you book".
- Instead say:
  "Would you like information on how to schedule...?"
  or
  "Would you like the IKEA FAQ steps for scheduling...?"
- If scheduling requires contacting IKEA, clearly state the supported contact method
  from the retrieved FAQ.
  
""".strip()


def parse_generated_answer(
    response_text: str,
    sources: list[str],
) -> GeneratedAnswer:
    """
    Parse the model JSON response.
    """

    try:

        parsed = json.loads(response_text)

    except json.JSONDecodeError as error:

        raise ValueError("Answer generator returned invalid JSON.") from error

    answer = str(
        parsed.get(
            "answer",
            "",
        )
    ).strip()

    raw_options = parsed.get(
        "options",
        [],
    )

    if not isinstance(
        raw_options,
        list,
    ):

        raw_options = []

    options = []

    seen_options = set()

    for option in raw_options:

        cleaned_option = str(option).strip()

        if not cleaned_option:
            continue

        normalized_option = cleaned_option.lower()

        if normalized_option in seen_options:
            continue

        seen_options.add(normalized_option)

        options.append(cleaned_option)

    follow_up_question = str(
        parsed.get(
            "follow_up_question",
            "",
        )
    ).strip()

    grounded = bool(
        parsed.get(
            "grounded",
            False,
        )
    )

    return GeneratedAnswer(
        answer=answer,
        options=options,
        follow_up_question=(follow_up_question),
        sources=sources,
        grounded=grounded,
    )


def generate_grounded_answer(
    retrieval_result: IntentRetrievalResult,
) -> GeneratedAnswer:
    """
    Generate the final grounded IKEA answer from
    intent-selected retrieval documents.
    """

    intent = retrieval_result.intent_result.intent

    # =========================================================
    # OUT OF SCOPE
    # =========================================================

    if intent == QueryIntent.OUT_OF_SCOPE:

        return GeneratedAnswer(
            answer=("I don't know based on the IKEA FAQ " "information I have."),
            options=[],
            follow_up_question="",
            sources=[],
            grounded=False,
        )

    # =========================================================
    # BUILD CONTEXT
    # =========================================================

    context = build_grounded_context(retrieval_result.selected_documents)

    if not context.documents:

        return GeneratedAnswer(
            answer=("I don't know based on the IKEA FAQ " "information I have."),
            options=[],
            follow_up_question="",
            sources=[],
            grounded=False,
        )

    # =========================================================
    # MODEL GENERATION
    # =========================================================

    model = create_answer_model()

    prompt = build_generation_prompt(
        query=retrieval_result.original_query,
        intent=intent,
        context=context,
    )

    response = model.invoke(prompt)

    response_text = str(response.content).strip()

    return parse_generated_answer(
        response_text=response_text,
        sources=context.source_urls,
    )


def print_generated_answer(
    result: GeneratedAnswer,
) -> None:
    """
    Print generated-answer diagnostics.
    """

    print()
    print("=" * 70)
    print("GROUNDED IKEA ANSWER")
    print("=" * 70)

    print()
    print(result.answer)

    if result.options:

        print()
        print("OPTIONS")
        print("-" * 70)

        for index, option in enumerate(
            result.options,
            start=1,
        ):

            print(f"{index}. {option}")

    if result.follow_up_question:

        print()
        print(result.follow_up_question)

    if result.sources:

        print()
        print("SOURCES")
        print("-" * 70)

        for source in result.sources:

            print(source)

    print()
    print(f"Grounded: " f"{result.grounded}")
