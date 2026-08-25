from __future__ import annotations

import json
import re

from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI

from backend.a01_config.a01_1_settings import settings


class ConversationAction(str, Enum):
    """
    Conversational action detected before retrieval.
    """

    STANDALONE = "standalone"
    CONTINUE = "continue"
    CONFIRM = "confirm"
    REJECT = "reject"
    OPTION_SELECTION = "option_selection"
    CONTEXTUAL = "contextual"
    CLARIFY = "clarify"


@dataclass
class ConversationResolutionResult:
    """
    Result of resolving one user message.

    original_query:
        What the user actually typed.

    resolved_query:
        Standalone query suitable for retrieval.

    action:
        Detected conversational action.

    needs_retrieval:
        Whether Semantic + BM25 retrieval should run.

    needs_clarification:
        Whether we should ask the user for clarification.

    direct_response:
        Response that can be returned without retrieval.

    was_resolved:
        Whether conversation history changed the query.

    reason:
        Debugging explanation.
    """

    original_query: str
    resolved_query: str
    action: ConversationAction
    needs_retrieval: bool
    needs_clarification: bool
    direct_response: str
    was_resolved: bool
    reason: str


# ============================================================
# CONVERSATIONAL COMMANDS
# ============================================================

AFFIRMATIVE_WORDS = {
    "yes",
    "yes please",
    "yeah",
    "yep",
    "sure",
    "okay",
    "ok",
    "please",
    "sounds good",
}


CONTINUE_WORDS = {
    "go",
    "go ahead",
    "next",
    "proceed",
    "continue",
    "carry on",
    "move on",
    "keep going",
}


NEGATIVE_WORDS = {
    "no",
    "no thanks",
    "no thank you",
    "nope",
    "not now",
}


# ============================================================
# EXPLICIT REFERENCES
# ============================================================

REFERENCE_PATTERNS = [
    r"^it$",
    r"^that$",
    r"^this$",
    r"^that one$",
    r"^this one$",
    r"^the first one$",
    r"^the second one$",
    r"^the third one$",
    r"^the fourth one$",
    r"^the fifth one$",
    r"^the last one$",
    r"^first$",
    r"^second$",
    r"^third$",
    r"^fourth$",
    r"^fifth$",
    r"^last$",
    r"^\d+$",
]


# ============================================================
# CONTEXT-DEPENDENT QUESTIONS
# ============================================================

CONTEXTUAL_PATTERNS = [
    r"^what about\b",
    r"^how about\b",
    r"^tell me more\b",
    r"^more details\b",
    r"^more about\b",
    r"^how long\b",
    r"^how much\b",
    r"^when does it\b",
    r"^where does it\b",
    r"^why does it\b",
    r"^what does it\b",
    r"^how does it\b",
]


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_query(
    query: str,
) -> str:
    """
    Normalize whitespace.
    """

    cleaned = query.strip()

    return re.sub(
        r"\s+",
        " ",
        cleaned,
    )


def normalize_for_matching(
    query: str,
) -> str:
    """
    Normalize for local command matching.
    """

    normalized = normalize_query(query).lower()

    return normalized.rstrip(" .!?;")


# ============================================================
# BASIC HELPERS
# ============================================================


def matches_reference_pattern(
    query: str,
) -> bool:
    """
    Detect explicit references such as:
        5
        last one
        second
        that one
    """

    for pattern in REFERENCE_PATTERNS:

        if re.search(
            pattern,
            query,
        ):
            return True

    return False


def matches_contextual_pattern(
    query: str,
) -> bool:
    """
    Detect phrases that clearly depend on
    previous conversation context.
    """

    for pattern in CONTEXTUAL_PATTERNS:

        if re.search(
            pattern,
            query,
        ):
            return True

    return False


# ============================================================
# LOCAL ACTION DETECTION
# ============================================================


def detect_local_action(
    query: str,
    has_history: bool,
) -> ConversationAction:
    """
    Detect obvious conversation controls.

    Important rule:

    A short phrase is NOT automatically a follow-up.

    Example:

        New conversation:
            "kitchen installation"

        This is a valid standalone query.

    Short phrases become context candidates only
    when useful conversation history already exists.
    """

    normalized = normalize_for_matching(query)

    # --------------------------------------------------------
    # AFFIRMATIVE
    # --------------------------------------------------------

    if normalized in AFFIRMATIVE_WORDS:

        return ConversationAction.CONFIRM

    # --------------------------------------------------------
    # CONTINUE
    # --------------------------------------------------------

    if normalized in CONTINUE_WORDS:

        return ConversationAction.CONTINUE

    # --------------------------------------------------------
    # REJECT
    # --------------------------------------------------------

    if normalized in NEGATIVE_WORDS:

        return ConversationAction.REJECT

    # --------------------------------------------------------
    # EXPLICIT OPTION REFERENCES
    # --------------------------------------------------------

    if matches_reference_pattern(normalized):

        return ConversationAction.OPTION_SELECTION

    # --------------------------------------------------------
    # CLEAR CONTEXTUAL LANGUAGE
    # --------------------------------------------------------

    if matches_contextual_pattern(normalized):

        return ConversationAction.CONTEXTUAL

    # --------------------------------------------------------
    # SHORT PHRASE + HISTORY
    # --------------------------------------------------------

    if has_history:

        word_count = len(normalized.split())

        if word_count <= 3:

            return ConversationAction.OPTION_SELECTION

    # --------------------------------------------------------
    # OTHERWISE STANDALONE
    # --------------------------------------------------------

    return ConversationAction.STANDALONE


# ============================================================
# HISTORY FORMATTER
# ============================================================


def format_history(
    conversation_history: list[dict],
    max_messages: int = 6,
) -> str:
    """
    Format recent conversation history.
    """

    recent_messages = conversation_history[-max_messages:]

    lines = []

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

        lines.append(f"{role.upper()}: {content}")

    return "\n".join(lines)


# ============================================================
# RESOLVER MODEL
# ============================================================


def create_resolution_model() -> ChatOpenAI:
    """
    Create the model used only for ambiguous
    conversation resolution.
    """

    if not settings.openai_api_key:

        raise ValueError("OPENAI_API_KEY is missing.")

    if not settings.openai_chat_model:

        raise ValueError("OPENAI_CHAT_MODEL is missing.")

    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,  # type: ignore
        temperature=0,
    )


# ============================================================
# RESOLUTION PROMPT
# ============================================================


def build_resolution_prompt(
    query: str,
    action: ConversationAction,
    conversation_history: list[dict],
) -> str:
    """
    Build conversation-resolution prompt.
    """

    history_text = format_history(conversation_history)

    return f"""
You are the conversation-resolution component of an IKEA FAQ assistant.

You MUST NOT answer the IKEA question.

Your only task is to determine what the customer's current
message means in the context of the recent conversation.

DETECTED ACTION:
{action.value}

RECENT CONVERSATION:
{history_text}

CURRENT USER MESSAGE:
{query}

RESOLUTION RULES:

1. Produce a standalone retrieval query when the customer's
   meaning is clear.

2. Do not invent IKEA information.

3. "go", "next", "proceed", "continue", "go ahead",
   "carry on" and similar messages mean:
   continue with the specific topic or action most recently
   offered by the assistant.

4. "yes", "sure", "okay", "ok" and similar messages confirm
   the assistant's most recent offer.

5. IMPORTANT:
   If the assistant's last question offered MULTIPLE choices
   and the user only says "yes", do NOT guess which choice
   they want.

   Example:

   Assistant:
   "Would you like scheduling information or lead-time details?"

   User:
   "yes"

   Correct:
   needs_retrieval = false
   needs_clarification = true

   clarification_question:
   "Would you like scheduling information, lead-time details, or both?"

6. If the assistant asked ONE yes/no continuation question:

   Assistant:
   "Would you like information about installation scheduling?"

   User:
   "yes"

   Resolve to:

   "How do I schedule IKEA kitchen installation?"

7. Resolve option references using the most recent option list.

   Example:

   Assistant:
   1. Measuring
   2. In-store planning
   3. Online planning
   4. Kitchen validation
   5. Kitchen installation

   User:
   "5"

   Resolve to:
   "Tell me about IKEA kitchen installation."

8. Short named topics may be selections when conversation
   history supports them.

   Example:

   Assistant:
   "Which kitchen service would you like?"

   User:
   "installation"

   Resolve to:
   "Tell me about IKEA kitchen installation."

9. If the current message clearly starts a new topic,
   preserve it as a standalone query.

10. If the user rejects the previous offer, retrieval is
    normally unnecessary.

Return ONLY valid JSON:

{{
  "resolved_query": "",
  "needs_retrieval": true,
  "needs_clarification": false,
  "clarification_question": "",
  "reason": "brief explanation"
}}
""".strip()


# ============================================================
# RESPONSE PARSER
# ============================================================


def parse_resolution_response(
    response_text: str,
    original_query: str,
    action: ConversationAction,
) -> ConversationResolutionResult:
    """
    Parse resolver JSON.
    """

    try:

        parsed = json.loads(response_text)

    except json.JSONDecodeError as error:

        raise ValueError("Conversation resolver returned " "invalid JSON.") from error

    resolved_query = str(
        parsed.get(
            "resolved_query",
            "",
        )
    ).strip()

    needs_retrieval = bool(
        parsed.get(
            "needs_retrieval",
            False,
        )
    )

    needs_clarification = bool(
        parsed.get(
            "needs_clarification",
            False,
        )
    )

    clarification_question = str(
        parsed.get(
            "clarification_question",
            "",
        )
    ).strip()

    reason = str(
        parsed.get(
            "reason",
            "",
        )
    ).strip()

    direct_response = ""

    if needs_clarification:

        direct_response = clarification_question

    return ConversationResolutionResult(
        original_query=original_query,
        resolved_query=resolved_query,
        action=action,
        needs_retrieval=needs_retrieval,
        needs_clarification=(needs_clarification),
        direct_response=(direct_response),
        was_resolved=(
            bool(resolved_query)
            and (
                normalize_for_matching(resolved_query)
                != normalize_for_matching(original_query)
            )
        ),
        reason=reason,
    )


# ============================================================
# MAIN RESOLVER
# ============================================================


def resolve_conversation_query(
    query: str,
    conversation_history: list[dict] | None = None,
) -> ConversationResolutionResult:
    """
    Resolve user input before retrieval.
    """

    if conversation_history is None:

        conversation_history = []

    cleaned_query = normalize_query(query)

    if not cleaned_query:

        raise ValueError("Query cannot be empty.")

    has_history = bool(conversation_history)

    action = detect_local_action(
        query=cleaned_query,
        has_history=has_history,
    )

    # ========================================================
    # STANDALONE
    # ========================================================

    if action == ConversationAction.STANDALONE:

        return ConversationResolutionResult(
            original_query=cleaned_query,
            resolved_query=cleaned_query,
            action=action,
            needs_retrieval=True,
            needs_clarification=False,
            direct_response="",
            was_resolved=False,
            reason=("Query is sufficiently standalone " "for normal retrieval."),
        )

    # ========================================================
    # CONTEXTUAL COMMAND WITHOUT HISTORY
    # ========================================================

    if not has_history:

        # A real content phrase should still be usable
        # as a standalone query.
        #
        # Explicit controls such as "yes", "next", "5",
        # "that one" genuinely require history.

        normalized = normalize_for_matching(cleaned_query)

        requires_history = (
            normalized in AFFIRMATIVE_WORDS
            or normalized in CONTINUE_WORDS
            or normalized in NEGATIVE_WORDS
            or matches_reference_pattern(normalized)
            or matches_contextual_pattern(normalized)
        )

        if not requires_history:

            return ConversationResolutionResult(
                original_query=cleaned_query,
                resolved_query=cleaned_query,
                action=ConversationAction.STANDALONE,
                needs_retrieval=True,
                needs_clarification=False,
                direct_response="",
                was_resolved=False,
                reason=(
                    "No prior conversation exists; "
                    "content phrase is treated as a "
                    "standalone IKEA query."
                ),
            )

        return ConversationResolutionResult(
            original_query=cleaned_query,
            resolved_query="",
            action=ConversationAction.CLARIFY,
            needs_retrieval=False,
            needs_clarification=True,
            direct_response=("What IKEA topic would you like " "to continue with?"),
            was_resolved=False,
            reason=(
                "The message depends on previous " "context, but no history exists."
            ),
        )

    # ========================================================
    # REJECTION
    # ========================================================

    if action == ConversationAction.REJECT:

        return ConversationResolutionResult(
            original_query=cleaned_query,
            resolved_query="",
            action=action,
            needs_retrieval=False,
            needs_clarification=False,
            direct_response=("Okay. What else would you like " "to know about IKEA?"),
            was_resolved=True,
            reason=("Customer rejected the previous " "offered continuation."),
        )

    # ========================================================
    # RESOLVE USING RECENT CONTEXT
    # ========================================================

    model = create_resolution_model()

    prompt = build_resolution_prompt(
        query=cleaned_query,
        action=action,
        conversation_history=(conversation_history),
    )

    response = model.invoke(prompt)

    response_text = str(response.content).strip()

    return parse_resolution_response(
        response_text=response_text,
        original_query=cleaned_query,
        action=action,
    )


# ============================================================
# DEBUG
# ============================================================


def print_resolution_result(
    result: ConversationResolutionResult,
) -> None:
    """
    Print conversation-resolution diagnostics.
    """

    print()
    print("=" * 70)
    print("CONVERSATION RESOLUTION")
    print("=" * 70)

    print(f"Original query: " f"{result.original_query}")

    print(f"Action: " f"{result.action.value}")

    print(f"Resolved query: " f"{result.resolved_query}")

    print(f"Needs retrieval: " f"{result.needs_retrieval}")

    print(f"Needs clarification: " f"{result.needs_clarification}")

    print(f"Direct response: " f"{result.direct_response}")

    print(f"Was resolved: " f"{result.was_resolved}")

    print(f"Reason: " f"{result.reason}")
