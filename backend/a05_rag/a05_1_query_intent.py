from __future__ import annotations

import re

from dataclasses import dataclass
from enum import Enum

# ============================================================
# BUSINESS INTENTS
# ============================================================


class QueryIntent(str, Enum):
    """
    Primary business intent of a customer query.

    This determines the retrieval strategy.
    """

    FACTUAL = "factual"

    ENUMERATION = "enumeration"

    COMPARISON = "comparison"

    PROCEDURAL = "procedural"

    POLICY = "policy"

    TROUBLESHOOTING = "troubleshooting"

    DEFINITION = "definition"

    RECOMMENDATION = "recommendation"

    CONFIRMATION = "confirmation"

    OUT_OF_SCOPE = "out_of_scope"


# ============================================================
# CONVERSATION MODIFIERS
# ============================================================


class ConversationType(str, Enum):
    """
    Indicates whether the query can stand alone
    or depends on previous conversation context.
    """

    STANDALONE = "standalone"

    FOLLOW_UP = "follow_up"

    CLARIFICATION = "clarification"


# ============================================================
# CLASSIFICATION RESULT
# ============================================================


@dataclass
class QueryIntentResult:
    """
    Complete result of query analysis.

    Attributes:
        query:
            Cleaned user query.

        intent:
            Primary business intent.

        conversation_type:
            Standalone, follow-up, or clarification.

        confidence:
            Rule-based confidence value.

        reason:
            Human-readable explanation.

        requires_parent_expansion:
            Whether all children of a matched parent
            may be needed.

        requires_multiple_targets:
            Whether retrieval should intentionally
            cover multiple entities/topics.

        requires_conversation_context:
            Whether previous conversation is needed.
    """

    query: str

    intent: QueryIntent

    conversation_type: ConversationType

    confidence: float

    reason: str

    requires_parent_expansion: bool

    requires_multiple_targets: bool

    requires_conversation_context: bool


# ============================================================
# CONVERSATION PATTERNS
# ============================================================


FOLLOW_UP_PATTERNS = [
    r"^what about\b",
    r"^how about\b",
    r"^and what about\b",
    r"^what about that\b",
    r"^what about this\b",
    r"^tell me more\b",
    r"^more about\b",
    r"^what else\b",
    r"^and then\b",
    r"^what happens next\b",
]


CLARIFICATION_PATTERNS = [
    r"^what do you mean\b",
    r"^what does that mean\b",
    r"^can you explain that\b",
    r"^explain that\b",
    r"^explain more\b",
    r"^can you clarify\b",
    r"^clarify\b",
]


# ============================================================
# ENUMERATION PATTERNS
# ============================================================


ENUMERATION_PATTERNS = [
    r"\blist all\b",
    r"\bshow all\b",
    r"\ball available\b",
    r"\ball options\b",
    r"\bavailable options\b",
    r"\bwhat options\b",
    r"\bwhich options\b",
    r"\bwhat services\b",
    r"\bwhich services\b",
    r"\bwhat kinds? of\b",
    r"\bwhat types? of\b",
    r"\bwhat .* are available\b",
    r"\bwhat .* do you have\b",
    r"\bwhat can .* offer\b",
]


# ============================================================
# COMPARISON PATTERNS
# ============================================================


COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bdifferences between\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bwhich is better\b",
    r"\bwhich one is better\b",
    r"\bbetter option\b",
]


# ============================================================
# PROCEDURAL PATTERNS
# ============================================================


PROCEDURAL_PATTERNS = [
    r"^how do i\b",
    r"^how can i\b",
    r"^how should i\b",
    r"^what steps\b",
    r"\bsteps to\b",
    r"\bprocess for\b",
    r"\bprocedure for\b",
    r"\bhow to\b",
]


# ============================================================
# POLICY PATTERNS
# ============================================================


POLICY_PATTERNS = [
    r"\bpolicy\b",
    r"\breturn policy\b",
    r"\brefund policy\b",
    r"\bexchange policy\b",
    r"\bcancellation policy\b",
    r"\bterms and conditions\b",
    r"\bcan i reload\b",
    r"\bcan i change the value\b",
    r"\bcan .* be returned\b",
    r"\bcan .* be refunded\b",
]


# ============================================================
# TROUBLESHOOTING PATTERNS
# ============================================================


TROUBLESHOOTING_PATTERNS = [
    r"\bnot working\b",
    r"\bdoesn't work\b",
    r"\bdoes not work\b",
    r"\bmissing\b",
    r"\bdamaged\b",
    r"\bbroken\b",
    r"\bdidn't receive\b",
    r"\bdid not receive\b",
    r"\bmarked as delivered\b",
    r"\bproblem\b",
    r"\bissue\b",
    r"\berror\b",
    r"\bwrong item\b",
]


# ============================================================
# DEFINITION PATTERNS
# ============================================================


DEFINITION_PATTERNS = [
    r"^what is\b",
    r"^what are\b",
    r"^what does .* mean\b",
    r"^define\b",
    r"^explain\b",
]


# ============================================================
# RECOMMENDATION PATTERNS
# ============================================================


RECOMMENDATION_PATTERNS = [
    r"\bwhat should i choose\b",
    r"\bwhich should i choose\b",
    r"\bwhich option should i\b",
    r"\bwhich is best for me\b",
    r"\bwhat is best for me\b",
    r"\brecommend\b",
    r"\brecommendation\b",
    r"\bsuggest\b",
]


# ============================================================
# CONFIRMATION PATTERNS
# ============================================================


CONFIRMATION_PATTERNS = [
    r"^so .* right\??$",
    r"^so .* correct\??$",
    r"^does that mean\b",
    r"^am i correct\b",
    r"^is that correct\b",
    r"^just to confirm\b",
    r"^to confirm\b",
]


# ============================================================
# OUT-OF-SCOPE SIGNALS
# ============================================================


OUT_OF_SCOPE_PATTERNS = [
    r"\bweather\b",
    r"\bstock market\b",
    r"\bfootball\b",
    r"\bcricket\b",
    r"\bpolitics\b",
    r"\bwrite python code\b",
    r"\btranslate\b",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================


def normalize_query(
    query: str,
) -> str:
    """
    Normalize whitespace while preserving meaning.
    """

    cleaned_query = query.strip()

    cleaned_query = re.sub(
        r"\s+",
        " ",
        cleaned_query,
    )

    return cleaned_query


# ============================================================
# PATTERN MATCHING
# ============================================================


def matches_any_pattern(
    query: str,
    patterns: list[str],
) -> bool:
    """
    Check whether query matches any supplied regex pattern.
    """

    lowered_query = query.lower()

    for pattern in patterns:

        if re.search(
            pattern,
            lowered_query,
        ):
            return True

    return False


# ============================================================
# CONVERSATION TYPE
# ============================================================


def classify_conversation_type(
    query: str,
) -> ConversationType:
    """
    Determine whether query is standalone,
    a follow-up, or clarification.
    """

    if matches_any_pattern(
        query=query,
        patterns=CLARIFICATION_PATTERNS,
    ):
        return ConversationType.CLARIFICATION

    if matches_any_pattern(
        query=query,
        patterns=FOLLOW_UP_PATTERNS,
    ):
        return ConversationType.FOLLOW_UP

    return ConversationType.STANDALONE


# ============================================================
# PRIMARY INTENT
# ============================================================


def classify_primary_intent(
    query: str,
) -> tuple[
    QueryIntent,
    float,
    str,
]:
    """
    Determine the primary business intent.

    Returns:
        intent
        confidence
        reason
    """

    # --------------------------------------------------------
    # OUT OF SCOPE
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=OUT_OF_SCOPE_PATTERNS,
    ):

        return (
            QueryIntent.OUT_OF_SCOPE,
            0.95,
            ("Query appears unrelated to " "the IKEA FAQ knowledge domain."),
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=COMPARISON_PATTERNS,
    ):

        return (
            QueryIntent.COMPARISON,
            0.95,
            ("Query explicitly requests comparison " "between choices or concepts."),
        )

    # --------------------------------------------------------
    # ENUMERATION
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=ENUMERATION_PATTERNS,
    ):

        return (
            QueryIntent.ENUMERATION,
            0.95,
            ("Query asks for multiple or all " "available options."),
        )

    # --------------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=RECOMMENDATION_PATTERNS,
    ):

        return (
            QueryIntent.RECOMMENDATION,
            0.90,
            ("Query asks for guidance in choosing " "between available options."),
        )

    # --------------------------------------------------------
    # TROUBLESHOOTING
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=TROUBLESHOOTING_PATTERNS,
    ):

        return (
            QueryIntent.TROUBLESHOOTING,
            0.90,
            ("Query describes a problem or issue " "that needs resolution."),
        )

    # --------------------------------------------------------
    # POLICY
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=POLICY_PATTERNS,
    ):

        return (
            QueryIntent.POLICY,
            0.90,
            ("Query asks about an IKEA rule, " "policy, restriction, or condition."),
        )

    # --------------------------------------------------------
    # PROCEDURAL
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=PROCEDURAL_PATTERNS,
    ):

        return (
            QueryIntent.PROCEDURAL,
            0.90,
            ("Query asks how to perform " "an action or process."),
        )

    # --------------------------------------------------------
    # CONFIRMATION
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=CONFIRMATION_PATTERNS,
    ):

        return (
            QueryIntent.CONFIRMATION,
            0.90,
            ("Query asks to confirm previously " "understood information."),
        )

    # --------------------------------------------------------
    # DEFINITION
    # --------------------------------------------------------

    if matches_any_pattern(
        query=query,
        patterns=DEFINITION_PATTERNS,
    ):

        return (
            QueryIntent.DEFINITION,
            0.85,
            ("Query asks for a definition " "or explanation."),
        )

    # --------------------------------------------------------
    # FACTUAL FALLBACK
    # --------------------------------------------------------

    return (
        QueryIntent.FACTUAL,
        0.80,
        (
            "Query appears to request a specific fact "
            "and does not strongly match another intent."
        ),
    )


# ============================================================
# FULL CLASSIFICATION
# ============================================================


def classify_query_intent(
    query: str,
) -> QueryIntentResult:
    """
    Run complete query analysis.

    Determines:
        conversation dependency
        primary business intent
        retrieval requirements
    """

    cleaned_query = normalize_query(query)

    if not cleaned_query:

        raise ValueError("Query cannot be empty.")

    conversation_type = classify_conversation_type(cleaned_query)

    intent, confidence, reason = classify_primary_intent(cleaned_query)

    # ========================================================
    # RETRIEVAL FLAGS
    # ========================================================

    requires_parent_expansion = intent == QueryIntent.ENUMERATION

    requires_multiple_targets = intent in {
        QueryIntent.COMPARISON,
        QueryIntent.RECOMMENDATION,
    }

    requires_conversation_context = (
        conversation_type
        in {
            ConversationType.FOLLOW_UP,
            ConversationType.CLARIFICATION,
        }
        or intent == QueryIntent.CONFIRMATION
    )

    return QueryIntentResult(
        query=cleaned_query,
        intent=intent,
        conversation_type=(conversation_type),
        confidence=confidence,
        reason=reason,
        requires_parent_expansion=(requires_parent_expansion),
        requires_multiple_targets=(requires_multiple_targets),
        requires_conversation_context=(requires_conversation_context),
    )


# ============================================================
# DEBUG DISPLAY
# ============================================================


def print_intent_result(
    result: QueryIntentResult,
) -> None:
    """
    Print intent-classification diagnostics.
    """

    print()
    print("=" * 70)
    print("QUERY INTENT")
    print("=" * 70)

    print(f"Query: " f"{result.query}")

    print(f"Intent: " f"{result.intent.value}")

    print(f"Conversation type: " f"{result.conversation_type.value}")

    print(f"Confidence: " f"{result.confidence}")

    print(f"Parent expansion: " f"{result.requires_parent_expansion}")

    print(f"Multiple targets: " f"{result.requires_multiple_targets}")

    print(f"Conversation context: " f"{result.requires_conversation_context}")

    print(f"Reason: " f"{result.reason}")


# ============================================================
# TEST
# ============================================================


def intent_test() -> None:
    """
    Test representative IKEA queries.

    No OpenAI API call occurs.
    """

    test_queries = [
        "How long does kitchen installation take?",
        "What kitchen services are available?",
        "Compare online and in-store kitchen planning.",
        "How do I return an item?",
        "What is the return policy?",
        ("My delivery is marked as delivered " "but I didn't receive it."),
        "What is Collect Near You?",
        ("Which kitchen planning option " "should I choose?"),
        ("So I can return the product, " "is that correct?"),
        "What about installation?",
        "What do you mean by kitchen validation?",
        "What is the weather today?",
    ]

    print("=" * 70)
    print("QUERY INTENT TEST")
    print("=" * 70)

    for query in test_queries:

        result = classify_query_intent(query)

        print()
        print(f"Query: {query}")

        print(f"Intent: " f"{result.intent.value}")

        print(f"Conversation: " f"{result.conversation_type.value}")

        print(f"Expand parent: " f"{result.requires_parent_expansion}")

        print(f"Multiple targets: " f"{result.requires_multiple_targets}")

        print(f"Needs history: " f"{result.requires_conversation_context}")
