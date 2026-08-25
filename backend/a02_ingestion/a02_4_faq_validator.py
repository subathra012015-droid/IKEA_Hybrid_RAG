from __future__ import annotations

MIN_QUESTION_LENGTH = 8
MAX_QUESTION_LENGTH = 250

MAX_ANSWER_LENGTH = 5000

# A short answer may still be perfectly valid.
# Therefore this threshold creates a warning,
# not a validation error.
SHORT_ANSWER_WARNING_LENGTH = 20


GENERIC_QUESTIONS = {
    "how can we help you?",
}


def validate_faq(
    faq: dict,
) -> dict:
    """
    Validate one IKEA FAQ record.

    Validation produces two types of results:

    errors:
        Serious problems that make the FAQ unsuitable
        for the knowledge base.

    warnings:
        Suspicious conditions that should be reviewed
        but do not automatically make the FAQ invalid.

    Returns:
        dict:
            {
                "errors": [...],
                "warnings": [...]
            }
    """

    errors = []
    warnings = []

    faq_id = str(
        faq.get(
            "faq_id",
            "",
        )
    ).strip()

    question = str(
        faq.get(
            "question",
            "",
        )
    ).strip()

    answer = str(
        faq.get(
            "answer",
            "",
        )
    ).strip()

    category = str(
        faq.get(
            "category",
            "",
        )
    ).strip()

    source = str(
        faq.get(
            "source",
            "",
        )
    ).strip()

    # =========================================================
    # FAQ ID VALIDATION
    # =========================================================

    if not faq_id:

        errors.append("missing_faq_id")

    # =========================================================
    # QUESTION VALIDATION
    # =========================================================

    if not question:

        errors.append("missing_question")

    else:

        if len(question) < MIN_QUESTION_LENGTH:

            errors.append("question_too_short")

        if len(question) > MAX_QUESTION_LENGTH:

            errors.append("question_too_long")

        if not question.endswith("?"):

            errors.append("question_missing_question_mark")

        if question.lower() in GENERIC_QUESTIONS:

            errors.append("generic_question")

    # =========================================================
    # ANSWER VALIDATION
    # =========================================================

    if not answer:

        errors.append("missing_answer")

    else:

        # Short answers are allowed.
        # Example:
        # "Not at this time."
        if len(answer) < SHORT_ANSWER_WARNING_LENGTH:

            warnings.append("short_answer")

        if len(answer) > MAX_ANSWER_LENGTH:

            errors.append("answer_too_long")

    # =========================================================
    # METADATA VALIDATION
    # =========================================================

    if not category:

        errors.append("missing_category")

    if not source:

        errors.append("missing_source")

    return {
        "errors": errors,
        "warnings": warnings,
    }


def validate_faqs(
    faqs: list[dict],
) -> dict:
    """
    Validate all extracted IKEA FAQs.

    Also detects duplicate FAQ IDs.

    FAQs containing errors are invalid.

    FAQs containing warnings but no errors
    remain valid.

    Returns:
        dict containing validation results.
    """

    valid_faqs = []
    invalid_faqs = []
    warning_faqs = []

    seen_ids = set()
    duplicate_ids = []

    for faq in faqs:

        faq_id = str(
            faq.get(
                "faq_id",
                "",
            )
        ).strip()

        result = validate_faq(faq)

        errors = result["errors"]

        warnings = result["warnings"]

        # =====================================================
        # DUPLICATE DETECTION
        # =====================================================

        if faq_id in seen_ids:

            errors.append("duplicate_faq_id")

            duplicate_ids.append(faq_id)

        else:

            seen_ids.add(faq_id)

        # =====================================================
        # INVALID FAQ
        # =====================================================

        if errors:

            invalid_faqs.append(
                {
                    "faq": faq,
                    "errors": errors,
                    "warnings": warnings,
                }
            )

            continue

        # =====================================================
        # VALID FAQ
        # =====================================================

        valid_faqs.append(faq)

        # =====================================================
        # VALID BUT HAS WARNING
        # =====================================================

        if warnings:

            warning_faqs.append(
                {
                    "faq": faq,
                    "warnings": warnings,
                }
            )

    return {
        "total_count": len(faqs),
        "valid_count": len(valid_faqs),
        "invalid_count": len(invalid_faqs),
        "warning_count": len(warning_faqs),
        "duplicate_ids": duplicate_ids,
        "valid_faqs": valid_faqs,
        "invalid_faqs": invalid_faqs,
        "warning_faqs": warning_faqs,
    }
