from __future__ import annotations

import re
from hashlib import sha256

from bs4 import BeautifulSoup, Tag


def normalize_text(text: str) -> str:
    """
    Normalize whitespace without changing the meaning
    of the original FAQ text.
    """

    text = text.replace("\xa0", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def build_faq_id(question: str) -> str:
    """
    Create a stable FAQ ID from the normalized question.

    The same normalized question should generate
    the same ID every time.
    """

    normalized_question = normalize_text(question).lower()

    full_hash = sha256(normalized_question.encode("utf-8")).hexdigest()

    return full_hash[:16]


def looks_like_question(text: str) -> bool:
    """
    Determine whether a piece of text looks like
    an FAQ question.
    """

    text = normalize_text(text)

    if not text:
        return False

    # Reject unusually short strings.
    if len(text) < 8:
        return False

    # Reject large containers / paragraphs.
    if len(text) > 250:
        return False

    # IKEA FAQ questions normally end with '?'.
    if not text.endswith("?"):
        return False

    return True


def find_answer_from_container(
    question_element: Tag,
    question: str,
) -> str:
    """
    Find the answer associated with a question.

    IKEA stores FAQ questions and answers inside
    nested HTML containers. We therefore inspect
    parent containers until we find one containing
    both the question and its answer.
    """

    parent = question_element.parent

    for _ in range(5):

        if parent is None:
            break

        if not isinstance(parent, Tag):
            break

        container_text = normalize_text(
            parent.get_text(
                " ",
                strip=True,
            )
        )

        if (
            container_text
            and container_text != question
            and container_text.startswith(question)
        ):

            answer = container_text[len(question) :].strip()

            # Prevent empty or excessively large
            # parent containers from being accepted.
            if 10 <= len(answer) <= 5000:
                return answer

        parent = parent.parent

    return ""


def get_category(
    question_element: Tag,
) -> str:
    """
    Find the nearest previous heading and use it
    as the FAQ category.
    """

    previous_heading = question_element.find_previous(
        [
            "h2",
            "h3",
            "h4",
        ]
    )

    if previous_heading:

        category = normalize_text(
            previous_heading.get_text(
                " ",
                strip=True,
            )
        )

        if category and not looks_like_question(category):
            return category

    return "IKEA FAQ"


def extract_faqs(
    html: str,
    source: str,
) -> list[dict]:
    """
    Extract structured IKEA FAQ records from raw HTML.

    Each record contains:
        faq_id
        question
        answer
        category
        source
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # Remove HTML areas that should never
    # become RAG knowledge.
    for tag in soup.find_all(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
        ]
    ):
        tag.decompose()

    faqs = []

    candidate_elements = soup.find_all(
        [
            "h2",
            "h3",
            "h4",
            "button",
        ]
    )

    for element in candidate_elements:

        question = normalize_text(
            element.get_text(
                " ",
                strip=True,
            )
        )

        if not looks_like_question(question):
            continue

        answer = find_answer_from_container(
            element,
            question,
        )

        if not answer:
            continue

        faq = {
            "faq_id": build_faq_id(question),
            "question": question,
            "answer": answer,
            "category": get_category(element),
            "source": source,
        }

        duplicate = any(existing["faq_id"] == faq["faq_id"] for existing in faqs)

        if not duplicate:
            faqs.append(faq)

    return faqs
