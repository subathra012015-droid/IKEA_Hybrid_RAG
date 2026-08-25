from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SNAPSHOT_PATH = Path("data/snapshots/ikea_faq_snapshot.json")


def load_faq_snapshot(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> list[dict]:
    """
    Load previously validated IKEA FAQ records
    from the local snapshot.

    Normal chatbot startup uses this file instead
    of fetching the IKEA website again.

    Supported snapshot structures:

        [
            {...},
            {...}
        ]

    or:

        {
            "faqs": [...]
        }

    or:

        {
            "records": [...]
        }

    Returns:
        list[dict]:
            Valid IKEA FAQ records.
    """

    path = Path(snapshot_path)

    if not path.exists():

        raise FileNotFoundError(
            f"IKEA FAQ snapshot was not found: {path}\n"
            "Run 'python update_knowledge.py' first."
        )

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            snapshot_data = json.load(file)

    except json.JSONDecodeError as error:

        raise ValueError(f"Invalid JSON in IKEA FAQ snapshot: {path}") from error

    # =========================================================
    # SNAPSHOT IS DIRECTLY A LIST
    # =========================================================

    if isinstance(
        snapshot_data,
        list,
    ):

        faq_records = snapshot_data

    # =========================================================
    # SNAPSHOT IS A DICTIONARY
    # =========================================================

    elif isinstance(
        snapshot_data,
        dict,
    ):

        possible_keys = [
            "faqs",
            "records",
            "data",
        ]

        faq_records = None

        for key in possible_keys:

            value = snapshot_data.get(key)

            if isinstance(
                value,
                list,
            ):

                faq_records = value

                break

        if faq_records is None:

            raise ValueError(
                "Snapshot JSON is a dictionary, "
                "but no FAQ list was found under "
                "'faqs', 'records', or 'data'."
            )

    else:

        raise ValueError("Unsupported IKEA FAQ snapshot structure.")

    # =========================================================
    # BASIC RECORD VALIDATION
    # =========================================================

    valid_records = []

    for record in faq_records:

        if not isinstance(
            record,
            dict,
        ):

            continue

        question = str(
            record.get(
                "question",
                "",
            )
        ).strip()

        answer = str(
            record.get(
                "answer",
                "",
            )
        ).strip()

        faq_id = str(
            record.get(
                "faq_id",
                "",
            )
        ).strip()

        if not question:
            continue

        if not answer:
            continue

        if not faq_id:
            continue

        valid_records.append(record)

    if not valid_records:

        raise ValueError("The IKEA FAQ snapshot contains " "no usable FAQ records.")

    return valid_records


def print_snapshot_summary(
    faq_records: list[dict],
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> None:
    """
    Print local snapshot diagnostics.
    """

    print()
    print("=" * 70)
    print("LOAD LOCAL IKEA FAQ SNAPSHOT")
    print("=" * 70)

    print(f"Snapshot: {snapshot_path}")

    print(f"FAQ records loaded: " f"{len(faq_records)}")

    print("Website fetch performed: No")
