import json

from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_DIRECTORY = Path("data/snapshots")

SNAPSHOT_FILE = SNAPSHOT_DIRECTORY / "ikea_faq_snapshot.json"


def save_faq_snapshot(
    faqs: list[dict],
) -> Path:
    """
    Save the current structured IKEA FAQs
    into a JSON snapshot.

    The snapshot will later be used for
    Level-2 change detection.

    Args:
        faqs:
            List of structured FAQ records.

    Returns:
        Path:
            Location of the saved JSON file.
    """

    SNAPSHOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_at = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "saved_at": saved_at,
        "faq_count": len(faqs),
        "faqs": faqs,
    }

    with SNAPSHOT_FILE.open(
        mode="w",
        encoding="utf-8",
    ) as file:

        json.dump(
            snapshot,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return SNAPSHOT_FILE
