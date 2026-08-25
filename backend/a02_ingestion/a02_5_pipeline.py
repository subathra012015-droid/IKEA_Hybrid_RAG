from backend.a02_ingestion.a02_1_website_loader import (
    load_ikea_faq,
)

from backend.a02_ingestion.a02_2_faq_extractor import (
    extract_faqs,
)

from backend.a02_ingestion.a02_3_snapshot_writer import (
    save_faq_snapshot,
)

from backend.a02_ingestion.a02_4_faq_validator import (
    validate_faqs,
)


def run_ingestion() -> list[dict]:
    """
    Run the complete IKEA FAQ ingestion pipeline.

    Execution order:

        1. Fetch IKEA website
        2. Extract structured FAQ records
        3. Validate FAQ records
        4. Review errors and warnings
        5. Save valid FAQ snapshot
        6. Display valid FAQ preview

    Returns:
        list[dict]:
            Valid structured IKEA FAQ records.
    """

    # =========================================================
    # STEP 1 - FETCH WEBSITE
    # =========================================================

    print()
    print("STEP 1 - FETCH IKEA WEBSITE")

    print("-" * 70)

    website = load_ikea_faq()

    print()
    print("Website fetch completed.")

    print(f"HTTP status: " f"{website['status_code']}")

    print(f"HTML length: " f"{len(website['html'])} characters")

    # =========================================================
    # STEP 2 - EXTRACT FAQ RECORDS
    # =========================================================

    print()
    print("STEP 2 - EXTRACT FAQ RECORDS")

    print("-" * 70)

    faqs = extract_faqs(
        html=website["html"],
        source=website["url"],
    )

    print()
    print("FAQ extraction completed.")

    print(f"FAQs extracted: " f"{len(faqs)}")

    # =========================================================
    # STEP 3 - VALIDATE FAQ RECORDS
    # =========================================================

    print()
    print("STEP 3 - VALIDATE FAQ RECORDS")

    print("-" * 70)

    validation = validate_faqs(faqs)

    valid_faqs = validation["valid_faqs"]

    invalid_faqs = validation["invalid_faqs"]

    warning_faqs = validation["warning_faqs"]

    print()
    print("FAQ validation completed.")

    print(f"Total extracted: " f"{validation['total_count']}")

    print(f"Valid FAQs: " f"{validation['valid_count']}")

    print(f"Invalid FAQs: " f"{validation['invalid_count']}")

    print(f"FAQs with warnings: " f"{validation['warning_count']}")

    print(f"Duplicate IDs: " f"{len(validation['duplicate_ids'])}")

    # =========================================================
    # DISPLAY INVALID RECORDS
    # =========================================================

    if invalid_faqs:

        print()
        print("=" * 70)

        print("INVALID FAQ RECORDS")

        print("=" * 70)

        for index, item in enumerate(
            invalid_faqs,
            start=1,
        ):

            faq = item["faq"]

            errors = item["errors"]

            warnings = item["warnings"]

            print()
            print(f"Invalid FAQ {index}")

            print("-" * 70)

            print(f"Question: " f"{faq.get('question', '')}")

            print(f"Answer: " f"{faq.get('answer', '')}")

            print(f"Errors: " f"{', '.join(errors)}")

            if warnings:

                print(f"Warnings: " f"{', '.join(warnings)}")

    # =========================================================
    # DISPLAY WARNING RECORDS
    # =========================================================

    if warning_faqs:

        print()
        print("=" * 70)

        print("FAQ WARNINGS")

        print("=" * 70)

        for index, item in enumerate(
            warning_faqs,
            start=1,
        ):

            faq = item["faq"]

            warnings = item["warnings"]

            print()
            print(f"Warning FAQ {index}")

            print("-" * 70)

            print(f"Question: " f"{faq.get('question', '')}")

            print(f"Answer: " f"{faq.get('answer', '')}")

            print(f"Warnings: " f"{', '.join(warnings)}")

    # =========================================================
    # STEP 4 - SAVE VALID FAQ SNAPSHOT
    # =========================================================

    print()
    print("STEP 4 - SAVE VALID FAQ SNAPSHOT")

    print("-" * 70)

    snapshot_path = save_faq_snapshot(valid_faqs)

    print()
    print("FAQ snapshot saved.")

    print(f"Snapshot path: " f"{snapshot_path}")

    print(f"FAQs saved: " f"{len(valid_faqs)}")

    # =========================================================
    # STEP 5 - VALID FAQ PREVIEW
    # =========================================================

    print()
    print("=" * 70)

    print("VALID FAQ PREVIEW")

    print("=" * 70)

    for index, faq in enumerate(
        valid_faqs[:10],
        start=1,
    ):

        print()
        print(f"FAQ {index}")

        print("-" * 70)

        print(f"FAQ ID: " f"{faq['faq_id']}")

        print(f"Category: " f"{faq['category']}")

        print(f"Question: " f"{faq['question']}")

        print()
        print("Answer:")

        print(faq["answer"][:1000])

        print()
        print(f"Source: " f"{faq['source']}")

    return valid_faqs
