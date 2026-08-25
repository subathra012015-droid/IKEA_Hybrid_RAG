import requests

from backend.a01_config.a01_1_settings import settings


def load_ikea_faq() -> dict:
    """
    Fetch the configured IKEA FAQ webpage as raw HTML.

    Returns:
        dict:
            url         - IKEA FAQ source URL
            html        - Raw HTML returned by IKEA
            status_code - HTTP response status
    """

    url = settings.ikea_faq_urls

    print("Loading IKEA FAQ website...")
    print(f"URL: {url}")

    headers = {"User-Agent": settings.user_agent}

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return {
        "url": url,
        "html": response.text,
        "status_code": response.status_code,
    }
