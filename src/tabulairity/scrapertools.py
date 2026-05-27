import requests
import re

from bs4 import BeautifulSoup
from random import choice, uniform
from time import sleep


# ---------------------------------------------------------------------------
# Soft-error detection
# ---------------------------------------------------------------------------

# Phrases that appear near the top of error/gate pages rather than articles.
# Checked case-insensitively against the first 1500 chars of extracted text.
_SOFT_ERROR_PHRASES = [
    # Bot / JS gates
    "enable javascript",
    "javascript is required",
    "please enable javascript",
    "just a moment",
    "checking your browser",
    "are you human",
    "verify you are human",
    "ddos protection",
    "cloudflare",
    "ray id",
    # Access / auth
    "access denied",
    "403 forbidden",
    "404 not found",
    "page not found",
    "this page does not exist",
    "no longer available",
    "content has been removed",
    # Paywall / subscription
    "subscribe to read",
    "subscription required",
    "subscribers only",
    "sign in to read",
    "register to continue",
    "create a free account",
    "already a subscriber",
    # Rate limiting
    "too many requests",
    "rate limit",
    "slow down",
    # Generic CMS / redirect noise
    "this site uses cookies",
    "we use cookies",
]

# Minimum word count for a page to be considered real article content.
_MIN_WORD_COUNT = 80


def _is_soft_error(text: str) -> tuple[bool, str]:
    """Return (True, reason) if the scraped text looks like an error/gate page."""
    if not text:
        return True, "empty"

    word_count = len(text.split())
    if word_count < _MIN_WORD_COUNT:
        return True, f"too short ({word_count} words)"

    probe = text[:1500].lower()
    for phrase in _SOFT_ERROR_PHRASES:
        if phrase in probe:
            return True, f"soft-error phrase: '{phrase}'"

    return False, ""


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/15.0 Safari/605.1.15",
]


def scrapePageText(url: str, maxLen: int = 100_000) -> str:
    """Fetch a webpage and return clean prose text, or an empty string if the
    page cannot be retrieved or is detected as a soft-error / gate page.

    Returns empty string on failure so callers can use a simple truthiness
    check: ``if not scrapePageText(url): skip``.
    """
    sleep(uniform(0.5, 1.5))

    try:
        response = requests.get(
            url,
            headers={"User-Agent": choice(_USER_AGENTS)},
            timeout=10,
        )
        response.raise_for_status()
        html = response.text

    except requests.exceptions.RequestException:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
            tag.decompose()

        raw_lines = (line.strip() for line in soup.get_text().splitlines())
        prose_lines = [line for line in raw_lines if line and line.count(" ") > 2]

        joined_lines = []
        for line in prose_lines:
            if joined_lines and joined_lines[-1][-1] in ".!?:":
                joined_lines.append(" " + line)
            else:
                joined_lines.append(line)

        text = re.sub(r"\n\s*\n", "\n\n", "\n".join(joined_lines)).strip()
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

    except Exception:
        return ""

    is_error, _ = _is_soft_error(text)
    if is_error:
        return ""

    return text[:maxLen]