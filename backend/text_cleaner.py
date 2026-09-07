"""
text_cleaner.py

Small, self-contained text-processing layer that sits between raw
extraction (PyMuPDF / Tesseract) and downstream AI use (search,
Q&A, future summarization).

Responsibilities:
- Clean OCR/text noise per page (extra spaces, blank lines, junk lines).
- Preserve structure that matters: headings, paragraph breaks, numbers.
- Flag pages whose OCR output is too poor to be useful.
- Combine all pages into one clean document string with page markers.

Nothing here touches Flask, storage, or Gemini - it's pure text
processing so it can be reused/unit-tested independently.
"""

import re

# ---------------------------------------
# CONFIG
# ---------------------------------------

# A page is considered "low quality" (likely garbled OCR) if, after
# cleaning, it has fewer than this many characters...
MIN_QUALITY_LENGTH = 15

# ...or if fewer than this fraction of its characters are
# letters/digits (i.e. it's mostly symbols/noise).
MIN_ALNUM_RATIO = 0.35

# Lines made up only of repeated punctuation/symbols (e.g. "-----",
# "......", "======") are almost always scan artifacts, not content.
_NOISE_LINE_RE = re.compile(r"^[\W_]+$", re.UNICODE)

# 3+ blank lines collapse down to a single paragraph break.
_MULTI_BLANK_RE = re.compile(r"\n\s*\n\s*(\n\s*)+")

# Runs of horizontal whitespace collapse to one space.
_MULTI_SPACE_RE = re.compile(r"[ \t]+")

# Stray control characters that sometimes leak in from OCR.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_line(line):
    """Normalize a single line: trim, collapse internal whitespace."""

    line = _CONTROL_CHARS_RE.sub("", line)
    line = _MULTI_SPACE_RE.sub(" ", line)
    return line.strip()


def clean_text(raw_text):
    """
    Clean a block of raw extracted/OCR text while preserving
    paragraph breaks, headings, and line-based structure (numbered
    lists, section titles, etc).

    Returns an empty string if there's nothing usable.
    """

    if not raw_text:
        return ""

    lines = raw_text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = _clean_line(line)

        # Drop pure-noise lines (e.g. "----", "***", OCR garbage
        # made only of symbols) but keep blank lines - they mark
        # paragraph breaks and get collapsed below.
        if line and _NOISE_LINE_RE.match(line):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines down to one blank line
    # (a single paragraph break), and trim outer whitespace.
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


def assess_quality(cleaned_text):
    """
    Decide whether a page's cleaned text is usable.

    Returns (is_low_quality: bool, reason: str | None)
    """

    if not cleaned_text:
        return True, "empty"

    if len(cleaned_text) < MIN_QUALITY_LENGTH:
        return True, "too_short"

    alnum_count = sum(ch.isalnum() for ch in cleaned_text)
    alnum_ratio = alnum_count / len(cleaned_text)

    if alnum_ratio < MIN_ALNUM_RATIO:
        return True, "low_alnum_ratio"

    return False, None


def clean_page(page_number, raw_text):
    """
    Clean a single page's text and attach a quality assessment.

    Returns a dict:
        {
            "page": <int>,
            "cleaned_text": <str>,
            "is_low_quality": <bool>,
            "quality_reason": <str | None>,
            "char_count": <int>
        }
    """

    cleaned = clean_text(raw_text)

    is_low_quality, reason = assess_quality(cleaned)

    return {
        "page": page_number,
        "cleaned_text": cleaned,
        "is_low_quality": is_low_quality,
        "quality_reason": reason,
        "char_count": len(cleaned)
    }


def clean_document_pages(pages):
    """
    Clean every page in a document's `content` list.

    `pages` is the existing list of page dicts, each with at least
    "page" and "text" keys (as produced by /api/upload).

    Returns a list of cleaned-page dicts (see clean_page), in the
    same page order as the input.
    """

    cleaned_pages = []

    for index, page in enumerate(pages, start=1):

        page_number = page.get("page") or index

        raw_text = page.get("text", "")

        cleaned_pages.append(clean_page(page_number, raw_text))

    return cleaned_pages


def build_combined_text(cleaned_pages):
    """
    Join cleaned per-page text into one document-level string,
    with clear page markers so downstream AI steps (summarization,
    Q&A) can still trace an answer back to a page.

    Pages flagged as low quality are still included (marked), so
    the combined text accounts for every page without silently
    dropping content.
    """

    sections = []

    for page in cleaned_pages:

        page_number = page.get("page")

        text = page.get("cleaned_text", "")

        if page.get("is_low_quality"):

            if text:
                body = text
            else:
                body = "[No readable text extracted for this page]"

        else:
            body = text

        sections.append(f"----- Page {page_number} -----\n{body}")

    return "\n\n".join(sections).strip()


def process_document(pages):
    """
    Convenience wrapper: clean all pages and build the combined
    document text in one call.

    Returns (cleaned_pages, combined_text).
    """

    cleaned_pages = clean_document_pages(pages)

    combined_text = build_combined_text(cleaned_pages)

    return cleaned_pages, combined_text