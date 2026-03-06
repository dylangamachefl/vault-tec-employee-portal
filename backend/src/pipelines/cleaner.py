"""
Stage 2: Text cleaning.

All functions are pure (str → str) and applied in order by clean_text().
The order matters — unicode normalization must happen first so subsequent
regex patterns work on ASCII-safe text.

BUG FIX (Phase 1 refactor): _remove_page_headers_footers now tracks code-fence
state so it never strips lines inside ``` blocks. Previously, short all-caps
lines inside fences (e.g. "3000-BIO-RESET-GAMMA") were treated as page headers
and discarded, leaving empty code blocks in the corpus.
"""

import re
import unicodedata

# Vault-Tec legal boilerplate sentinel (appears at end of many source docs)
_BOILERPLATE_PATTERN = re.compile(
    r"VAULT-TEC CORPORATION accepts no liability.*$",
    re.IGNORECASE | re.DOTALL,
)

# Form-fill placeholders: any line that contains 3+ consecutive underscores, or
# a standalone empty checkbox marker like "[ ]"
_FORM_FILL_LINE_PATTERN = re.compile(
    r"^[^\n]*_{3,}[^\n]*$|^\s*\[\s*\]\s*$",
    re.MULTILINE,
)

# Pure-uppercase header/footer lines (doc title repeats), < 60 chars
_UPPERCASE_HEADER_PATTERN = re.compile(
    r"^[A-Z0-9 \-.,!?:/'\"]{1,59}$",
    re.MULTILINE,
)

# Three or more consecutive newlines
_EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# Code fence opener/closer
_CODE_FENCE_PATTERN = re.compile(r"^```")


# ---------------------------------------------------------------------------
# Unicode normalisation map — smart quotes, em/en dashes, ellipsis, etc.
# ---------------------------------------------------------------------------
_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2018", "'"),  # left single quotation mark
    ("\u2019", "'"),  # right single quotation mark
    ("\u201c", '"'),  # left double quotation mark
    ("\u201d", '"'),  # right double quotation mark
    ("\u2013", "-"),  # en dash
    ("\u2014", "--"),  # em dash
    ("\u2026", "..."),  # horizontal ellipsis
    ("\u00a0", " "),  # non-breaking space
    ("\u2022", "-"),  # bullet
    ("\u2023", "-"),  # triangular bullet
    ("\u25cf", "-"),  # black circle
    ("\ufeff", ""),  # BOM
]


def _normalize_unicode(text: str) -> str:
    """Replace smart quotes, em dashes, etc. with ASCII equivalents."""
    for src, dst in _UNICODE_REPLACEMENTS:
        text = text.replace(src, dst)
    # Decompose remaining accented characters where possible
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def _remove_excess_whitespace(text: str) -> str:
    """Collapse 3+ consecutive blank lines into exactly 2."""
    return _EXCESS_NEWLINES_PATTERN.sub("\n\n", text)


def _remove_page_headers_footers(text: str) -> str:
    """
    Remove lines that are pure uppercase and shorter than 60 chars — these are
    typically document titles repeated as page headers/footers in the PDFs.

    IMPORTANT: Lines inside code fences (``` ... ```) are NEVER stripped,
    even if they match the uppercase heuristic. This prevents stripping
    credential strings, reset codes, and command output from code blocks.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    in_fence = False

    for line in lines:
        stripped = line.strip()

        # Track code fence state FIRST, before any filtering
        if _CODE_FENCE_PATTERN.match(stripped):
            in_fence = not in_fence
            cleaned.append(line)
            continue

        # Keep empty lines (they carry whitespace structure)
        if not stripped:
            cleaned.append(line)
            continue

        # Inside a code fence: always keep the line verbatim
        if in_fence:
            cleaned.append(line)
            continue

        # Outside a fence: drop pure-uppercase short lines (likely headers/footers)
        if stripped == stripped.upper() and len(stripped) < 60 and re.search(r"[A-Z]", stripped):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def _strip_form_fill_fields(text: str) -> str:
    """Remove lines consisting of underscores or empty checkbox markers."""
    return _FORM_FILL_LINE_PATTERN.sub("", text)


def _strip_legal_boilerplate(text: str) -> str:
    """Remove Vault-Tec legal footer beginning with the sentinel phrase."""
    return _BOILERPLATE_PATTERN.sub("", text).rstrip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(raw: str) -> str:
    """
    Apply the five canonical cleaning steps in order:
      1. Unicode normalisation
      2. Remove excess whitespace
      3. Remove page headers/footers (pure-uppercase short lines, NOT inside fences)
      4. Strip form-fill fields (underscores, checkboxes)
      5. Strip Vault-Tec legal boilerplate footer

    Returns: cleaned string
    """
    text = _normalize_unicode(raw)
    text = _remove_excess_whitespace(text)
    text = _remove_page_headers_footers(text)
    text = _strip_form_fill_fields(text)
    text = _strip_legal_boilerplate(text)
    # Final whitespace pass after removal steps may have introduced new gaps
    text = _remove_excess_whitespace(text)
    return text.strip()
