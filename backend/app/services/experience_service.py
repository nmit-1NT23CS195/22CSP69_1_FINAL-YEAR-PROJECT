"""
experience_service.py
=====================
Years of Experience (YoE) extraction service using SpaCy NER and date parsing.

Capabilities:
    1. SpaCy NER to extract ORG (organization) and DATE entities.
    2. Regex patterns to identify date ranges (e.g., "Jan 2020 – Dec 2022").
    3. Date range math to calculate duration of each employment entry.
    4. Aggregation of total estimated Years of Experience.
    5. Programmatic auto-download of en_core_web_sm if not installed.

Graceful degradation: if SpaCy fails to load, returns a zeroed-out
result dict without crashing the pipeline.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpaCy Lazy Loading with Auto-Download
# ---------------------------------------------------------------------------
_nlp: Optional[object] = None
_nlp_load_attempted: bool = False


def _get_spacy_nlp():
    """
    Lazily load the SpaCy en_core_web_sm model.

    If the model is not installed, attempts to download it programmatically
    using spacy.cli.download(). If that also fails, returns None.

    Returns:
        A loaded SpaCy Language model, or None if unavailable.
    """
    global _nlp, _nlp_load_attempted

    if _nlp is not None:
        return _nlp
    if _nlp_load_attempted:
        return None

    _nlp_load_attempted = True

    try:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy en_core_web_sm loaded successfully")
        except OSError:
            # Model not installed — attempt programmatic download
            logger.info("en_core_web_sm not found — downloading...")
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
                logger.info("en_core_web_sm downloaded and loaded successfully")
            except Exception as download_exc:
                logger.error(
                    "Failed to download en_core_web_sm: %s. "
                    "Experience extraction will be disabled.",
                    download_exc,
                )
                return None

        return _nlp

    except ImportError:
        logger.warning(
            "SpaCy not installed — experience extraction DISABLED. "
            "Install with: pip install spacy"
        )
        return None


# ---------------------------------------------------------------------------
# Date Range Regex Patterns
# ---------------------------------------------------------------------------
# Matches patterns like:
#   "Jan 2020 - Dec 2022", "January 2020 – Present", "2019-2022",
#   "06/2020 - 08/2022", "March 2018 to Present"

_MONTH_NAMES = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_DATE_RANGE_PATTERNS: List[re.Pattern] = [
    # "Month Year - Month Year" or "Month Year – Present"
    re.compile(
        rf"({_MONTH_NAMES}\s+\d{{4}})\s*[-–—to]+\s*({_MONTH_NAMES}\s+\d{{4}}|present|current|now|ongoing)",
        re.IGNORECASE,
    ),
    # "MM/YYYY - MM/YYYY" or "MM/YYYY – Present"
    re.compile(
        r"(\d{1,2}/\d{4})\s*[-–—to]+\s*(\d{1,2}/\d{4}|present|current|now|ongoing)",
        re.IGNORECASE,
    ),
    # "YYYY - YYYY" or "YYYY – Present"
    re.compile(
        r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now|ongoing)",
        re.IGNORECASE,
    ),
]

# Month name → month number mapping
_MONTH_MAP: Dict[str, int] = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6,
    "jul": 7, "july": 7, "aug": 8, "august": 8,
    "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


# ===========================================================================
# Date Parsing Utilities
# ===========================================================================

def _parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.

    Handles formats:
        - "Jan 2020", "January 2020"
        - "01/2020"
        - "2020" (assumes January)
        - "present", "current", "now", "ongoing" → today's date

    Args:
        date_str: The date string to parse.

    Returns:
        A datetime object, or None if parsing fails.
    """
    date_str = date_str.strip().lower()

    # Handle "present" / "current" / "now" / "ongoing"
    if date_str in {"present", "current", "now", "ongoing"}:
        return datetime.now()

    # Try "Month Year" format (e.g., "Jan 2020", "January 2020")
    for month_name, month_num in _MONTH_MAP.items():
        if date_str.startswith(month_name):
            # Extract the year
            year_match = re.search(r"\d{4}", date_str)
            if year_match:
                try:
                    return datetime(int(year_match.group()), month_num, 1)
                except ValueError:
                    continue

    # Try "MM/YYYY" format
    mm_yyyy_match = re.match(r"(\d{1,2})/(\d{4})", date_str)
    if mm_yyyy_match:
        try:
            month = int(mm_yyyy_match.group(1))
            year = int(mm_yyyy_match.group(2))
            if 1 <= month <= 12 and 1900 <= year <= 2100:
                return datetime(year, month, 1)
        except ValueError:
            pass

    # Try plain "YYYY" format
    yyyy_match = re.match(r"^(\d{4})$", date_str)
    if yyyy_match:
        try:
            year = int(yyyy_match.group(1))
            if 1900 <= year <= 2100:
                return datetime(year, 1, 1)
        except ValueError:
            pass

    return None


def _calculate_duration_years(start: datetime, end: datetime) -> float:
    """
    Calculate the duration in years between two dates.

    Args:
        start: Start date.
        end:   End date.

    Returns:
        Duration in years (float), minimum 0.0.
    """
    if end < start:
        return 0.0
    delta = end - start
    years = delta.days / 365.25  # Account for leap years
    return round(max(0.0, years), 1)


# ===========================================================================
# Core Experience Extraction
# ===========================================================================

def extract_experience(text: str) -> Dict[str, Any]:
    """
    Extract employment experience information from resume text.

    Uses a combination of:
        1. Regex date-range detection to find employment periods.
        2. SpaCy NER to extract ORG entities near date ranges.
        3. Duration calculation for each entry.
        4. Aggregation of total estimated Years of Experience.

    Note: Date ranges are primary; NER provides organization context.
    Overlapping date ranges are NOT double-counted (we take the max span).

    Args:
        text: Raw resume text (or the "experience" section text).

    Returns:
        Dict containing:
            - "total_yoe": float — estimated total years of experience
            - "experience_entries": list of dicts, each with:
                - "org": str — organization name (if detected, else "Unknown")
                - "start_date": str — parsed start date
                - "end_date": str — parsed end date
                - "duration_years": float — duration of this entry
    """
    if not text or not text.strip():
        return {"total_yoe": 0.0, "experience_entries": []}

    # ------------------------------------------------------------------
    # 1. Extract all date ranges from the text
    # ------------------------------------------------------------------
    date_ranges: List[Tuple[datetime, datetime, str, str]] = []

    for pattern in _DATE_RANGE_PATTERNS:
        for match in pattern.finditer(text):
            start_str = match.group(1)
            end_str = match.group(2)

            start_date = _parse_date_string(start_str)
            end_date = _parse_date_string(end_str)

            if start_date and end_date:
                date_ranges.append((start_date, end_date, start_str, end_str))

    # ------------------------------------------------------------------
    # 2. Use SpaCy NER to find ORG entities near date ranges
    # ------------------------------------------------------------------
    nlp = _get_spacy_nlp()
    org_entities: List[Tuple[str, int]] = []

    if nlp is not None:
        try:
            # Process text with SpaCy (limit doc length for performance)
            doc = nlp(text[:10000])

            for ent in doc.ents:
                if ent.label_ == "ORG":
                    org_entities.append((ent.text, ent.start_char))

        except Exception as exc:
            logger.error("SpaCy NER extraction failed: %s", exc)

    # ------------------------------------------------------------------
    # 3. Build experience entries by pairing date ranges with nearby ORGs
    # ------------------------------------------------------------------
    experience_entries: List[Dict[str, Any]] = []

    for start_date, end_date, start_str, end_str in date_ranges:
        duration = _calculate_duration_years(start_date, end_date)

        # Skip implausibly long durations (>50 years) or zero durations
        if duration > 50.0 or duration <= 0.0:
            continue

        # Find the nearest ORG entity to this date range
        # (within 200 characters before the date range in text)
        org_name = "Unknown"

        # Find the position of the start date string in the text
        date_pos_match = re.search(re.escape(start_str), text, re.IGNORECASE)
        if date_pos_match and org_entities:
            date_pos = date_pos_match.start()
            # Look for ORG entities within 200 chars before the date
            nearest_org = None
            min_distance = float("inf")

            for org_text, org_pos in org_entities:
                distance = date_pos - org_pos
                if 0 <= distance <= 200 and distance < min_distance:
                    min_distance = distance
                    nearest_org = org_text

            if nearest_org:
                org_name = nearest_org

        experience_entries.append({
            "org": org_name,
            "start_date": start_str.strip(),
            "end_date": end_str.strip(),
            "duration_years": duration,
        })

    # ------------------------------------------------------------------
    # 4. Calculate total YoE (merge overlapping ranges to avoid double-counting)
    # ------------------------------------------------------------------
    total_yoe = _calculate_total_yoe(
        [(entry["start_date"], entry["end_date"]) for entry in experience_entries]
        if experience_entries else [],
        date_ranges,
    )

    logger.info(
        "Experience extraction: %d entries, %.1f total YoE",
        len(experience_entries), total_yoe,
    )

    return {
        "total_yoe": total_yoe,
        "experience_entries": experience_entries,
    }


def _calculate_total_yoe(
    date_strings: List[Tuple[str, str]],
    parsed_ranges: List[Tuple[datetime, datetime, str, str]],
) -> float:
    """
    Calculate total YoE by merging overlapping date ranges.

    This prevents double-counting when a candidate held multiple
    positions at the same time (e.g., part-time overlapping roles).

    Args:
        date_strings: List of (start_str, end_str) pairs (unused, kept for compat).
        parsed_ranges: List of (start_date, end_date, start_str, end_str) tuples.

    Returns:
        Float representing total non-overlapping years of experience.
    """
    if not parsed_ranges:
        return 0.0

    # Filter out invalid ranges
    valid_ranges: List[Tuple[datetime, datetime]] = []
    for start, end, _, _ in parsed_ranges:
        duration = _calculate_duration_years(start, end)
        if 0.0 < duration <= 50.0:
            valid_ranges.append((start, end))

    if not valid_ranges:
        return 0.0

    # Sort by start date
    valid_ranges.sort(key=lambda x: x[0])

    # Merge overlapping ranges
    merged: List[Tuple[datetime, datetime]] = [valid_ranges[0]]

    for start, end in valid_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            # Overlapping — extend the current range
            merged[-1] = (last_start, max(last_end, end))
        else:
            # Non-overlapping — start a new range
            merged.append((start, end))

    # Sum up durations of merged ranges
    total: float = sum(
        _calculate_duration_years(start, end) for start, end in merged
    )

    return round(total, 1)
