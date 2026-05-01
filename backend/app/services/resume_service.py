"""
resume_service.py
=================
Resume text extraction and structural parsing service.

Capabilities:
    1. Multi-format text extraction: PDF (PyMuPDF), DOCX (python-docx), TXT
    2. OCR fallback: if PyMuPDF extracts empty text from a PDF (scanned image),
       routes through pytesseract for optical character recognition
    3. Section segmentation: parses raw resume text into structured blocks
       (Experience, Education, Projects, Skills, Summary, Other)

All original function signatures are preserved.
"""

import io
import logging
import re
from typing import Dict

import docx
import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR Dependencies — graceful degradation if not installed
# ---------------------------------------------------------------------------
_OCR_AVAILABLE: bool = False

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
    logger.info("pytesseract + Pillow loaded — OCR fallback ENABLED")
except ImportError:
    logger.warning(
        "pytesseract or Pillow not installed — OCR fallback DISABLED. "
        "Scanned image PDFs will return empty text. "
        "Install with: pip install pytesseract Pillow"
    )

# ---------------------------------------------------------------------------
# Section Header Patterns for Resume Parsing
# ---------------------------------------------------------------------------
# These regex patterns match common resume section headers.
# They are case-insensitive and match headers that appear on their own line.
_SECTION_PATTERNS: Dict[str, re.Pattern] = {
    "experience": re.compile(
        r"^\s*(?:work\s+experience|professional\s+experience|employment\s+history|"
        r"experience|work\s+history|career\s+history|professional\s+background)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "education": re.compile(
        r"^\s*(?:education|academic\s+background|qualifications|"
        r"educational\s+background|academic\s+qualifications|degrees?)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "projects": re.compile(
        r"^\s*(?:projects?|personal\s+projects?|academic\s+projects?|"
        r"key\s+projects?|notable\s+projects?|side\s+projects?)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "skills": re.compile(
        r"^\s*(?:skills?|technical\s+skills?|core\s+competenc(?:ies|y)|"
        r"technologies|tech\s+stack|tools?\s*(?:&|and)?\s*technologies|"
        r"programming\s+skills?|software\s+skills?|areas?\s+of\s+expertise)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    "summary": re.compile(
        r"^\s*(?:summary|professional\s+summary|objective|career\s+objective|"
        r"profile|about\s+me|personal\s+statement|overview)\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
}


# ===========================================================================
# Text Extraction (original API preserved)
# ===========================================================================

def extract_text(file_bytes: bytes, filename: str = "") -> str:
    """
    Extract text content from a resume file.

    Supports PDF, DOCX/DOC, and TXT formats.
    For PDFs: if PyMuPDF returns empty text (scanned image PDF), automatically
    falls back to pytesseract OCR if available.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to determine format).

    Returns:
        Extracted text string. Empty string on failure.
    """
    text: str = ""
    filename = filename.lower().strip()

    try:
        if filename.endswith(".docx") or filename.endswith(".doc"):
            # ----------------------------------------------------------
            # Handle Microsoft Word documents
            # ----------------------------------------------------------
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            logger.info("DOCX extraction: %d characters from '%s'", len(text), filename)

        elif filename.endswith(".txt"):
            # ----------------------------------------------------------
            # Handle raw text files
            # ----------------------------------------------------------
            text = file_bytes.decode("utf-8", errors="ignore")
            logger.info("TXT extraction: %d characters from '%s'", len(text), filename)

        else:
            # ----------------------------------------------------------
            # Default to PDF (PyMuPDF)
            # ----------------------------------------------------------
            text = _extract_pdf_text(file_bytes, filename)

    except Exception as e:
        logger.error("File extraction error for '%s': %s", filename, e, exc_info=True)
        return ""

    return text


def _extract_pdf_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from PDF bytes using PyMuPDF.
    Falls back to OCR if the extracted text is empty (scanned image PDF).

    Args:
        file_bytes: Raw PDF file bytes.
        filename:   Filename for logging.

    Returns:
        Extracted text string.
    """
    text: str = ""

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.error("PyMuPDF extraction failed for '%s': %s", filename, e)
        # Don't return yet — try OCR fallback below

    # ------------------------------------------------------------------
    # OCR Fallback: if fitz extracted empty/whitespace-only text,
    # the PDF is likely a scanned image. Route through pytesseract.
    # ------------------------------------------------------------------
    if not text.strip():
        logger.info(
            "PyMuPDF returned empty text for '%s' — attempting OCR fallback",
            filename,
        )
        text = _ocr_extract_text(file_bytes, filename)

    if text.strip():
        logger.info("PDF extraction: %d characters from '%s'", len(text), filename)
    else:
        logger.warning("No text could be extracted from '%s'", filename)

    return text


def _ocr_extract_text(file_bytes: bytes, filename: str) -> str:
    """
    OCR fallback: convert each PDF page to an image and run pytesseract.

    Graceful degradation: if pytesseract or Pillow is not installed,
    or if Tesseract binary is not found, logs a warning and returns "".

    Args:
        file_bytes: Raw PDF file bytes.
        filename:   Filename for logging.

    Returns:
        OCR-extracted text string, or "" on failure.
    """
    if not _OCR_AVAILABLE:
        logger.warning(
            "OCR requested for '%s' but pytesseract/Pillow not installed", filename
        )
        return ""

    extracted_pages: list = []

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                # Render page to a high-resolution pixmap (300 DPI)
                mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI scaling
                pix = page.get_pixmap(matrix=mat)

                # Convert pixmap to PIL Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))

                # Run OCR on the image
                page_text: str = pytesseract.image_to_string(img)
                extracted_pages.append(page_text)

                logger.debug(
                    "OCR page %d/%d: %d characters", page_num + 1, len(doc), len(page_text)
                )

        full_text = "\n".join(extracted_pages)
        logger.info(
            "OCR extraction complete for '%s': %d total characters across %d pages",
            filename, len(full_text), len(extracted_pages),
        )
        return full_text

    except Exception as e:
        logger.error(
            "OCR fallback failed for '%s': %s. "
            "Ensure Tesseract is installed as a system binary.",
            filename, e,
        )
        return ""


# ===========================================================================
# Section Segmentation
# ===========================================================================

def parse_resume_sections(text: str) -> Dict[str, str]:
    """
    Parse resume text into structured sections using NLP heuristics
    and common header detection.

    Identifies and extracts the following sections:
        - experience: Work history, employment details
        - education: Academic background, degrees
        - projects: Personal/academic/professional projects
        - skills: Technical skills, competencies
        - summary: Professional summary, objective
        - other: Everything that doesn't fit the above categories

    Algorithm:
        1. Scan text line by line for section headers using regex patterns.
        2. Record the position and type of each header found.
        3. Slice text between consecutive headers to extract section content.
        4. Any text before the first header goes into "summary" or "other".

    Args:
        text: Raw resume text to parse.

    Returns:
        Dict mapping section names to their text content.
        Always returns at least {"other": text} if no sections are detected.
    """
    if not text or not text.strip():
        return {"other": ""}

    # ------------------------------------------------------------------
    # 1. Find all section header positions
    # ------------------------------------------------------------------
    # Each entry: (start_position, end_position, section_name)
    header_positions: list = []

    for section_name, pattern in _SECTION_PATTERNS.items():
        for match in pattern.finditer(text):
            header_positions.append((match.start(), match.end(), section_name))

    # ------------------------------------------------------------------
    # 2. Sort headers by their position in the text
    # ------------------------------------------------------------------
    header_positions.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # 3. If no headers found, return all text as "other"
    # ------------------------------------------------------------------
    if not header_positions:
        logger.debug("No section headers detected — returning all text as 'other'")
        return {"other": text.strip()}

    # ------------------------------------------------------------------
    # 4. Extract section content between consecutive headers
    # ------------------------------------------------------------------
    sections: Dict[str, str] = {}

    # Text before the first header → "summary" or "other"
    preamble = text[: header_positions[0][0]].strip()
    if preamble:
        sections["summary"] = preamble

    for i, (start, end, section_name) in enumerate(header_positions):
        # Content runs from end of this header to start of the next header
        if i + 1 < len(header_positions):
            content = text[end : header_positions[i + 1][0]]
        else:
            content = text[end:]

        content = content.strip()

        # If the same section appears twice (rare), concatenate
        if section_name in sections:
            sections[section_name] += "\n\n" + content
        else:
            sections[section_name] = content

    # Ensure all standard sections exist (even if empty)
    for section_name in ["experience", "education", "projects", "skills", "summary", "other"]:
        if section_name not in sections:
            sections[section_name] = ""

    logger.info(
        "Section segmentation: %s",
        {k: len(v) for k, v in sections.items() if v},
    )

    return sections