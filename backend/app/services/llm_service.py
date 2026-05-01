"""
llm_service.py
==============
Stubbed LLM (Large Language Model) interface for enriched skill extraction.

Purpose:
    Takes noisy parsed resume text and sends it to an external LLM API
    to extract a clean, structured JSON payload mapping each detected skill
    to its estimated years of use.

Design:
    - Provider-agnostic: uses standard HTTP POST via httpx.
    - Prompt template is structured for OpenAI / Google Gemini JSON-mode
      compatibility (system + user message format).
    - Currently returns an empty dict with a log message indicating
      that no LLM API is configured — NO NotImplementedError raised.
    - To activate: set LLM_API_URL and LLM_API_KEY environment variables.

Integration:
    When activated, the output can be used to:
        1. Cross-validate FlashText extraction results.
        2. Provide richer YoE estimates per skill.
        3. Feed into downstream XGBoost/LSTM feature vectors.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------
_LLM_API_URL: Optional[str] = os.environ.get("LLM_API_URL")
_LLM_API_KEY: Optional[str] = os.environ.get("LLM_API_KEY")
_LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
# This prompt is engineered for JSON-mode compatibility with OpenAI and Gemini.
# It instructs the LLM to return ONLY a JSON object — no markdown, no explanation.

_SYSTEM_PROMPT: str = """You are an expert technical recruiter and resume analyzer.
Your task is to extract technical skills from the provided resume text and estimate
how many years the candidate has used each skill.

STRICT RULES:
1. Return ONLY a valid JSON object. No markdown, no code fences, no explanation.
2. The JSON must map each extracted skill to its estimated years of use as a number.
3. If years cannot be determined, use 0.
4. Only include technical/professional skills — exclude soft skills.
5. Normalize skill names to their canonical forms (e.g., "JS" → "JavaScript").
6. Round years to 1 decimal place.

EXAMPLE OUTPUT:
{"Python": 3.0, "React": 2.5, "AWS": 1.0, "Docker": 0.5, "SQL": 4.0}"""

_USER_PROMPT_TEMPLATE: str = """Analyze the following resume text and extract all technical skills with estimated years of experience for each.

RESUME TEXT:
{resume_text}

Return ONLY a JSON object mapping skills to years. No other text."""


# ===========================================================================
# Core LLM Interface
# ===========================================================================

def extract_skills_via_llm(resume_text: str) -> Dict[str, float]:
    """
    Send resume text to an external LLM API for structured skill extraction.

    The LLM is prompted to return a JSON payload in the format:
        {"extracted_skill": estimated_years_used}
    e.g.:
        {"Python": 3.0, "React": 2.5, "AWS": 1.0}

    Graceful degradation: if the LLM API is not configured, or the request
    fails, returns an empty dict without crashing.

    Args:
        resume_text: Raw or parsed resume text to analyze.

    Returns:
        Dict mapping skill names (str) to estimated years of use (float).
        Returns {} if LLM is not configured or the request fails.
    """
    # ------------------------------------------------------------------
    # Guard: check if LLM API is configured
    # ------------------------------------------------------------------
    if not _LLM_API_URL or not _LLM_API_KEY:
        logger.info(
            "LLM service not configured — set LLM_API_URL and LLM_API_KEY "
            "environment variables to enable LLM-based skill extraction. "
            "Returning empty result."
        )
        return {}

    if not resume_text or not resume_text.strip():
        logger.warning("LLM extraction skipped — empty resume text")
        return {}

    # ------------------------------------------------------------------
    # Build the request payload (OpenAI / Gemini compatible format)
    # ------------------------------------------------------------------
    # Truncate very long resumes to stay within token limits
    truncated_text: str = resume_text[:8000]

    # OpenAI-compatible payload structure
    payload: Dict[str, Any] = {
        "model": _LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(resume_text=truncated_text)},
        ],
        "temperature": 0.1,  # Low temperature for deterministic output
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},  # JSON mode (OpenAI/Gemini)
    }

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {_LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    # ------------------------------------------------------------------
    # Send HTTP request to LLM API
    # ------------------------------------------------------------------
    try:
        import httpx

        logger.info("Sending resume to LLM API: %s", _LLM_API_URL)

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                _LLM_API_URL,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()

        # ------------------------------------------------------------------
        # Parse the LLM response
        # ------------------------------------------------------------------
        response_data: Dict[str, Any] = response.json()

        # Extract the assistant's message content
        # OpenAI format: response_data["choices"][0]["message"]["content"]
        # Gemini format may differ — handle both
        content: str = ""

        if "choices" in response_data:
            # OpenAI format
            content = response_data["choices"][0]["message"]["content"]
        elif "candidates" in response_data:
            # Gemini format
            content = response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            logger.warning("Unexpected LLM response format: %s", list(response_data.keys()))
            return {}

        # Parse JSON from the content
        skills_dict: Dict[str, float] = json.loads(content)

        # Validate: ensure all values are numeric
        validated: Dict[str, float] = {}
        for skill, years in skills_dict.items():
            try:
                validated[str(skill)] = round(float(years), 1)
            except (ValueError, TypeError):
                validated[str(skill)] = 0.0

        logger.info(
            "LLM extraction: %d skills extracted from resume",
            len(validated),
        )

        return validated

    except ImportError:
        logger.warning(
            "httpx not installed — LLM service requires httpx. "
            "Install with: pip install httpx"
        )
        return {}

    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM response as JSON: %s", exc)
        return {}

    except Exception as exc:
        logger.error(
            "LLM API request failed: %s. The pipeline will continue "
            "without LLM-enriched skill extraction.",
            exc,
        )
        return {}
