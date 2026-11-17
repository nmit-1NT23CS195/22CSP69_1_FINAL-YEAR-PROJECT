"""
app/core/llm_client.py
======================
Centralized, resilient Gemini LLM orchestration for PlaceBuddy.

ARCHITECTURE
------------
  ResilientGeminiClient
      │
      ├── Primary   model: gemini-2.5-flash      (native structured JSON output)
      ├── Fallback1 model: gemini-3.5-flash-lite  (500 RPD / 15 RPM — manual JSON)
      └── Fallback2 model: gemini-3.1-flash-lite  (500 RPD / 15 RPM — manual JSON)

CALLING STRATEGY PER MODEL
---------------------------
  Primary (gemini-2.5-flash):
    Uses the native SDK `response_schema` parameter so Gemini guarantees
    schema-conformant JSON output without extra parsing.

  Fallbacks (gemini-3.5-flash-lite, gemini-3.1-flash-lite):
    These models do NOT support `response_schema` in v1beta.  Instead:
    1. The JSON schema is appended directly to the prompt as instructions.
    2. The raw text is returned (may include ``` markdown fences).
    3. We strip markdown and parse with json.loads() + model_validate().

RETRY STRATEGY (per model)
--------------------------
  tenacity: wait_exponential(multiplier=2, min=2, max=10), stop_after_attempt(3)
  Catches google.genai.errors.ClientError for 429 / 503.
  On exhausted retries, cascades to the next model.

DESIGN GUARANTEES
-----------------
1. ZERO caller crashes: generate_structured_batch always returns a list.
2. BATCHING: one prompt = N items. batch_size is for logging only.
3. NEW SDK: uses google.genai (not google.generativeai).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Tenacity ──────────────────────────────────────────────────────────────────
try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    _TENACITY_AVAILABLE = False
    logger.warning("tenacity not installed — retry/backoff disabled. Run: pip install tenacity")

# ── google.genai ──────────────────────────────────────────────────────────────
try:
    from google import genai as _genai                # type: ignore
    from google.genai import errors as _genai_errors  # type: ignore
    from google.genai import types as _genai_types    # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    _genai = None         # type: ignore
    _genai_errors = None  # type: ignore
    _genai_types = None   # type: ignore
    logger.warning("google-genai not installed. Run: pip install google-genai")


# ---------------------------------------------------------------------------
# Pydantic schemas — re-exported for use by the router & hydration script
# ---------------------------------------------------------------------------

class MCQItem(BaseModel):
    """Single MCQ question returned by Gemini."""
    question_text:  str
    options:        list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation:    str


class MCQBatch(BaseModel):
    """Top-level envelope that Gemini must return."""
    questions: list[MCQItem]


# ---------------------------------------------------------------------------
# API key resolver
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str | None:
    """Read GEMINI_API_KEY from environment or app/.env file."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(("GEMINI_API_KEY=", "LLM_API_KEY=")):
                    return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rate_or_transient(exc: Exception) -> bool:
    """True if the exception is a 429 / 503 that warrants cascading."""
    if _genai_errors and isinstance(exc, _genai_errors.ClientError):
        msg = str(exc).lower()
        return any(k in msg for k in ("429", "quota", "resource_exhausted", "503", "unavailable"))
    return False


def _strip_markdown_fences(text: str) -> str:
    """
    Remove ```json / ``` markdown code fences that some models wrap output in.
    Returns the inner text trimmed of surrounding whitespace.
    """
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_fallback_response(text: str, response_schema: type[BaseModel]) -> BaseModel:
    """
    Parse raw text from a fallback model:
      1. Strip markdown fences.
      2. json.loads().
      3. model_validate() via Pydantic.
    Raises ValueError / json.JSONDecodeError on failure (caught by caller).
    """
    cleaned = _strip_markdown_fences(text)
    data = json.loads(cleaned)
    return response_schema.model_validate(data)


# ---------------------------------------------------------------------------
# Model cascade
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)

# Primary:   gemini-2.5-flash        — native structured output via response_schema
# Fallback1: gemini-3.5-flash-lite   — 500 RPD / 15 RPM, manual JSON parsing
# Fallback2: gemini-3.1-flash-lite   — 500 RPD / 15 RPM, manual JSON parsing
_MODEL_CASCADE: list[str] = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# The PRIMARY model gets native schema enforcement.
# All others receive prompt-injected schema + manual parsing.
_PRIMARY_MODEL = _MODEL_CASCADE[0]


# ---------------------------------------------------------------------------
# ResilientGeminiClient
# ---------------------------------------------------------------------------

class ResilientGeminiClient:
    """
    Async wrapper around google.genai with:
      - Dynamic per-model calling strategy (native vs. manual JSON)
      - Per-model exponential back-off via tenacity
      - Model cascade on 429 / 503 exhaustion
      - Always returns a list — NEVER raises to the caller
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or _resolve_api_key()
        self._client = None
        if _GENAI_AVAILABLE and self._api_key:
            self._client = _genai.Client(api_key=self._api_key)
        else:
            logger.warning(
                "ResilientGeminiClient: google-genai unavailable or key missing — "
                "all calls will return []."
            )

    # ── Internal: per-model synchronous call ──────────────────────────────────

    def _call_primary_sync(
        self,
        model: str,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> str:
        """
        Call the primary model with native response_schema enforcement.
        Returns raw JSON text (guaranteed schema-conformant by Gemini).
        Raises ClientError on 429 / 503 so tenacity can catch and retry.
        """
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=_genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.7,
            ),
        )
        return response.text

    def _call_fallback_sync(
        self,
        model: str,
        prompt: str,
        response_schema: type[BaseModel],
    ) -> str:
        """
        Call a fallback model WITHOUT response_schema.
        The schema is injected into the prompt as a plain-text instruction.
        Returns raw text that may include markdown fences — caller strips them.
        Raises ClientError on 429 / 503.
        """
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        augmented_prompt = (
            f"{prompt}\n\n"
            f"Return ONLY strictly valid JSON that matches this exact schema "
            f"(no markdown, no explanation):\n{schema_json}"
        )
        response = self._client.models.generate_content(
            model=model,
            contents=augmented_prompt,
            config=_genai_types.GenerateContentConfig(
                temperature=0.7,
            ),
        )
        return response.text

    # ── Internal: tenacity wrapper factory ───────────────────────────────────

    def _build_retrying_caller(self, model: str, is_primary: bool):
        """
        Return a tenacity-wrapped version of the appropriate sync caller.
        Each model in the cascade gets its own fresh retry state.
        """
        raw_fn = self._call_primary_sync if is_primary else self._call_fallback_sync

        if not _TENACITY_AVAILABLE or not _genai_errors:
            # No tenacity — just call directly
            return raw_fn

        @retry(
            retry=retry_if_exception_type(_genai_errors.ClientError),
            wait=wait_exponential(multiplier=2, min=2, max=10),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _wrapped(model_: str, prompt_: str, schema_: type[BaseModel]) -> str:
            return raw_fn(model_, prompt_, schema_)

        return _wrapped

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_structured_batch(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        batch_size: int,
    ) -> list[BaseModel]:
        """
        Generate a structured batch from Gemini.

        Parameters
        ----------
        prompt          : Full prompt asking for ``batch_size`` items.
        response_schema : Pydantic model class the response must conform to.
        batch_size      : Expected item count (logging only).

        Returns
        -------
        list[BaseModel]  — `.questions` list extracted from the parsed schema,
                           or [] on total failure. Never raises.
        """
        if not self._client:
            logger.warning("generate_structured_batch: client unavailable → []")
            return []

        loop = asyncio.get_event_loop()

        for model in _MODEL_CASCADE:
            is_primary = (model == _PRIMARY_MODEL)
            caller = self._build_retrying_caller(model, is_primary)

            try:
                logger.info(
                    "LLM request | model=%s is_primary=%s batch_size=%d",
                    model, is_primary, batch_size,
                )

                raw_text: str = await loop.run_in_executor(
                    None,
                    lambda m=model, ip=is_primary: caller(m, prompt, response_schema),
                )

                # ── Parse response ───────────────────────────────────────────
                if is_primary:
                    # Native schema — model_validate_json is sufficient
                    parsed = response_schema.model_validate_json(raw_text)
                else:
                    # Fallback — strip fences, then json.loads + model_validate
                    parsed = _parse_fallback_response(raw_text, response_schema)

                items = getattr(parsed, "questions", [parsed])
                logger.info(
                    "LLM success | model=%s returned=%d/%d items",
                    model, len(items), batch_size,
                )
                return items

            except Exception as exc:
                # Unwrap RetryError to inspect the actual last exception
                inner = exc
                last_attempt = getattr(exc, "last_attempt", None)
                if last_attempt is not None:
                    try:
                        inner = last_attempt.exception()
                    except Exception:
                        pass

                is_rate = _is_rate_or_transient(inner)
                logger.warning(
                    "LLM failed  | model=%s rate_or_transient=%s error=%s",
                    model, is_rate, exc,
                )
                # Always cascade — whether 404, 429, parse error, etc.
                logger.error(
                    "LLM cascading from model=%s: %s", model, exc, exc_info=False
                )
                continue

        logger.error(
            "generate_structured_batch: all %d models exhausted for batch_size=%d → []",
            len(_MODEL_CASCADE), batch_size,
        )
        return []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_client_singleton: ResilientGeminiClient | None = None


def get_llm_client() -> ResilientGeminiClient:
    """Return (or lazily create) the process-level singleton."""
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = ResilientGeminiClient()
    return _client_singleton
