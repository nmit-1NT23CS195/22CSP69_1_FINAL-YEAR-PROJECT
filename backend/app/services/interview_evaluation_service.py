"""
app/services/interview_evaluation_service.py
=============================================
Video-Based Mock Interview — Batched LLM Evaluator (Module D)

DESIGN
------
InterviewEvaluator receives 5 (question, transcript, rubric) triples and
issues ONE single Gemini call that evaluates all 5 answers simultaneously.

Batching avoids:
  - 5× API round-trip overhead
  - 5× token-level cold-start latency
  - Rate-limit pressure from rapid sequential requests

SCHEMA ENFORCEMENT
------------------
The response is validated against `InterviewEvaluationResponse`, a strict
Pydantic V2 model that guarantees exactly 5 AnswerEvaluation objects.
On fallback models the schema is injected into the prompt text and the raw
response is manually parsed.

COMPOSITE SCORING & DIFFICULTY MULTIPLIER
------------------------------------------
Each answer is scored on 4 rubric dimensions (0–10).  A weighted average
produces the per-answer score (0–100).  The five scores are averaged into a
raw composite, then a difficulty multiplier is applied:

  easy   → 0.85×  (max achievable: 85 / 100)
  medium → 1.00×  (standard, no adjustment)
  hard   → 1.15×  (bonus for harder questions, capped at 100)

The final ``adjusted_composite_score`` is what appears in the JSON report.
"""

from __future__ import annotations

import json
import logging
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class AnswerEvaluation(BaseModel):
    """Evaluation for one interview answer."""

    question_id: int = Field(..., description="Index 0-4 matching the input array.")
    technical_correctness: float = Field(
        ...,
        ge=0.0, le=10.0,
        description="Factual accuracy and depth of technical knowledge shown (0-10).",
    )
    relevance: float = Field(
        ...,
        ge=0.0, le=10.0,
        description="How well the answer addresses what was actually asked (0-10).",
    )
    completeness: float = Field(
        ...,
        ge=0.0, le=10.0,
        description=(
            "Whether the answer covers all required sub-parts of the question. "
            "An incomplete answer that trails off should score lower (0-10)."
        ),
    )
    star_structure_score: float = Field(
        ...,
        ge=0.0, le=10.0,
        description=(
            "Adherence to Situation-Task-Action-Result narrative structure. "
            "Factual/technical questions may score 0 here without penalty to overall (0-10)."
        ),
    )
    weighted_score: float = Field(
        ...,
        ge=0.0, le=100.0,
        description="Pre-computed weighted composite score out of 100.",
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="Bullet-point strengths observed in the answer.",
    )
    improvements: list[str] = Field(
        default_factory=list,
        description="Specific, actionable improvement suggestions.",
    )
    feedback: str = Field(
        ...,
        description="2-3 sentence holistic qualitative feedback for the candidate.",
    )

    @field_validator("technical_correctness", "relevance", "completeness", "star_structure_score", mode="before")
    @classmethod
    def _clamp_score(cls, v: Any) -> float:
        try:
            return max(0.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    @field_validator("weighted_score", mode="before")
    @classmethod
    def _clamp_weighted(cls, v: Any) -> float:
        try:
            return max(0.0, min(100.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class InterviewEvaluationResponse(BaseModel):
    """
    Top-level envelope returned by Gemini.
    Must contain exactly 5 AnswerEvaluation objects.
    """
    evaluations: list[AnswerEvaluation] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 answer evaluations, ordered by question_id 0-4.",
    )

    # Populated server-side after LLM parse — not part of the LLM schema
    adjusted_composite_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Raw composite score (mean of weighted_scores) multiplied by the "
            "difficulty multiplier, capped at 100."
        ),
    )
    difficulty_multiplier: float = Field(
        default=1.0,
        description="Multiplier applied to the raw composite: 0.85 / 1.00 / 1.15.",
    )

    @field_validator("evaluations")
    @classmethod
    def _require_five(cls, v: list[AnswerEvaluation]) -> list[AnswerEvaluation]:
        if len(v) != 5:
            raise ValueError(
                f"Expected exactly 5 evaluations, got {len(v)}. "
                "The LLM must evaluate all 5 answers."
            )
        return v


# ---------------------------------------------------------------------------
# Score weight constants
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "technical_correctness": 0.35,
    "relevance":             0.30,
    "completeness":          0.20,
    "star_structure_score":  0.15,
}
assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Difficulty multiplier constants
# ---------------------------------------------------------------------------

_DIFFICULTY_MULTIPLIERS: dict[str, float] = {
    "easy":   0.85,   # Cap: 85 / 100
    "medium": 1.00,   # Standard
    "hard":   1.15,   # Bonus for harder depth, still capped at 100
}
_DEFAULT_MULTIPLIER: float = 1.00


def _get_difficulty_multiplier(difficulties: list[str]) -> float:
    """Return the multiplier for a session based on the *majority* difficulty.

    Strategy: take the most common difficulty value across the 5 questions.
    Ties are broken by selecting the more demanding level (hard > medium > easy).
    Falls back to 1.0 if the list is empty or all values are unrecognised.
    """
    if not difficulties:
        return _DEFAULT_MULTIPLIER

    normalised = [d.strip().lower() for d in difficulties]
    priority   = {"hard": 3, "medium": 2, "easy": 1}

    # Count occurrences, ignore unknown values
    counts: dict[str, int] = {}
    for d in normalised:
        if d in _DIFFICULTY_MULTIPLIERS:
            counts[d] = counts.get(d, 0) + 1

    if not counts:
        return _DEFAULT_MULTIPLIER

    # Pick the difficulty with the highest count; break ties by priority
    dominant = max(counts, key=lambda k: (counts[k], priority.get(k, 0)))
    multiplier = _DIFFICULTY_MULTIPLIERS[dominant]
    logger.debug(
        "[difficulty_multiplier] counts=%s  dominant=%r  multiplier=%.2f",
        counts, dominant, multiplier,
    )
    return multiplier


def _compute_weighted_score(eval_obj: AnswerEvaluation) -> float:
    """Return a 0-100 weighted score for one answer."""
    raw = (
        eval_obj.technical_correctness * _WEIGHTS["technical_correctness"]
        + eval_obj.relevance             * _WEIGHTS["relevance"]
        + eval_obj.completeness          * _WEIGHTS["completeness"]
        + eval_obj.star_structure_score  * _WEIGHTS["star_structure_score"]
    )
    return round(raw * 10, 2)  # scale 0–10 → 0–100


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a world-class Senior Technical Interviewer and Talent Assessment Expert.
Your task is to evaluate a candidate's responses to 5 interview questions from a mock interview session.

EVALUATION PHILOSOPHY:
- Be rigorous but constructive. Identify both strengths and gaps.
- For behavioral questions: assess STAR structure (Situation, Task, Action, Result).
- For technical questions: assess factual accuracy, depth, and practical applicability.
- Base scores ONLY on what the transcript contains — do not reward vague or incomplete answers.
- An answer that trails off, repeats itself, or stays entirely abstract should score lower on completeness and relevance.

SCHEMA COMPLIANCE — NON-NEGOTIABLE:
You MUST return ONLY a valid JSON object matching the provided response schema exactly.
- Output exactly 5 evaluations, one per question, in the same order (question_id 0 through 4).
- All numeric scores are floats in [0.0, 10.0] except weighted_score which is [0.0, 100.0].
- Do NOT include markdown, code fences, or any text outside the JSON object.
"""


def _build_evaluation_prompt(
    questions: list[str],
    transcripts: list[str],
    rubrics: list[str],
    schema_json: str,
) -> str:
    """
    Construct the single batched evaluation prompt.

    Parameters
    ----------
    questions   : 5 interview question strings.
    transcripts : 5 candidate answer transcripts (may be empty strings).
    rubrics     : 5 role/question-specific rubric hints.
    schema_json : JSON Schema string for the response model.
    """
    lines: list[str] = [
        _SYSTEM_PROMPT,
        "",
        "INTERVIEW SESSION — 5 QUESTIONS & ANSWERS:",
        "=" * 60,
    ]
    for i in range(5):
        transcript = (transcripts[i] or "").strip()
        display_transcript = transcript if transcript else "[No audio transcribed — candidate may have been silent]"
        lines += [
            f"\n--- QUESTION {i} (question_id: {i}) ---",
            f"Question: {questions[i]}",
            f"Rubric Hint: {rubrics[i]}",
            f"Candidate Answer Transcript:\n{display_transcript}",
        ]

    lines += [
        "",
        "=" * 60,
        "SCORING INSTRUCTIONS:",
        "- technical_correctness : 0-10 (factual depth and accuracy)",
        "- relevance             : 0-10 (how directly it answers the question)",
        "- completeness          : 0-10 (all parts covered, no trailing off)",
        "- star_structure_score  : 0-10 (STAR adherence; 0 for pure technical Q&A)",
        f"- weighted_score        : pre-compute as (tc*0.35 + rel*0.30 + comp*0.20 + star*0.15) * 10",
        "- strengths             : list of 1-3 bullet strings",
        "- improvements          : list of 1-3 actionable suggestions",
        "- feedback              : 2-3 sentence holistic qualitative paragraph",
        "",
        "REQUIRED JSON SCHEMA:",
        schema_json,
        "",
        "Return ONLY valid JSON matching the schema above. No markdown. No explanation.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# InterviewEvaluator
# ---------------------------------------------------------------------------

class InterviewEvaluator:
    """
    One-shot batched evaluator for a 5-question mock interview session.

    Usage
    -----
        evaluator = InterviewEvaluator()
        result = await evaluator.evaluate(
            questions=[...],
            transcripts=[...],
            rubrics=[...],
        )
    """

    def __init__(self) -> None:
        # Import the existing ResilientGeminiClient from the project's llm_client
        try:
            from app.core.llm_client import get_llm_client  # type: ignore
            self._llm_client = get_llm_client()
            logger.info("[InterviewEvaluator] ResilientGeminiClient loaded.")
        except Exception as exc:
            self._llm_client = None
            logger.warning(
                "[InterviewEvaluator] Could not load ResilientGeminiClient: %s. "
                "Evaluation will return fallback scores.",
                exc,
            )

    # ── Fallback: build a minimal zeroed response when LLM is unavailable ─────

    @staticmethod
    def _build_fallback_response(n: int = 5) -> InterviewEvaluationResponse:
        """Return a zero-scored response with an error note in feedback."""
        evals = [
            AnswerEvaluation(
                question_id=i,
                technical_correctness=0.0,
                relevance=0.0,
                completeness=0.0,
                star_structure_score=0.0,
                weighted_score=0.0,
                strengths=[],
                improvements=["LLM evaluation unavailable — please retry."],
                feedback="Evaluation could not be completed due to an LLM service error.",
            )
            for i in range(n)
        ]
        return InterviewEvaluationResponse(evaluations=evals)

    # ── Core async method ─────────────────────────────────────────────────────

    async def evaluate(
        self,
        questions: list[str],
        transcripts: list[str],
        rubrics: list[str],
        difficulties: list[str] | None = None,
    ) -> InterviewEvaluationResponse:
        """
        Evaluate 5 interview answers with ONE batched LLM call.

        Parameters
        ----------
        questions    : Exactly 5 question strings.
        transcripts  : Exactly 5 transcript strings (empty string if audio failed).
        rubrics      : Exactly 5 rubric/guidance strings for the evaluator.
        difficulties : Optional list of difficulty values per question (e.g.
                       ``["medium", "medium", "hard", "easy", "medium"]``).
                       Used to compute the difficulty multiplier applied to the
                       final composite score.  Defaults to all ``"medium"`` if
                       omitted.

        Returns
        -------
        InterviewEvaluationResponse — always returned, never raises.
        """
        # ── Validate input lengths ────────────────────────────────────────────
        n = 5
        if not (len(questions) == len(transcripts) == len(rubrics) == n):
            logger.error(
                "[InterviewEvaluator] Expected %d items in each list, got q=%d t=%d r=%d",
                n, len(questions), len(transcripts), len(rubrics),
            )
            return self._build_fallback_response(n)

        if self._llm_client is None:
            logger.warning("[InterviewEvaluator] No LLM client — returning fallback.")
            return self._build_fallback_response(n)

        # ── Resolve difficulty multiplier (before any LLM call) ───────────────
        effective_difficulties = difficulties or ["medium"] * n
        multiplier = _get_difficulty_multiplier(effective_difficulties)
        logger.info(
            "[InterviewEvaluator] Difficulty multiplier=%.2f  (difficulties=%s)",
            multiplier, effective_difficulties,
        )

        # ── Build the batched prompt ──────────────────────────────────────────
        schema_json = json.dumps(
            InterviewEvaluationResponse.model_json_schema(), indent=2
        )
        prompt = _build_evaluation_prompt(questions, transcripts, rubrics, schema_json)

        logger.info(
            "[InterviewEvaluator] Sending batched prompt (%d chars) for %d answers.",
            len(prompt), n,
        )

        # ── Fire the LLM via ResilientGeminiClient ───────────────────────────
        # generate_structured_batch returns list[BaseModel] from the `questions`
        # attribute; for our schema the top-level key is `evaluations`, so we
        # call it with the response schema directly and parse the whole envelope.

        try:
            # We use the client's raw async infrastructure but target our schema.
            # Because ResilientGeminiClient.generate_structured_batch expects
            # a schema with a `.questions` attribute, we call the underlying
            # Gemini client directly here for the interview-specific schema.
            import asyncio

            loop = asyncio.get_event_loop()
            raw_text: str = ""

            # Access the underlying google.genai client
            inner_client = getattr(self._llm_client, "_client", None)
            if inner_client is None:
                logger.warning("[InterviewEvaluator] Inner Gemini client is None.")
                return self._build_fallback_response(n)

            try:
                from google.genai import types as _genai_types  # type: ignore

                def _sync_call() -> str:
                    resp = inner_client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=_genai_types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=InterviewEvaluationResponse,
                            temperature=0.3,   # lower temp for consistent scoring
                        ),
                    )
                    return resp.text

                raw_text = await loop.run_in_executor(None, _sync_call)
                logger.info(
                    "[InterviewEvaluator] Raw LLM response received (%d chars).",
                    len(raw_text),
                )

            except Exception as primary_exc:
                logger.warning(
                    "[InterviewEvaluator] Primary model call failed: %s — trying fallback.",
                    primary_exc,
                )
                # Fallback: inject schema into prompt, call without response_schema
                try:
                    def _sync_fallback() -> str:
                        resp = inner_client.models.generate_content(
                            model="gemini-3.1-flash-lite",
                            contents=prompt,
                        )
                        return resp.text

                    raw_text = await loop.run_in_executor(None, _sync_fallback)
                except Exception as fallback_exc:
                    logger.error(
                        "[InterviewEvaluator] Both LLM calls failed: %s",
                        fallback_exc,
                    )
                    return self._build_fallback_response(n)

            # ── Parse the response ────────────────────────────────────────────
            if not raw_text or not raw_text.strip():
                logger.error("[InterviewEvaluator] LLM returned empty response.")
                return self._build_fallback_response(n)

            # Strip markdown fences if present
            import re
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
            clean_text = match.group(1).strip() if match else raw_text.strip()

            parsed = InterviewEvaluationResponse.model_validate_json(clean_text)

            # ── Recompute weighted_score server-side for safety ───────────────
            for ev in parsed.evaluations:
                ev.weighted_score = _compute_weighted_score(ev)

            # ── Apply difficulty multiplier to get adjusted composite ─────────
            raw_composite = mean([ev.weighted_score for ev in parsed.evaluations])
            adjusted      = min(round(raw_composite * multiplier, 2), 100.0)
            parsed.difficulty_multiplier     = multiplier
            parsed.adjusted_composite_score  = adjusted

            logger.info(
                "[InterviewEvaluator] Evaluation complete — raw=%.2f  "
                "multiplier=%.2f  adjusted=%.2f  per-answer=%s",
                raw_composite, multiplier, adjusted,
                [f"{ev.weighted_score:.1f}" for ev in parsed.evaluations],
            )
            return parsed

        except Exception as exc:
            logger.error(
                "[InterviewEvaluator] Unexpected error during evaluation: %s",
                exc, exc_info=True,
            )
            return self._build_fallback_response(n)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_evaluator_singleton: InterviewEvaluator | None = None


def get_interview_evaluator() -> InterviewEvaluator:
    """Return (or lazily create) the process-level InterviewEvaluator singleton."""
    global _evaluator_singleton
    if _evaluator_singleton is None:
        _evaluator_singleton = InterviewEvaluator()
    return _evaluator_singleton
