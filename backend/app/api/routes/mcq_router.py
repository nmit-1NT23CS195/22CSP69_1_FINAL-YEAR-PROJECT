"""
mcq_router.py
=============
POST /api/mcq/generate — Self-Populating AI Cache MCQ Engine

FLOW OVERVIEW
─────────────
1. Accept skill lists from the frontend (derived from ATS Phase 1 output).
2. Determine the target 10-question distribution:
       2 × Green  (verified competencies — consolidation questions)
       4 × Yellow (unverified skills     — probe questions)
       4 × Red    (missing/gap skills    — challenge questions)
3. For each (skill, tier) slot, check the PostgreSQL cache first.
4. If cache is insufficient, call Gemini (gemini-2.0-flash) with a Pydantic
   structured-output schema to generate exactly the missing questions.
5. Bulk-insert all newly generated questions into the DB (permanent cache).
6. Shuffle the final 10 questions and return them to the frontend.

ARCHITECTURAL GUARANTEES
────────────────────────
• ZERO REGRESSIONS: this router is completely independent — it does not import
  from or modify ats_service.py, llm_service.py, or any existing route.
• If Gemini is unavailable or the DB is empty, the endpoint falls back
  gracefully using a curated static set rather than returning a 500 error.
• Idempotent cache writes — the same question is never stored twice because
  we only insert what was freshly generated (not already in the DB).
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Question

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Target quiz composition
# ---------------------------------------------------------------------------
_TARGET = {
    "Green":  2,   # Verified competencies
    "Yellow": 4,   # Unverified / stated-only skills
    "Red":    4,   # Missing / critical gaps
}
_TOTAL_QUESTIONS = sum(_TARGET.values())   # 10

# ---------------------------------------------------------------------------
# Helper — resolve Gemini API key (same strategy as llm_service.py)
# ---------------------------------------------------------------------------
def _get_gemini_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if key:
        return key
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") or line.startswith("LLM_API_KEY="):
                    return line.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# Pydantic schemas for request / response
# ---------------------------------------------------------------------------

class QuizRequest(BaseModel):
    """
    Payload from the frontend.  All three lists come directly from the
    cognitive_analysis section of the /ats/score/core response.
    """
    verified_skills:   list[str] = Field(default_factory=list, description="Skills the candidate has demonstrated (Green tier)")
    unverified_skills: list[str] = Field(default_factory=list, description="Skills listed but not evidenced (Yellow tier)")
    missing_skills:    list[str] = Field(default_factory=list, description="Skills absent from resume (Red tier)")

    @field_validator("verified_skills", "unverified_skills", "missing_skills", mode="before")
    @classmethod
    def _normalise(cls, v: list[str]) -> list[str]:
        """Lowercase + strip whitespace for consistent DB lookups."""
        return [s.strip().lower() for s in v if s and s.strip()]


class QuizQuestion(BaseModel):
    """Single question returned to the frontend."""
    skill:          str
    forensic_tier:  str
    question:       str
    options:        list[str]
    correct_answer: str
    explanation:    str


class QuizResponse(BaseModel):
    """Envelope returned by the endpoint."""
    total:     int
    questions: list[QuizQuestion]
    source:    str  # "cache" | "ai_generated" | "mixed" | "fallback"


# ---------------------------------------------------------------------------
# Pydantic schema for Gemini Structured Output
# ---------------------------------------------------------------------------

class _GeminiQuestionItem(BaseModel):
    """One question as returned by Gemini."""
    question_text:  str
    options:        list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation:    str


class _GeminiQuizOutput(BaseModel):
    """Root schema enforced on Gemini's response."""
    questions: list[_GeminiQuestionItem]


# ---------------------------------------------------------------------------
# Static fallback bank (used when Gemini is unavailable AND cache is empty)
# ---------------------------------------------------------------------------
_STATIC_FALLBACK: list[dict[str, Any]] = [
    {
        "skill": "data structures", "forensic_tier": "Green",
        "question_text": "What is the time complexity of searching in a balanced BST?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "correct_answer": "O(log n)",
        "explanation": "Each comparison in a balanced BST halves the search space, giving O(log n)."
    },
    {
        "skill": "algorithms", "forensic_tier": "Green",
        "question_text": "Which algorithm guarantees O(n log n) worst-case sorting time?",
        "options": ["Quick Sort", "Bubble Sort", "Merge Sort", "Insertion Sort"],
        "correct_answer": "Merge Sort",
        "explanation": "Merge Sort's divide-and-conquer always splits and merges in O(n log n) regardless of input."
    },
    {
        "skill": "operating systems", "forensic_tier": "Yellow",
        "question_text": "What is a zombie process in Unix/Linux?",
        "options": [
            "A process consuming 100% CPU",
            "A child that exited but hasn't been wait()ed by the parent",
            "A background daemon with no controlling terminal",
            "A process in uninterruptible sleep"
        ],
        "correct_answer": "A child that exited but hasn't been wait()ed by the parent",
        "explanation": "After exit(), the child's PCB remains until the parent calls wait() to collect the exit status."
    },
    {
        "skill": "networking", "forensic_tier": "Yellow",
        "question_text": "What does ARP resolve?",
        "options": [
            "Domain names to IP addresses",
            "IP addresses to MAC addresses on a local network",
            "MAC addresses to port numbers",
            "IP addresses to domain names"
        ],
        "correct_answer": "IP addresses to MAC addresses on a local network",
        "explanation": "ARP broadcasts 'Who has IP X?' on the LAN. The host responds with its MAC address."
    },
    {
        "skill": "databases", "forensic_tier": "Yellow",
        "question_text": "What does ACID stand for in database transactions?",
        "options": [
            "Atomicity, Consistency, Isolation, Durability",
            "Availability, Consistency, Isolation, Durability",
            "Atomicity, Concurrency, Integrity, Durability",
            "Atomicity, Consistency, Integrity, Distribution"
        ],
        "correct_answer": "Atomicity, Consistency, Isolation, Durability",
        "explanation": "ACID properties guarantee reliable database transactions even in the face of errors and concurrent access."
    },
    {
        "skill": "system design", "forensic_tier": "Yellow",
        "question_text": "What is the primary purpose of a Load Balancer?",
        "options": [
            "Encrypt traffic between client and server",
            "Distribute incoming requests across multiple backend servers",
            "Cache static assets closer to users",
            "Monitor server health metrics"
        ],
        "correct_answer": "Distribute incoming requests across multiple backend servers",
        "explanation": "A load balancer prevents any single server from becoming a bottleneck by spreading traffic evenly."
    },
    {
        "skill": "docker", "forensic_tier": "Red",
        "question_text": "What is the difference between a Docker image and a container?",
        "options": [
            "An image is running; a container is stored",
            "An image is a read-only template; a container is a running instance of it",
            "They are identical",
            "A container is a collection of images"
        ],
        "correct_answer": "An image is a read-only template; a container is a running instance of it",
        "explanation": "Images are immutable blueprints. Containers are live, writable instances created from images."
    },
    {
        "skill": "kubernetes", "forensic_tier": "Red",
        "question_text": "What Kubernetes resource manages stateless application deployments?",
        "options": ["StatefulSet", "DaemonSet", "Deployment", "CronJob"],
        "correct_answer": "Deployment",
        "explanation": "A Deployment manages replica sets for stateless pods, handling rolling updates and rollbacks."
    },
    {
        "skill": "system design", "forensic_tier": "Red",
        "question_text": "In the CAP theorem, which two properties does a partition-tolerant system trade off between?",
        "options": [
            "Consistency and Availability",
            "Availability and Partition Tolerance",
            "Consistency and Partition Tolerance",
            "Scalability and Durability"
        ],
        "correct_answer": "Consistency and Availability",
        "explanation": "When a network partition occurs, a system must choose between staying consistent (rejecting writes) or staying available (allowing stale reads)."
    },
    {
        "skill": "security", "forensic_tier": "Red",
        "question_text": "What attack does input sanitisation primarily prevent?",
        "options": [
            "Man-in-the-Middle (MitM)",
            "Distributed Denial of Service (DDoS)",
            "SQL Injection and Cross-Site Scripting (XSS)",
            "Brute Force"
        ],
        "correct_answer": "SQL Injection and Cross-Site Scripting (XSS)",
        "explanation": "Sanitising user input removes or escapes special characters before they reach the database or HTML renderer, preventing injection attacks."
    }
]

# ---------------------------------------------------------------------------
# Core DB helpers
# ---------------------------------------------------------------------------

def _fetch_cached(db: Session, skills: list[str], tier: str, limit: int) -> list[Question]:
    """Return up to `limit` cached questions for any of the given skills at a tier."""
    if not skills:
        return []
    rows = (
        db.query(Question)
        .filter(Question.skill.in_(skills), Question.forensic_tier == tier)
        .order_by(Question.id)
        .limit(limit)
        .all()
    )
    return rows


def _bulk_insert(db: Session, items: list[dict[str, Any]]) -> None:
    """Insert a list of question dicts into the DB and commit."""
    if not items:
        return
    db.bulk_insert_mappings(Question, items)  # type: ignore[arg-type]
    db.commit()


# ---------------------------------------------------------------------------
# Gemini generation helper
# ---------------------------------------------------------------------------

async def _generate_with_gemini(
    skill: str,
    tier: str,
    count: int,
) -> list[dict[str, Any]]:
    """
    Call Gemini with a structured-output schema to generate `count` questions
    for `skill` at `tier` difficulty.

    Returns a list of question dicts ready for DB insertion.
    """
    if count <= 0:
        return []

    api_key = _get_gemini_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY not found — skipping AI generation for skill=%s", skill)
        return []

    tier_descriptors = {
        "Green":  "consolidation-level questions that reinforce a solid understanding",
        "Yellow": "intermediate-level probe questions that expose shallow understanding",
        "Red":    "advanced challenge questions on concepts the candidate likely lacks entirely",
    }

    prompt = f"""Generate exactly {count} technical MCQ question(s) about "{skill}".
These should be {tier_descriptors.get(tier, "technical")} questions.

Requirements:
- Each question must have exactly 4 answer options.
- The correct_answer must be an exact copy of one of the 4 options.
- The explanation must be 1-3 sentences, technically accurate, and educational.
- Questions must be distinct — no two questions should test the same sub-concept.
- Do NOT include question numbers or labels in the question_text.

Return ONLY valid JSON matching this schema:
{{
  "questions": [
    {{
      "question_text": "<full question string>",
      "options": ["<option1>", "<option2>", "<option3>", "<option4>"],
      "correct_answer": "<must exactly match one option>",
      "explanation": "<forensic explanation>"
    }}
  ]
}}"""

    try:
        # Use the new google.genai SDK (google-genai package)
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_GeminiQuizOutput,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )

        # Parse structured output
        raw_text = response.text or ""
        parsed: _GeminiQuizOutput = _GeminiQuizOutput.model_validate_json(raw_text)

        results = []
        for item in parsed.questions[:count]:
            # Guard: correct_answer must be in options
            if item.correct_answer not in item.options:
                logger.warning(
                    "Gemini returned mismatched correct_answer for skill=%s — skipping", skill
                )
                continue
            results.append({
                "skill":          skill,
                "forensic_tier":  tier,
                "question_text":  item.question_text.strip(),
                "options":        item.options,
                "correct_answer": item.correct_answer,
                "explanation":    item.explanation.strip(),
            })

        logger.info("Gemini generated %d/%d questions for skill=%s tier=%s", len(results), count, skill, tier)
        return results

    except Exception as exc:
        logger.error("Gemini generation failed for skill=%s tier=%s: %s", skill, tier, exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Question serialiser (ORM → dict for response)
# ---------------------------------------------------------------------------

def _to_quiz_question(q: Question | dict[str, Any]) -> QuizQuestion:
    if isinstance(q, Question):
        return QuizQuestion(
            skill=q.skill,
            forensic_tier=q.forensic_tier,
            question=q.question_text,
            options=q.options,
            correct_answer=q.correct_answer,
            explanation=q.explanation,
        )
    # From static fallback dict
    return QuizQuestion(
        skill=q["skill"],
        forensic_tier=q["forensic_tier"],
        question=q["question_text"],
        options=q["options"],
        correct_answer=q["correct_answer"],
        explanation=q["explanation"],
    )


# ---------------------------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=QuizResponse,
    summary="Generate a 10-question skill-adaptive MCQ quiz",
    description=(
        "Accepts verified, unverified, and missing skill lists from the ATS analysis. "
        "Returns a personalised 10-question quiz (2 Green + 4 Yellow + 4 Red). "
        "Questions are fetched from the PostgreSQL cache first; any gaps are filled "
        "by Gemini and permanently cached."
    ),
)
async def generate_quiz(payload: QuizRequest, db: Session = Depends(get_db)) -> QuizResponse:
    """
    Self-Populating AI Cache MCQ Generation.

    Tier mapping:
        verified_skills   → Green  (2 questions)
        unverified_skills → Yellow (4 questions)
        missing_skills    → Red    (4 questions)
    """

    skill_map = {
        "Green":  payload.verified_skills,
        "Yellow": payload.unverified_skills,
        "Red":    payload.missing_skills,
    }

    # Fallback: if a tier has no skills, pull from the opposite tier's overflow
    # or use generic labels so Gemini can still generate useful questions.
    _GENERIC_SKILLS = {
        "Green":  ["data structures", "algorithms", "python"],
        "Yellow": ["system design", "databases", "networking"],
        "Red":    ["docker", "kubernetes", "cloud computing"],
    }
    for tier, skills in skill_map.items():
        if not skills:
            skill_map[tier] = _GENERIC_SKILLS[tier]

    final_questions: list[Question | dict[str, Any]] = []
    ai_generated_count = 0

    for tier, needed in _TARGET.items():
        skills = skill_map[tier]

        # ── 1. Fetch from cache ──────────────────────────────────────────
        cached = _fetch_cached(db, skills, tier, limit=needed)
        final_questions.extend(cached)
        gap = needed - len(cached)

        if gap <= 0:
            logger.info("Cache hit: tier=%s needed=%d got=%d", tier, needed, len(cached))
            continue

        logger.info("Cache miss: tier=%s needed=%d cached=%d generating=%d", tier, needed, len(cached), gap)

        # ── 2. Generate the missing questions with Gemini ────────────────
        # Pick skills round-robin to diversify generated questions
        skills_for_gen = [skills[i % len(skills)] for i in range(gap)]
        new_questions: list[dict[str, Any]] = []

        for skill in set(skills_for_gen):
            count_for_skill = skills_for_gen.count(skill)
            generated = await _generate_with_gemini(skill, tier, count_for_skill)
            new_questions.extend(generated)

        # ── 3. Persist to DB (permanent cache) ──────────────────────────
        if new_questions:
            try:
                _bulk_insert(db, new_questions)
                ai_generated_count += len(new_questions)
            except Exception as db_exc:
                logger.error("DB bulk insert failed: %s", db_exc, exc_info=True)
                db.rollback()

            final_questions.extend(new_questions)

        # ── 4. Still short? Use static fallback ─────────────────────────
        remaining_gap = needed - (len(cached) + len(new_questions))
        if remaining_gap > 0:
            tier_fallbacks = [q for q in _STATIC_FALLBACK if q["forensic_tier"] == tier]
            fallback_picks = random.sample(
                tier_fallbacks,
                min(remaining_gap, len(tier_fallbacks))
            )
            final_questions.extend(fallback_picks)
            logger.warning(
                "Used %d static fallback questions for tier=%s", len(fallback_picks), tier
            )

    # ── Trim to exactly 10 and shuffle ──────────────────────────────────────
    final_questions = final_questions[:_TOTAL_QUESTIONS]
    random.shuffle(final_questions)

    # ── Determine source label ───────────────────────────────────────────────
    if ai_generated_count == 0:
        source = "cache"
    elif ai_generated_count == _TOTAL_QUESTIONS:
        source = "ai_generated"
    elif any(isinstance(q, dict) and q.get("skill") in [fb["skill"] for fb in _STATIC_FALLBACK] for q in final_questions):
        source = "fallback"
    else:
        source = "mixed"

    return QuizResponse(
        total=len(final_questions),
        questions=[_to_quiz_question(q) for q in final_questions],
        source=source,
    )
