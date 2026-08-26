"""
mcq_router.py
=============
POST /api/mcq/generate -- Self-Populating AI Cache MCQ Engine (v2)

FLOW OVERVIEW
-------------
1. Accept skill lists from the frontend (derived from ATS Phase 1 output).
2. Infer seniority from the optional `target_role` string; determine the
   (easy, medium, hard) difficulty budget for the 10-question quiz.
3. For each (skill, difficulty, forensic_tier) slot, execute a fast random
   sample query against the PostgreSQL cache.
4. Hit evaluation:
   - All 10 resolved from DB  ->  return immediately (sub-100ms path).
   - Deficit exists           ->  call Gemini for the missing count only,
                                  bulk-insert them, combine + shuffle, return.
5. If Gemini fails or key is absent, fall back to the curated static bank.

ARCHITECTURAL GUARANTEES
-------------------------
- ZERO REGRESSIONS: does not import from ats_service.py, llm_service.py,
  or any existing route.
- Cache writes are idempotent -- we only insert freshly generated rows.
- The endpoint is backward-compatible: target_role is optional.
"""

import logging
import os
import random
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Question

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Forensic-tier target distribution (unchanged from v1)
# ---------------------------------------------------------------------------
_TIER_TARGET = {
    "Green":  2,   # verified competencies
    "Yellow": 4,   # unverified / stated-only
    "Red":    4,   # missing / gap skills
}
_TOTAL_QUESTIONS = sum(_TIER_TARGET.values())  # 10

# ---------------------------------------------------------------------------
# Seniority -> difficulty budget  (easy + medium + hard must equal 10)
# ---------------------------------------------------------------------------
_DIFFICULTY_BUDGET: dict[str, dict[str, int]] = {
    "junior": {"easy": 4, "medium": 5, "hard": 1},
    "mid":    {"easy": 2, "medium": 5, "hard": 3},
    "senior": {"easy": 1, "medium": 4, "hard": 5},
}

_SENIORITY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(senior|lead|architect|principal|staff|director)\b", "senior"),
    (r"\b(junior|intern|associate|entry|trainee|graduate)\b", "junior"),
]

# ---------------------------------------------------------------------------
# Helper -- resolve Gemini API key
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
# Seniority resolver
# ---------------------------------------------------------------------------
def _infer_seniority(role: str | None) -> str:
    """
    Infer seniority level from a target role string.

    Rules (case-insensitive):
      senior / lead / architect / principal / staff / director  -> "senior"
      junior / intern / associate / entry / trainee / graduate  -> "junior"
      anything else (or None)                                    -> "mid"
    """
    if not role:
        return "mid"
    role_lower = role.lower()
    for pattern, level in _SENIORITY_PATTERNS:
        if re.search(pattern, role_lower):
            return level
    return "mid"


# ---------------------------------------------------------------------------
# Difficulty allocator
# ---------------------------------------------------------------------------
def _build_difficulty_slots(seniority: str) -> list[str]:
    """
    Return an ordered list of 10 difficulty labels based on seniority budget.
    e.g. junior -> ["easy","easy","easy","easy","medium","medium","medium",
                     "medium","medium","hard"]
    """
    budget = _DIFFICULTY_BUDGET[seniority]
    slots: list[str] = []
    for diff, count in budget.items():
        slots.extend([diff] * count)
    random.shuffle(slots)          # randomise order for variety
    return slots[:_TOTAL_QUESTIONS]


# ---------------------------------------------------------------------------
# Pydantic schemas -- request / response
# ---------------------------------------------------------------------------

class QuizRequest(BaseModel):
    """
    Payload from the frontend.  All three skill lists come directly from the
    cognitive_analysis section of the /ats/score/core response.
    `target_role` is optional but enables seniority-aware difficulty scaling.
    """
    verified_skills:   list[str] = Field(default_factory=list)
    unverified_skills: list[str] = Field(default_factory=list)
    missing_skills:    list[str] = Field(default_factory=list)
    target_role:       str | None = Field(default=None, description="e.g. 'Senior Backend Developer'")

    @field_validator("verified_skills", "unverified_skills", "missing_skills", mode="before")
    @classmethod
    def _normalise(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s and s.strip()]


class QuizQuestion(BaseModel):
    """Single question returned to the frontend."""
    skill:          str
    forensic_tier:  str
    difficulty:     str
    question:       str
    options:        list[str]
    correct_answer: str
    explanation:    str


class QuizResponse(BaseModel):
    """Envelope returned by the endpoint."""
    total:         int
    seniority:     str        # "junior" | "mid" | "senior"
    questions:     list[QuizQuestion]
    source:        str        # "cache" | "ai_generated" | "mixed" | "fallback"


# ---------------------------------------------------------------------------
# Pydantic schema for Gemini structured output
# ---------------------------------------------------------------------------

class _GeminiQuestionItem(BaseModel):
    question_text:  str
    options:        list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation:    str


class _GeminiQuizOutput(BaseModel):
    questions: list[_GeminiQuestionItem]


# ---------------------------------------------------------------------------
# Static fallback bank (used when Gemini AND cache are both exhausted)
# ---------------------------------------------------------------------------
_STATIC_FALLBACK: list[dict[str, Any]] = [
    {
        "skill": "data structures", "forensic_tier": "Green", "difficulty": "medium",
        "question_text": "What is the time complexity of searching in a balanced BST?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "correct_answer": "O(log n)",
        "explanation": "Each comparison halves the search space, giving O(log n)."
    },
    {
        "skill": "algorithms", "forensic_tier": "Green", "difficulty": "medium",
        "question_text": "Which algorithm guarantees O(n log n) worst-case sorting?",
        "options": ["Quick Sort", "Bubble Sort", "Merge Sort", "Insertion Sort"],
        "correct_answer": "Merge Sort",
        "explanation": "Merge Sort always splits and merges in O(n log n)."
    },
    {
        "skill": "operating systems", "forensic_tier": "Yellow", "difficulty": "medium",
        "question_text": "What is a zombie process?",
        "options": [
            "A process consuming 100% CPU",
            "A child that exited but hasn't been wait()ed by the parent",
            "A background daemon with no controlling terminal",
            "A process in uninterruptible sleep"
        ],
        "correct_answer": "A child that exited but hasn't been wait()ed by the parent",
        "explanation": "After exit(), the child's PCB remains until the parent calls wait()."
    },
    {
        "skill": "networking", "forensic_tier": "Yellow", "difficulty": "easy",
        "question_text": "What does ARP resolve?",
        "options": [
            "Domain names to IP addresses",
            "IP addresses to MAC addresses on a local network",
            "MAC addresses to port numbers",
            "IP addresses to domain names"
        ],
        "correct_answer": "IP addresses to MAC addresses on a local network",
        "explanation": "ARP broadcasts 'Who has IP X?' on the LAN. The host responds with its MAC."
    },
    {
        "skill": "databases", "forensic_tier": "Yellow", "difficulty": "easy",
        "question_text": "What does ACID stand for in database transactions?",
        "options": [
            "Atomicity, Consistency, Isolation, Durability",
            "Availability, Consistency, Isolation, Durability",
            "Atomicity, Concurrency, Integrity, Durability",
            "Atomicity, Consistency, Integrity, Distribution"
        ],
        "correct_answer": "Atomicity, Consistency, Isolation, Durability",
        "explanation": "ACID guarantees reliable transactions even under errors and concurrency."
    },
    {
        "skill": "system design", "forensic_tier": "Yellow", "difficulty": "medium",
        "question_text": "What is the primary purpose of a Load Balancer?",
        "options": [
            "Encrypt traffic between client and server",
            "Distribute incoming requests across multiple backend servers",
            "Cache static assets closer to users",
            "Monitor server health metrics"
        ],
        "correct_answer": "Distribute incoming requests across multiple backend servers",
        "explanation": "A load balancer prevents any single server from becoming a bottleneck."
    },
    {
        "skill": "docker", "forensic_tier": "Red", "difficulty": "easy",
        "question_text": "What is the difference between a Docker image and a container?",
        "options": [
            "An image is running; a container is stored",
            "An image is a read-only template; a container is a running instance of it",
            "They are identical",
            "A container is a collection of images"
        ],
        "correct_answer": "An image is a read-only template; a container is a running instance of it",
        "explanation": "Images are immutable blueprints; containers are live writable instances."
    },
    {
        "skill": "kubernetes", "forensic_tier": "Red", "difficulty": "medium",
        "question_text": "What Kubernetes resource manages stateless application deployments?",
        "options": ["StatefulSet", "DaemonSet", "Deployment", "CronJob"],
        "correct_answer": "Deployment",
        "explanation": "A Deployment manages replica sets for stateless pods with rolling updates."
    },
    {
        "skill": "system design", "forensic_tier": "Red", "difficulty": "hard",
        "question_text": "In the CAP theorem, which two properties does a partition-tolerant system trade off between?",
        "options": [
            "Consistency and Availability",
            "Availability and Partition Tolerance",
            "Consistency and Partition Tolerance",
            "Scalability and Durability"
        ],
        "correct_answer": "Consistency and Availability",
        "explanation": "Under a network partition a system must choose: consistent or available."
    },
    {
        "skill": "security", "forensic_tier": "Red", "difficulty": "hard",
        "question_text": "What attack does input sanitisation primarily prevent?",
        "options": [
            "Man-in-the-Middle (MitM)",
            "Distributed Denial of Service (DDoS)",
            "SQL Injection and Cross-Site Scripting (XSS)",
            "Brute Force"
        ],
        "correct_answer": "SQL Injection and Cross-Site Scripting (XSS)",
        "explanation": "Sanitising input removes special characters before they hit the DB or HTML renderer."
    }
]


# ---------------------------------------------------------------------------
# Core DB helpers -- granular (skill, difficulty, forensic_tier) lookup
# ---------------------------------------------------------------------------

def _fetch_cached_granular(
    db: Session,
    skills: list[str],
    tier: str,
    difficulty: str,
    limit: int,
) -> list[Question]:
    """
    Fetch up to `limit` cached questions matching ANY of `skills` at a
    specific (forensic_tier, difficulty) combination.

    Uses ORDER BY random() for variety across requests; the composite index
    ix_mcq_skill_diff_tier makes this efficient even on large tables.
    """
    if not skills or limit <= 0:
        return []
    rows = (
        db.query(Question)
        .filter(
            Question.skill.in_(skills),
            Question.forensic_tier == tier,
            Question.difficulty == difficulty,
        )
        .order_by(func.random())
        .limit(limit)
        .all()
    )
    return rows


def _fetch_cached_tier_fallback(
    db: Session,
    skills: list[str],
    tier: str,
    limit: int,
    exclude_ids: list[int] | None = None,
) -> list[Question]:
    """
    Relaxed fallback: fetch up to `limit` cached questions matching ANY of
    `skills` at `tier`, ignoring difficulty. Used when the strict granular
    lookup returns fewer rows than needed (e.g. no Green/hard questions seeded).
    """
    if not skills or limit <= 0:
        return []
    q = (
        db.query(Question)
        .filter(
            Question.skill.in_(skills),
            Question.forensic_tier == tier,
        )
    )
    if exclude_ids:
        q = q.filter(Question.id.notin_(exclude_ids))
    return q.order_by(func.random()).limit(limit).all()



def _bulk_insert(db: Session, items: list[dict[str, Any]]) -> None:
    """Insert question dicts into the DB and commit. Rolls back on error."""
    if not items:
        return
    try:
        db.bulk_insert_mappings(Question, items)  # type: ignore[arg-type]
        db.commit()
    except Exception as exc:
        logger.error("DB bulk insert failed: %s", exc, exc_info=True)
        db.rollback()


# ---------------------------------------------------------------------------
# Gemini generation helper (difficulty-aware)
# ---------------------------------------------------------------------------

async def _generate_with_gemini(
    skill: str,
    tier: str,
    difficulty: str,
    count: int,
    seniority: str,
) -> list[dict[str, Any]]:
    """
    Call Gemini (gemini-2.0-flash) for exactly `count` questions on `skill`
    at the given forensic tier and difficulty level.

    Returns a list of dicts ready for DB insertion.
    """
    if count <= 0:
        return []

    api_key = _get_gemini_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY not set -- skipping AI gen for skill=%s", skill)
        return []

    tier_descriptors = {
        "Green":  "consolidation-level questions that reinforce solid understanding",
        "Yellow": "intermediate probe questions that expose shallow understanding",
        "Red":    "advanced challenge questions on concepts the candidate likely lacks",
    }
    diff_descriptors = {
        "easy":   "straightforward definitional or conceptual questions suitable for beginners",
        "medium": "practical application questions requiring moderate depth",
        "hard":   "expert-level questions involving trade-offs, internals, or edge cases",
    }

    prompt = (
        f'Generate exactly {count} technical MCQ question(s) about "{skill}".\n'
        f"These should be {tier_descriptors.get(tier, 'technical')} questions "
        f"at {diff_descriptors.get(difficulty, 'medium')} difficulty, "
        f"appropriate for a {seniority}-level candidate.\n\n"
        "Requirements:\n"
        "- Each question must have exactly 4 answer options.\n"
        "- correct_answer must be an exact copy of one of the 4 options.\n"
        "- explanation must be 1-3 sentences, technically accurate, educational.\n"
        "- Questions must be distinct -- no two questions should test the same sub-concept.\n"
        "- Do NOT include question numbers in the question_text.\n\n"
        "Return ONLY valid JSON matching this schema:\n"
        '{"questions": [{"question_text": "<full question>", '
        '"options": ["<o1>","<o2>","<o3>","<o4>"], '
        '"correct_answer": "<must match one option>", '
        '"explanation": "<forensic explanation>"}]}'
    )

    try:
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

        raw_text = response.text or ""
        parsed: _GeminiQuizOutput = _GeminiQuizOutput.model_validate_json(raw_text)

        results: list[dict[str, Any]] = []
        for item in parsed.questions[:count]:
            if item.correct_answer not in item.options:
                logger.warning("Gemini mismatched correct_answer for skill=%s -- skipping", skill)
                continue
            results.append({
                "skill":          skill,
                "forensic_tier":  tier,
                "difficulty":     difficulty,
                "seniority_tier": seniority,
                "question_text":  item.question_text.strip(),
                "options":        item.options,
                "correct_answer": item.correct_answer,
                "explanation":    item.explanation.strip(),
            })

        logger.info(
            "Gemini: generated %d/%d for skill=%s tier=%s diff=%s",
            len(results), count, skill, tier, difficulty,
        )
        return results

    except Exception as exc:
        logger.error(
            "Gemini failed: skill=%s tier=%s diff=%s -- %s", skill, tier, difficulty, exc,
            exc_info=True,
        )
        return []


# ---------------------------------------------------------------------------
# Question serialiser -- ORM row or dict -> QuizQuestion
# ---------------------------------------------------------------------------

def _to_quiz_question(q: "Question | dict[str, Any]") -> QuizQuestion:
    if isinstance(q, Question):
        return QuizQuestion(
            skill=q.skill,
            forensic_tier=q.forensic_tier,
            difficulty=getattr(q, "difficulty", "medium"),
            question=q.question_text,
            options=q.options,
            correct_answer=q.correct_answer,
            explanation=q.explanation,
        )
    return QuizQuestion(
        skill=q["skill"],
        forensic_tier=q["forensic_tier"],
        difficulty=q.get("difficulty", "medium"),
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
    summary="Generate a 10-question seniority-adaptive MCQ quiz",
    description=(
        "Accepts verified, unverified, and missing skill lists from the ATS analysis "
        "plus an optional target_role string for seniority detection. "
        "Returns a personalised 10-question quiz (2 Green + 4 Yellow + 4 Red) with "
        "difficulty scaled to junior / mid / senior. "
        "Questions are fetched from the PostgreSQL cache first; any deficit is filled "
        "by Gemini and permanently cached for sub-100ms future hits."
    ),
)
async def generate_quiz(payload: QuizRequest, db: Session = Depends(get_db)) -> QuizResponse:
    """
    Granular Cache-First MCQ Generation.

    Tier mapping:
        verified_skills   -> Green  (2 questions)
        unverified_skills -> Yellow (4 questions)
        missing_skills    -> Red    (4 questions)

    Difficulty allocation is determined by seniority inferred from target_role.
    """

    # ── Seniority & difficulty budget ──────────────────────────────────────
    seniority = _infer_seniority(payload.target_role)
    difficulty_slots = _build_difficulty_slots(seniority)   # 10-item list

    logger.info(
        "Quiz request: role=%r seniority=%s budget=%s",
        payload.target_role, seniority, _DIFFICULTY_BUDGET[seniority],
    )

    # ── Skill map (tier -> skill list) ─────────────────────────────────────
    skill_map = {
        "Green":  payload.verified_skills,
        "Yellow": payload.unverified_skills,
        "Red":    payload.missing_skills,
    }
    _GENERIC_SKILLS = {
        "Green":  ["data structures", "algorithms", "python"],
        "Yellow": ["system design", "databases", "networking"],
        "Red":    ["docker", "kubernetes", "cloud computing"],
    }
    for tier, skills in skill_map.items():
        if not skills:
            skill_map[tier] = _GENERIC_SKILLS[tier]

    # ── Assign difficulty slots to tiers (round-robin) ─────────────────────
    # Build a per-tier queue of difficulty labels that sums correctly.
    slot_idx = 0
    tier_difficulty_queue: dict[str, list[str]] = {"Green": [], "Yellow": [], "Red": []}
    for tier, needed in _TIER_TARGET.items():
        for _ in range(needed):
            tier_difficulty_queue[tier].append(difficulty_slots[slot_idx % len(difficulty_slots)])
            slot_idx += 1

    # ── Granular cache-first retrieval ─────────────────────────────────────
    final_questions: list[Question | dict[str, Any]] = []
    ai_generated_count = 0
    used_fallback = False

    for tier, needed in _TIER_TARGET.items():
        skills    = skill_map[tier]
        diff_reqs = tier_difficulty_queue[tier]   # e.g. ["easy","medium","medium"]

        # Group by difficulty to minimise DB round-trips
        diff_groups: dict[str, int] = {}
        for d in diff_reqs:
            diff_groups[d] = diff_groups.get(d, 0) + 1

        tier_questions: list[Question | dict[str, Any]] = []
        used_ids: list[int] = []

        for diff, diff_count in diff_groups.items():
            # ── Step 1: Granular DB cache lookup ────────────────────────
            cached = _fetch_cached_granular(db, skills, tier, diff, diff_count)
            tier_questions.extend(cached)
            used_ids.extend(q.id for q in cached if isinstance(q, Question))
            gap = diff_count - len(cached)

            if gap <= 0:
                logger.info("Cache HIT: tier=%s diff=%s needed=%d", tier, diff, diff_count)
                continue

            logger.info(
                "Cache MISS: tier=%s diff=%s cached=%d gap=%d", tier, diff, len(cached), gap,
            )

            # ── Step 2: Tier-relaxed DB fallback (any difficulty, same tier) ─
            tier_fb = _fetch_cached_tier_fallback(db, skills, tier, gap, exclude_ids=used_ids)
            if tier_fb:
                tier_questions.extend(tier_fb)
                used_ids.extend(q.id for q in tier_fb)
                gap -= len(tier_fb)
                logger.info(
                    "Tier-fallback DB: tier=%s filled %d from any-difficulty cache",
                    tier, len(tier_fb),
                )

            if gap <= 0:
                continue

            logger.info(
                "Calling Gemini for tier=%s diff=%s count=%d", tier, diff, gap,
            )

            # ── Step 3: Gemini for the remaining deficit ─────────────────
            skills_for_gen = [skills[i % len(skills)] for i in range(gap)]
            new_questions: list[dict[str, Any]] = []

            for skill in dict.fromkeys(skills_for_gen):
                count_for_skill = skills_for_gen.count(skill)
                generated = await _generate_with_gemini(
                    skill, tier, diff, count_for_skill, seniority
                )
                new_questions.extend(generated)

            # ── Step 4: Persist to DB ─────────────────────────────────────
            if new_questions:
                _bulk_insert(db, new_questions)
                ai_generated_count += len(new_questions)
                tier_questions.extend(new_questions)

            # ── Step 5: Static fallback if still short ────────────────────
            still_short = needed - len(tier_questions)
            if still_short > 0:
                fb_pool = [
                    q for q in _STATIC_FALLBACK
                    if q["forensic_tier"] == tier
                ]
                picks = random.sample(fb_pool, min(still_short, len(fb_pool)))
                tier_questions.extend(picks)
                used_fallback = True
                logger.warning(
                    "Static fallback: tier=%s used %d questions", tier, len(picks)
                )

        final_questions.extend(tier_questions[:needed])


    # ── Trim to exactly 10 and shuffle ────────────────────────────────────
    final_questions = final_questions[:_TOTAL_QUESTIONS]
    random.shuffle(final_questions)

    # ── Determine source label ─────────────────────────────────────────────
    if used_fallback:
        source = "fallback"
    elif ai_generated_count == 0:
        source = "cache"
    elif ai_generated_count == _TOTAL_QUESTIONS:
        source = "ai_generated"
    else:
        source = "mixed"

    return QuizResponse(
        total=len(final_questions),
        seniority=seniority,
        questions=[_to_quiz_question(q) for q in final_questions],
        source=source,
    )
