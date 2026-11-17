"""
scripts/mega_hydrate_db.py
===========================
Offline mega-hydration script for PlaceBuddy.

PURPOSE
-------
Pre-populates the database with 700 Technical + 500 Aptitude placement-grade
MCQ questions using the Gemini API OFFLINE, so live assessment requests are
served entirely from the DB with zero LLM calls and sub-30ms latency.

MODEL & RATE LIMITS
-------------------
Primary model  : gemini-3.5-flash-lite  (500 RPD / 15 RPM)
Fallback model : gemini-3.1-flash-lite  (500 RPD / 15 RPM)
Batch size     : 10 questions per prompt
Sleep interval : 5 seconds between API calls  (~12 RPM, safely under 15 RPM)

DATA TARGETS
------------
Technical (700 total) → mcq_questions (Model: Question)
  Topics : Data Structures, Algorithms, System Design, Python, Java,
           React, SQL, Docker, Kubernetes, Git
  Per topic: 20 Easy + 30 Medium + 20 Hard = 70 questions × 10 topics

Aptitude (500 total) → aptitude_questions (Model: AptitudeQuestion)
  Quantitative (200): Percentages, Profit & Loss, Time & Work,
                      Speed & Distance, Probability
  Logical      (180): Syllogisms, Blood Relations, Seating Arrangements,
                      Series Completion
  Verbal       (120): Sentence Correction, Critical Reasoning, Reading Inference

IDEMPOTENCY
-----------
Before each batch call, the existing count per (topic/category, difficulty)
is checked against the target. If count >= target, that batch is skipped.

RESILIENCY
----------
Each batch is committed in its own try/except/rollback block. A single
batch failure never corrupts previously committed data.

USAGE
-----
    # Run from the repo root:
    cd 22CSP69_1_FINAL-YEAR-PROJECT/backend
    python ../scripts/mega_hydrate_db.py

    # Dry-run (no API calls, no DB writes — just prints what would happen):
    python ../scripts/mega_hydrate_db.py --dry-run

    # Hydrate only technical questions:
    python ../scripts/mega_hydrate_db.py --skip-aptitude

    # Hydrate only aptitude questions:
    python ../scripts/mega_hydrate_db.py --skip-technical

PROGRESS OUTPUT FORMAT
----------------------
    [14/210] Seeded 10 Docker (Medium) questions | DB total: 140
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap: add backend/ to sys.path and load .env
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent / "backend"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_ENV_PATH = _BACKEND_DIR / "app" / ".env"
if _ENV_PATH.exists():
    with open(_ENV_PATH, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

from sqlalchemy.exc import SQLAlchemyError                       # noqa: E402
# pyrefly: ignore [missing-import]
from app.db.database import Base, SessionLocal, engine           # noqa: E402
# pyrefly: ignore [missing-import]
from app.db.models import AptitudeQuestion, Question             # noqa: E402

logging.basicConfig(level=logging.WARNING)  # quiet SQLAlchemy noise

# ─────────────────────────────────────────────────────────────────────────────
# google.genai import
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google import genai as _genai                           # type: ignore
    from google.genai import types as _genai_types               # type: ignore
    from google.genai import errors as _genai_errors             # type: ignore
    _GENAI_OK = True
except ImportError:
    _genai = None  # type: ignore
    _genai_types = None  # type: ignore
    _genai_errors = None  # type: ignore
    _GENAI_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Model cascade — lite models only (stays within 500 RPD / 15 RPM)
_MODEL_PRIMARY  = "gemini-3.5-flash-lite"
_MODEL_FALLBACK = "gemini-3.1-flash-lite"

BATCH_SIZE          = 10   # questions per Gemini call
SLEEP_BETWEEN_CALLS = 5    # seconds (~12 RPM, comfortably under 15 RPM limit)

# ── Technical targets ────────────────────────────────────────────────────────
TECH_TOPICS = [
    "Data Structures",
    "Algorithms",
    "System Design",
    "Python",
    "Java",
    "React",
    "SQL",
    "Docker",
    "Kubernetes",
    "Git",
]

# Per-topic difficulty distribution (must sum to 70)
TECH_DIFFICULTY_TARGETS: dict[str, int] = {
    "easy":   20,
    "medium": 30,
    "hard":   20,
}

# Difficulty → forensic_tier mapping for the Question model
TECH_TIER_MAP: dict[str, str] = {
    "easy":   "Green",
    "medium": "Yellow",
    "hard":   "Red",
}

# ── Aptitude targets ─────────────────────────────────────────────────────────
# Structure: { category: { topic: target_count } }
APTITUDE_PLAN: dict[str, dict[str, int]] = {
    "quantitative": {
        "Percentages":      40,
        "Profit & Loss":    40,
        "Time & Work":      40,
        "Speed & Distance": 40,
        "Probability":      40,
    },
    "logical": {
        "Syllogisms":            45,
        "Blood Relations":       45,
        "Seating Arrangements":  45,
        "Series Completion":     45,
    },
    "verbal": {
        "Sentence Correction":   40,
        "Critical Reasoning":    40,
        "Reading Inference":     40,
    },
}

# Each aptitude topic is split across difficulties (fractions round down;
# leftover goes to "medium").
APTITUDE_DIFFICULTY_SPLIT: dict[str, float] = {
    "easy":   0.30,
    "medium": 0.40,
    "hard":   0.30,
}

# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char: str = "─", width: int = 72) -> None:
    print(char * width, flush=True)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]  {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_tables() -> None:
    """Create all missing tables (idempotent)."""
    Base.metadata.create_all(bind=engine)
    _log("✓ DB tables verified / created.")


def _tech_count(db, topic: str, difficulty: str) -> int:
    return (
        db.query(Question)
        .filter(
            Question.skill == topic.lower(),
            Question.difficulty == difficulty,
        )
        .count()
    )


def _aptitude_count(db, category: str, topic: str, difficulty: str) -> int:
    return (
        db.query(AptitudeQuestion)
        .filter(
            AptitudeQuestion.category == category,
            AptitudeQuestion.topic == topic,
            AptitudeQuestion.difficulty == difficulty,
        )
        .count()
    )


def _safe_commit(db, rows: list[dict], model_cls, label: str) -> int:
    """
    Bulk-insert `rows` in a single transaction with full rollback on failure.
    Returns the number of rows actually inserted.
    """
    if not rows:
        return 0
    try:
        db.bulk_insert_mappings(model_cls, rows)  # type: ignore[arg-type]
        db.commit()
        return len(rows)
    except SQLAlchemyError as exc:
        db.rollback()
        _log(f"  ✗ DB commit failed for {label}: {exc}")
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — placement-grade quality constraints injected
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = (
    "You are a senior placement-assessment engineer generating MCQ questions "
    "for campus recruitment drives at top-tier tech companies.\n"
    "STRICT RULES:\n"
    "1. Avoid trivial definitional questions like 'What is Python?' or 'What is SQL?'.\n"
    "2. For Medium and Hard technical questions, include concise code snippets to "
    "   trace OR real-world system architecture scenarios.\n"
    "3. All 3 distractors must represent REALISTIC misconceptions a candidate might "
    "   hold — not obviously wrong options.\n"
    "4. Explanations must (a) explain WHY the correct option is right AND "
    "   (b) briefly explain why the most common wrong alternative fails.\n"
    "5. For Quantitative aptitude questions, show step-by-step arithmetic in the "
    "   explanation (e.g., 'Step 1: ... Step 2: ...').\n"
    "6. Every question must be self-contained — no references to previous questions.\n"
)

_JSON_SCHEMA_HINT = (
    'Return ONLY a valid JSON object matching this exact schema — no markdown, '
    'no prose:\n'
    '{"questions": [{"question_text": "...", "options": ["A", "B", "C", "D"], '
    '"correct_answer": "...", "explanation": "..."}]}\n'
    "correct_answer MUST be an exact copy of one of the four options."
)


def _tech_prompt(topic: str, difficulty: str, count: int) -> str:
    diff_desc = {
        "easy":   "straightforward conceptual questions that test working knowledge "
                  "(avoid trivial definitions; prefer 'which of these behaviours' style)",
        "medium": "practical application questions with short code snippets to trace "
                  "or real scenarios (e.g. 'A developer does X — what happens?')",
        "hard":   "expert-level trade-off questions: performance implications, "
                  "edge-case behaviour, architectural decisions, or complex code output",
    }
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n"
        f"Topic      : {topic}\n"
        f"Difficulty : {difficulty.upper()} — {diff_desc.get(difficulty, 'moderate depth')}\n"
        f"Count      : Generate exactly {count} unique questions.\n\n"
        f"{_JSON_SCHEMA_HINT}"
    )


def _aptitude_prompt(category: str, topic: str, difficulty: str, count: int) -> str:
    cat_desc = {
        "quantitative": (
            f"quantitative aptitude specifically on the sub-topic '{topic}'. "
            "Use numerical values and require 1–3 step arithmetic. "
            "Show step-by-step solution in the explanation."
        ),
        "logical": (
            f"logical reasoning specifically on the sub-topic '{topic}'. "
            "Questions should require deductive reasoning or pattern identification."
        ),
        "verbal": (
            f"verbal ability specifically on the sub-topic '{topic}'. "
            "Focus on sentence-level language tasks: grammar, inference, or correction."
        ),
    }
    diff_desc = {
        "easy":   "solvable in under 30 seconds with direct rule application",
        "medium": "1–2 reasoning steps, moderate difficulty, realistic workplace phrasing",
        "hard":   "multi-step problem; requires careful analysis and eliminaton of traps",
    }
    return (
        f"{_SYSTEM_INSTRUCTIONS}\n"
        f"Category   : {category}\n"
        f"Sub-topic  : {topic}\n"
        f"Difficulty : {difficulty.upper()} — {diff_desc.get(difficulty, 'moderate')}\n"
        f"Context    : {cat_desc.get(category, category)}\n"
        f"Count      : Generate exactly {count} unique questions.\n\n"
        f"{_JSON_SCHEMA_HINT}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gemini client (direct, bypasses llm_client.py to target lite models only)
# ─────────────────────────────────────────────────────────────────────────────

_MODELS = [_MODEL_PRIMARY, _MODEL_FALLBACK]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _JSON_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _call_gemini(client, model: str, prompt: str) -> str:
    """Single synchronous call; raises on error so caller can cascade."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=_genai_types.GenerateContentConfig(temperature=0.75),
    )
    return response.text


def _generate_questions(client, prompt: str, label: str) -> list[dict[str, Any]]:
    """
    Try each model in _MODELS cascade.
    Returns a list of parsed question dicts, or [] on total failure.
    """
    for model in _MODELS:
        try:
            raw = _call_gemini(client, model, prompt)
            cleaned = _strip_fences(raw)
            data = json.loads(cleaned)
            questions = data.get("questions", [])
            if not isinstance(questions, list) or not questions:
                _log(f"  ⚠  {model}: parsed OK but 'questions' list is empty for {label}")
                continue
            # Basic validation: each item must have the 4 required fields
            valid = []
            for q in questions:
                if all(k in q for k in ("question_text", "options", "correct_answer", "explanation")):
                    if isinstance(q["options"], list) and len(q["options"]) == 4:
                        # Ensure correct_answer is one of the options
                        if q["correct_answer"] in q["options"]:
                            valid.append(q)
                        else:
                            _log(f"  ⚠  Skipping question: correct_answer not in options ({label})")
                    else:
                        _log(f"  ⚠  Skipping question: not exactly 4 options ({label})")
                else:
                    _log(f"  ⚠  Skipping question: missing required fields ({label})")
            if valid:
                return valid
            _log(f"  ⚠  {model}: all parsed questions failed validation for {label}")
        except json.JSONDecodeError as exc:
            _log(f"  ✗ {model}: JSON parse error for {label}: {exc}")
        except Exception as exc:
            if _genai_errors and isinstance(exc, _genai_errors.ClientError):
                _log(f"  ✗ {model}: API error for {label}: {exc}")
            else:
                _log(f"  ✗ {model}: unexpected error for {label}: {type(exc).__name__}: {exc}")
    _log(f"  ✗ All models exhausted for {label}. Skipping batch.")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Technical hydration
# ─────────────────────────────────────────────────────────────────────────────

def _hydrate_technical(client, db, dry_run: bool) -> dict[str, int]:
    """
    Seed mcq_questions with 700 technical questions:
    10 topics × 70 questions each (20 Easy + 30 Medium + 20 Hard).
    Batches of 10. Idempotent per (topic, difficulty).
    """
    stats = {"api_calls": 0, "inserted": 0, "skipped": 0, "errors": 0}

    # Calculate total batch calls for progress display
    # Each difficulty target / batch_size = number of batches per (topic, diff)
    total_batches = sum(
        # ceil division
        (target + BATCH_SIZE - 1) // BATCH_SIZE
        for target in TECH_DIFFICULTY_TARGETS.values()
    ) * len(TECH_TOPICS)
    batch_num = 0

    _sep("═")
    _log(
        f"TECHNICAL HYDRATION\n"
        f"  Topics    : {len(TECH_TOPICS)}  ({', '.join(TECH_TOPICS)})\n"
        f"  Per topic : 20 Easy + 30 Medium + 20 Hard = 70\n"
        f"  Total     : {len(TECH_TOPICS) * sum(TECH_DIFFICULTY_TARGETS.values())} questions\n"
        f"  Batches   : {total_batches} API calls (est.)\n"
        f"  dry-run   : {dry_run}"
    )
    _sep()

    for topic in TECH_TOPICS:
        _log(f"\n▶  Topic: {topic!r}")

        for difficulty, target in TECH_DIFFICULTY_TARGETS.items():
            tier = TECH_TIER_MAP[difficulty]
            existing = _tech_count(db, topic, difficulty)
            remaining = max(0, target - existing)

            if remaining == 0:
                _log(
                    f"   ↷ SKIP  {topic} / {difficulty}  "
                    f"(DB: {existing} / {target} — target reached)"
                )
                stats["skipped"] += (target + BATCH_SIZE - 1) // BATCH_SIZE
                batch_num += (target + BATCH_SIZE - 1) // BATCH_SIZE
                continue

            _log(
                f"   ▸ {topic} / {difficulty}  "
                f"(DB: {existing} / {target}, need {remaining} more)"
            )

            # Break remaining into batches of BATCH_SIZE
            seeded_this_diff = 0
            while seeded_this_diff < remaining:
                batch_num += 1
                count = min(BATCH_SIZE, remaining - seeded_this_diff)
                label = f"{topic} ({difficulty.capitalize()})"

                db_total = _tech_count(db, topic, difficulty) + seeded_this_diff
                prefix = f"   [{batch_num:03d}/{total_batches}] {label}"

                if dry_run:
                    _log(f"{prefix} → DRY-RUN: would request {count} questions")
                    stats["skipped"] += 1
                    seeded_this_diff += count   # pretend we seeded
                    continue

                _log(f"{prefix} → Requesting {count} questions…")

                prompt = _tech_prompt(topic, difficulty, count)
                stats["api_calls"] += 1

                questions = _generate_questions(client, prompt, label)

                if not questions:
                    stats["errors"] += 1
                    # Sleep before retrying next batch
                    _log(f"   ⏳ Sleeping {SLEEP_BETWEEN_CALLS}s after error…")
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    # Skip this batch but continue
                    seeded_this_diff += count
                    continue

                rows = [
                    {
                        "skill":          topic.lower(),
                        "forensic_tier":  tier,
                        "difficulty":     difficulty,
                        "seniority_tier": None,
                        "question_text":  q["question_text"],
                        "options":        q["options"],      # JSON column — list
                        "correct_answer": q["correct_answer"],
                        "explanation":    q["explanation"],
                    }
                    for q in questions
                ]

                n = _safe_commit(db, rows, Question, label)
                stats["inserted"] += n
                seeded_this_diff += len(questions)

                db_total_after = _tech_count(db, topic, difficulty)
                _log(
                    f"   [{batch_num:03d}/{total_batches}] "
                    f"Seeded {n} {label} questions | DB total: {db_total_after}"
                )

                if n < len(questions):
                    stats["errors"] += 1

                _log(f"   ⏳ Sleeping {SLEEP_BETWEEN_CALLS}s (rate limit guard)…")
                time.sleep(SLEEP_BETWEEN_CALLS)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Aptitude hydration
# ─────────────────────────────────────────────────────────────────────────────

def _aptitude_topic_targets(topic_total: int) -> dict[str, int]:
    """
    Distribute `topic_total` questions across easy/medium/hard.
    Leftover from integer rounding goes to medium.
    """
    easy   = int(topic_total * APTITUDE_DIFFICULTY_SPLIT["easy"])
    hard   = int(topic_total * APTITUDE_DIFFICULTY_SPLIT["hard"])
    medium = topic_total - easy - hard
    return {"easy": easy, "medium": medium, "hard": hard}


def _hydrate_aptitude(client, db, dry_run: bool) -> dict[str, int]:
    """
    Seed aptitude_questions with 500 aptitude questions across 3 categories.
    Idempotent per (category, topic, difficulty).
    """
    stats = {"api_calls": 0, "inserted": 0, "skipped": 0, "errors": 0}

    # Pre-compute total API calls for progress display
    total_batches = 0
    for cat_topics in APTITUDE_PLAN.values():
        for total in cat_topics.values():
            for diff_count in _aptitude_topic_targets(total).values():
                if diff_count > 0:
                    total_batches += (diff_count + BATCH_SIZE - 1) // BATCH_SIZE
    batch_num = 0

    _sep("═")
    grand_total = sum(sum(v.values()) for v in APTITUDE_PLAN.values())
    _log(
        f"APTITUDE HYDRATION\n"
        f"  Categories: Quantitative (200), Logical (180), Verbal (120)\n"
        f"  Total     : {grand_total} questions\n"
        f"  Batches   : {total_batches} API calls (est.)\n"
        f"  dry-run   : {dry_run}"
    )
    _sep()

    for category, topics in APTITUDE_PLAN.items():
        _log(f"\n▶  Category: {category.upper()}")

        for topic, topic_total in topics.items():
            diff_targets = _aptitude_topic_targets(topic_total)
            _log(f"   ▸ Topic: {topic!r}  "
                 f"(Easy:{diff_targets['easy']} "
                 f"Med:{diff_targets['medium']} "
                 f"Hard:{diff_targets['hard']})")

            for difficulty, target in diff_targets.items():
                if target == 0:
                    continue

                existing  = _aptitude_count(db, category, topic, difficulty)
                remaining = max(0, target - existing)

                if remaining == 0:
                    n_batches = (target + BATCH_SIZE - 1) // BATCH_SIZE
                    _log(
                        f"      ↷ SKIP  {topic} / {difficulty}  "
                        f"(DB: {existing} / {target} — target reached)"
                    )
                    stats["skipped"] += n_batches
                    batch_num += n_batches
                    continue

                seeded_this_diff = 0
                while seeded_this_diff < remaining:
                    batch_num += 1
                    count = min(BATCH_SIZE, remaining - seeded_this_diff)
                    label = f"{topic} ({difficulty.capitalize()})"
                    prefix = f"      [{batch_num:03d}/{total_batches}] {category}/{label}"

                    if dry_run:
                        _log(f"{prefix} → DRY-RUN: would request {count} questions")
                        stats["skipped"] += 1
                        seeded_this_diff += count
                        continue

                    _log(f"{prefix} → Requesting {count} questions…")

                    prompt = _aptitude_prompt(category, topic, difficulty, count)
                    stats["api_calls"] += 1

                    questions = _generate_questions(client, prompt, label)

                    if not questions:
                        stats["errors"] += 1
                        _log(f"   ⏳ Sleeping {SLEEP_BETWEEN_CALLS}s after error…")
                        time.sleep(SLEEP_BETWEEN_CALLS)
                        seeded_this_diff += count
                        continue

                    rows = [
                        {
                            "category":      category,
                            "topic":         topic,
                            "difficulty":    difficulty,
                            "question_text": q["question_text"],
                            "options":       json.dumps(q["options"]),  # TEXT column
                            "correct_answer": q["correct_answer"],
                            "explanation":   q["explanation"],
                        }
                        for q in questions
                    ]

                    n = _safe_commit(db, rows, AptitudeQuestion, label)
                    stats["inserted"] += n
                    seeded_this_diff += len(questions)

                    db_total_after = _aptitude_count(db, category, topic, difficulty)
                    _log(
                        f"      [{batch_num:03d}/{total_batches}] "
                        f"Seeded {n} {label} questions | DB total: {db_total_after}"
                    )

                    if n < len(questions):
                        stats["errors"] += 1

                    _log(f"   ⏳ Sleeping {SLEEP_BETWEEN_CALLS}s (rate limit guard)…")
                    time.sleep(SLEEP_BETWEEN_CALLS)

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PlaceBuddy Mega DB Hydration Script\n"
            "Offline pre-population of 700 Technical + 500 Aptitude questions\n"
            "using gemini-3.5-flash-lite (500 RPD / 15 RPM)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without making any API calls or DB writes.",
    )
    parser.add_argument(
        "--skip-technical", action="store_true",
        help="Skip technical question hydration (mcq_questions).",
    )
    parser.add_argument(
        "--skip-aptitude", action="store_true",
        help="Skip aptitude question hydration (aptitude_questions).",
    )
    args = parser.parse_args()

    _sep("═")
    _log("PlaceBuddy Mega DB Hydration Script  (v1.0)")
    _log(f"  Primary model  : {_MODEL_PRIMARY}")
    _log(f"  Fallback model : {_MODEL_FALLBACK}")
    _log(f"  Batch size     : {BATCH_SIZE} questions per call")
    _log(f"  Sleep interval : {SLEEP_BETWEEN_CALLS}s between calls "
         f"(~{60 // SLEEP_BETWEEN_CALLS} RPM)")
    _log(f"  dry-run        : {args.dry_run}")
    _log(f"  skip-technical : {args.skip_technical}")
    _log(f"  skip-aptitude  : {args.skip_aptitude}")
    _sep("═")

    # ── Preflight checks ───────────────────────────────────────────────────────
    if not _GENAI_OK:
        print(
            "\n✗ google-genai package not installed.\n"
            "  Run: pip install google-genai\n",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    if not api_key and not args.dry_run:
        print(
            "\n✗ GEMINI_API_KEY not set.\n"
            "  Add it to backend/app/.env or export it before running.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Initialise tables and client ──────────────────────────────────────────
    _ensure_tables()

    client = None
    if not args.dry_run:
        client = _genai.Client(api_key=api_key)
        _log("✓ Gemini client initialised.")

    db = SessionLocal()
    all_stats: dict[str, dict[str, int]] = {}

    try:
        if not args.skip_technical:
            all_stats["technical"] = _hydrate_technical(client, db, args.dry_run)
        else:
            _log("\n⏭  Skipping technical hydration (--skip-technical).")

        if not args.skip_aptitude:
            all_stats["aptitude"] = _hydrate_aptitude(client, db, args.dry_run)
        else:
            _log("\n⏭  Skipping aptitude hydration (--skip-aptitude).")

    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted by user. All committed batches are safe in the DB.", flush=True)
    finally:
        db.close()

    # ── Summary report ────────────────────────────────────────────────────────
    _sep("═")
    _log("HYDRATION COMPLETE — Summary")
    _sep()
    total_inserted = 0
    total_errors   = 0
    for module, s in all_stats.items():
        _log(
            f"  {module.upper():12s}  "
            f"API calls: {s['api_calls']:4d}  |  "
            f"Inserted:  {s['inserted']:5d}  |  "
            f"Skipped: {s['skipped']:4d}  |  "
            f"Errors: {s['errors']:3d}"
        )
        total_inserted += s["inserted"]
        total_errors   += s["errors"]
    _sep()
    _log(f"  TOTAL INSERTED : {total_inserted}")
    _log(f"  TOTAL ERRORS   : {total_errors}")
    if total_errors:
        _log("  ⚠  Errors occurred. Rerun the script — it is fully idempotent.")
    _sep("═")


if __name__ == "__main__":
    main()
