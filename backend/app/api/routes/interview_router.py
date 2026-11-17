"""
interview_router.py — Module D: Video-Based Mock Interview
==========================================================
FastAPI router exposing two endpoints:

  GET  /api/interview/questions   — Fetch 5 questions for a target role.
  POST /api/interview/submit      — Accept 5 video uploads, process them
                                    concurrently, evaluate answers with one
                                    batched LLM call, and return a full report.

Architectural constraints enforced here:
  * ``asyncio.to_thread`` wraps all CPU-bound ``MediaProcessor.process_video``
    calls to prevent event-loop stalls.
  * ALL temporary files (video + extracted WAV) are deleted in a ``finally``
    block regardless of success or failure.
  * A single video failure does NOT abort the remaining 4 evaluations —
    ``asyncio.gather(..., return_exceptions=True)`` is used and each exception
    is converted to a stub result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
from datetime import datetime, timezone
from statistics import mean
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import InterviewQuestion, InterviewResult
from app.services.interview_evaluation_service import get_interview_evaluator
from app.services.media_processing_service import MediaProcessor

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Hardcoded fallback questions — used to pad when the DB has fewer than 5
# rows for the requested role.  Covers the most common placement scenarios.
# ---------------------------------------------------------------------------
_FALLBACK_QUESTIONS: list[dict] = [
    {
        "id": -1,
        "question_text": "Tell me about yourself and your background.",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": (
            "Look for a structured self-introduction covering education, relevant "
            "experience, and career goals. Award points for conciseness, relevance "
            "to the role, and professional tone."
        ),
        "tags": ["introduction", "communication"],
    },
    {
        "id": -2,
        "question_text": (
            "Describe a challenging project you worked on. What was your role, "
            "what obstacles did you face, and what was the outcome?"
        ),
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": (
            "Evaluate using the STAR framework: Situation clarity, specific Task "
            "ownership, concrete Actions taken, and measurable Result. Deduct for "
            "vague or hypothetical answers."
        ),
        "tags": ["STAR", "problem-solving", "teamwork"],
    },
    {
        "id": -3,
        "question_text": (
            "Explain the difference between a stack and a queue, and give a "
            "real-world use-case for each."
        ),
        "question_type": "technical",
        "difficulty": "easy",
        "rubric": (
            "Award full marks for correct LIFO vs FIFO explanation, appropriate "
            "use-cases (e.g. undo-redo for stack; print spooler/BFS for queue), "
            "and clear articulation. Partial marks for correct concept with weak example."
        ),
        "tags": ["data-structures", "fundamentals"],
    },
    {
        "id": -4,
        "question_text": "Where do you see yourself in the next 3 to 5 years?",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": (
            "Look for realistic career goals aligned with the applied role, "
            "evidence of self-awareness, and genuine motivation. Flag overly "
            "generic or company-specific answers."
        ),
        "tags": ["career-goals", "motivation"],
    },
    {
        "id": -5,
        "question_text": (
            "What is the time complexity of searching in a balanced BST versus "
            "an unsorted array? Explain your reasoning."
        ),
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Full marks: O(log n) for balanced BST with correct justification "
            "(height ≈ log n), O(n) for linear scan. Partial: correct values "
            "without justification. Zero: incorrect values."
        ),
        "tags": ["algorithms", "complexity", "data-structures"],
    },
    {
        "id": -6,
        "question_text": (
            "Tell me about a time you disagreed with a team member. How did you "
            "resolve it?"
        ),
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": (
            "Assess empathy, communication strategy, and outcome. Reward "
            "collaborative resolution over 'I convinced them I was right.' "
            "STAR structure expected."
        ),
        "tags": ["conflict-resolution", "teamwork", "STAR"],
    },
    {
        "id": -7,
        "question_text": "Explain REST vs GraphQL and when you would choose each.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": (
            "Award points for: REST statelessness/HTTP verbs, GraphQL single "
            "endpoint/query flexibility, over-fetching/under-fetching contrast, "
            "and a justified real-world scenario for each. Deduct for factual errors."
        ),
        "tags": ["api-design", "REST", "GraphQL"],
    },
    {
        "id": -8,
        "question_text": "What is your greatest strength and how have you demonstrated it professionally?",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": (
            "Look for a specific, role-relevant strength backed by a concrete "
            "example. Generic self-praise without evidence should score low."
        ),
        "tags": ["self-awareness", "communication"],
    },
]


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------
class QuestionOut(BaseModel):
    id: int
    question_text: str
    question_type: str
    difficulty: str
    rubric: str
    tags: Optional[list] = None


class QuestionsResponse(BaseModel):
    role: str
    questions: List[QuestionOut]
    source: str  # 'database' | 'fallback' | 'mixed'


# ---------------------------------------------------------------------------
# GET /api/interview/questions
# ---------------------------------------------------------------------------
@router.get(
    "/questions",
    response_model=QuestionsResponse,
    summary="Fetch 5 interview questions for a target role",
    tags=["Mock Interview"],
)
async def get_interview_questions(
    role: str,
    question_type: Optional[str] = None,
    difficulty: str = "medium",
    db: Session = Depends(get_db),
) -> QuestionsResponse:
    """Return exactly 5 interview questions for *role*.

    Parameters
    ----------
    role:
        Target job role string (case-insensitive, e.g. ``"backend developer"``).
    question_type:
        Optional filter — ``"behavioral"`` or ``"technical"``.  When omitted
        both types are returned.
    difficulty:
        Preferred difficulty — ``"easy"``, ``"medium"`` (default), or ``"hard"``.
        If fewer than 5 questions exist for that exact difficulty the pool is
        padded with questions of *any* difficulty to guarantee 5 results.

    Returns
    -------
    QuestionsResponse
        ``questions`` list (always length 5), ``role``, and a ``source`` hint
        (``'database'``, ``'fallback'``, or ``'mixed'``).
    """
    role_normalised       = role.strip().lower()
    difficulty_normalised = difficulty.strip().lower() or "medium"

    def _row_to_dict(row: InterviewQuestion) -> dict:
        return {
            "id": row.id,
            "question_text": row.question_text,
            "question_type": row.question_type,
            "difficulty": row.difficulty,
            "rubric": row.rubric,
            "tags": row.tags,
        }

    # -- Base query: filter by role (and optionally question_type) -------------
    base_query = db.query(InterviewQuestion).filter(
        InterviewQuestion.role.ilike(f"%{role_normalised}%")
    )
    if question_type:
        base_query = base_query.filter(
            InterviewQuestion.question_type == question_type.lower()
        )

    # -- Primary pool: exact difficulty match ----------------------------------
    primary_rows = base_query.filter(
        InterviewQuestion.difficulty == difficulty_normalised
    ).all()
    primary_pool = [_row_to_dict(r) for r in primary_rows]

    if len(primary_pool) >= 5:
        # Enough exact-difficulty DB rows — sample and return.
        selected = random.sample(primary_pool, 5)
        source   = "database"
    else:
        # -- Fallback padding strategy (3-tier priority) ----------------------
        # 1. Use all exact-difficulty DB rows we do have.
        # 2. Pad from other-difficulty DB rows for the same role.
        # 3. Last resort: use hardcoded _FALLBACK_QUESTIONS.

        already_ids = {q["id"] for q in primary_pool}

        # Other-difficulty DB rows for the same role
        other_rows = [
            _row_to_dict(r)
            for r in base_query.filter(
                InterviewQuestion.difficulty != difficulty_normalised
            ).all()
            if r.id not in already_ids
        ]

        combined_db = primary_pool + other_rows

        if len(combined_db) >= 5:
            # We have enough DB rows across all difficulties — prefer primary.
            padding = random.sample(other_rows, 5 - len(primary_pool))
            selected = primary_pool + padding
            random.shuffle(selected)
            source = "database" if not primary_pool else "mixed_difficulty"
        else:
            # Need to reach into hardcoded fallbacks.
            db_ids = {q["id"] for q in combined_db}
            fallback_pool = [
                q for q in _FALLBACK_QUESTIONS
                if q["id"] not in db_ids
            ]
            need    = 5 - len(combined_db)
            padding = random.sample(fallback_pool, min(need, len(fallback_pool)))
            selected = combined_db + padding
            random.shuffle(selected)
            source   = "fallback" if not combined_db else "mixed"
            logger.warning(
                "Insufficient DB questions for role=%r difficulty=%r "
                "(found %d DB rows) — padded with fallbacks.",
                role, difficulty_normalised, len(combined_db),
            )

    questions_out = [QuestionOut(**q) for q in selected]
    return QuestionsResponse(role=role_normalised, questions=questions_out, source=source)


# ---------------------------------------------------------------------------
# POST /api/interview/submit
# ---------------------------------------------------------------------------
@router.post(
    "/submit",
    summary="Submit 5 video responses for evaluation",
    tags=["Mock Interview"],
)
async def submit_interview(
    video_1: UploadFile = File(..., description="Video for question 1"),
    video_2: UploadFile = File(..., description="Video for question 2"),
    video_3: UploadFile = File(..., description="Video for question 3"),
    video_4: UploadFile = File(..., description="Video for question 4"),
    video_5: UploadFile = File(..., description="Video for question 5"),
    question_ids: str = Form(
        ...,
        description='JSON-encoded list of 5 question IDs, e.g. "[1,2,3,4,5]"',
    ),
    target_role: str = Form(
        "general",
        description="Target role string used to tag the saved result (e.g. 'backend developer').",
    ),
    user_id: str = Form(
        "anonymous",
        description="Caller identifier (email or UUID). Defaults to 'anonymous' until auth is wired in.",
    ),
    difficulty: str = Form(
        "medium",
        description="Difficulty level selected by the user: 'easy', 'medium', or 'hard'.",
    ),
    db: Session = Depends(get_db),
) -> dict:
    """Accept 5 video uploads + 5 question IDs, process and evaluate them.

    Processing pipeline per video (runs concurrently via ``asyncio.gather``):
      1. Save upload to a temp ``.webm`` file.
      2. ``MediaProcessor.process_video`` (in a thread) → transcript + metrics.
      3. All 5 transcripts batched into ONE ``InterviewEvaluator.evaluate`` call.
      4. Composite score computed; full JSON report returned.

    Failure isolation: an error on any single video produces a stub result with
    ``errors`` populated; the other 4 evaluations continue unaffected.
    """
    # -- Parse & validate question IDs -----------------------------------------
    try:
        q_ids: list[int] = json.loads(question_ids)
        if not isinstance(q_ids, list) or len(q_ids) != 5:
            raise ValueError("Must provide exactly 5 question IDs.")
        q_ids = [int(qid) for qid in q_ids]
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid question_ids: {exc}",
        ) from exc

    # -- Fetch questions from DB (fallback rows have negative ids) -------------
    db_questions: list[dict] = []
    for qid in q_ids:
        if qid < 0:
            # Fallback question — look up in _FALLBACK_QUESTIONS
            fb = next((q for q in _FALLBACK_QUESTIONS if q["id"] == qid), None)
            if fb is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Fallback question id={qid} not found.",
                )
            db_questions.append(fb)
        else:
            row = db.query(InterviewQuestion).filter(InterviewQuestion.id == qid).first()
            if row is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"InterviewQuestion id={qid} not found in database.",
                )
            db_questions.append(
                {
                    "id": row.id,
                    "question_text": row.question_text,
                    "question_type": row.question_type,
                    "difficulty": row.difficulty,
                    "rubric": row.rubric,
                    "tags": row.tags,
                }
            )

    uploads: list[UploadFile] = [video_1, video_2, video_3, video_4, video_5]
    temp_paths: list[str] = []

    # -- Save uploads to temp files -------------------------------------------
    # Must happen BEFORE the try/finally so we can register paths for cleanup.
    for upload in uploads:
        suffix = (
            os.path.splitext(upload.filename or "")[1] or ".webm"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await upload.read()
            tmp.write(contents)
            temp_paths.append(tmp.name)
        logger.debug("Saved upload %r → %s", upload.filename, temp_paths[-1])

    # -- Process & evaluate (cleanup guaranteed by finally) --------------------
    try:
        processor = MediaProcessor()

        # Run all 5 videos concurrently; return_exceptions keeps failures local.
        raw_results = await asyncio.gather(
            *[asyncio.to_thread(processor.process_video, path) for path in temp_paths],
            return_exceptions=True,
        )

        # Normalise results — convert exceptions to stub dicts
        processing_errors: list[Optional[str]] = []
        results: list[dict] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                err_msg = f"Video {i + 1} processing failed: {res}"
                logger.error(err_msg)
                processing_errors.append(err_msg)
                results.append(
                    {
                        "transcript": "",
                        "acoustics": {},
                        "vision": {},
                        "errors": [str(res)],
                    }
                )
            else:
                processing_errors.append(None)
                results.append(res)

        # -- Build inputs for evaluator ----------------------------------------
        transcripts: list[str] = [r.get("transcript", "") for r in results]
        questions_text: list[str] = [q["question_text"] for q in db_questions]
        rubrics: list[str] = [q["rubric"] for q in db_questions]
        difficulties: list[str] = [q["difficulty"] for q in db_questions]

        # -- Single batched LLM evaluation call --------------------------------
        evaluator = get_interview_evaluator()
        evaluation = await evaluator.evaluate(
            questions=questions_text,
            transcripts=transcripts,
            rubrics=rubrics,
            difficulties=difficulties,
        )

        # -- Composite score (difficulty-adjusted, computed inside evaluator) --
        scores = [ev.weighted_score for ev in evaluation.evaluations]
        raw_composite   = round(mean(scores) if scores else 0.0, 2)
        composite_score = evaluation.adjusted_composite_score

        # -- Build the full report payload -------------------------------------
        generated_at = datetime.now(timezone.utc).isoformat()
        report = {
            "composite_score": composite_score,
            "difficulty": difficulty.strip().lower() or "medium",
            "difficulty_multiplier": evaluation.difficulty_multiplier,
            "raw_composite_score": round(mean([ev.weighted_score for ev in evaluation.evaluations]) if evaluation.evaluations else 0.0, 2),
            "total_questions": 5,
            "evaluations": [ev.model_dump() for ev in evaluation.evaluations],
            "acoustic_metrics": [r.get("acoustics", {}) for r in results],
            "vision_metrics": [r.get("vision", {}) for r in results],
            "questions": [
                {
                    "id": q["id"],
                    "question_text": q["question_text"],
                    "question_type": q["question_type"],
                    "difficulty": q["difficulty"],
                }
                for q in db_questions
            ],
            "processing_errors": processing_errors,
            "generated_at": generated_at,
        }

        # -- Persist to DB -----------------------------------------------------
        # Non-fatal: a DB write failure must NOT prevent the client from
        # receiving the evaluation report it already paid compute cost for.
        result_id: Optional[int] = None
        try:
            db_result = InterviewResult(
                user_id=user_id.strip() or "anonymous",
                target_role=target_role.strip().lower() or "general",
                overall_score=composite_score,
                detailed_report=report,
            )
            db.add(db_result)
            db.commit()
            db.refresh(db_result)
            result_id = db_result.id
            logger.info(
                "Persisted InterviewResult id=%d  user=%r  role=%r  score=%.2f",
                result_id,
                db_result.user_id,
                db_result.target_role,
                composite_score,
            )
        except Exception as db_err:
            db.rollback()
            logger.error(
                "Failed to persist InterviewResult — DB error: %s", db_err, exc_info=True
            )

        # -- Return report to client -------------------------------------------
        return {**report, "result_id": result_id}

    finally:
        # -- Cleanup: delete ALL temp video files (WAVs deleted inside process_video)
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug("Cleaned up temp file: %s", path)
            except OSError as cleanup_err:
                logger.warning("Could not delete temp file %s: %s", path, cleanup_err)
