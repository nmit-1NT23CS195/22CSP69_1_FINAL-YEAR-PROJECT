"""
models.py
=========
SQLAlchemy ORM models for the PlaceBuddy assessment engine.

TWO TABLES
----------
1. mcq_questions  (Model: Question)
   Existing schema — untouched. Stores technical/forensic MCQ questions
   generated and cached by the Gemini-powered mcq_router.

2. aptitude_questions  (Model: AptitudeQuestion)
   Dedicated table for the Foundational Aptitude & Reasoning Assessment.
   Stores pre-seeded and Gemini-generated quantitative, logical, and
   verbal ability questions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE 1: mcq_questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column Details
~~~~~~~~~~~~~~
id              SERIAL PRIMARY KEY
skill           VARCHAR(120) NOT NULL
                Normalized lowercase skill name (e.g. "react", "kubernetes").
forensic_tier   VARCHAR(10) NOT NULL
                One of: 'Green' | 'Yellow' | 'Red'.
difficulty      VARCHAR(20) NOT NULL  default='medium'
                One of: 'easy' | 'medium' | 'hard'.
seniority_tier  VARCHAR(20) nullable
                One of: 'junior' | 'mid' | 'senior'.
question_text   TEXT NOT NULL
options         JSON NOT NULL  (Array of exactly 4 answer strings)
correct_answer  VARCHAR(512) NOT NULL
explanation     TEXT NOT NULL

Indexes
~~~~~~~
ix_mcq_skill            -- single-column fast lookup by skill name
ix_mcq_skill_diff_tier  -- composite (skill, difficulty, forensic_tier)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TABLE 2: aptitude_questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column Details
~~~~~~~~~~~~~~
id              SERIAL PRIMARY KEY
category        VARCHAR(50) NOT NULL
                One of: 'quantitative' | 'logical' | 'verbal'.
                Indexed for category-filtered lookups.
topic           VARCHAR(100) nullable
                Sub-topic within the category.
                e.g. 'Percentages', 'Syllogisms', 'Reading Comprehension'.
                Indexed for granular topic lookups.
difficulty      VARCHAR(20) NOT NULL  default='medium'
                One of: 'easy' | 'medium' | 'hard'.
                Indexed for difficulty-filtered lookups.
question_text   TEXT NOT NULL
options         TEXT NOT NULL  (JSON-serialised list of exactly 4 strings)
correct_answer  VARCHAR(255) NOT NULL
explanation     TEXT nullable

Indexes
~~~~~~~
ix_aptitude_category    -- single-column for category lookups
ix_aptitude_topic       -- single-column for topic lookups
ix_aptitude_difficulty  -- single-column for difficulty lookups
ix_aptitude_lookup      -- composite (category, difficulty); primary cache access pattern
3. interview_questions  (Model: InterviewQuestion)
   Question bank for the Video-Based Mock Interview module (Module D).
   Stores behavioral and technical questions keyed by target role.

4. interview_results  (Model: InterviewResult)
   Persists each completed mock interview evaluation session.
   ``detailed_report`` stores the full JSON report (evaluations, acoustic
   metrics, vision metrics) for downstream PRS engine consumption.
"""

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, JSON, Index
from sqlalchemy.sql import func
from app.db.database import Base


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 1 — mcq_questions
#  Technical / forensic assessment cache.  ORIGINAL SCHEMA — DO NOT MODIFY.
# ═══════════════════════════════════════════════════════════════════════════

class Question(Base):
    """Technical MCQ question cache — one row per question."""

    __tablename__ = "mcq_questions"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # -- Skill metadata --------------------------------------------------------
    skill: str          = Column(String(120), nullable=False)
    forensic_tier: str  = Column(String(10),  nullable=False)   # Green | Yellow | Red

    # -- Difficulty / seniority metadata ---------------------------------------
    difficulty: str      = Column(String(20), nullable=False, default="medium")
    seniority_tier: str  = Column(String(20), nullable=True)

    # -- Question content ------------------------------------------------------
    question_text: str  = Column(Text,        nullable=False)
    options: list       = Column(JSON,        nullable=False)    # List[str] — 4 items
    correct_answer: str = Column(String(512), nullable=False)
    explanation: str    = Column(Text,        nullable=False)

    # -- Indexes ---------------------------------------------------------------
    __table_args__ = (
        Index("ix_mcq_skill",           "skill"),
        Index("ix_mcq_skill_diff_tier", "skill", "difficulty", "forensic_tier"),
    )

    def __repr__(self) -> str:
        return (
            f"<Question id={self.id} skill={self.skill!r} "
            f"tier={self.forensic_tier!r} diff={self.difficulty!r} "
            f"q={self.question_text[:40]!r}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 2 — aptitude_questions
#  Foundational Aptitude & Reasoning Assessment (quantitative / logical / verbal).
# ═══════════════════════════════════════════════════════════════════════════

class AptitudeQuestion(Base):
    """
    Aptitude assessment question — one row per question.

    `options` is stored as a JSON string (TEXT column) so that the table
    is portable across DB engines.  The router serialises/deserialises with
    json.loads / json.dumps.
    """

    __tablename__ = "aptitude_questions"

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # -- Category & topic ------------------------------------------------------
    category: str  = Column(String(50),  nullable=False)   # quantitative | logical | verbal
    topic: str     = Column(String(100), nullable=True)     # e.g. "Percentages", "Syllogisms"

    # -- Difficulty ------------------------------------------------------------
    difficulty: str = Column(String(20), nullable=False, default="medium")

    # -- Question content ------------------------------------------------------
    question_text: str  = Column(Text,        nullable=False)
    options: str        = Column(Text,        nullable=False)   # JSON string → List[str]
    correct_answer: str = Column(String(255), nullable=False)
    explanation: str    = Column(Text,        nullable=True)

    # -- Indexes ---------------------------------------------------------------
    __table_args__ = (
        Index("ix_aptitude_category",   "category"),
        Index("ix_aptitude_topic",      "topic"),
        Index("ix_aptitude_difficulty", "difficulty"),
        Index("ix_aptitude_lookup",     "category", "difficulty"),
    )

    def __repr__(self) -> str:
        return (
            f"<AptitudeQuestion id={self.id} category={self.category!r} "
            f"topic={self.topic!r} diff={self.difficulty!r} "
            f"q={self.question_text[:40]!r}>"
        )


# ---------------------------------------------------------------------------
# InterviewQuestion — Module D: Video-Based Mock Interview
# ---------------------------------------------------------------------------
class InterviewQuestion(Base):
    """Stores behavioral and technical interview questions used in the
    video-based mock interview module.

    Columns
    -------
    id            : Auto-incrementing primary key.
    role          : Target job role (lowercase, e.g. "backend developer").
    question_text : Full question text presented to the candidate.
    question_type : ``'behavioral'`` or ``'technical'``.
    difficulty    : ``'easy'``, ``'medium'``, or ``'hard'`` (default ``'medium'``).
    rubric        : Guidance text passed to the LLM evaluator to grade the answer.
    tags          : Optional JSON list of skill/topic tags (e.g. ``["python", "api"]``).
    """

    __tablename__ = "interview_questions"

    id            = Column(Integer,     primary_key=True, index=True, autoincrement=True)
    role          = Column(String(120), nullable=False)
    question_text = Column(Text,        nullable=False)
    question_type = Column(String(30),  nullable=False)   # 'behavioral' | 'technical'
    difficulty    = Column(String(20),  nullable=False, server_default="medium")
    rubric        = Column(Text,        nullable=False)
    tags          = Column(JSON,        nullable=True)     # List[str] serialised as JSON

    # -- Indexes ---------------------------------------------------------------
    __table_args__ = (
        Index("ix_interview_role",      "role"),
        Index("ix_interview_role_type", "role", "question_type"),
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewQuestion id={self.id} role={self.role!r} "
            f"type={self.question_type!r} diff={self.difficulty!r} "
            f"q={self.question_text[:40]!r}>"
        )


# ---------------------------------------------------------------------------
# InterviewResult — Module D: persisted evaluation sessions
# ---------------------------------------------------------------------------
class InterviewResult(Base):
    """Stores the outcome of each completed mock interview session.

    One row is written by ``POST /api/interview/submit`` immediately before
    the JSON report is returned to the client.  The ``detailed_report``
    column holds the entire evaluation payload (per-question scores,
    transcripts, acoustic metrics, vision metrics) so the PRS engine can
    consume it without re-running the pipeline.

    Columns
    -------
    id              : Auto-incrementing primary key.
    user_id         : Caller-supplied identifier (email or UUID).  Defaults to
                      ``"anonymous"`` until authentication is wired in.
    target_role     : Role the candidate practiced (e.g. ``"backend developer"``).
    overall_score   : Composite weighted score (0–10 float), mean of 5 answers.
    detailed_report : Full JSON report dict — evaluations, acoustic_metrics,
                      vision_metrics, questions, processing_errors.
    created_at      : UTC timestamp set automatically at insert time.
    """

    __tablename__ = "interview_results"

    id              = Column(Integer,  primary_key=True, index=True, autoincrement=True)
    user_id         = Column(String(255), nullable=False, server_default="anonymous")
    target_role     = Column(String(120), nullable=False)
    overall_score   = Column(Float,       nullable=False)
    detailed_report = Column(JSON,        nullable=False)   # full report dict
    created_at      = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # -- Indexes ---------------------------------------------------------------
    __table_args__ = (
        Index("ix_result_user_id",    "user_id"),
        Index("ix_result_role",       "target_role"),
        Index("ix_result_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewResult id={self.id} user={self.user_id!r} "
            f"role={self.target_role!r} score={self.overall_score}>"
        )
