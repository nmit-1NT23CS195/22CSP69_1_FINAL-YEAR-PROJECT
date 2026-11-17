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
"""

from sqlalchemy import Column, Integer, String, Text, JSON, Index
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
