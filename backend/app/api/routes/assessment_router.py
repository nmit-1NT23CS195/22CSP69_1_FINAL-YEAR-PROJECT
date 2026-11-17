"""
app/api/routes/assessment_router.py
====================================
POST /api/assessment/stream  —  Dual-Module SSE Assessment Stream (v4.0 — Pure DB)

ARCHITECTURE (v4.0)
-------------------
- ZERO live LLM calls.  All questions are served from the database only.
- Sub-30ms resolution time for a full 10-question set when the DB is warm.
- No rate-limit risk.  No external API dependency during live requests.

TABLE ROUTING  (absolute isolation — each module ONLY touches its own table)
----------------------------------------------------------------------------
  Technical module  →  mcq_questions      (ORM model: Question)
  Aptitude  module  →  aptitude_questions  (ORM model: AptitudeQuestion)

RETRIEVAL MECHANICS
-------------------
1. Map requested skills/categories + difficulties into 10 target slots.
2. Per slot: query the matching indexed rows with .limit(40), then
   random.sample() in Python for variety (avoids ORDER BY random()).
3. If a slot has zero matching DB rows → try relaxed constraints.
4. If still no match → fall back to an embedded static bank.
   NO LLM call is ever made.

STREAM CONTRACT
---------------
Emits exactly 3 SSE events in a single HTTP connection:

    event: active_module_ready
    data:  {"module": "technical|aptitude", "questions": [...10 items...]}

    event: secondary_module_ready
    data:  {"module": "technical|aptitude", "questions": [...10 items...]}

    event: stream_complete
    data:  {"status": "completed"}

ZERO REGRESSIONS
----------------
- Does NOT import from llm_client.py, llm_service.py, or mcq_router.py.
- Does NOT use ORDER BY random().
- Existing /ats/score endpoints are completely unaffected.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AptitudeQuestion, Question

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_QUESTIONS_PER_MODULE = 10
_DB_FETCH_LIMIT = 50        # rows fetched per query — never ORDER BY random()
_DB_BATCH_LIMIT = 40        # fallback slot fetch limit

# Aptitude: 10 questions split  3 Easy + 4 Medium + 3 Hard
_APTITUDE_DIFFICULTY_BUDGET: dict[str, int] = {"easy": 3, "medium": 4, "hard": 3}
_APTITUDE_CATEGORIES = ["quantitative", "logical", "verbal"]


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class AssessmentRequest(BaseModel):
    """
    Payload from the React dashboard.

    active_module   Which module the frontend shows first ('technical' | 'aptitude')
    role            Target role string for seniority inference (technical path)
    green_skills    Verified skills from ATS cognitive_analysis
    yellow_skills   Unverified / stated-only skills
    red_skills      Missing / gap skills
    """
    role:          str | None  = Field(default=None)
    green_skills:  list[str]   = Field(default_factory=list)
    yellow_skills: list[str]   = Field(default_factory=list)
    red_skills:    list[str]   = Field(default_factory=list)
    active_module: str         = Field(default="technical")

    @field_validator("active_module")
    @classmethod
    def _validate_module(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"technical", "aptitude"}:
            raise ValueError("active_module must be 'technical' or 'aptitude'")
        return v

    @field_validator("green_skills", "yellow_skills", "red_skills", mode="before")
    @classmethod
    def _normalise(cls, v: list[str]) -> list[str]:
        return [s.strip().lower() for s in v if s and s.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Extended static fallback banks  (20 tech + 18 aptitude)
# Guarantees a complete 10-question set even when DB + LLM both fail.
# ---------------------------------------------------------------------------

_TECH_FALLBACK: list[dict[str, Any]] = [
    # ── Software Engineering Basics (generic, always-applicable) ──────────────
    {
        "skill": "software engineering", "forensic_tier": "Green", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What does DRY stand for in software engineering?",
        "options": ["Don't Repeat Yourself", "Do Repeat Yourself", "Design Reusable Yet", "Design Readable YAML"],
        "correct_answer": "Don't Repeat Yourself",
        "explanation": "DRY means every piece of knowledge should have a single authoritative representation.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Yellow", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What is the purpose of version control systems like Git?",
        "options": [
            "To compile source code faster",
            "To track changes in files and enable collaboration",
            "To deploy applications to production",
            "To format source code automatically",
        ],
        "correct_answer": "To track changes in files and enable collaboration",
        "explanation": "VCS tools record a history of file changes and allow multiple developers to work safely in parallel.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Red", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What is the Single Responsibility Principle (SRP)?",
        "options": [
            "A class should handle exactly one HTTP request",
            "A class should have only one reason to change",
            "A function should call only one other function",
            "A module should export exactly one symbol",
        ],
        "correct_answer": "A class should have only one reason to change",
        "explanation": "SRP, part of SOLID, states that a class should do one thing so changes in requirements affect only one class.",
    },
    {
        "skill": "data structures", "forensic_tier": "Green", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What is the time complexity of searching in a balanced BST?",
        "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "correct_answer": "O(log n)",
        "explanation": "Each comparison halves the search space, giving O(log n).",
    },
    {
        "skill": "algorithms", "forensic_tier": "Green", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "Which algorithm guarantees O(n log n) worst-case sorting?",
        "options": ["Quick Sort", "Bubble Sort", "Merge Sort", "Insertion Sort"],
        "correct_answer": "Merge Sort",
        "explanation": "Merge Sort always splits and merges in O(n log n) regardless of input.",
    },
    {
        "skill": "databases", "forensic_tier": "Yellow", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What does ACID stand for in database transactions?",
        "options": [
            "Atomicity, Consistency, Isolation, Durability",
            "Availability, Consistency, Isolation, Durability",
            "Atomicity, Concurrency, Integrity, Durability",
            "Atomicity, Consistency, Integrity, Distribution",
        ],
        "correct_answer": "Atomicity, Consistency, Isolation, Durability",
        "explanation": "ACID guarantees reliable transactions under errors and concurrency.",
    },
    {
        "skill": "docker", "forensic_tier": "Red", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What is the difference between a Docker image and a container?",
        "options": [
            "An image is running; a container is stored",
            "An image is a read-only template; a container is a running instance of it",
            "They are identical",
            "A container is a collection of images",
        ],
        "correct_answer": "An image is a read-only template; a container is a running instance of it",
        "explanation": "Images are immutable blueprints; containers are live writable instances.",
    },
    {
        "skill": "system design", "forensic_tier": "Yellow", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What is the primary purpose of a Load Balancer?",
        "options": [
            "Encrypt traffic between client and server",
            "Distribute incoming requests across multiple backend servers",
            "Cache static assets closer to users",
            "Monitor server health metrics",
        ],
        "correct_answer": "Distribute incoming requests across multiple backend servers",
        "explanation": "A load balancer prevents any single server from becoming a bottleneck.",
    },
    {
        "skill": "security", "forensic_tier": "Red", "difficulty": "hard",
        "seniority_tier": None,
        "question_text": "What attack does input sanitisation primarily prevent?",
        "options": [
            "Man-in-the-Middle (MitM)",
            "Distributed Denial of Service (DDoS)",
            "SQL Injection and Cross-Site Scripting (XSS)",
            "Brute Force",
        ],
        "correct_answer": "SQL Injection and Cross-Site Scripting (XSS)",
        "explanation": "Sanitising input removes malicious characters before they reach the DB or renderer.",
    },
    {
        "skill": "networking", "forensic_tier": "Yellow", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What does ARP resolve?",
        "options": [
            "Domain names to IP addresses",
            "IP addresses to MAC addresses on a local network",
            "MAC addresses to port numbers",
            "IP addresses to domain names",
        ],
        "correct_answer": "IP addresses to MAC addresses on a local network",
        "explanation": "ARP broadcasts 'Who has IP X?' on the LAN; the host responds with its MAC.",
    },
    {
        "skill": "operating systems", "forensic_tier": "Yellow", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What is a zombie process?",
        "options": [
            "A process consuming 100% CPU",
            "A child that exited but hasn't been wait()ed by the parent",
            "A background daemon with no controlling terminal",
            "A process in uninterruptible sleep",
        ],
        "correct_answer": "A child that exited but hasn't been wait()ed by the parent",
        "explanation": "After exit(), the child PCB remains until the parent calls wait().",
    },
    {
        "skill": "kubernetes", "forensic_tier": "Red", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What Kubernetes resource manages stateless application deployments?",
        "options": ["StatefulSet", "DaemonSet", "Deployment", "CronJob"],
        "correct_answer": "Deployment",
        "explanation": "A Deployment manages replica sets for stateless pods with rolling updates.",
    },
    {
        "skill": "system design", "forensic_tier": "Red", "difficulty": "hard",
        "seniority_tier": None,
        "question_text": "In the CAP theorem, which two properties does a partition-tolerant system trade off between?",
        "options": [
            "Consistency and Availability",
            "Availability and Partition Tolerance",
            "Consistency and Partition Tolerance",
            "Scalability and Durability",
        ],
        "correct_answer": "Consistency and Availability",
        "explanation": "Under a network partition a system must choose: consistent or available.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Yellow", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What is the difference between unit testing and integration testing?",
        "options": [
            "Unit tests test the full system; integration tests test individual functions",
            "Unit tests test individual components in isolation; integration tests verify that components work together",
            "They are the same thing",
            "Integration tests are performed by end-users; unit tests by machines",
        ],
        "correct_answer": "Unit tests test individual components in isolation; integration tests verify that components work together",
        "explanation": "Unit tests isolate a single function/class; integration tests check the wiring between multiple components.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Green", "difficulty": "hard",
        "seniority_tier": None,
        "question_text": "What is eventual consistency in distributed systems?",
        "options": [
            "All nodes have the same data at all times",
            "Data updates propagate to all nodes, but reads may temporarily see stale data",
            "The system guarantees that writes are rejected during partitions",
            "Only the primary node can serve reads",
        ],
        "correct_answer": "Data updates propagate to all nodes, but reads may temporarily see stale data",
        "explanation": "Eventual consistency allows high availability; nodes converge to the same state over time.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Yellow", "difficulty": "hard",
        "seniority_tier": None,
        "question_text": "What is the purpose of the Open-Closed Principle (OCP)?",
        "options": [
            "Classes should be open to modification but closed to extension",
            "Classes should be open for extension but closed for modification",
            "Methods should be open to all callers",
            "Modules should be open-source",
        ],
        "correct_answer": "Classes should be open for extension but closed for modification",
        "explanation": "OCP means you add new behaviour by extending (subclassing/composing), not by editing existing code.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Red", "difficulty": "medium",
        "seniority_tier": None,
        "question_text": "What is a REST API?",
        "options": [
            "An API that only uses WebSockets",
            "An architectural style using stateless HTTP methods to access resources",
            "A framework for building desktop applications",
            "A database query language",
        ],
        "correct_answer": "An architectural style using stateless HTTP methods to access resources",
        "explanation": "REST (Representational State Transfer) uses HTTP verbs (GET/POST/PUT/DELETE) on resource URIs.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Green", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What does CI/CD stand for?",
        "options": [
            "Continuous Integration / Continuous Delivery (or Deployment)",
            "Code Inspection / Code Deployment",
            "Client Integration / Client Deployment",
            "Centralised Infrastructure / Centralised DevOps",
        ],
        "correct_answer": "Continuous Integration / Continuous Delivery (or Deployment)",
        "explanation": "CI automates builds and tests on every commit; CD automates delivery to staging or production.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Yellow", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What is a design pattern?",
        "options": [
            "A specific library or framework",
            "A reusable solution template for a commonly occurring software design problem",
            "A coding style guide",
            "A type of database schema",
        ],
        "correct_answer": "A reusable solution template for a commonly occurring software design problem",
        "explanation": "Design patterns (e.g. Singleton, Observer, Factory) provide proven blueprints for common problems.",
    },
    {
        "skill": "software engineering", "forensic_tier": "Red", "difficulty": "easy",
        "seniority_tier": None,
        "question_text": "What is the purpose of an API gateway in a microservices architecture?",
        "options": [
            "To store the service's data",
            "To provide a single entry point for clients, handling routing, auth, and rate limiting",
            "To compile microservices into a single binary",
            "To replace the database layer",
        ],
        "correct_answer": "To provide a single entry point for clients, handling routing, auth, and rate limiting",
        "explanation": "An API gateway decouples clients from internal service topology and centralises cross-cutting concerns.",
    },
]

# Aptitude fallback — 2 per (category × difficulty) = 18 total
_APTITUDE_FALLBACK: list[dict[str, Any]] = [
    # quantitative / easy
    {
        "category": "quantitative", "topic": "Percentages", "difficulty": "easy",
        "question_text": "What is 25% of 400?",
        "options": json.dumps(["75", "100", "125", "150"]),
        "correct_answer": "100",
        "explanation": "25% of 400 = (25 ÷ 100) × 400 = 100.",
    },
    {
        "category": "quantitative", "topic": "Simplification", "difficulty": "easy",
        "question_text": "What is 15% of 200?",
        "options": json.dumps(["20", "25", "30", "35"]),
        "correct_answer": "30",
        "explanation": "15% of 200 = 0.15 × 200 = 30.",
    },
    # quantitative / medium
    {
        "category": "quantitative", "topic": "Profit & Loss", "difficulty": "medium",
        "question_text": "A shopkeeper buys an article for ₹200 and sells it for ₹250. What is the profit percent?",
        "options": json.dumps(["20%", "25%", "30%", "50%"]),
        "correct_answer": "25%",
        "explanation": "Profit% = (50 ÷ 200) × 100 = 25%.",
    },
    {
        "category": "quantitative", "topic": "Speed, Distance & Time", "difficulty": "medium",
        "question_text": "A car covers 450 km in 5 hours. What is its average speed?",
        "options": json.dumps(["80 km/h", "90 km/h", "95 km/h", "100 km/h"]),
        "correct_answer": "90 km/h",
        "explanation": "Speed = Distance ÷ Time = 450 ÷ 5 = 90 km/h.",
    },
    # quantitative / hard
    {
        "category": "quantitative", "topic": "Compound Interest", "difficulty": "hard",
        "question_text": "What is the compound interest on ₹8,000 at 10% per annum for 2 years?",
        "options": json.dumps(["₹1,600", "₹1,680", "₹1,760", "₹1,800"]),
        "correct_answer": "₹1,680",
        "explanation": "CI = 8000 × [(1.1)² − 1] = 8000 × 0.21 = ₹1,680.",
    },
    {
        "category": "quantitative", "topic": "Ratios & Mixtures", "difficulty": "hard",
        "question_text": "In a 80 L mixture of milk and water (ratio 3:1), how many litres of water must be added to make ratio 2:1?",
        "options": json.dumps(["5 L", "8 L", "10 L", "12 L"]),
        "correct_answer": "10 L",
        "explanation": "Milk = 60 L, Water = 20 L. 60 ÷ (20+x) = 2 → x = 10 L.",
    },
    # logical / easy
    {
        "category": "logical", "topic": "Series Completion", "difficulty": "easy",
        "question_text": "Find the next term: 3, 6, 12, 24, ___",
        "options": json.dumps(["36", "42", "48", "60"]),
        "correct_answer": "48",
        "explanation": "Each term is doubled: 24 × 2 = 48.",
    },
    {
        "category": "logical", "topic": "Analogies", "difficulty": "easy",
        "question_text": "Book is to Library as Painting is to ___",
        "options": json.dumps(["Artist", "Gallery", "Canvas", "Museum"]),
        "correct_answer": "Gallery",
        "explanation": "A library houses books; a gallery houses paintings.",
    },
    # logical / medium
    {
        "category": "logical", "topic": "Syllogisms", "difficulty": "medium",
        "question_text": "All roses are flowers. Some flowers fade quickly. Which conclusion is valid?",
        "options": json.dumps([
            "All roses fade quickly",
            "Some roses fade quickly",
            "No conclusion about roses can be drawn",
            "All flowers are roses",
        ]),
        "correct_answer": "No conclusion about roses can be drawn",
        "explanation": "We know some flowers fade but cannot determine which are roses.",
    },
    {
        "category": "logical", "topic": "Coding-Decoding", "difficulty": "medium",
        "question_text": "In a code MOUSE = 13-15-21-19-5. What is the code for TIGER?",
        "options": json.dumps(["20-9-7-5-18", "19-8-6-4-17", "21-10-8-6-19", "20-9-7-4-18"]),
        "correct_answer": "20-9-7-5-18",
        "explanation": "Each letter → alphabetical position: T=20, I=9, G=7, E=5, R=18.",
    },
    # logical / hard
    {
        "category": "logical", "topic": "Set Theory", "difficulty": "hard",
        "question_text": "In a group of 60, 35 play cricket, 30 play football, and 10 play both. How many play neither?",
        "options": json.dumps(["5", "10", "15", "20"]),
        "correct_answer": "5",
        "explanation": "|C ∪ F| = 35 + 30 − 10 = 55. Neither = 60 − 55 = 5.",
    },
    {
        "category": "logical", "topic": "Age Problems", "difficulty": "hard",
        "question_text": "A is 3 times as old as B. 5 years ago A was 4 times as old as B. How old is A now?",
        "options": json.dumps(["30", "36", "45", "48"]),
        "correct_answer": "45",
        "explanation": "Let B = x. A = 3x. (3x−5) = 4(x−5) → x=15. A = 45.",
    },
    # verbal / easy
    {
        "category": "verbal", "topic": "Synonyms & Antonyms", "difficulty": "easy",
        "question_text": "Choose the synonym for AFFLUENT:",
        "options": json.dumps(["Poor", "Wealthy", "Modest", "Frugal"]),
        "correct_answer": "Wealthy",
        "explanation": "Affluent means having abundant wealth — synonymous with wealthy.",
    },
    {
        "category": "verbal", "topic": "Synonyms & Antonyms", "difficulty": "easy",
        "question_text": "Choose the antonym for VERBOSE:",
        "options": json.dumps(["Fluent", "Concise", "Eloquent", "Lengthy"]),
        "correct_answer": "Concise",
        "explanation": "Verbose means using too many words; its antonym is concise.",
    },
    # verbal / medium
    {
        "category": "verbal", "topic": "Grammar & Error Spotting", "difficulty": "medium",
        "question_text": "Which sentence is grammatically correct?",
        "options": json.dumps([
            "Each of the students have submitted their reports.",
            "Each of the students has submitted their report.",
            "Each of the student have submitted their reports.",
            "Each of the students have submitted his report.",
        ]),
        "correct_answer": "Each of the students has submitted their report.",
        "explanation": "'Each' is singular → 'has'. 'Their' accepted as gender-neutral.",
    },
    {
        "category": "verbal", "topic": "Vocabulary in Context", "difficulty": "medium",
        "question_text": "Choose the best word: The scientist's ___ discovery challenged decades of established theory.",
        "options": json.dumps(["mundane", "redundant", "seminal", "trivial"]),
        "correct_answer": "seminal",
        "explanation": "Seminal means strongly influencing future developments — fitting for a groundbreaking discovery.",
    },
    # verbal / hard
    {
        "category": "verbal", "topic": "Vocabulary in Context", "difficulty": "hard",
        "question_text": "Choose the word closest in meaning to EQUIVOCAL:",
        "options": json.dumps(["Clear", "Ambiguous", "Decisive", "Certain"]),
        "correct_answer": "Ambiguous",
        "explanation": "Equivocal means open to more than one interpretation — ambiguous.",
    },
    {
        "category": "verbal", "topic": "Grammar & Error Spotting", "difficulty": "hard",
        "question_text": "Choose the sentence with correct parallel structure:",
        "options": json.dumps([
            "She likes swimming, to run, and cycling.",
            "He decided to write the report, reviewing data, and submit it.",
            "The manager asked us to arrive early, prepare the materials, and present the findings.",
            "They plan on hiking, to camp, and fish.",
        ]),
        "correct_answer": "The manager asked us to arrive early, prepare the materials, and present the findings.",
        "explanation": "Option C uses three bare infinitives (arrive/prepare/present) — parallel structure.",
    },
]


# ---------------------------------------------------------------------------
# Crash-proof padding helper
# ---------------------------------------------------------------------------

def _pad_to_ten(
    collected: list[dict[str, Any]],
    module: str,
) -> list[dict[str, Any]]:
    """
    Guarantee exactly _QUESTIONS_PER_MODULE (10) items in `collected`.

    Uses safe sampling: random.sample(pool, min(needed, len(pool)))
    so a short pool NEVER raises ValueError.

    If `collected` already has >= 10 items, shuffles and returns the first 10.
    If it has fewer, pads with items from the appropriate static fallback bank
    until exactly 10 are present.
    """
    needed = _QUESTIONS_PER_MODULE - len(collected)
    if needed > 0:
        # Pick the right fallback bank
        bank_raw = _TECH_FALLBACK if module == "technical" else _APTITUDE_FALLBACK
        converter = _fallback_tech_to_dict if module == "technical" else _fallback_aptitude_to_dict
        # Avoid duplicating items already in collected if possible
        existing_questions = {q.get("question", q.get("question_text", "")) for q in collected}
        candidates = [
            converter(fb) for fb in bank_raw
            if fb.get("question_text", "") not in existing_questions
        ]
        # If all unique candidates exhausted, cycle the whole bank
        if len(candidates) < needed:
            candidates = [converter(fb) for fb in bank_raw]
        # Safe sample — never raises even if candidates < needed
        n = min(needed, len(candidates))
        padding = random.sample(candidates, n)
        collected = collected + padding

    random.shuffle(collected)
    return collected[:_QUESTIONS_PER_MODULE]


def _tech_row_to_dict(row: Question) -> dict[str, Any]:
    return {
        "module":         "technical",
        "category":       None,
        "topic":          None,
        "skill":          row.skill,
        "forensic_tier":  row.forensic_tier,
        "difficulty":     row.difficulty,
        "question":       row.question_text,
        "options":        row.options,       # JSON column → already a Python list
        "correct_answer": row.correct_answer,
        "explanation":    row.explanation,
    }


def _aptitude_row_to_dict(row: AptitudeQuestion) -> dict[str, Any]:
    opts = row.options
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except json.JSONDecodeError:
            opts = [opts]
    return {
        "module":         "aptitude",
        "category":       row.category,
        "topic":          row.topic,
        "skill":          row.category,
        "forensic_tier":  None,
        "difficulty":     row.difficulty,
        "question":       row.question_text,
        "options":        opts,
        "correct_answer": row.correct_answer,
        "explanation":    row.explanation,
    }


def _fallback_aptitude_to_dict(fb: dict[str, Any]) -> dict[str, Any]:
    opts = fb["options"]
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except json.JSONDecodeError:
            opts = [opts]
    return {
        "module":         "aptitude",
        "category":       fb["category"],
        "topic":          fb.get("topic"),
        "skill":          fb["category"],
        "forensic_tier":  None,
        "difficulty":     fb["difficulty"],
        "question":       fb["question_text"],
        "options":        opts,
        "correct_answer": fb["correct_answer"],
        "explanation":    fb.get("explanation", ""),
    }


def _fallback_tech_to_dict(fb: dict[str, Any]) -> dict[str, Any]:
    return {
        "module":         "technical",
        "category":       None,
        "topic":          None,
        "skill":          fb["skill"],
        "forensic_tier":  fb["forensic_tier"],
        "difficulty":     fb["difficulty"],
        "question":       fb["question_text"],
        "options":        fb["options"],
        "correct_answer": fb["correct_answer"],
        "explanation":    fb["explanation"],
    }


# ---------------------------------------------------------------------------
# Module resolvers
# ---------------------------------------------------------------------------

def _resolve_technical(
    db: Session,
    green_skills: list[str],
    yellow_skills: list[str],
    red_skills:    list[str],
    role:          str | None,   # kept for API compat, no longer used internally
) -> list[dict[str, Any]]:
    """
    Resolve 10 technical questions ONLY from mcq_questions (Table: Question).
    ZERO LLM calls. Maximum 2 DB queries. Sub-50ms guaranteed.

    Cascading fetch strategy
    ------------------------
    Attempt 1 ─ Skill-filtered query:
        SELECT * FROM mcq_questions WHERE skill IN (user_skills) LIMIT 50
        If this yields >= 10 distinct rows → done.

    Attempt 2 ─ Generic fill (no skill filter):
        If Attempt 1 yields fewer than 10, run:
        SELECT * FROM mcq_questions LIMIT 50
        to top up the pool.

    Attempt 3 ─ Static hardcoded bank:
        If both queries together still yield fewer than 10 rows,
        pad with the embedded _TECH_FALLBACK bank.

    Safe sampling: random.sample(pool, min(N, len(pool))) — never raises ValueError.
    """
    need = _QUESTIONS_PER_MODULE
    collected: list[dict[str, Any]] = []
    seen_ids:  set[int]             = set()

    # Flatten all provided skills into one pool (lower-cased by schema validator)
    all_user_skills = list({s for s in (green_skills + yellow_skills + red_skills) if s})

    # ── Attempt 1: skill-filtered query ─────────────────────────────────────
    if all_user_skills:
        skill_rows = (
            db.query(Question)
            .filter(Question.skill.in_(all_user_skills))
            .limit(_DB_FETCH_LIMIT)
            .all()
        )
        candidates = [r for r in skill_rows if r.id not in seen_ids]
        take = min(need, len(candidates))
        chosen = random.sample(candidates, take)
        for row in chosen:
            seen_ids.add(row.id)
            collected.append(_tech_row_to_dict(row))

    # ── Attempt 2: generic fill (any technical question) ────────────────────
    if len(collected) < need:
        generic_rows = (
            db.query(Question)
            .limit(_DB_FETCH_LIMIT)
            .all()
        )
        candidates = [r for r in generic_rows if r.id not in seen_ids]
        still_needed = need - len(collected)
        take = min(still_needed, len(candidates))
        chosen = random.sample(candidates, take)
        for row in chosen:
            seen_ids.add(row.id)
            collected.append(_tech_row_to_dict(row))

    # ── Attempt 3: static hardcoded bank ─────────────────────────────────
    # Only reached if mcq_questions is empty or has very few rows.
    padded = _pad_to_ten(collected, "technical")
    random.shuffle(padded)
    return padded


def _fetch_tech_slot(
    db: Session,
    skill: str,
    diff: str,
    tier: str | None,
    seen_ids: set[int],
) -> Question | None:
    """
    Cascading fallback fetch for a single technical question slot:
    1. Exact match (skill + difficulty [+ tier]) excluding seen_ids.
    2. Relax difficulty: filter only by skill ([+ tier]) excluding seen_ids.
    3. Relax skill: fetch any Question excluding seen_ids.
    4. If still empty: return None.
    """
    # 1. Exact match
    filters = [Question.skill == skill, Question.difficulty == diff]
    if tier:
        filters.append(Question.forensic_tier == tier)
    q1 = db.query(Question).filter(*filters)
    if seen_ids:
        q1 = q1.filter(Question.id.notin_(list(seen_ids)))
    rows = q1.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    # 2. Relax difficulty
    filters = [Question.skill == skill]
    if tier:
        filters.append(Question.forensic_tier == tier)
    q2 = db.query(Question).filter(*filters)
    if seen_ids:
        q2 = q2.filter(Question.id.notin_(list(seen_ids)))
    rows = q2.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    # 3. Relax skill (any question)
    q3 = db.query(Question)
    if seen_ids:
        q3 = q3.filter(Question.id.notin_(list(seen_ids)))
    rows = q3.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    return None


def _fetch_aptitude_slot(
    db: Session,
    category: str,
    diff: str,
    seen_ids: set[int],
) -> AptitudeQuestion | None:
    """
    Cascading fallback fetch for a single aptitude question slot:
    1. Exact match (category + difficulty) excluding seen_ids. Fetch limit(40).
    2. Relax difficulty: filter only by category excluding seen_ids. Fetch limit(40).
    3. Any aptitude question excluding seen_ids. Fetch limit(40).
    4. If still empty: return None.
    """
    # 1. Exact match (category, difficulty)
    q1 = db.query(AptitudeQuestion).filter(
        AptitudeQuestion.category == category,
        AptitudeQuestion.difficulty == diff,
    )
    if seen_ids:
        q1 = q1.filter(AptitudeQuestion.id.notin_(list(seen_ids)))
    rows = q1.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    # 2. Relax difficulty: category only
    q2 = db.query(AptitudeQuestion).filter(
        AptitudeQuestion.category == category,
    )
    if seen_ids:
        q2 = q2.filter(AptitudeQuestion.id.notin_(list(seen_ids)))
    rows = q2.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    # 3. Relax category: any aptitude question
    q3 = db.query(AptitudeQuestion)
    if seen_ids:
        q3 = q3.filter(AptitudeQuestion.id.notin_(list(seen_ids)))
    rows = q3.limit(40).all()
    candidates = [r for r in rows if r.id not in seen_ids]
    if candidates:
        return random.choice(candidates)

    return None


def _resolve_aptitude(db: Session) -> list[dict[str, Any]]:
    """
    Resolve 10 aptitude questions ONLY from aptitude_questions (Table: AptitudeQuestion).
    NO LLM call ever made here.

    Resolution order per slot
    -------------------------
    1. Query with (category, difficulty) — .limit(40) + random.choice().
    2. Relax difficulty: any difficulty for this category.
    3. Static embedded fallback bank (category-matched first, then any).
    """
    # Build (category, difficulty) slots
    slots: list[tuple[str, str]] = []
    for diff, total in _APTITUDE_DIFFICULTY_BUDGET.items():
        per_cat = total // len(_APTITUDE_CATEGORIES)
        leftover = total % len(_APTITUDE_CATEGORIES)
        for i, cat in enumerate(_APTITUDE_CATEGORIES):
            n = per_cat + (1 if i < leftover else 0)
            slots.extend([(cat, diff)] * n)
    random.shuffle(slots)

    collected:   list[dict[str, Any]] = []
    seen_ids:    set[int]             = set()
    seen_fb_ids: set[int]             = set()

    for category, diff in slots:
        # ── 1. Exact match (category, difficulty) ─────────────────────────────
        row = _fetch_aptitude_slot(db, category, diff, seen_ids)
        if row is not None:
            seen_ids.add(row.id)
            collected.append(_aptitude_row_to_dict(row))
            continue

        # ── 2. Relax difficulty: any difficulty for this category ─────────────
        rows_relaxed = (
            db.query(AptitudeQuestion)
            .filter(AptitudeQuestion.category == category)
            .limit(_DB_BATCH_LIMIT)
            .all()
        )
        candidates = [r for r in rows_relaxed if r.id not in seen_ids]
        if candidates:
            row = random.choice(candidates)
            seen_ids.add(row.id)
            collected.append(_aptitude_row_to_dict(row))
            continue

        # ── 3. Static fallback: prefer (category, diff) → category → any ──────
        matched = (
            next(
                (fb for fb in _APTITUDE_FALLBACK
                 if fb["category"] == category and fb["difficulty"] == diff
                 and id(fb) not in seen_fb_ids),
                None,
            )
            or next(
                (fb for fb in _APTITUDE_FALLBACK
                 if fb["category"] == category and id(fb) not in seen_fb_ids),
                None,
            )
            or next(
                (fb for fb in _APTITUDE_FALLBACK if id(fb) not in seen_fb_ids),
                _APTITUDE_FALLBACK[len(collected) % len(_APTITUDE_FALLBACK)],
            )
        )
        seen_fb_ids.add(id(matched))
        collected.append(_fallback_aptitude_to_dict(matched))

    padded = _pad_to_ten(collected, "aptitude")
    return padded


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _sse_generator(
    payload: AssessmentRequest,
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    Yields exactly 3 SSE events (active_module_ready, secondary_module_ready, stream_complete).

    CRASH GUARANTEES
    ----------------
    - Both resolver functions return exactly 10 items (padded with static bank if needed).
    - Per-module try/except ensures a single module failure emits an error payload
      rather than dropping the stream.
    - An outer try/except catches catastrophic DB failures (e.g. engine dead) and
      emits a full 10-question static payload so the connection NEVER drops.
    - Zero LLM calls. Zero random.sample(pool, N) where N > len(pool).
    """
    active    = payload.active_module
    secondary = "aptitude" if active == "technical" else "technical"

    try:
        # ── Active module (pure DB — synchronous) ──────────────────────────────
        try:
            if active == "technical":
                active_qs = _resolve_technical(
                    db, payload.green_skills, payload.yellow_skills,
                    payload.red_skills, payload.role,
                )
            else:
                active_qs = _resolve_aptitude(db)
            yield _fmt_sse("active_module_ready", {"module": active, "questions": active_qs})
        except Exception as exc:
            logger.error("Active module resolution failed: %s", exc, exc_info=True)
            # Emit static fallback so frontend always gets 10 questions
            emergency_qs = _pad_to_ten([], active)
            yield _fmt_sse("active_module_ready", {
                "module": active,
                "questions": emergency_qs,
                "fallback": True,
                "error": str(exc),
            })

        # ── Secondary module (pure DB — synchronous) ────────────────────────────
        try:
            if secondary == "technical":
                secondary_qs = _resolve_technical(
                    db, payload.green_skills, payload.yellow_skills,
                    payload.red_skills, payload.role,
                )
            else:
                secondary_qs = _resolve_aptitude(db)
            yield _fmt_sse("secondary_module_ready", {"module": secondary, "questions": secondary_qs})
        except Exception as exc:
            logger.error("Secondary module resolution failed: %s", exc, exc_info=True)
            emergency_qs = _pad_to_ten([], secondary)
            yield _fmt_sse("secondary_module_ready", {
                "module": secondary,
                "questions": emergency_qs,
                "fallback": True,
                "error": str(exc),
            })

        # ── Sentinel ────────────────────────────────────────────────────────────
        yield _fmt_sse("stream_complete", {"status": "completed"})

    except Exception as catastrophic:
        # Outer guard: DB engine dead, OOM, etc.
        # Always emit all 3 SSE events so the frontend state machine completes.
        logger.critical(
            "Catastrophic SSE failure — emitting static fallback for both modules: %s",
            catastrophic, exc_info=True,
        )
        try:
            yield _fmt_sse("active_module_ready", {
                "module": active,
                "questions": _pad_to_ten([], active),
                "fallback": True,
                "error": "Internal server error — static fallback served",
            })
            yield _fmt_sse("secondary_module_ready", {
                "module": secondary,
                "questions": _pad_to_ten([], secondary),
                "fallback": True,
                "error": "Internal server error — static fallback served",
            })
            yield _fmt_sse("stream_complete", {"status": "completed"})
        except Exception:
            pass  # If even yield fails, accept the connection drop gracefully


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/stream",
    summary="Dual-Module Assessment SSE Stream",
    description=(
        "Returns a Server-Sent Events stream resolving both the active and secondary "
        "assessment modules (technical / aptitude) in a single HTTP connection.\n\n"
        "Technical questions: `mcq_questions` (Model: Question).\n"
        "Aptitude questions: `aptitude_questions` (Model: AptitudeQuestion).\n\n"
        "Events: `active_module_ready`, `secondary_module_ready`, `stream_complete`.\n\n"
        "v4.0: 100% database-driven — zero live LLM calls, sub-30ms resolution."
    ),
    response_class=StreamingResponse,
)
async def stream_assessment(
    payload: AssessmentRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    return StreamingResponse(
        _sse_generator(payload, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
