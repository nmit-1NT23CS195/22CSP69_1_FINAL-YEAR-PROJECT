"""
scripts/seed_aptitude.py
========================
Standalone, idempotent seeder for the PlaceBuddy Aptitude Assessment module.

Run from the backend/ directory:
    python ../scripts/seed_aptitude.py

Targets: aptitude_questions table  (Model: AptitudeQuestion)
Does NOT touch: mcq_questions table  (Model: Question)

Seeds exactly 30 high-quality aptitude questions:
    • 10 Quantitative Aptitude   (3 Easy + 4 Medium + 3 Hard)
    • 10 Logical Reasoning       (3 Easy + 4 Medium + 3 Hard)
    • 10 Verbal Ability          (3 Easy + 4 Medium + 3 Hard)

Idempotency guard:
    If db.query(AptitudeQuestion).count() >= 30, the script exits without
    inserting anything — safe to re-run at any time.
"""

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add backend/ to sys.path and load .env manually
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
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

# pyrefly: ignore [missing-import]
from app.db.database import SessionLocal, engine, Base   # noqa: E402
# pyrefly: ignore [missing-import]
from app.db.models import AptitudeQuestion               # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("seed_aptitude")


# ---------------------------------------------------------------------------
# Helper — build a question dict
# options is stored as a JSON string per the AptitudeQuestion schema
# ---------------------------------------------------------------------------
def _q(
    category: str,
    topic: str,
    difficulty: str,
    text: str,
    options: list[str],
    correct: str,
    explanation: str,
) -> dict:
    return {
        "category":      category,
        "topic":         topic,
        "difficulty":    difficulty,
        "question_text": text,
        "options":       json.dumps(options),   # TEXT column stores JSON string
        "correct_answer": correct,
        "explanation":   explanation,
    }


# ---------------------------------------------------------------------------
# Aptitude question bank  (30 questions)
# ---------------------------------------------------------------------------
APTITUDE_BANK: list[dict] = [

    # ══════════════════════════════════════════════════════════════
    #  QUANTITATIVE APTITUDE  (10 questions: 3 Easy · 4 Medium · 3 Hard)
    # ══════════════════════════════════════════════════════════════

    # ── Easy ×3 ──────────────────────────────────────────────────
    _q("quantitative", "Percentages", "easy",
       "What is 15% of 240?",
       ["32", "36", "40", "48"], "36",
       "15% of 240 = (15 ÷ 100) × 240 = 36."),

    _q("quantitative", "Speed, Distance & Time", "easy",
       "A train travels 360 km in 4 hours. What is its speed in km/h?",
       ["80", "90", "100", "120"], "90",
       "Speed = Distance ÷ Time = 360 ÷ 4 = 90 km/h."),

    _q("quantitative", "Simple Interest", "easy",
       "What is the simple interest on ₹5,000 at 8% per annum for 3 years?",
       ["₹800", "₹1,000", "₹1,200", "₹1,500"], "₹1,200",
       "SI = (P × R × T) ÷ 100 = (5000 × 8 × 3) ÷ 100 = ₹1,200."),

    # ── Medium ×4 ────────────────────────────────────────────────
    _q("quantitative", "Pipes & Cisterns", "medium",
       "Pipe A fills a tank in 12 hours; Pipe B empties it in 18 hours. "
       "If both are opened simultaneously, how many hours to fill the tank?",
       ["24", "30", "36", "48"], "36",
       "Net fill rate = 1/12 − 1/18 = 3/36 − 2/36 = 1/36. Time = 36 hours."),

    _q("quantitative", "Profit & Loss", "medium",
       "A merchant marks goods 40% above cost price and gives a 20% discount. "
       "What is the profit percent?",
       ["8%", "10%", "12%", "14%"], "12%",
       "Let CP = 100. MP = 140. SP = 140 × 0.80 = 112. Profit% = 12%."),

    _q("quantitative", "Speed, Distance & Time", "medium",
       "Two cars start from the same point in opposite directions at 60 km/h "
       "and 80 km/h. After how many hours are they 420 km apart?",
       ["2.5 h", "3 h", "3.5 h", "4 h"], "3 h",
       "Relative speed = 60 + 80 = 140 km/h. Time = 420 ÷ 140 = 3 hours."),

    _q("quantitative", "Ratios & Mixtures", "medium",
       "In a mixture of 80 litres the ratio of milk to water is 3:1. "
       "How many litres of water must be added to make the ratio 2:1?",
       ["5 L", "8 L", "10 L", "12 L"], "10 L",
       "Milk = 60 L, Water = 20 L. 60 ÷ (20 + x) = 2 → x = 10 L."),

    # ── Hard ×3 ──────────────────────────────────────────────────
    _q("quantitative", "Compound Interest", "hard",
       "The compound interest on a sum at 10% per annum for 2 years is ₹2,100. "
       "What is the principal?",
       ["₹8,000", "₹9,000", "₹10,000", "₹12,000"], "₹10,000",
       "CI = P[(1.1)² − 1] = 0.21P = 2100 → P = ₹10,000."),

    _q("quantitative", "Simple Interest", "hard",
       "A sum triples itself in 20 years under simple interest. "
       "What is the rate of interest per annum?",
       ["8%", "10%", "12%", "15%"], "10%",
       "SI = 2P = P × R × 20 ÷ 100 → R = 200 ÷ 20 = 10%."),

    _q("quantitative", "Percentages", "hard",
       "A shopkeeper gives successive discounts of 20% and 10% on an article "
       "marked at ₹5,000. What is the final selling price?",
       ["₹3,400", "₹3,500", "₹3,600", "₹3,800"], "₹3,600",
       "After 20%: 5000 × 0.80 = 4000. After 10%: 4000 × 0.90 = ₹3,600."),

    # ══════════════════════════════════════════════════════════════
    #  LOGICAL REASONING  (10 questions: 3 Easy · 4 Medium · 3 Hard)
    # ══════════════════════════════════════════════════════════════

    # ── Easy ×3 ──────────────────────────────────────────────────
    _q("logical", "Series Completion", "easy",
       "Find the next number in the series: 2, 4, 8, 16, ___",
       ["24", "28", "32", "64"], "32",
       "Each term is doubled: 16 × 2 = 32."),

    _q("logical", "Analogies", "easy",
       "Monday is to Week as January is to ___",
       ["Month", "Year", "Season", "Quarter"], "Year",
       "Monday is the first element of a week; January is the first month of a year."),

    _q("logical", "Syllogisms", "easy",
       "All blooms are gloms. All gloms are proms. Which conclusion is definitely true?",
       ["All proms are blooms",
        "Some proms are blooms",
        "All blooms are proms",
        "No blooms are proms"],
       "All blooms are proms",
       "Transitive syllogism: Blooms ⊆ Gloms ⊆ Proms, so all blooms are proms."),

    # ── Medium ×4 ────────────────────────────────────────────────
    _q("logical", "Coding-Decoding", "medium",
       "In a code, MOUSE = 13-15-21-19-5. What is the code for TIGER?",
       ["20-9-7-5-18", "19-8-6-4-17", "21-10-8-6-19", "20-9-7-4-18"],
       "20-9-7-5-18",
       "Each letter is replaced by its alphabetical position: T=20, I=9, G=7, E=5, R=18."),

    _q("logical", "Syllogisms", "medium",
       "Statements: Some cats are dogs. All dogs are birds. "
       "Which conclusions follow?\n"
       "I. Some cats are birds.  II. Some birds are cats.",
       ["Only I", "Only II", "Both I and II", "Neither"],
       "Both I and II",
       "Some cats → dogs → birds, so I holds. Converting I gives II. Both are valid."),

    _q("logical", "Clocks & Calendars", "medium",
       "A clock shows 3:00. What is the angle between the hour and minute hands?",
       ["60°", "75°", "90°", "120°"], "90°",
       "Minute hand at 0°, hour hand at 90° (3 × 30°). Angle = 90°."),

    _q("logical", "Seating Arrangements", "medium",
       "Five friends sit in a row. A is immediately right of B. C is at one end. "
       "D sits between A and E. E is not at either end. "
       "Who sits at the end opposite to C?",
       ["A", "B", "D", "E"], "B",
       "Arrangement: C – B – A – D – E. B is at the opposite end from C."),

    # ── Hard ×3 ──────────────────────────────────────────────────
    _q("logical", "Age Problems", "hard",
       "A is 3 times as old as B. 5 years ago A was 4 times as old as B. "
       "How old is A now?",
       ["30", "36", "45", "48"], "45",
       "Let B = x. A = 3x. (3x − 5) = 4(x − 5) → x = 15. A = 45."),

    _q("logical", "Permutations & Combinations", "hard",
       "In a tournament every team plays every other team exactly once. "
       "There are 10 teams. How many total matches are played?",
       ["40", "45", "50", "90"], "45",
       "C(10, 2) = 10 × 9 ÷ 2 = 45 matches."),

    _q("logical", "Set Theory", "hard",
       "In a group of 60, 35 play cricket, 30 play football, and 10 play both. "
       "How many play neither?",
       ["5", "10", "15", "20"], "5",
       "|C ∪ F| = 35 + 30 − 10 = 55. Neither = 60 − 55 = 5."),

    # ══════════════════════════════════════════════════════════════
    #  VERBAL ABILITY  (10 questions: 3 Easy · 4 Medium · 3 Hard)
    # ══════════════════════════════════════════════════════════════

    # ── Easy ×3 ──────────────────────────────────────────────────
    _q("verbal", "Synonyms & Antonyms", "easy",
       "Choose the correct synonym for BENEVOLENT:",
       ["Hostile", "Generous", "Timid", "Boastful"], "Generous",
       "Benevolent means kind and generous; 'Hostile' is its antonym."),

    _q("verbal", "Synonyms & Antonyms", "easy",
       "Choose the word most opposite in meaning to VERBOSE:",
       ["Fluent", "Concise", "Eloquent", "Lengthy"], "Concise",
       "Verbose means using too many words; its antonym is concise."),

    _q("verbal", "Fill in the Blanks", "easy",
       "Fill in the blank: The committee ___ the proposal after a lengthy debate.",
       ["approves", "approved", "approving", "approve"], "approved",
       "The sentence describes a completed past action, requiring simple past tense: 'approved'."),

    # ── Medium ×4 ────────────────────────────────────────────────
    _q("verbal", "Grammar & Error Spotting", "medium",
       "Identify the error: "
       "'Neither the manager nor the employees was present at the meeting.'",
       ["'Neither...nor' is wrong",
        "'was' should be 'were'",
        "'present' should be 'presenting'",
        "No error"],
       "'was' should be 'were'",
       "With 'Neither...nor', the verb agrees with the nearest noun. "
       "'employees' is plural, so 'were' is correct."),

    _q("verbal", "Vocabulary in Context", "medium",
       "Choose the best word: The scientist's ___ discovery challenged decades "
       "of established theory.",
       ["mundane", "redundant", "seminal", "trivial"], "seminal",
       "Seminal means strongly influencing future developments — fitting for a groundbreaking discovery."),

    _q("verbal", "Reading Comprehension", "medium",
       "'He was so tired that he could scarcely keep his eyes open.' "
       "What does 'scarcely' imply?",
       ["Completely unable", "Barely able", "Easily able", "Intermittently able"],
       "Barely able",
       "Scarcely means 'only just' or 'barely' — he could barely manage to keep his eyes open."),

    _q("verbal", "Word Usage", "medium",
       "Which sentence uses the word 'effect' correctly?",
       ["The new policy will effect sales positively.",
        "The medicine had little effect on the patient.",
        "She tried to effect her mood with music.",
        "The storm effected the crops badly."],
       "The medicine had little effect on the patient.",
       "'Effect' as a noun means a result or outcome; only option B uses it as a noun correctly."),

    # ── Hard ×3 ──────────────────────────────────────────────────
    _q("verbal", "Reading Comprehension", "hard",
       "Passage: 'The proliferation of misinformation on digital platforms has "
       "engendered a crisis of epistemic trust.' "
       "Which word is closest in meaning to 'epistemic'?",
       ["Emotional", "Knowledge-related", "Economic", "Political"],
       "Knowledge-related",
       "Epistemic relates to epistemology — the study of knowledge and justified belief."),

    _q("verbal", "Grammar & Error Spotting", "hard",
       "Choose the sentence with correct parallel structure:",
       ["She likes swimming, to run, and cycling.",
        "He decided to write the report, reviewing data, and submit it.",
        "The manager asked us to arrive early, prepare the materials, and present the findings.",
        "They plan on hiking, to camp, and fish."],
       "The manager asked us to arrive early, prepare the materials, and present the findings.",
       "Parallel structure requires consistent form. Option C uses three bare infinitives "
       "(arrive / prepare / present) — grammatically parallel."),

    _q("verbal", "Figures of Speech", "hard",
       "Identify the rhetorical device: 'The pen is mightier than the sword.'",
       ["Simile", "Metaphor", "Hyperbole", "Personification"],
       "Metaphor",
       "A metaphor directly equates two unlike things without 'like' or 'as'. "
       "Writing (pen) is equated to military power (sword)."),
]


# ---------------------------------------------------------------------------
# Seeder logic
# ---------------------------------------------------------------------------

def _ensure_tables() -> None:
    """Create aptitude_questions table if it does not exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
    logger.info("Table check / creation complete.")


def seed(db) -> None:
    existing_count = db.query(AptitudeQuestion).count()
    if existing_count >= 30:
        logger.info(
            "Aptitude table already has %d rows (≥ 30). Skipping seed.",
            existing_count,
        )
        return

    logger.info(
        "Found %d existing aptitude rows. Inserting up to %d questions...",
        existing_count,
        len(APTITUDE_BANK),
    )

    inserted = skipped = errors = 0
    for item in APTITUDE_BANK:
        try:
            # Row-level idempotency: skip if same question text already exists
            snippet = item["question_text"][:80]
            exists = (
                db.query(AptitudeQuestion)
                .filter(
                    AptitudeQuestion.category == item["category"],
                    AptitudeQuestion.question_text.startswith(snippet),
                )
                .first()
            )
            if exists:
                skipped += 1
                continue

            q = AptitudeQuestion(**item)
            db.add(q)
            inserted += 1
        except Exception as exc:
            errors += 1
            logger.error(
                "Error seeding: %s | %s",
                item.get("question_text", "")[:60],
                exc,
            )

    if inserted > 0:
        try:
            db.commit()
            logger.info("Committed %d new aptitude questions.", inserted)
        except Exception as exc:
            db.rollback()
            logger.error("Commit failed: %s", exc)
            raise

    logger.info(
        "Aptitude seed: %d total | %d inserted | %d skipped | %d errors",
        len(APTITUDE_BANK), inserted, skipped, errors,
    )


def main() -> None:
    logger.info("Starting PlaceBuddy Aptitude seed script...")
    db = SessionLocal()
    try:
        _ensure_tables()
        seed(db)
    except Exception as exc:
        logger.error("Seed failed: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        db.close()
    logger.info("Done. Aptitude cache is ready.")


if __name__ == "__main__":
    main()
