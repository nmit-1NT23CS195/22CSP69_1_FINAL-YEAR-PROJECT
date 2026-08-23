import asyncio
from app.api.routes.mcq_router import _generate_with_gemini, _bulk_insert
from app.db.database import SessionLocal
from app.db.models import Question


async def test():
    print("=== Testing Gemini Structured Output Generation ===")
    result = await _generate_with_gemini("react", "Green", 2)
    print(f"Generated {len(result)} questions for skill=react tier=Green")
    for i, q in enumerate(result, 1):
        qt = q["question_text"][:70]
        ca = q["correct_answer"]
        print(f"  Q{i}: {qt}...")
        print(f"       Correct: {ca}")

    if result:
        print("\n=== Persisting to DB ===")
        db = SessionLocal()
        _bulk_insert(db, result)
        count = db.query(Question).count()
        print(f"Total rows in mcq_questions: {count}")
        db.close()
        print("Cache write: SUCCESS")
    else:
        print("No questions generated (check GEMINI_API_KEY)")


asyncio.run(test())
