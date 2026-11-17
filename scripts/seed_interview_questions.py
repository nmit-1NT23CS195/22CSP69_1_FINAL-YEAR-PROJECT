"""
seed_interview_questions.py
===========================
Populates the ``interview_questions`` table with a starter set of ~30
behavioral and technical questions covering the most common placement roles.

Usage (from repo root, with venv activated):
    python scripts/seed_interview_questions.py

Idempotent: skips insertion if the table already has ≥ 20 rows.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — ensure the backend package is importable when the script is run
# from the repo root (22CSP69_1_FINAL-YEAR-PROJECT/).
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

# Load .env from backend/app/.env if it exists (mirrors run.py behaviour)
_ENV_FILE = _BACKEND_DIR / "app" / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE)

from app.db.database import SessionLocal, engine      # noqa: E402
from app.db import models                             # noqa: E402  registers all ORM models
from app.db.models import InterviewQuestion            # noqa: E402

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------
_QUESTIONS: list[dict] = [
    # ── Backend Developer ─────────────────────────────────────────────────
    {
        "role": "backend developer",
        "question_text": "Explain the difference between synchronous and asynchronous programming. When would you choose async?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award marks for I/O-bound vs CPU-bound distinction, event-loop explanation, concrete async use-cases (web APIs, file I/O). Deduct for confusing async with multi-threading.",
        "tags": ["python", "async", "concurrency"],
    },
    {
        "role": "backend developer",
        "question_text": "What is database indexing and how does it improve query performance? What are the trade-offs?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: B-tree/hash index mechanics, faster SELECT at cost of slower INSERT/UPDATE and extra storage. Partial: vague 'it makes it faster'. Zero: incorrect.",
        "tags": ["databases", "SQL", "performance"],
    },
    {
        "role": "backend developer",
        "question_text": "Describe the REST architectural constraints and how you apply them in API design.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Look for: statelessness, uniform interface, client-server, layered system, cacheable responses. Bonus for HATEOAS. Deduct for confusing REST with HTTP.",
        "tags": ["api-design", "REST", "HTTP"],
    },
    {
        "role": "backend developer",
        "question_text": "Walk me through how you would design a URL shortener service (like bit.ly).",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": "Assess: hash/base62 encoding, DB schema choice (SQL vs NoSQL), redirect mechanics, collision handling, analytics, caching (Redis), scalability discussion.",
        "tags": ["system-design", "databases", "caching"],
    },
    {
        "role": "backend developer",
        "question_text": "Tell me about a time you had to debug a production issue under pressure. What was your approach?",
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": "STAR structure expected. Reward: systematic debugging (logs, metrics, isolation), calm communication with stakeholders, post-mortem mindset. Deduct for blaming others.",
        "tags": ["debugging", "incident-response", "STAR"],
    },
    {
        "role": "backend developer",
        "question_text": "How do you approach writing unit tests for a service that makes external HTTP calls?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Look for mocking/patching (unittest.mock, pytest-httpx), dependency injection, test isolation, and coverage of edge cases (timeouts, 500s). Deduct for 'I test manually'.",
        "tags": ["testing", "mocking", "python"],
    },

    # ── Data Scientist ────────────────────────────────────────────────────
    {
        "role": "data scientist",
        "question_text": "Explain the bias-variance trade-off and how you manage it in practice.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: clear definition of both, underfitting vs overfitting, regularisation/ensemble/cross-validation as mitigations. Partial: correct concept, weak mitigation strategies.",
        "tags": ["machine-learning", "model-evaluation", "statistics"],
    },
    {
        "role": "data scientist",
        "question_text": "How would you handle a dataset where 95% of samples belong to one class?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award points for: SMOTE/oversampling, class weights, precision-recall vs accuracy, threshold tuning, ensemble methods. Deduct for suggesting accuracy as the sole metric.",
        "tags": ["imbalanced-data", "classification", "machine-learning"],
    },
    {
        "role": "data scientist",
        "question_text": "Walk me through your process for feature selection before training a model.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Assess: correlation analysis, mutual information, RFE, L1 regularisation (Lasso), domain knowledge, multicollinearity checks. Bonus for mentioning SHAP for interpretability.",
        "tags": ["feature-engineering", "EDA", "machine-learning"],
    },
    {
        "role": "data scientist",
        "question_text": "Describe a data science project you're proud of. What was your contribution and what impact did it have?",
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": "STAR structure. Reward quantified impact (e.g. 'reduced churn by 12%'), ownership language, honest limitations, and learnings. Deduct for vague 'we built a model'.",
        "tags": ["project", "impact", "STAR"],
    },
    {
        "role": "data scientist",
        "question_text": "What is the difference between bagging and boosting? Give an example algorithm for each.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: bagging = parallel (Random Forest), reduces variance; boosting = sequential (XGBoost/AdaBoost), reduces bias. Partial: correct examples, weak theory.",
        "tags": ["ensemble-methods", "machine-learning"],
    },
    {
        "role": "data scientist",
        "question_text": "How do you validate that a machine learning model is not leaking data?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": "Look for: train-test split before any preprocessing, pipeline usage, careful time-series splitting, checking feature importance for suspiciously high signals, AUC sanity checks.",
        "tags": ["data-leakage", "validation", "machine-learning"],
    },

    # ── Full Stack Engineer ───────────────────────────────────────────────
    {
        "role": "full stack engineer",
        "question_text": "Explain the difference between server-side rendering (SSR) and client-side rendering (CSR). When would you use each?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award marks for: SEO benefit of SSR, faster initial paint, CSR interactivity/SPA tradeoffs, hydration concept, TTI vs FCP metrics. Bonus for mentioning ISR/SSG.",
        "tags": ["frontend", "SSR", "CSR", "performance"],
    },
    {
        "role": "full stack engineer",
        "question_text": "How do you manage state in a large React application?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Look for: Context API, Redux/Zustand/Jotai trade-offs, server state (React Query/SWR) vs client state distinction, component co-location principle. Deduct for 'I use setState everywhere'.",
        "tags": ["react", "state-management", "frontend"],
    },
    {
        "role": "full stack engineer",
        "question_text": "Describe how you would implement JWT-based authentication across a React frontend and a FastAPI backend.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: access + refresh token split, httpOnly cookie vs localStorage security, backend middleware validation, token expiry handling, CORS config. Deduct for storing tokens in localStorage without noting XSS risk.",
        "tags": ["authentication", "JWT", "security", "fullstack"],
    },
    {
        "role": "full stack engineer",
        "question_text": "Tell me about a time you improved the performance of a web application. What metrics did you use?",
        "question_type": "behavioral",
        "difficulty": "medium",
        "rubric": "STAR format. Reward: LCP/FID/CLS/TTI awareness, specific tool (Lighthouse, WebPageTest), concrete change (lazy loading, CDN, DB index), measurable improvement. Deduct for vague 'it felt faster'.",
        "tags": ["performance", "optimization", "STAR"],
    },
    {
        "role": "full stack engineer",
        "question_text": "What are WebSockets and when would you use them instead of HTTP polling?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award points for: persistent bidirectional connection, latency advantage, use-cases (chat, live dashboards, collaborative editing), trade-offs (stateful server, scaling complexity).",
        "tags": ["websockets", "real-time", "networking"],
    },

    # ── Software Engineer (generic) ───────────────────────────────────────
    {
        "role": "software engineer",
        "question_text": "What are the SOLID principles? Give an example of one you have applied.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: correct definitions of all five, concrete code example for at least one. Partial: correct definitions, no example. Zero: confuses or cannot name them.",
        "tags": ["OOP", "design-principles", "software-design"],
    },
    {
        "role": "software engineer",
        "question_text": "Explain how Git rebase differs from merge. When should you avoid rebasing?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award marks for: linear history via rebase, merge preserves topology, golden rule (never rebase shared branches), force-push implications. Deduct for confusing interactive rebase with squash-merge.",
        "tags": ["git", "version-control", "collaboration"],
    },
    {
        "role": "software engineer",
        "question_text": "Describe a situation where you had to learn a new technology quickly to deliver a project. How did you approach it?",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": "STAR structure. Reward: structured learning strategy, documentation/community use, applied practice over passive reading, sharing knowledge with team. Deduct for 'I just Googled it'.",
        "tags": ["learning", "adaptability", "STAR"],
    },
    {
        "role": "software engineer",
        "question_text": "What is the difference between a process and a thread? How does Python handle concurrency with the GIL?",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": "Full marks: memory isolation of processes, shared memory of threads, GIL preventing true thread parallelism for CPU-bound work, multiprocessing as the workaround, asyncio for I/O-bound. Partial: correct GIL definition, weak implications.",
        "tags": ["concurrency", "python", "OS"],
    },
    {
        "role": "software engineer",
        "question_text": "How do you approach code review — both as an author and as a reviewer?",
        "question_type": "behavioral",
        "difficulty": "easy",
        "rubric": "Look for: author self-review before submission, small focused PRs, respectful reviewer tone, focus on correctness/maintainability over style, constructive feedback culture.",
        "tags": ["code-review", "collaboration", "communication"],
    },

    # ── DevOps / Cloud Engineer ───────────────────────────────────────────
    {
        "role": "devops engineer",
        "question_text": "Explain the concept of Infrastructure as Code (IaC) and name the tools you have used.",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Award marks for: repeatability, version control, declarative vs imperative styles, Terraform/Pulumi/CloudFormation examples, drift detection. Deduct for confusing IaC with scripting.",
        "tags": ["IaC", "terraform", "cloud", "devops"],
    },
    {
        "role": "devops engineer",
        "question_text": "What is a Kubernetes pod, deployment, and service? How do they relate?",
        "question_type": "technical",
        "difficulty": "medium",
        "rubric": "Full marks: pod = smallest schedulable unit, deployment manages replicas + rolling updates, service = stable network endpoint. Partial: correct definitions, no relationship articulated.",
        "tags": ["kubernetes", "containers", "devops"],
    },
    {
        "role": "devops engineer",
        "question_text": "Describe how you would design a CI/CD pipeline for a FastAPI application deployed on AWS.",
        "question_type": "technical",
        "difficulty": "hard",
        "rubric": "Assess: source trigger (GitHub Actions/CodePipeline), build stage (Docker), test stage (pytest), push to ECR, deploy to ECS/EKS, blue-green or rolling strategy, rollback plan.",
        "tags": ["CI/CD", "AWS", "docker", "fastapi"],
    },
    {
        "role": "devops engineer",
        "question_text": "Tell me about a production outage you helped resolve. What was the root cause and how did you prevent recurrence?",
        "question_type": "behavioral",
        "difficulty": "hard",
        "rubric": "STAR format. Reward: calm escalation, structured debugging (logs/metrics/traces), clear RCA, post-mortem culture, concrete preventive change. Deduct for blame without fix.",
        "tags": ["incident-response", "SRE", "STAR"],
    },
]

# ---------------------------------------------------------------------------
# Main seed logic
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure tables exist (idempotent if already created by main.py)
    models.Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        existing_count: int = db.query(InterviewQuestion).count()
        if existing_count >= 20:
            print(
                f"[seed_interview_questions] Skipping — table already has "
                f"{existing_count} rows (threshold: 20)."
            )
            return

        inserted = 0
        for q in _QUESTIONS:
            # Idempotency guard: skip if identical question text already exists
            exists = (
                db.query(InterviewQuestion)
                .filter(InterviewQuestion.question_text == q["question_text"])
                .first()
            )
            if exists:
                continue

            row = InterviewQuestion(
                role=q["role"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                rubric=q["rubric"],
                tags=q.get("tags"),
            )
            db.add(row)
            inserted += 1

        db.commit()
        print(
            f"[seed_interview_questions] Inserted {inserted} interview questions "
            f"(total now: {db.query(InterviewQuestion).count()})."
        )
    except Exception as exc:
        db.rollback()
        print(f"[seed_interview_questions] ERROR — rolled back: {exc}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
