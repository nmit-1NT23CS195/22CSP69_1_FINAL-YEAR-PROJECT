import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import roles
from app.api.routes import resume, skills, ats
from app.api.routes import mcq_router
from app.db.database import engine
from app.db import models  # noqa: F401 — ensures ORM models are registered before create_all

# ---------------------------------------------------------------------------
# Global ROLES_DB — loaded ONCE at startup from the enriched dictionary.
# ats_service.py imports this to avoid per-request disk I/O.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent / "data"
_ROLES_DB_PATH = _DATA_DIR / "enriched_roles_dictionary.json"

try:
    with open(_ROLES_DB_PATH, "r", encoding="utf-8") as _fh:
        ROLES_DB: dict = json.load(_fh)
except Exception as _e:
    import logging
    logging.getLogger(__name__).error("Failed to load ROLES_DB: %s", _e)
    ROLES_DB: dict = {}

app = FastAPI(
    title="PlaceBuddy ATS System",
    description="Intelligent Applicant Tracking System backend for the PlaceBuddy placement readiness platform.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow the React/Vite dev server to call the API from the browser
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",      # Vite default
        "http://localhost:5174",      # Vite fallback (port in use)
        "http://localhost:5175",
        "http://localhost:5176",
        "http://localhost:5177",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        "http://127.0.0.1:5177",
        "http://localhost:3000",      # CRA / Next.js default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Database — create all tables on startup (idempotent: skipped if they exist)
# ---------------------------------------------------------------------------
models.Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(resume.router, prefix="/resume", tags=["Resume"])
app.include_router(skills.router, prefix="/skills", tags=["Skills"])
app.include_router(ats.router, prefix="/ats", tags=["ATS Score"])
app.include_router(roles.router)
app.include_router(mcq_router.router, prefix="/api/mcq", tags=["MCQ Engine"])