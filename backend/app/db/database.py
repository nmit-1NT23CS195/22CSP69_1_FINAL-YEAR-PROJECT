"""
database.py
===========
SQLAlchemy engine + session factory for the PlaceBuddy PostgreSQL database.

The DATABASE_URL is read from the .env file located at:
    backend/app/.env

Connection string format:
    postgresql://user:password@host:port/dbname

Thread-safe session management:
    - SessionLocal is a factory that creates independent DB sessions.
    - get_db() is a FastAPI dependency that yields one session per request
      and guarantees the session is closed even if an exception is raised.
"""

import os
import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load DATABASE_URL
# We read it manually from .env so the module works with or without
# python-dotenv being imported, and without changing the existing startup flow.
# ---------------------------------------------------------------------------
def _read_env_var(key: str) -> str | None:
    """Read a variable from the .env next to this file, or from os.environ."""
    value = os.environ.get(key)
    if value:
        return value

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1]
    return None


DATABASE_URL: str = _read_env_var("DATABASE_URL") or ""

if not DATABASE_URL:
    logger.error(
        "DATABASE_URL is not set. Add it to backend/app/.env.\n"
        "Example: DATABASE_URL=postgresql://postgres:password@localhost:5432/placebuddy_db"
    )

# ---------------------------------------------------------------------------
# SQLAlchemy Engine
# ---------------------------------------------------------------------------
# pool_pre_ping=True  — detects stale connections and recycles them before use,
#                       which avoids "server closed the connection unexpectedly"
#                       errors that occur after long idle periods.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,          # Set to True to log all SQL statements (useful for debugging)
)

# ---------------------------------------------------------------------------
# Session factory
# autocommit=False — we commit manually; autoflush=False — we flush manually
# ---------------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ---------------------------------------------------------------------------
# Declarative base — all ORM models inherit from this
# ---------------------------------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------------------------------
# FastAPI dependency — yields one DB session per request
# ---------------------------------------------------------------------------
def get_db():
    """
    Dependency for FastAPI route handlers.

    Usage:
        from app.db.database import get_db
        from sqlalchemy.orm import Session
        from fastapi import Depends

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
