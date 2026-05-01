"""
skill_loader.py
===============
Backward-compatible wrappers around the DataRepository for loading skills.

These functions preserve the original API surface (load_skills, load_priority_skills)
so that any existing code importing them continues to work unchanged.

Internally, all data access is delegated to the Repository Pattern layer,
which serves cached JSON from memory instead of parsing CSV on every call.
"""

import logging
from typing import List, Set

from app.services.data_repository import get_repository

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORIGINAL API — preserved for backward compatibility
# ---------------------------------------------------------------------------

def load_skills() -> List[str]:
    """
    Return a flat, deduplicated, lowercase list of all known technical skills.

    Previously read from skills.csv on every call.
    Now delegates to LocalJsonRepository with LRU-cached JSON.
    """
    repo = get_repository()
    skills: List[str] = repo.get_all_skills()
    logger.debug("load_skills() returned %d skills from repository", len(skills))
    return skills


def load_priority_skills() -> Set[str]:
    """
    Return the set of high-priority skills that receive 5x weight in scoring.

    Previously read from priority_skills.csv via pandas on every call.
    Now delegates to LocalJsonRepository with LRU-cached JSON.
    """
    repo = get_repository()
    priority: Set[str] = repo.get_priority_skills()
    logger.debug("load_priority_skills() returned %d priority skills", len(priority))
    return priority