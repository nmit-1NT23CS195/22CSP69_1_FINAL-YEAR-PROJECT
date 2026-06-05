"""
role_service.py
===============
Service for looking up job roles and their required skills/certifications.

Refactored from CSV-based pandas reads to the Repository Pattern data layer.
Retains rapidfuzz for fuzzy role-name matching when exact match fails.
"""

import logging
from typing import List

from rapidfuzz import process, fuzz

from app.services.data_repository import get_repository

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GET ALL ROLES (USED BY API)
# ---------------------------------------------------------------------------

def get_all_roles() -> List[str]:
    """
    Return a list of all known job role display names.

    Previously loaded from CSV via pandas on every call.
    Now served from LRU-cached JSON via the repository.
    """
    repo = get_repository()
    roles: List[str] = repo.get_all_role_names()
    logger.debug("get_all_roles() returned %d roles", len(roles))
    return roles


# ---------------------------------------------------------------------------
# GET ROLE REQUIREMENTS (USED IN ATS PIPELINE)
# ---------------------------------------------------------------------------

def get_role_requirements(role_name: str) -> List[str]:
    """
    Return a combined list of skills and certifications required for a given role.

    Matching strategy:
        1. Exact match (case-insensitive) against the roles dictionary.
        2. If no exact match, use rapidfuzz fuzzy matching (threshold ≥ 70).

    Args:
        role_name: The job role title to look up.

    Returns:
        A list of skill and certification strings. Empty list if no match.
    """
    repo = get_repository()

    # ------------------------------------------------------------------
    # 1. Attempt exact match via repository
    # ------------------------------------------------------------------
    requirements: List[str] = repo.get_role_requirements(role_name)

    if requirements:
        logger.info("Exact match found for role '%s' — %d requirements", role_name, len(requirements))
        return requirements

    # ------------------------------------------------------------------
    # 2. Fuzzy match fallback using rapidfuzz
    # ------------------------------------------------------------------
    all_display_names: List[str] = repo.get_all_role_names()

    if not all_display_names:
        logger.warning("No roles available for fuzzy matching")
        return []

    match = process.extractOne(role_name, all_display_names, scorer=fuzz.WRatio)

    if match and match[1] >= 70:
        matched_role: str = match[0]
        logger.info(
            "Fuzzy match: '%s' → '%s' (score: %.1f)",
            role_name, matched_role, match[1],
        )
        # Use the matched display name to look up requirements
        return repo.get_role_requirements(matched_role)

    logger.warning("No match found for role '%s' (best fuzzy score: %.1f)", role_name, match[1] if match else 0)
    return []