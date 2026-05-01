"""
data_repository.py
==================
Repository Pattern implementation for the ATS data access layer.

Provides an abstract interface (DataRepository) and a concrete LocalJsonRepository
implementation that loads structured JSON files into memory using functools.lru_cache.

Architecture:
    DataRepository (Abstract Base)
        └── LocalJsonRepository (JSON files + LRU cache)
        └── CloudDatabaseRepository (future — swap in without touching service code)

Usage:
    from app.services.data_repository import get_repository
    repo = get_repository()
    skills = repo.get_all_skills()
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve the absolute path to the data directory once at module level.
# This avoids path-resolution bugs when the working directory changes.
# ---------------------------------------------------------------------------
_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


# ===========================================================================
# Abstract Base — swap to any backend by implementing this interface
# ===========================================================================
class DataRepository(ABC):
    """
    Abstract contract for all data access.

    Every concrete repository (JSON files, PostgreSQL, Firebase, etc.)
    must implement these methods so the service layer remains agnostic.
    """

    @abstractmethod
    def get_skills_dictionary(self) -> Dict[str, List[str]]:
        """Return the full skills dictionary grouped by category."""
        ...

    @abstractmethod
    def get_all_skills(self) -> List[str]:
        """Return a flat, deduplicated list of every known skill (lowercase)."""
        ...

    @abstractmethod
    def get_synonyms_mapping(self) -> Dict[str, List[str]]:
        """Return canonical_skill → [variant1, variant2, …] mapping."""
        ...

    @abstractmethod
    def get_priority_skills(self) -> Set[str]:
        """Return the set of high-priority skills (lowercase)."""
        ...

    @abstractmethod
    def get_roles_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Return {role_key: {display_name, skills, certifications}}."""
        ...

    @abstractmethod
    def get_all_role_names(self) -> List[str]:
        """Return a list of all human-readable role display names."""
        ...

    @abstractmethod
    def get_role_requirements(self, role_name: str) -> List[str]:
        """Return combined skills + certifications for a given role."""
        ...


# ===========================================================================
# Concrete Implementation — Local JSON Files + LRU Cache
# ===========================================================================
class LocalJsonRepository(DataRepository):
    """
    Loads JSON data files from disk ONCE (via lru_cache) and serves them
    from memory for the lifetime of the process.

    File layout expected under app/data/:
        skills_dictionary.json
        synonyms_mapping.json
        priority_skills.json
        roles_dictionary.json
    """

    def __init__(self, data_dir: Path = _DATA_DIR) -> None:
        self._data_dir = data_dir
        logger.info("LocalJsonRepository initialized — data dir: %s", self._data_dir)

    # -------------------------------------------------------------------
    # Private: cached JSON loaders (class-level caching via module funcs)
    # -------------------------------------------------------------------

    def _load_json(self, filename: str) -> Any:
        """Load and parse a JSON file from the data directory."""
        filepath: Path = self._data_dir / filename
        if not filepath.exists():
            logger.error("Data file not found: %s", filepath)
            return {}
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -------------------------------------------------------------------
    # Public API — each backed by a module-level @lru_cache function
    # -------------------------------------------------------------------

    def get_skills_dictionary(self) -> Dict[str, List[str]]:
        """Return full category → skills mapping from skills_dictionary.json."""
        return _cached_skills_dictionary(str(self._data_dir))

    def get_all_skills(self) -> List[str]:
        """Flatten every category into a single deduplicated lowercase list."""
        return _cached_all_skills(str(self._data_dir))

    def get_synonyms_mapping(self) -> Dict[str, List[str]]:
        """Return canonical → [synonyms] mapping from synonyms_mapping.json."""
        return _cached_synonyms_mapping(str(self._data_dir))

    def get_priority_skills(self) -> Set[str]:
        """Return priority skill set from priority_skills.json."""
        return _cached_priority_skills(str(self._data_dir))

    def get_roles_dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Return roles dict keyed by lowercase role name."""
        return _cached_roles_dictionary(str(self._data_dir))

    def get_all_role_names(self) -> List[str]:
        """Return display names of all roles."""
        roles: Dict[str, Dict[str, Any]] = self.get_roles_dictionary()
        return [v["display_name"] for v in roles.values() if "display_name" in v]

    def get_role_requirements(self, role_name: str) -> List[str]:
        """
        Look up a role by name (case-insensitive) and return its combined
        skills + certifications as a flat list.  Returns [] if not found.
        """
        roles: Dict[str, Dict[str, Any]] = self.get_roles_dictionary()
        key: str = role_name.strip().lower()
        entry = roles.get(key)
        if entry is None:
            return []
        combined: List[str] = list(entry.get("skills", []))
        combined.extend(entry.get("certifications", []))
        return combined


# ===========================================================================
# Module-level @lru_cache loaders — hashable argument (str path) required
# ===========================================================================

@lru_cache(maxsize=1)
def _cached_skills_dictionary(data_dir: str) -> Dict[str, List[str]]:
    """Load and cache skills_dictionary.json."""
    filepath = os.path.join(data_dir, "skills_dictionary.json")
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data: Dict[str, List[str]] = json.load(fh)
            logger.info(
                "Loaded skills_dictionary.json — %d categories, %d total skills",
                len(data),
                sum(len(v) for v in data.values()),
            )
            return data
    except Exception as exc:
        logger.error("Failed to load skills_dictionary.json: %s", exc)
        return {}


@lru_cache(maxsize=1)
def _cached_all_skills(data_dir: str) -> List[str]:
    """Flatten the skills dictionary into a unique lowercase list."""
    skills_dict = _cached_skills_dictionary(data_dir)
    all_skills: Set[str] = set()
    for category_skills in skills_dict.values():
        for skill in category_skills:
            all_skills.add(skill.strip().lower())
    result = sorted(all_skills)
    logger.info("Flattened skills list: %d unique skills", len(result))
    return result


@lru_cache(maxsize=1)
def _cached_synonyms_mapping(data_dir: str) -> Dict[str, List[str]]:
    """Load and cache synonyms_mapping.json."""
    filepath = os.path.join(data_dir, "synonyms_mapping.json")
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data: Dict[str, List[str]] = json.load(fh)
            logger.info("Loaded synonyms_mapping.json — %d canonical entries", len(data))
            return data
    except Exception as exc:
        logger.error("Failed to load synonyms_mapping.json: %s", exc)
        return {}


@lru_cache(maxsize=1)
def _cached_priority_skills(data_dir: str) -> Set[str]:
    """Load and cache priority_skills.json as a frozen set."""
    filepath = os.path.join(data_dir, "priority_skills.json")
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data: List[str] = json.load(fh)
            result = frozenset(s.strip().lower() for s in data)
            logger.info("Loaded priority_skills.json — %d priority skills", len(result))
            # Return as a regular set for downstream compatibility
            return set(result)
    except Exception as exc:
        logger.error("Failed to load priority_skills.json: %s", exc)
        return set()


@lru_cache(maxsize=1)
def _cached_roles_dictionary(data_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load and cache roles_dictionary.json."""
    filepath = os.path.join(data_dir, "roles_dictionary.json")
    try:
        with open(filepath, "r", encoding="utf-8") as fh:
            data: Dict[str, Dict[str, Any]] = json.load(fh)
            logger.info("Loaded roles_dictionary.json — %d roles", len(data))
            return data
    except Exception as exc:
        logger.error("Failed to load roles_dictionary.json: %s", exc)
        return {}


# ===========================================================================
# Singleton Factory — service layer imports this
# ===========================================================================

# Module-level singleton — instantiated once on first import
_repository_instance: DataRepository | None = None


def get_repository() -> DataRepository:
    """
    Return the singleton DataRepository instance.

    Currently returns LocalJsonRepository. To switch to a cloud database,
    change this function to return a CloudDatabaseRepository instead.
    """
    global _repository_instance
    if _repository_instance is None:
        _repository_instance = LocalJsonRepository()
    return _repository_instance
