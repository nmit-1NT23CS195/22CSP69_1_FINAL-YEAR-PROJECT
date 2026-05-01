"""
nlp_service.py
==============
Core NLP extraction engine for the ATS pipeline.

Key upgrades from the original:
    1. FlashText KeywordProcessor replaces the O(M×N) regex loop with
       an O(N) single-pass Aho-Corasick trie scan for skill extraction.
    2. Synonym resolution is built into the trie — variants like "nodejs",
       "node", "node.js" all resolve to the canonical "node.js".
    3. Section-aware contextual weighting: skills found in Experience/Projects
       sections receive a 2x score multiplier vs. skills in a raw Skills list.
    4. Soft skills and action verbs also use FlashText for consistent O(N) perf.

All original function signatures are preserved for backward compatibility.
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from flashtext import KeywordProcessor

from app.services.data_repository import get_repository

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ===========================================================================
# FlashText Trie Builders (cached — built once per process lifetime)
# ===========================================================================

@lru_cache(maxsize=1)
def _build_skills_processor() -> KeywordProcessor:
    """
    Build and cache a FlashText KeywordProcessor trie containing:
        - Every skill from skills_dictionary.json (canonical → canonical)
        - Every synonym from synonyms_mapping.json (variant → canonical)

    FlashText is case-insensitive by default, which eliminates the need
    for manual .lower() normalization before matching.

    Returns:
        A ready-to-use KeywordProcessor instance.
    """
    repo = get_repository()
    processor = KeywordProcessor(case_sensitive=False)

    # ------------------------------------------------------------------
    # 1. Add all canonical skills (the skill maps to itself)
    # ------------------------------------------------------------------
    all_skills: List[str] = repo.get_all_skills()
    for skill in all_skills:
        # add_keyword(keyword_to_search, clean_name_to_return)
        processor.add_keyword(skill.lower(), skill.lower())

    # ------------------------------------------------------------------
    # 2. Add synonym variants (each variant maps to its canonical form)
    # ------------------------------------------------------------------
    synonyms: Dict[str, List[str]] = repo.get_synonyms_mapping()
    for canonical, variants in synonyms.items():
        canonical_lower = canonical.lower()
        # Ensure the canonical skill itself is in the trie
        processor.add_keyword(canonical_lower, canonical_lower)
        for variant in variants:
            processor.add_keyword(variant.lower(), canonical_lower)

    logger.info(
        "FlashText skills processor built — %d total keywords in trie",
        len(processor),
    )
    return processor


@lru_cache(maxsize=1)
def _build_soft_skills_processor() -> KeywordProcessor:
    """
    Build a FlashText trie for industry-standard soft skills.
    Returns title-cased clean names for display.
    """
    processor = KeywordProcessor(case_sensitive=False)

    # Industry-standard soft skills dictionary
    soft_skills_db: List[str] = [
        "leadership", "communication", "teamwork", "problem solving",
        "critical thinking", "adaptability", "time management", "conflict resolution",
        "agile", "scrum", "mentoring", "collaboration", "creativity", "innovation",
        "analytical", "attention to detail", "decision making", "presentation",
        "public speaking", "negotiation", "empathy", "project management",
    ]

    for skill in soft_skills_db:
        # Map lowercase keyword → Title Cased display name
        processor.add_keyword(skill.lower(), skill.title())

    logger.info("FlashText soft skills processor built — %d keywords", len(processor))
    return processor


@lru_cache(maxsize=1)
def _build_action_verbs_processor() -> KeywordProcessor:
    """
    Build a FlashText trie for high-impact action verbs.
    Returns title-cased clean names for display.
    """
    processor = KeywordProcessor(case_sensitive=False)

    # High-impact action verbs
    action_verbs_db: List[str] = [
        "architected", "spearheaded", "engineered", "developed", "managed",
        "orchestrated", "implemented", "designed", "led", "optimized",
        "streamlined", "transformed", "modernized", "pioneered", "launched",
        "executed", "directed", "formulated", "conceptualized", "mentored",
        "delivered", "resolved", "automated", "maximized", "accelerated",
        "reduced", "increased", "improved", "negotiated", "secured",
    ]

    for verb in action_verbs_db:
        processor.add_keyword(verb.lower(), verb.title())

    logger.info("FlashText action verbs processor built — %d keywords", len(processor))
    return processor


# ===========================================================================
# Core Extraction Functions
# ===========================================================================

def extract_skills(
    text: str,
    custom_skills: Optional[List[str]] = None,
    sections: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Extract technical skills from text using FlashText O(N) trie matching.

    This replaces the original O(M×N) regex loop. FlashText performs a
    single pass over the text, matching against the entire skills dictionary
    and synonym mapping simultaneously.

    Args:
        text:          The raw text to extract skills from.
        custom_skills: Optional list of additional skills to search for
                       (e.g., role-specific JD requirements). These are
                       added to a copy of the processor for this call.
        sections:      Optional dict of resume sections (from parse_resume_sections).
                       If provided, enables contextual weighting.

    Returns:
        A deduplicated list of matched skill names (lowercase canonical forms).
    """
    if not text or not text.strip():
        return []

    # ------------------------------------------------------------------
    # Get the base processor (cached trie with all skills + synonyms)
    # ------------------------------------------------------------------
    base_processor: KeywordProcessor = _build_skills_processor()

    # ------------------------------------------------------------------
    # If custom_skills are provided, create a temporary processor copy
    # with the additional keywords injected.
    # ------------------------------------------------------------------
    if custom_skills:
        processor = KeywordProcessor(case_sensitive=False)
        # Copy all existing keywords from base processor by re-adding them
        # (FlashText doesn't support deep copy, so we rebuild)
        repo = get_repository()
        all_skills = repo.get_all_skills()
        synonyms = repo.get_synonyms_mapping()

        for skill in all_skills:
            processor.add_keyword(skill.lower(), skill.lower())
        for canonical, variants in synonyms.items():
            canonical_lower = canonical.lower()
            processor.add_keyword(canonical_lower, canonical_lower)
            for variant in variants:
                processor.add_keyword(variant.lower(), canonical_lower)

        # Inject custom skills
        for cs in custom_skills:
            cs_clean = cs.strip().lower()
            if cs_clean:
                processor.add_keyword(cs_clean, cs_clean)
    else:
        processor = base_processor

    # ------------------------------------------------------------------
    # Single-pass extraction
    # ------------------------------------------------------------------
    extracted: List[str] = processor.extract_keywords(text.lower())

    # Deduplicate while preserving canonical forms
    unique_skills: List[str] = list(set(extracted))

    logger.debug(
        "extract_skills: found %d unique skills in %d chars of text",
        len(unique_skills), len(text),
    )

    return unique_skills


def extract_skills_with_context(
    text: str,
    sections: Dict[str, str],
    custom_skills: Optional[List[str]] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract skills with contextual weighting based on resume sections.

    Skills found in "experience" or "projects" sections receive a 2x multiplier.
    Skills found only in "skills" section receive a 1x multiplier.

    Args:
        text:          Full resume text (used for basic extraction fallback).
        sections:      Dict of section_name → section_text from parse_resume_sections.
        custom_skills: Optional additional skills to search for.

    Returns:
        Tuple of:
            - List of all unique skills found (lowercase canonical)
            - Dict mapping skill → contextual weight multiplier (1.0 or 2.0)
    """
    if not sections:
        # Fallback: no sections available, extract from full text with 1x weight
        skills = extract_skills(text, custom_skills=custom_skills)
        weights = {s: 1.0 for s in skills}
        return skills, weights

    # ------------------------------------------------------------------
    # Section-aware extraction
    # ------------------------------------------------------------------
    # High-value sections: skills here get 2x multiplier
    high_value_sections = {"experience", "projects"}
    # Standard sections: skills here get 1x multiplier
    standard_sections = {"skills", "education", "other", "summary"}

    all_skills: Set[str] = set()
    contextual_weights: Dict[str, float] = {}

    for section_name, section_text in sections.items():
        if not section_text or not section_text.strip():
            continue

        # Extract skills from this section
        section_skills = extract_skills(section_text, custom_skills=custom_skills)

        # Determine the multiplier for this section
        section_key = section_name.lower().strip()
        if section_key in high_value_sections:
            multiplier = 2.0
        else:
            multiplier = 1.0

        for skill in section_skills:
            all_skills.add(skill)
            # Use the HIGHEST multiplier if a skill appears in multiple sections
            if skill not in contextual_weights or multiplier > contextual_weights[skill]:
                contextual_weights[skill] = multiplier

    logger.info(
        "Contextual extraction: %d skills, %d with 2x weight",
        len(all_skills),
        sum(1 for w in contextual_weights.values() if w == 2.0),
    )

    return list(all_skills), contextual_weights


# ===========================================================================
# Advanced NLP Extractions (preserved API)
# ===========================================================================

def extract_soft_skills(text: str) -> List[str]:
    """
    Extract soft skills from text using FlashText O(N) trie matching.

    Returns a deduplicated list of Title Cased soft skill names.
    """
    if not text or not text.strip():
        return []

    processor: KeywordProcessor = _build_soft_skills_processor()
    extracted: List[str] = processor.extract_keywords(text.lower())

    unique_skills: List[str] = list(set(extracted))
    logger.debug("extract_soft_skills: found %d soft skills", len(unique_skills))
    return unique_skills


def extract_action_verbs(text: str) -> List[str]:
    """
    Extract high-impact action verbs from text using FlashText O(N) trie matching.

    Returns a deduplicated list of Title Cased action verb names.
    """
    if not text or not text.strip():
        return []

    processor: KeywordProcessor = _build_action_verbs_processor()
    extracted: List[str] = processor.extract_keywords(text.lower())

    unique_verbs: List[str] = list(set(extracted))
    logger.debug("extract_action_verbs: found %d action verbs", len(unique_verbs))
    return unique_verbs