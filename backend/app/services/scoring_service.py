"""
scoring_service.py
==================
Multi-signal weighted scoring engine for the ATS pipeline.

Upgrades from the original:
    1. Stop-word filtering: compute_keyword_score now uses SpaCy's stop-word
       list to strip common English stop words before set intersection,
       eliminating false matches on "the", "and", "is", etc.
    2. Goldilocks quality scoring: penalizes both too-short AND too-long resumes.
    3. Contextual weighting: skills found in Experience/Projects sections
       receive 2x score multiplier in the technical score calculation.
    4. New weight distribution (40/15/10/10/10/5/10) incorporating
       semantic similarity and TF-IDF relevance scores.

All original function signatures are preserved for backward compatibility.
New functions accept additional optional parameters.
"""

import logging
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SpaCy Stop Words — loaded lazily, fallback to basic set if unavailable
# ---------------------------------------------------------------------------
_STOP_WORDS: Optional[Set[str]] = None


def _get_stop_words() -> Set[str]:
    """
    Load SpaCy's English stop-word list.
    Falls back to a basic hardcoded set if SpaCy is unavailable.

    Returns:
        A set of lowercase stop words.
    """
    global _STOP_WORDS

    if _STOP_WORDS is not None:
        return _STOP_WORDS

    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Try auto-download
            try:
                from spacy.cli import download
                download("en_core_web_sm")
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                raise ImportError("SpaCy model unavailable")

        _STOP_WORDS = nlp.Defaults.stop_words
        logger.info("Loaded %d SpaCy stop words", len(_STOP_WORDS))
        return _STOP_WORDS

    except (ImportError, Exception) as exc:
        logger.warning(
            "SpaCy stop words unavailable (%s) — using fallback set", exc
        )
        # Fallback: basic English stop words
        _STOP_WORDS = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "can", "need", "must", "it", "its", "this", "that", "these",
            "those", "i", "me", "my", "we", "our", "you", "your", "he",
            "him", "his", "she", "her", "they", "them", "their", "what",
            "which", "who", "whom", "how", "when", "where", "why", "not",
            "no", "nor", "so", "if", "then", "than", "too", "very", "just",
            "about", "above", "after", "again", "all", "also", "any", "as",
            "because", "before", "between", "both", "each", "few", "here",
            "into", "more", "most", "other", "out", "over", "own", "same",
            "some", "such", "through", "under", "until", "up", "while",
        }
        return _STOP_WORDS


# ===========================================================================
# Weight Function (preserved)
# ===========================================================================

def get_weight(skill: str, priority_skills: Set[str]) -> int:
    """
    Return the weight multiplier for a skill.

    Priority skills receive 5x weight; all others receive 1x.

    Args:
        skill:           The skill name (lowercase).
        priority_skills: Set of high-priority skill names.

    Returns:
        Integer weight: 5 for priority skills, 1 for standard.
    """
    return 5 if skill in priority_skills else 1


# ===========================================================================
# Weighted Skill Score (enhanced with contextual weights)
# ===========================================================================

def compute_weighted_skill_score(
    required_skills: List[str],
    resume_skills: List[str],
    priority_skills: Set[str],
    contextual_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate a weighted skill match score.

    Each required skill has a base weight (5x for priority, 1x for standard).
    If contextual_weights are provided (from section-aware extraction),
    matched skills are further multiplied by their contextual weight
    (2x for Experience/Projects, 1x for Skills section).

    Args:
        required_skills:    List of skills required by the JD.
        resume_skills:      List of skills found in the resume.
        priority_skills:    Set of high-priority skills.
        contextual_weights: Optional dict of skill → contextual multiplier.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not required_skills:
        return 0.0

    required_set: Set[str] = set(s.strip().lower() for s in required_skills if s)
    resume_set: Set[str] = set(s.strip().lower() for s in resume_skills if s)

    total_weight: float = 0.0
    matched_weight: float = 0.0

    for skill in required_set:
        base_weight: int = get_weight(skill, priority_skills)

        # The maximum possible weight includes contextual bonus
        # (assume 2x as max contextual multiplier for normalization)
        max_contextual = 2.0 if contextual_weights else 1.0
        total_weight += base_weight * max_contextual

        if skill in resume_set:
            # Apply contextual multiplier if available
            ctx_multiplier: float = 1.0
            if contextual_weights and skill in contextual_weights:
                ctx_multiplier = contextual_weights[skill]

            matched_weight += base_weight * ctx_multiplier

    return matched_weight / total_weight if total_weight > 0 else 0.0


# ===========================================================================
# Backward Compatibility Wrapper
# ===========================================================================

def compute_ats_score(required_skills: List[str], resume_skills: List[str]) -> float:
    """
    Old function wrapper (used by existing APIs).
    Now internally calls weighted logic with default weights.

    Args:
        required_skills: Skills required by the JD.
        resume_skills:   Skills found in the resume.

    Returns:
        Float score in [0.0, 1.0].
    """
    required_set: Set[str] = set(s.strip().lower() for s in required_skills if s)
    resume_set: Set[str] = set(s.strip().lower() for s in resume_skills if s)

    if not required_set:
        return 0.0

    matched: int = len(required_set & resume_set)
    return matched / len(required_set)


# ===========================================================================
# Keyword Score (FIXED: stop-word filtering)
# ===========================================================================

def compute_keyword_score(resume_text: str, jd_text: str) -> float:
    """
    Calculate keyword overlap between resume and JD text.

    FIX: Now filters out common English stop words before calculating
    set intersection, preventing false matches on words like
    "the", "and", "is", "for", etc.

    Args:
        resume_text: Raw resume text.
        jd_text:     Raw job description text.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not jd_text or not jd_text.strip():
        return 0.0
    if not resume_text or not resume_text.strip():
        return 0.0

    stop_words: Set[str] = _get_stop_words()

    # Tokenize and filter stop words
    resume_words: Set[str] = {
        word.lower() for word in resume_text.split()
        if word.lower() not in stop_words and len(word) > 1
    }
    jd_words: Set[str] = {
        word.lower() for word in jd_text.split()
        if word.lower() not in stop_words and len(word) > 1
    }

    if not jd_words:
        return 0.0

    overlap: int = len(resume_words & jd_words)
    score: float = overlap / len(jd_words)

    logger.debug(
        "Keyword score: %d/%d JD words matched (%.3f) — %d stop words filtered",
        overlap, len(jd_words), score,
        len(set(jd_text.lower().split())) - len(jd_words),
    )

    return score


# ===========================================================================
# Quality Score (UPGRADED: Goldilocks curve)
# ===========================================================================

def compute_quality_score(resume_text: str) -> float:
    """
    Calculate a quality score based on resume length.

    Implements the "Goldilocks" scoring curve — real ATS systems penalize
    both too-short (sparse) AND too-long (text walls) resumes.

    Scoring table:
        < 200 words   → 0.4 (too sparse)
        200–400 words → 0.8 (acceptable)
        400–800 words → 1.0 (optimal — the sweet spot)
        800–1000 words → 0.7 (getting verbose)
        > 1000 words  → 0.4 (text wall — likely un-tailored)

    Args:
        resume_text: Raw resume text.

    Returns:
        Float score in [0.0, 1.0].
    """
    if not resume_text or not resume_text.strip():
        return 0.0

    word_count: int = len(resume_text.split())

    if word_count < 200:
        score = 0.4
    elif word_count <= 400:
        score = 0.8
    elif word_count <= 800:
        score = 1.0
    elif word_count <= 1000:
        score = 0.7
    else:
        score = 0.4

    logger.debug("Quality score: %d words → %.1f", word_count, score)
    return score


# ===========================================================================
# Smart ATS Score (UPGRADED: new weight distribution)
# ===========================================================================

def compute_smart_ats_score(
    jd_text: str,
    resume_text: str,
    jd_skills: List[str],
    resume_skills: List[str],
    priority_skills: Set[str],
    soft_skills: Optional[List[str]] = None,
    action_verbs: Optional[List[str]] = None,
    semantic_score: float = 0.0,
    tfidf_score: float = 0.0,
    contextual_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute the composite ATS score using multiple weighted signals.

    NEW weight distribution (approved):
        Technical Skill Score (weighted + contextual) = 40%
        Semantic Similarity                          = 15%
        TF-IDF Relevance                             = 10%
        Soft Skills                                  = 10%
        Action Verbs                                 = 10%
        Keyword Match (stop-word filtered)           =  5%
        Quality Score (Goldilocks)                   = 10%
                                                TOTAL = 100%

    All original return keys are preserved. New keys are ADDED to the
    breakdown dict for downstream ML feature engineering.

    Args:
        jd_text:             Raw job description text.
        resume_text:         Raw resume text.
        jd_skills:           Skills extracted from JD.
        resume_skills:       Skills extracted from resume.
        priority_skills:     Set of high-priority skills.
        soft_skills:         List of soft skills found (optional).
        action_verbs:        List of action verbs found (optional).
        semantic_score:      Semantic similarity score [0.0–1.0] (optional).
        tfidf_score:         TF-IDF relevance score [0.0–1.0] (optional).
        contextual_weights:  Skill → contextual weight mapping (optional).

    Returns:
        Dict with "overall_score" and detailed "breakdown" of all components.
    """
    soft_skills = soft_skills or []
    action_verbs = action_verbs or []

    # ------------------------------------------------------------------
    # 1. Technical Skill Score (weighted + contextual) — 40%
    # ------------------------------------------------------------------
    technical_score_raw: float = compute_weighted_skill_score(
        jd_skills, resume_skills, priority_skills, contextual_weights
    )
    technical_score: float = technical_score_raw * 100

    # ------------------------------------------------------------------
    # 2. Semantic Similarity Score — 15%
    # ------------------------------------------------------------------
    semantic_display: float = semantic_score * 100

    # ------------------------------------------------------------------
    # 3. TF-IDF Relevance Score — 10%
    # ------------------------------------------------------------------
    tfidf_display: float = tfidf_score * 100

    # ------------------------------------------------------------------
    # 4. Soft Skills Score — 10%
    # Expecting at least 3 soft skills for a perfect score
    # ------------------------------------------------------------------
    soft_skills_score: float = min(len(soft_skills) / 3.0, 1.0) * 100

    # ------------------------------------------------------------------
    # 5. Action Verbs Score — 10%
    # Expecting at least 5 action verbs for a perfect score
    # ------------------------------------------------------------------
    action_verbs_score: float = min(len(action_verbs) / 5.0, 1.0) * 100

    # ------------------------------------------------------------------
    # 6. Keyword Match Score (stop-word filtered) — 5%
    # ------------------------------------------------------------------
    keyword_score: float = compute_keyword_score(resume_text, jd_text) * 100

    # ------------------------------------------------------------------
    # 7. Quality Score (Goldilocks) — 10%
    # ------------------------------------------------------------------
    quality_score: float = compute_quality_score(resume_text) * 100

    # ------------------------------------------------------------------
    # Combine scores with approved weight distribution
    # ------------------------------------------------------------------
    final_score: float = (
        (technical_score * 0.40) +
        (semantic_display * 0.15) +
        (tfidf_display * 0.10) +
        (soft_skills_score * 0.10) +
        (action_verbs_score * 0.10) +
        (keyword_score * 0.05) +
        (quality_score * 0.10)
    )

    logger.info(
        "Smart ATS Score: %.2f | Tech=%.1f Sem=%.1f TFIDF=%.1f "
        "Soft=%.1f Verbs=%.1f KW=%.1f Qual=%.1f",
        final_score, technical_score, semantic_display, tfidf_display,
        soft_skills_score, action_verbs_score, keyword_score, quality_score,
    )

    return {
        "overall_score": round(final_score, 2),
        "breakdown": {
            # Original keys (preserved for backward compatibility)
            "technical_score": round(technical_score, 2),
            "soft_skills_score": round(soft_skills_score, 2),
            "action_verbs_score": round(action_verbs_score, 2),
            "keyword_score": round(keyword_score, 2),
            "quality_score": round(quality_score, 2),
            # New keys (additive — no existing keys removed)
            "semantic_similarity_score": round(semantic_display, 2),
            "tfidf_relevance_score": round(tfidf_display, 2),
            # Weight distribution metadata for transparency
            "weights": {
                "technical": 0.40,
                "semantic_similarity": 0.15,
                "tfidf_relevance": 0.10,
                "soft_skills": 0.10,
                "action_verbs": 0.10,
                "keyword_match": 0.05,
                "quality": 0.10,
            },
        },
    }
