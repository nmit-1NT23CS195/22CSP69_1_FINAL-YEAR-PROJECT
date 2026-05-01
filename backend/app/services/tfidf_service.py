"""
tfidf_service.py
================
TF-IDF based keyword relevance scoring between Job Descriptions and Resumes.

Uses scikit-learn's TfidfVectorizer to:
    1. Compare the JD and Resume as a 2-document corpus.
    2. Identify highly unique, relevant keywords that distinguish the JD.
    3. Score how many of these high-value JD terms appear in the resume.
    4. Return both the relevance score and the top keyword details.

This catches important terms that basic keyword matching misses because
TF-IDF weights rare, domain-specific terms higher than generic jargon.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def compute_tfidf_analysis(
    jd_text: str,
    resume_text: str,
    top_n: int = 15,
) -> Dict[str, Any]:
    """
    Perform TF-IDF analysis comparing a Job Description against a Resume.

    Algorithm:
        1. Fit a TF-IDF vectorizer on the 2-document corpus [JD, Resume].
        2. Extract feature names and their TF-IDF weights for the JD.
        3. Rank JD terms by TF-IDF weight (high weight = rare, important term).
        4. Check which of these top JD terms also appear in the resume.
        5. Calculate a relevance score based on overlap of top terms.

    Args:
        jd_text:     Raw job description text.
        resume_text: Raw resume text.
        top_n:       Number of top TF-IDF keywords to analyze (default 15).

    Returns:
        Dict containing:
            - "tfidf_score":     Float [0.0–1.0] relevance score.
            - "top_jd_keywords": List of dicts with keyword details:
                [{"keyword": str, "tfidf_weight": float, "found_in_resume": bool}, ...]
            - "matched_count":   Int count of top JD keywords found in resume.
            - "total_analyzed":  Int count of keywords analyzed.
    """
    # ------------------------------------------------------------------
    # Guard against empty inputs
    # ------------------------------------------------------------------
    if not jd_text or not jd_text.strip() or not resume_text or not resume_text.strip():
        logger.warning("TF-IDF analysis skipped — empty JD or resume text")
        return {
            "tfidf_score": 0.0,
            "top_jd_keywords": [],
            "matched_count": 0,
            "total_analyzed": 0,
        }

    try:
        # ------------------------------------------------------------------
        # 1. Build TF-IDF matrix on the 2-document corpus
        # ------------------------------------------------------------------
        vectorizer = TfidfVectorizer(
            stop_words="english",       # Remove common stop words
            max_features=5000,          # Limit vocabulary size
            ngram_range=(1, 2),         # Unigrams + bigrams for phrases
            min_df=1,                   # Include terms appearing in at least 1 doc
            sublinear_tf=True,          # Apply log normalization to TF
        )

        # Corpus: [JD, Resume]
        corpus: List[str] = [jd_text.lower(), resume_text.lower()]
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # ------------------------------------------------------------------
        # 2. Extract feature names and JD term weights
        # ------------------------------------------------------------------
        feature_names: List[str] = vectorizer.get_feature_names_out().tolist()
        jd_vector = tfidf_matrix[0].toarray().flatten()  # JD is document 0
        resume_vector = tfidf_matrix[1].toarray().flatten()  # Resume is document 1

        # ------------------------------------------------------------------
        # 3. Rank JD terms by TF-IDF weight (descending)
        # ------------------------------------------------------------------
        jd_term_scores: List[Tuple[str, float]] = [
            (feature_names[i], float(jd_vector[i]))
            for i in range(len(feature_names))
            if jd_vector[i] > 0  # Only terms that actually appear in JD
        ]
        jd_term_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top N keywords
        top_keywords = jd_term_scores[:top_n]

        # ------------------------------------------------------------------
        # 4. Check which top JD keywords appear in the resume
        # ------------------------------------------------------------------
        resume_words_set = set(resume_text.lower().split())
        top_jd_keywords: List[Dict[str, Any]] = []
        matched_count: int = 0

        for keyword, weight in top_keywords:
            # Check if the keyword (or any word in a bigram) appears in resume
            keyword_words = set(keyword.split())
            # Explicit bool() cast to avoid numpy.bool_ serialization issues
            found: bool = bool(
                bool(keyword_words & resume_words_set)
                or float(resume_vector[feature_names.index(keyword)]) > 0
            )

            if found:
                matched_count += 1

            top_jd_keywords.append({
                "keyword": keyword,
                "tfidf_weight": round(float(weight), 4),
                "found_in_resume": found,
            })

        # ------------------------------------------------------------------
        # 5. Calculate TF-IDF relevance score
        # ------------------------------------------------------------------
        total_analyzed: int = len(top_keywords)
        tfidf_score: float = matched_count / total_analyzed if total_analyzed > 0 else 0.0

        # Also calculate cosine similarity between the full JD and Resume vectors
        cosine_sim: float = float(
            cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        )

        # Blend: 60% top-keyword overlap + 40% full-vector cosine similarity
        blended_score: float = (tfidf_score * 0.6) + (cosine_sim * 0.4)

        logger.info(
            "TF-IDF analysis: %d/%d top keywords matched, "
            "keyword_overlap=%.3f, cosine_sim=%.3f, blended=%.3f",
            matched_count, total_analyzed, tfidf_score, cosine_sim, blended_score,
        )

        return {
            "tfidf_score": round(blended_score, 4),
            "top_jd_keywords": top_jd_keywords,
            "matched_count": matched_count,
            "total_analyzed": total_analyzed,
            "cosine_similarity": round(cosine_sim, 4),
        }

    except Exception as exc:
        logger.error("TF-IDF analysis failed: %s", exc, exc_info=True)
        return {
            "tfidf_score": 0.0,
            "top_jd_keywords": [],
            "matched_count": 0,
            "total_analyzed": 0,
        }
