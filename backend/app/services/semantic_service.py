"""
semantic_service.py
===================
Semantic vector similarity service using sentence-transformers.

Loads the `all-MiniLM-L6-v2` model (~80MB) LAZILY on first use to protect
FastAPI startup times. If the model fails to load (missing dependency,
out of memory, etc.), gracefully degrades by returning 0.0 instead of crashing.

The cosine similarity between JD and Resume embeddings captures semantic
meaning that pure keyword matching misses:
    - "machine learning" ↔ "ML engineering" → high similarity
    - "REST API development" ↔ "building web services" → moderate similarity
"""

import logging
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------
_model: Optional[object] = None
_model_load_attempted: bool = False
_MODEL_NAME: str = "all-MiniLM-L6-v2"


def _get_model():
    """
    Lazily load the sentence-transformers model on first call.

    Uses a module-level singleton to avoid reloading the model on every request.
    If loading fails, sets a flag so we don't retry on subsequent calls.

    Returns:
        The loaded SentenceTransformer model, or None if unavailable.
    """
    global _model, _model_load_attempted

    # If we already loaded (or tried and failed), return cached result
    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    _model_load_attempted = True

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers model '%s'...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("Model '%s' loaded successfully", _MODEL_NAME)
        return _model

    except ImportError:
        logger.warning(
            "sentence-transformers not installed — semantic similarity DISABLED. "
            "Install with: pip install sentence-transformers"
        )
        return None

    except Exception as exc:
        logger.error(
            "Failed to load sentence-transformers model '%s': %s. "
            "Semantic similarity will be disabled for this session.",
            _MODEL_NAME, exc,
        )
        return None


def compute_semantic_similarity(jd_text: str, resume_text: str) -> float:
    """
    Compute cosine similarity between JD and Resume text embeddings.

    Uses the all-MiniLM-L6-v2 model to convert both texts into 384-dimensional
    dense vector embeddings, then calculates cosine similarity.

    Graceful degradation: returns 0.0 if:
        - sentence-transformers is not installed
        - The model fails to load
        - Either input text is empty

    Args:
        jd_text:     Raw job description text.
        resume_text: Raw resume text.

    Returns:
        Float in [0.0, 1.0] representing semantic similarity.
        0.0 indicates no similarity or model unavailable.
    """
    # ------------------------------------------------------------------
    # Guard against empty inputs
    # ------------------------------------------------------------------
    if not jd_text or not jd_text.strip():
        logger.warning("Semantic similarity skipped — empty JD text")
        return 0.0
    if not resume_text or not resume_text.strip():
        logger.warning("Semantic similarity skipped — empty resume text")
        return 0.0

    # ------------------------------------------------------------------
    # Load model (lazy initialization)
    # ------------------------------------------------------------------
    model = _get_model()
    if model is None:
        logger.debug("Semantic model unavailable — returning 0.0")
        return 0.0

    try:
        # ------------------------------------------------------------------
        # Encode both texts into dense vector embeddings
        # ------------------------------------------------------------------
        # Truncate very long texts to avoid memory issues
        # MiniLM has a max sequence length of 256 tokens
        jd_truncated: str = jd_text[:5000]
        resume_truncated: str = resume_text[:5000]

        embeddings = model.encode(
            [jd_truncated, resume_truncated],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        jd_embedding: np.ndarray = embeddings[0]
        resume_embedding: np.ndarray = embeddings[1]

        # ------------------------------------------------------------------
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        # ------------------------------------------------------------------
        dot_product: float = float(np.dot(jd_embedding, resume_embedding))
        norm_jd: float = float(np.linalg.norm(jd_embedding))
        norm_resume: float = float(np.linalg.norm(resume_embedding))

        if norm_jd == 0.0 or norm_resume == 0.0:
            logger.warning("Zero-norm embedding detected — returning 0.0")
            return 0.0

        similarity: float = dot_product / (norm_jd * norm_resume)

        # Clamp to [0.0, 1.0] (cosine sim can be negative for very different texts)
        similarity = max(0.0, min(1.0, similarity))

        logger.info(
            "Semantic similarity: %.4f (JD: %d chars, Resume: %d chars)",
            similarity, len(jd_text), len(resume_text),
        )

        return round(similarity, 4)

    except Exception as exc:
        logger.error("Semantic similarity computation failed: %s", exc, exc_info=True)
        return 0.0
