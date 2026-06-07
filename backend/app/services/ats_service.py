"""
ats_service.py
==============
Main ATS pipeline orchestrator.

This is the central entry point that coordinates all services:
    1. Text extraction (resume_service)
    2. Section segmentation (resume_service)
    3. NLP skill extraction with FlashText (nlp_service)
    4. Contextual weighting (nlp_service)
    5. Soft skills & action verbs extraction (nlp_service)
    6. TF-IDF keyword analysis (tfidf_service)
    7. Semantic vector similarity (semantic_service)
    8. Years of experience estimation (experience_service)
    9. LLM-enriched extraction (llm_service)
    10. Composite scoring (scoring_service)

CRITICAL CONTRACT:
    - run_pipeline() signature is UNCHANGED.
    - All original return dict keys are PRESERVED.
    - New keys are ADDED to the return payload.
    - All new service calls are wrapped in try/except for graceful degradation.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from app.services.resume_service import extract_text, parse_resume_sections
from app.services.nlp_service import (
    extract_skills,
    extract_skills_with_context,
    extract_soft_skills,
    extract_action_verbs,
)
from app.services.role_service import get_role_requirements
from app.services.skill_loader import load_priority_skills
from app.services.scoring_service import compute_smart_ats_score

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


async def run_pipeline(
    resume_bytes: bytes,
    resume_filename: str = "",
    jd_text: Optional[str] = None,
    jd_bytes: Optional[bytes] = None,
    jd_filename: str = "",
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute the full ATS analysis pipeline.

    SIGNATURE PRESERVED — no changes to input arguments or their types.

    Pipeline stages:
        1. Extract text from resume (with OCR fallback for scanned PDFs).
        2. Determine JD source (file upload > raw text > role lookup).
        3. Parse resume into structured sections.
        4. Extract skills with contextual weighting.
        5. Run TF-IDF analysis.
        6. Compute semantic similarity.
        7. Estimate years of experience.
        8. Attempt LLM-enriched extraction.
        9. Compute composite ATS score.
        10. Build and return the result payload.

    Args:
        resume_bytes:    Raw bytes of the uploaded resume file.
        resume_filename: Original filename of the resume.
        jd_text:         Raw job description text (optional).
        jd_bytes:        Raw bytes of an uploaded JD file (optional).
        jd_filename:     Original filename of the JD file (optional).
        role:            Job role name for role-based skill lookup (optional).

    Returns:
        Dict containing ATS score, matched/missing skills, and all analytics.
        Returns {"error": "..."} if no valid JD source is provided.
    """
    # ==================================================================
    # STAGE 1: Resume Text Extraction
    # ==================================================================
    resume_text: str = extract_text(resume_bytes, resume_filename)
    logger.info("Resume text extracted: %d characters", len(resume_text))

    # ==================================================================
    # STAGE 2: Job Description Source Selection (priority order)
    # ==================================================================
    final_jd_text: str = ""
    jd_skills: List[str] = []

    if jd_bytes:
        # Priority 1: Uploaded JD file
        final_jd_text = extract_text(jd_bytes, jd_filename)
        jd_skills = extract_skills(final_jd_text)

    elif jd_text and jd_text.strip():
        # Priority 2: Raw JD text
        final_jd_text = jd_text
        jd_skills = extract_skills(final_jd_text)

    elif role and role.strip():
        # Priority 3: Role-based skill lookup
        jd_skills = get_role_requirements(role)
        # Normalize to lowercase
        jd_skills = [s.lower() for s in jd_skills]
        # Build synthetic JD text from skills for keyword/TF-IDF analysis
        final_jd_text = " ".join(jd_skills)

    # ------------------------------------------------------------------
    # Validate: at least one JD source must be provided
    # ------------------------------------------------------------------
    if not jd_skills and not final_jd_text:
        return {"error": "No valid job description or role provided"}

    # ==================================================================
    # STAGE 3: Resume Section Segmentation
    # ==================================================================
    resume_sections: Dict[str, str] = {}
    try:
        resume_sections = parse_resume_sections(resume_text)
        logger.info(
            "Resume sections parsed: %s",
            {k: len(v) for k, v in resume_sections.items() if v},
        )
    except Exception as exc:
        logger.error("Section parsing failed: %s — using full text", exc)
        resume_sections = {"other": resume_text}

    # ==================================================================
    # STAGE 4: Skill Extraction with Contextual Weighting
    # ==================================================================
    # Dynamic Extraction: ensure role-specific requirements are searched
    resume_skills: List[str] = extract_skills(resume_text, custom_skills=jd_skills)

    # Section-aware extraction for contextual weights
    contextual_weights: Dict[str, float] = {}
    try:
        _, contextual_weights = extract_skills_with_context(
            resume_text, resume_sections, custom_skills=jd_skills
        )
        logger.info(
            "Contextual weights computed: %d skills with 2x weight",
            sum(1 for w in contextual_weights.values() if w == 2.0),
        )
    except Exception as exc:
        logger.error("Contextual extraction failed: %s", exc)
        contextual_weights = {s: 1.0 for s in resume_skills}

    # ==================================================================
    # STAGE 5: Advanced NLP Extractions
    # ==================================================================
    soft_skills: List[str] = extract_soft_skills(resume_text)
    action_verbs: List[str] = extract_action_verbs(resume_text)

    # ==================================================================
    # STAGE 6: TF-IDF Keyword Analysis (graceful degradation)
    # ==================================================================
    tfidf_result: Dict[str, Any] = {
        "tfidf_score": 0.0,
        "top_jd_keywords": [],
        "matched_count": 0,
        "total_analyzed": 0,
    }
    try:
        from app.services.tfidf_service import compute_tfidf_analysis
        tfidf_result = compute_tfidf_analysis(final_jd_text, resume_text)
        logger.info("TF-IDF score: %.4f", tfidf_result["tfidf_score"])
    except Exception as exc:
        logger.error("TF-IDF analysis failed: %s — using default 0.0", exc)

    # ==================================================================
    # STAGE 7: Semantic Vector Similarity (graceful degradation)
    # ==================================================================
    semantic_score: float = 0.0
    try:
        from app.services.semantic_service import compute_semantic_similarity
        semantic_score = compute_semantic_similarity(final_jd_text, resume_text)
        logger.info("Semantic similarity: %.4f", semantic_score)
    except Exception as exc:
        logger.error("Semantic similarity failed: %s — using default 0.0", exc)

    # ==================================================================
    # STAGE 8: Years of Experience Estimation (graceful degradation)
    # ==================================================================
    experience_data: Dict[str, Any] = {"total_yoe": 0.0, "experience_entries": []}
    try:
        from app.services.experience_service import extract_experience
        # Prefer the experience section if available, else use full text
        experience_text: str = resume_sections.get("experience", resume_text)
        experience_data = extract_experience(experience_text)
        logger.info("Estimated YoE: %.1f", experience_data["total_yoe"])
    except Exception as exc:
        logger.error("Experience extraction failed: %s", exc)

    # ==================================================================
    # STAGE 9: Concurrent LLM Cognitive Analysis
    # Fires CoreScoringSchema + DeepAnalysisSchema in parallel via
    # asyncio.gather(). Wall-clock time ≈ max(T_core, T_deep) not their sum.
    # ==================================================================
    cognitive_analysis: Dict[str, Any] = {}
    try:
        from app.services.llm_service import run_cognitive_analysis_concurrent
        cognitive_analysis = await run_cognitive_analysis_concurrent(resume_text, final_jd_text)
        if cognitive_analysis:
            logger.info("Concurrent cognitive analysis completed via LLM.")
    except Exception as exc:
        logger.error("LLM Cognitive analysis failed: %s", exc)

    llm_skills = {}
    llm_diagnosis_score = 0.0
    if cognitive_analysis:
        skill_matrix = cognitive_analysis.get("skill_matrix", [])
        if isinstance(skill_matrix, list):
            for item in skill_matrix:
                if isinstance(item, dict):
                    skill_name = item.get("skill_name")
                    if skill_name:
                        llm_skills[skill_name] = float(item.get("estimated_yoe", 0.0))
        llm_diagnosis_score = cognitive_analysis.get("llm_diagnosis_score", 0.0)

    # ==================================================================
    # STAGE 10: Composite Scoring
    # ==================================================================
    priority_skills: Set[str] = load_priority_skills()

    score_data: Dict[str, Any] = compute_smart_ats_score(
        jd_text=final_jd_text,
        resume_text=resume_text,
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        priority_skills=priority_skills,
        soft_skills=soft_skills,
        action_verbs=action_verbs,
        semantic_score=semantic_score,
        tfidf_score=tfidf_result["tfidf_score"],
        contextual_weights=contextual_weights,
        llm_diagnosis_score=llm_diagnosis_score,
    )

    # ==================================================================
    # STAGE 11: Build Result Payload
    # ==================================================================
    # BUG FIX: Missing Skills — strict lowercase normalization + None safety
    # The original `list(set(jd_skills) - set(resume_skills))` failed when:
    #   - Case differed ("Python" vs "python")
    #   - Whitespace differed (" react " vs "react")
    #   - Either list was None/empty
    jd_normalized: Set[str] = {
        s.strip().lower() for s in (jd_skills or []) if s and s.strip()
    }
    resume_normalized: Set[str] = {
        s.strip().lower() for s in (resume_skills or []) if s and s.strip()
    }

    matched_skills: List[str] = sorted(jd_normalized & resume_normalized)
    missing_skills: List[str] = sorted(jd_normalized - resume_normalized)

    # ------------------------------------------------------------------
    # Construct the final return dict
    # CRITICAL: All original keys are preserved. New keys are ADDED.
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        # ── Original keys (PRESERVED) ──────────────────────────────
        "ats_score": score_data["overall_score"],
        "technical_skills": {
            "matched": matched_skills,
            "missing": missing_skills,
        },
        "soft_skills_found": soft_skills,
        "action_verbs_count": len(action_verbs),
        "action_verbs_found": action_verbs,
        "metrics_breakdown": score_data["breakdown"],

        # ── New keys (ADDED for downstream ML feature engineering) ──
        "keyword_metrics": {
            "semantic_similarity": round(semantic_score, 4),
            "tfidf_score": tfidf_result["tfidf_score"],
            "keyword_match": score_data["breakdown"]["keyword_score"],
        },
        "tfidf_analysis": {
            "score": tfidf_result["tfidf_score"],
            "top_jd_keywords": tfidf_result["top_jd_keywords"],
            "matched_count": tfidf_result["matched_count"],
            "total_analyzed": tfidf_result["total_analyzed"],
        },
        "estimated_experience": experience_data,
        "resume_sections": {
            k: v[:500] if v else ""  # Truncate section previews for payload size
            for k, v in resume_sections.items()
        },
        "contextual_skill_weights": contextual_weights,
        "llm_enriched_skills": llm_skills,
        "cognitive_analysis": cognitive_analysis,
    }

    logger.info(
        "Pipeline complete — ATS Score: %.2f | Skills: %d matched, %d missing | YoE: %.1f",
        result["ats_score"],
        len(matched_skills),
        len(missing_skills),
        experience_data["total_yoe"],
    )

    return result


# ===========================================================================
# TWO-STAGE PIPELINE FUNCTIONS
# Powers the /ats/score/core and /ats/score/deep endpoints.
# ===========================================================================

async def run_core_pipeline(
    resume_bytes: bytes,
    resume_filename: str = "",
    jd_text: Optional[str] = None,
    jd_bytes: Optional[bytes] = None,
    jd_filename: str = "",
    role: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stage 1 pipeline: runs ALL deterministic layers + CoreScoringSchema LLM call.

    Returns the full core payload PLUS `extracted_resume_text` (raw string) so
    the caller can hand it directly to run_deep_pipeline() without re-parsing.
    """
    # ── Text extraction ────────────────────────────────────────────────
    resume_text: str = extract_text(resume_bytes, resume_filename)
    logger.info("[CorePipeline] Resume extracted: %d chars", len(resume_text))

    # ── JD source resolution ───────────────────────────────────────────
    final_jd_text: str = ""
    jd_skills: List[str] = []

    if jd_bytes:
        final_jd_text = extract_text(jd_bytes, jd_filename)
        jd_skills = extract_skills(final_jd_text)
    elif jd_text and jd_text.strip():
        final_jd_text = jd_text
        jd_skills = extract_skills(final_jd_text)
    elif role and role.strip():
        jd_skills = [s.lower() for s in get_role_requirements(role)]
        final_jd_text = " ".join(jd_skills)

    if not jd_skills and not final_jd_text:
        return {"error": "No valid job description or role provided"}

    # ── Section segmentation ───────────────────────────────────────────
    resume_sections: Dict[str, str] = {}
    try:
        resume_sections = parse_resume_sections(resume_text)
    except Exception as exc:
        logger.error("[CorePipeline] Section parsing failed: %s", exc)
        resume_sections = {"other": resume_text}

    # ── Skill extraction ───────────────────────────────────────────────
    resume_skills: List[str] = extract_skills(resume_text, custom_skills=jd_skills)
    contextual_weights: Dict[str, float] = {}
    try:
        _, contextual_weights = extract_skills_with_context(
            resume_text, resume_sections, custom_skills=jd_skills
        )
    except Exception as exc:
        logger.error("[CorePipeline] Contextual extraction failed: %s", exc)
        contextual_weights = {s: 1.0 for s in resume_skills}

    soft_skills: List[str] = extract_soft_skills(resume_text)
    action_verbs: List[str] = extract_action_verbs(resume_text)

    # ── TF-IDF ────────────────────────────────────────────────────────
    tfidf_result: Dict[str, Any] = {"tfidf_score": 0.0, "top_jd_keywords": [], "matched_count": 0, "total_analyzed": 0}
    try:
        from app.services.tfidf_service import compute_tfidf_analysis
        tfidf_result = compute_tfidf_analysis(final_jd_text, resume_text)
    except Exception as exc:
        logger.error("[CorePipeline] TF-IDF failed: %s", exc)

    # ── Semantic similarity ────────────────────────────────────────────
    semantic_score: float = 0.0
    try:
        from app.services.semantic_service import compute_semantic_similarity
        semantic_score = compute_semantic_similarity(final_jd_text, resume_text)
    except Exception as exc:
        logger.error("[CorePipeline] Semantic failed: %s", exc)

    # ── Years of experience ────────────────────────────────────────────
    experience_data: Dict[str, Any] = {"total_yoe": 0.0, "experience_entries": []}
    try:
        from app.services.experience_service import extract_experience
        experience_data = extract_experience(resume_sections.get("experience", resume_text))
    except Exception as exc:
        logger.error("[CorePipeline] YoE extraction failed: %s", exc)

    # ── Core LLM call ─────────────────────────────────────────────────
    core_analysis: Dict[str, Any] = {}
    try:
        from app.services.llm_service import run_core_analysis
        core_analysis = await run_core_analysis(resume_text, final_jd_text)
        logger.info("[CorePipeline] Core LLM analysis complete.")
    except Exception as exc:
        logger.error("[CorePipeline] Core LLM failed: %s", exc)

    # ── Scoring ───────────────────────────────────────────────────────
    priority_skills: Set[str] = load_priority_skills()
    llm_diagnosis_score = float(core_analysis.get("llm_diagnosis_score", 0.0))

    score_data: Dict[str, Any] = compute_smart_ats_score(
        jd_text=final_jd_text,
        resume_text=resume_text,
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        priority_skills=priority_skills,
        soft_skills=soft_skills,
        action_verbs=action_verbs,
        semantic_score=semantic_score,
        tfidf_score=tfidf_result["tfidf_score"],
        contextual_weights=contextual_weights,
        llm_diagnosis_score=llm_diagnosis_score,
    )

    # ── Skill diff ────────────────────────────────────────────────────
    jd_normalized: Set[str] = {s.strip().lower() for s in (jd_skills or []) if s and s.strip()}
    resume_normalized: Set[str] = {s.strip().lower() for s in (resume_skills or []) if s and s.strip()}
    matched_skills: List[str] = sorted(jd_normalized & resume_normalized)
    missing_skills: List[str] = sorted(jd_normalized - resume_normalized)

    llm_skills = {}
    for item in core_analysis.get("skill_matrix", []):
        if isinstance(item, dict) and item.get("skill_name"):
            llm_skills[item["skill_name"]] = float(item.get("estimated_yoe", 0.0))

    logger.info(
        "[CorePipeline] Done — Score: %.2f | %d matched | %d missing",
        score_data["overall_score"], len(matched_skills), len(missing_skills),
    )

    return {
        # Core scoring data
        "ats_score":               score_data["overall_score"],
        "technical_skills":        {"matched": matched_skills, "missing": missing_skills},
        "soft_skills_found":       soft_skills,
        "action_verbs_count":      len(action_verbs),
        "action_verbs_found":      action_verbs,
        "metrics_breakdown":       score_data["breakdown"],
        "keyword_metrics": {
            "semantic_similarity": round(semantic_score, 4),
            "tfidf_score":         tfidf_result["tfidf_score"],
            "keyword_match":       score_data["breakdown"]["keyword_score"],
        },
        "tfidf_analysis":          {"score": tfidf_result["tfidf_score"], "top_jd_keywords": tfidf_result["top_jd_keywords"], "matched_count": tfidf_result["matched_count"], "total_analyzed": tfidf_result["total_analyzed"]},
        "estimated_experience":    experience_data,
        "resume_sections":         {k: v[:500] if v else "" for k, v in resume_sections.items()},
        "contextual_skill_weights": contextual_weights,
        "llm_enriched_skills":     llm_skills,
        "cognitive_analysis":      core_analysis,
        # ── Passed to /score/deep to skip re-parsing ──────────────────
        "extracted_resume_text":   resume_text,
        "extracted_jd_text":       final_jd_text,
        # Legacy schema keys
        "matched_skills":  matched_skills,
        "missing_skills":  missing_skills,
    }


async def run_deep_pipeline(
    resume_text: str,
    jd_text: str,
    skill_matrix: list = None,
    resume_sections: dict = None,
) -> Dict[str, Any]:
    """
    Stage 2 pipeline: receives pre-extracted text + optional structured data.

    When skill_matrix and resume_sections are supplied (from /score/core response),
    they are forwarded directly to run_deep_analysis to build a compact JSON prompt
    instead of sending raw resume_text — significantly reducing input token cost.

    Fallback: if structured data is not provided, raw resume_text is used.
    """
    deep_analysis: Dict[str, Any] = {}
    try:
        from app.services.llm_service import run_deep_analysis
        deep_analysis = await run_deep_analysis(
            resume_text=resume_text,
            jd_text=jd_text,
            skill_matrix=skill_matrix,
            resume_sections=resume_sections,
        )
        logger.info("[DeepPipeline] Deep LLM analysis complete.")
    except Exception as exc:
        logger.error("[DeepPipeline] Deep LLM failed: %s", exc)

    return {
        "targeted_questions":       deep_analysis.get("targeted_questions", []),
        "dsa_bridge":               deep_analysis.get("dsa_bridge", []),
        "micro_project_suggestion": deep_analysis.get("micro_project_suggestion", ""),
    }