"""
ats.py
======
ATS scoring endpoint.

Accepts multipart/form-data with:
    - resume    : UploadFile  (required) — the candidate's resume (PDF / DOCX / TXT)
    - jd_text   : str         (optional) — raw job description pasted by the user
    - role      : str         (optional) — job role name for role-based skill lookup

Priority for JD source (handled inside run_pipeline):
    jd_bytes (file) > jd_text (raw text) > role (lookup)
"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import Any, Dict, Optional

from app.services.ats_service import run_pipeline

router = APIRouter()

_SENTINEL_VALUES = {"", "string", "null", "none", "undefined"}


@router.post("/score", response_model=Dict[str, Any])
async def score_resume(
    resume: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd_text: Optional[str] = Form(None, description="Raw job description text"),
    role: Optional[str] = Form(None, description="Job role name for role-based skill lookup"),
) -> Dict[str, Any]:
    """
    Run the full ATS analysis pipeline on an uploaded resume.

    Returns a rich JSON payload containing:
    - `ats_score`                  — composite placement-readiness score (0–100)
    - `metrics_breakdown`          — 7-signal weighted breakdown
    - `technical_skills`           — matched & missing skills vs. the JD
    - `soft_skills_found`          — soft skills detected in the resume
    - `action_verbs_found`         — action verbs detected
    - `semantic_similarity_score`  — sentence-transformer cosine similarity
    - `tfidf_analysis`             — TF-IDF keyword analysis
    - `estimated_experience`       — NER-based years-of-experience estimate
    - `contextual_skill_weights`   — section-aware skill confidence weights
    - `llm_enriched_skills`        — LLM-extracted skill → YoE mapping
    - `resume_sections`            — structured section previews
    """
    # ── Sanitise text inputs ────────────────────────────────────────────
    # Ignore Swagger default placeholder values that can leak through
    if jd_text and jd_text.strip().lower() in _SENTINEL_VALUES:
        jd_text = None
    if role and role.strip().lower() in _SENTINEL_VALUES:
        role = None

    # ── Read resume bytes ───────────────────────────────────────────────
    resume_bytes: bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")

    resume_filename: str = resume.filename or ""

    # ── Validate: at least one JD source must be supplied ──────────────
    if not jd_text and not role:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'jd_text' (job description) or 'role' (job role name).",
        )

    # ── Run the pipeline ────────────────────────────────────────────────
    result: Dict[str, Any] = run_pipeline(
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_text=jd_text,
        role=role,
    )

    # Propagate pipeline-level errors as HTTP 422
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result