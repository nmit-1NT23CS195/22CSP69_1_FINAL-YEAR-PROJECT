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
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.services.ats_service import run_pipeline, run_core_pipeline, run_deep_pipeline
from app.api.schemas import ATSAnalysisResponse

router = APIRouter()

_SENTINEL_VALUES = {"", "string", "null", "none", "undefined"}


class DeepAnalysisRequest(BaseModel):
    """JSON payload for the /score/deep endpoint."""
    resume_text: str
    jd_text: str = ""          # Optional when job_role is provided
    job_role: Optional[str] = None
    skill_matrix: Optional[List[Dict[str, Any]]] = None
    resume_sections: Optional[Dict[str, str]] = None


@router.post("/score", response_model=ATSAnalysisResponse)
async def score_resume(
    resume: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd_file: Optional[UploadFile] = File(None, description="Job description file"),
    jd_text: str = Form("", description="Raw job description text"),
    role: str = Form("", description="Job role name for role-based skill lookup"),
) -> ATSAnalysisResponse:
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

    # ── Read JD bytes (if uploaded) ─────────────────────────────────────
    jd_bytes: Optional[bytes] = None
    jd_filename: str = ""
    if jd_file:
        jd_bytes = await jd_file.read()
        jd_filename = jd_file.filename or ""

    # ── Validate: at least one JD source must be supplied ──────────────
    if not jd_bytes and not jd_text and not role:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'jd_file', 'jd_text' (job description) or 'role' (job role name).",
        )

    # ── Run the pipeline (async — runs both LLM calls concurrently) ────
    result: Dict[str, Any] = await run_pipeline(
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_text=jd_text,
        jd_bytes=jd_bytes,
        jd_filename=jd_filename,
        role=role,
    )

    # Propagate pipeline-level errors as HTTP 422
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return ATSAnalysisResponse(
        ats_score=result["ats_score"],
        keyword_metrics=result.get("keyword_metrics", {}),
        matched_skills=result.get("technical_skills", {}).get("matched", []),
        missing_skills=result.get("technical_skills", {}).get("missing", []),
        contextual_skill_weights=result.get("contextual_skill_weights", {}),
        estimated_experience=result.get("estimated_experience", {}),
        soft_skills_found=result.get("soft_skills_found", []),
        action_verbs_found=result.get("action_verbs_found", []),
        llm_enriched_skills=result.get("llm_enriched_skills", {}),
        cognitive_analysis=result.get("cognitive_analysis") or {}
    )


# ---------------------------------------------------------------------------
# TWO-STAGE ENDPOINTS
# /score/core  — heavy: file upload + all deterministic layers + CoreLLM
# /score/deep  — light: raw strings only + DeepLLM (no parsing overhead)
# ---------------------------------------------------------------------------

@router.post("/score/core")
async def score_resume_core(
    resume: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    jd_file: Optional[UploadFile] = File(None, description="Job description file"),
    jd_text: str = Form("", description="Raw job description text"),
    role: str = Form("", description="Job role name for role-based skill lookup"),
) -> Dict[str, Any]:
    """
    Stage 1 of the two-stage pipeline.

    Runs all deterministic layers (TF-IDF, pgvector, NLP, YoE) and the
    CoreScoringSchema LLM call.  Returns the full core payload plus
    `extracted_resume_text` and `extracted_jd_text` — passthrough tokens
    for the frontend to hand to /score/deep without re-uploading the file.
    """
    if jd_text and jd_text.strip().lower() in _SENTINEL_VALUES:
        jd_text = None
    if role and role.strip().lower() in _SENTINEL_VALUES:
        role = None

    resume_bytes: bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume file is empty.")
    resume_filename: str = resume.filename or ""

    jd_bytes: Optional[bytes] = None
    jd_filename: str = ""
    if jd_file:
        jd_bytes = await jd_file.read()
        jd_filename = jd_file.filename or ""

    if not jd_bytes and not jd_text and not role:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'jd_file', 'jd_text' or 'role'.",
        )

    result: Dict[str, Any] = await run_core_pipeline(
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_text=jd_text,
        jd_bytes=jd_bytes,
        jd_filename=jd_filename,
        role=role,
    )

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result


@router.post("/score/deep")
async def score_resume_deep(payload: DeepAnalysisRequest) -> Dict[str, Any]:
    """
    Stage 2 of the two-stage pipeline.

    Receives a JSON payload containing:
      - resume_text:     Pre-extracted resume text (from /score/core).
      - jd_text:         Job description text.
      - skill_matrix:    (Optional) Pre-parsed LLM skill objects — reduces token cost.
      - resume_sections: (Optional) Pre-parsed section dict — reduces token cost.

    When skill_matrix + resume_sections are provided, the LLM prompt is built from
    compact structured JSON instead of raw resume_text, cutting input tokens by ~60-70%.
    """
    if not payload.resume_text or not payload.resume_text.strip():
        raise HTTPException(status_code=422, detail="resume_text must not be empty.")
    if not payload.jd_text.strip() and not payload.job_role:
        raise HTTPException(status_code=422, detail="Provide either jd_text or job_role.")

    return await run_deep_pipeline(
        resume_text=payload.resume_text,
        jd_text=payload.jd_text,
        job_role=payload.job_role,
        skill_matrix=payload.skill_matrix,
        resume_sections=payload.resume_sections,
    )

