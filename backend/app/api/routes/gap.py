from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional

from app.services.ats_service import run_pipeline

router = APIRouter()


@router.post("/missing-skills")
async def gap_analysis(
    request: Request,

    resume: UploadFile = File(...),

    jd_text: Optional[str] = Form(None),
    role: Optional[str] = Form(None)
):
    form = await request.form()

    # -------------------------------
    # RESUME
    # -------------------------------
    resume_bytes = await resume.read()
    resume_filename = getattr(resume, "filename", "")

    # -------------------------------
    # MANUAL JD FILE (NO VALIDATION)
    # -------------------------------
    jd_bytes = None
    jd_filename = ""
    jd_file = form.get("jd_file")

    if jd_file and hasattr(jd_file, "filename") and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_filename = jd_file.filename

    # -------------------------------
    # CLEAN INPUTS
    # -------------------------------
    if jd_text in ["", "string", "jd_text"]:
        jd_text = None

    if role in ["", "string"]:
        role = None

    # -------------------------------
    # PIPELINE
    # -------------------------------
    result = run_pipeline(
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_text=jd_text,
        jd_bytes=jd_bytes,
        jd_filename=jd_filename,
        role=role
    )

    # Handle error case from pipeline
    if "error" in result:
        return {"error": result["error"], "missing_skills": [], "matched_skills": []}

    # Extract from the correct keys in the pipeline response
    # Pipeline returns: result["technical_skills"]["matched"] and result["technical_skills"]["missing"]
    technical = result.get("technical_skills", {})

    return {
        "missing_skills": technical.get("missing", []),
        "matched_skills": technical.get("matched", [])
    }