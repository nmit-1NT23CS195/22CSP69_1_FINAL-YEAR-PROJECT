from fastapi import APIRouter, UploadFile, File, Form
from app.services.ats_service import run_pipeline

router = APIRouter()

@router.post("/missing-skills")
async def gap_analysis(
    resume: UploadFile = File(...),
    jd_file: UploadFile = File(None),
    jd_text: str = Form(None),
    role: str = Form(None)
):
    resume_bytes = await resume.read()
    jd_bytes = await jd_file.read() if jd_file else None

    result = run_pipeline(
        resume_bytes=resume_bytes,
        jd_text=jd_text,
        jd_bytes=jd_bytes,
        role=role
    )

    return {
        "missing_skills": result["missing_skills"],
        "matched_skills": result["matched_skills"]
    }