from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from app.services.ats_service import run_pipeline

router = APIRouter()


@router.post("/score")
async def score_resume(
    resume: UploadFile = File(...),
    jd_file: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
    role: Optional[str] = Form(None)
):
    resume_bytes = await resume.read()

    jd_bytes = None

    if jd_file and hasattr(jd_file, "filename") and jd_file.filename:
        jd_bytes = await jd_file.read()

    if jd_text == "":
        jd_text = None

    return run_pipeline(
        resume_bytes=resume_bytes,
        jd_bytes=jd_bytes,
        jd_text=jd_text,
        role=role
    )