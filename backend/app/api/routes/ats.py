from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import Optional
from app.services.ats_service import run_pipeline

router = APIRouter()


@router.post("/score")
async def score_resume(
    request: Request,
    resume: UploadFile = File(...),
    jd_file: Optional[UploadFile] = File(None, description="IMPORTANT: Do NOT check 'Send empty value' in Swagger if you are not uploading a file. Just leave it unselected."),
    jd_text: Optional[str] = Form(None),
    role: Optional[str] = Form(None)
):
    form = await request.form()
    
    resume_bytes = await resume.read()

    jd_bytes = None
    jd_file = form.get("jd_file")

    resume_filename = getattr(resume, "filename", "")
    
    # 🔥 FIX: ignore empty jd_file manually
    if jd_file and hasattr(jd_file, "filename") and jd_file.filename:
        jd_bytes = await jd_file.read()
        jd_filename = jd_file.filename
    else:
        jd_filename = ""

    # 🔥 FIX: ignore empty text inputs
    if jd_text in ["", "string", "jd_text"]:
        jd_text = None
        
    if role in ["", "string"]:
        role = None

    result = run_pipeline(
        resume_bytes=resume_bytes,
        resume_filename=resume_filename,
        jd_bytes=jd_bytes,
        jd_filename=jd_filename,
        jd_text=jd_text,
        role=role
    )

    return result