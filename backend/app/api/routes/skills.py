from fastapi import APIRouter, UploadFile, File
from app.services.resume_service import extract_text
from app.services.nlp_service import extract_skills

router = APIRouter()


@router.post("/extract")
async def extract_skills_api(resume: UploadFile = File(...)):
    resume_bytes = await resume.read()
    text = extract_text(resume_bytes, resume.filename)

    skills = extract_skills(text)

    return {"skills": skills}