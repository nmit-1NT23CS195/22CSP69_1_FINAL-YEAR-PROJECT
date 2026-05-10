from fastapi import APIRouter, UploadFile, File
from app.services.resume_service import extract_text
from app.api.schemas import ResumeUploadResponse

router = APIRouter()

@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text(await file.read(), file.filename)
    return ResumeUploadResponse(resume_text=text)