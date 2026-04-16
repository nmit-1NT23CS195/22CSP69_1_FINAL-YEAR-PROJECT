from fastapi import APIRouter, UploadFile, File
from app.services.resume_service import extract_text_from_pdf

router = APIRouter()

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text_from_pdf(await file.read())
    return {"resume_text": text}