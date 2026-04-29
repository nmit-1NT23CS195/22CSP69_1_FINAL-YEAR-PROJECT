from fastapi import APIRouter, UploadFile, File
from app.services.resume_service import extract_text

router = APIRouter()

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text(await file.read(), file.filename)
    return {"resume_text": text}