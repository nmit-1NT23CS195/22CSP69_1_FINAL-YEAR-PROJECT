from fastapi import APIRouter, UploadFile, File
from app.services.resume_service import extract_text_from_pdf
from app.services.skill_loader import load_skills
from app.services.nlp_service import build_skill_map, extract_skills

router = APIRouter(prefix="/skills", tags=["Skills"])


@router.post("/extract")
async def extract_resume_skills(file: UploadFile = File(...)):
    
    # read file
    resume_bytes = await file.read()

    # extract text
    text = extract_text_from_pdf(resume_bytes)

    # load dataset
    skills_list = load_skills()

    # build map
    skill_map = build_skill_map(skills_list)

    # extract skills
    skills = extract_skills(text, skill_map)

    return {"skills": skills}