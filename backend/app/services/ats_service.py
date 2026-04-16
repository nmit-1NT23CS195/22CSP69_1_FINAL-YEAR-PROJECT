from app.services.resume_service import extract_text_from_pdf
from app.services.scoring_service import compute_ats_score
from app.services.skill_loader import load_skills
from app.services.nlp_service import build_skill_map, extract_skills
from app.services.role_service import get_role_text


def run_pipeline(resume_bytes, jd_text=None, jd_bytes=None, role=None):

    # -------------------------------
    # Resume text
    # -------------------------------
    resume_text = extract_text_from_pdf(resume_bytes)

    # -------------------------------
    # Handle JD / Role
    # -------------------------------
    if jd_bytes:
        jd_text = extract_text_from_pdf(jd_bytes)

    elif role:
        jd_text = get_role_text(role)

    elif not jd_text:
        jd_text = ""

    # -------------------------------
    # Load skills dataset
    # -------------------------------
    skills = load_skills()
    skill_map = build_skill_map(skills)

    # -------------------------------
    # Extract skills
    # -------------------------------
    resume_skills = extract_skills(resume_text, skill_map)
    jd_skills = extract_skills(jd_text, skill_map)

    # -------------------------------
    # Score
    # -------------------------------
    result = compute_ats_score(
        resume_text,
        jd_text,
        resume_skills,
        jd_skills
    )

    return result