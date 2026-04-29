from app.services.resume_service import extract_text
from app.services.nlp_service import extract_skills, extract_soft_skills, extract_action_verbs
from app.services.role_service import get_role_requirements
from app.services.skill_loader import load_priority_skills
from app.services.scoring_service import compute_smart_ats_score


def run_pipeline(resume_bytes, resume_filename="", jd_text=None, jd_bytes=None, jd_filename="", role=None):

    # Resume text
    resume_text = extract_text(resume_bytes, resume_filename)

    # Variables
    final_jd_text = ""
    jd_skills = []

    # JD selection (priority order)
    if jd_bytes:
        final_jd_text = extract_text(jd_bytes, jd_filename)
        jd_skills = extract_skills(final_jd_text)
    elif jd_text and jd_text.strip():
        final_jd_text = jd_text
        jd_skills = extract_skills(final_jd_text)
    elif role and role.strip():
        jd_skills = get_role_requirements(role)
        # Convert skills to lowercase and use as jd text context for keywords (since JD text is gone)
        jd_skills = [s.lower() for s in jd_skills]
        final_jd_text = " ".join(jd_skills)

    if not jd_skills and not final_jd_text:
        return {"error": "No valid job description or role provided"}

    # Dynamic Extraction: Ensures role-specific requirements are strictly searched for in the resume!
    resume_skills = extract_skills(resume_text, custom_skills=jd_skills)
    
    # Advanced NLP Extractions
    soft_skills = extract_soft_skills(resume_text)
    action_verbs = extract_action_verbs(resume_text)

    # Load priority skills
    priority_skills = load_priority_skills()

    # Smart Weighted score combining skills, keywords, quality, soft skills, and verbs
    score_data = compute_smart_ats_score(
        final_jd_text,
        resume_text,
        jd_skills,
        resume_skills,
        priority_skills,
        soft_skills,
        action_verbs
    )

    return {
        "ats_score": score_data["overall_score"],
        "technical_skills": {
            "matched": list(set(jd_skills) & set(resume_skills)),
            "missing": list(set(jd_skills) - set(resume_skills))
        },
        "soft_skills_found": soft_skills,
        "action_verbs_count": len(action_verbs),
        "action_verbs_found": action_verbs,
        "metrics_breakdown": score_data["breakdown"]
    }