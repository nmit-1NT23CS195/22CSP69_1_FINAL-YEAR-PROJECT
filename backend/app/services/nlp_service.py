import re
from app.services.skill_loader import load_skills


def extract_skills(text, custom_skills=None):
    text = text.lower()

    skills_db = load_skills()
    
    # Inject specific role requirements dynamically if provided
    if custom_skills:
        skills_db.extend([s.lower() for s in custom_skills])
        skills_db = list(set(skills_db))

    extracted = []

    for skill in skills_db:
        # 🔥 exact word match (fixes false positives and handles special chars like c++)
        pattern = r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])'

        if re.search(pattern, text):
            extracted.append(skill)

    return list(set(extracted))


# -------------------------------
# ADVANCED NLP FEATURES
# -------------------------------

def extract_soft_skills(text):
    text = text.lower()
    
    # Industry-standard soft skills dictionary
    soft_skills_db = [
        "leadership", "communication", "teamwork", "problem solving", 
        "critical thinking", "adaptability", "time management", "conflict resolution",
        "agile", "scrum", "mentoring", "collaboration", "creativity", "innovation",
        "analytical", "attention to detail", "decision making", "presentation",
        "public speaking", "negotiation", "empathy", "project management"
    ]
    
    extracted = []
    for skill in soft_skills_db:
        # Avoid substring matches (e.g. "team" in "teaming")
        pattern = r'(?<![a-z])' + re.escape(skill) + r'(?![a-z])'
        if re.search(pattern, text):
            extracted.append(skill.title())
            
    return list(set(extracted))

def extract_action_verbs(text):
    text = text.lower()
    
    # High-impact action verbs
    action_verbs_db = [
        "architected", "spearheaded", "engineered", "developed", "managed",
        "orchestrated", "implemented", "designed", "led", "optimized",
        "streamlined", "transformed", "modernized", "pioneered", "launched",
        "executed", "directed", "formulated", "conceptualized", "mentored",
        "delivered", "resolved", "automated", "maximized", "accelerated",
        "reduced", "increased", "improved", "negotiated", "secured"
    ]
    
    extracted = []
    for verb in action_verbs_db:
        pattern = r'\b' + re.escape(verb) + r'\b'
        if re.search(pattern, text):
            extracted.append(verb.title())
            
    return list(set(extracted))