# -------------------------------
# WEIGHT FUNCTION
# -------------------------------
def get_weight(skill, priority_skills):
    return 5 if skill in priority_skills else 1


# -------------------------------
# NEW WEIGHTED SCORE
# -------------------------------
def compute_weighted_skill_score(required_skills, resume_skills, priority_skills):

    required_skills = set(required_skills)
    resume_skills = set(resume_skills)

    total_weight = 0
    matched_weight = 0

    for skill in required_skills:
        weight = get_weight(skill, priority_skills)

        total_weight += weight

        if skill in resume_skills:
            matched_weight += weight

    return matched_weight / total_weight if total_weight else 0


# -------------------------------
# 🔥 BACKWARD COMPATIBILITY FIX
# -------------------------------
def compute_ats_score(required_skills, resume_skills):
    """
    Old function wrapper (used by existing APIs)
    Now internally calls weighted logic with default weights
    """

    # fallback → treat all skills equally
    required_skills = set(required_skills)
    resume_skills = set(resume_skills)

    if not required_skills:
        return 0

    matched = len(required_skills & resume_skills)

    return matched / len(required_skills)


# -------------------------------
# OPTIONAL (if used elsewhere)
# -------------------------------
def compute_keyword_score(resume_text, jd_text):
    resume_words = set(resume_text.split())
    jd_words = set(jd_text.split())

    if not jd_words:
        return 0

    return len(resume_words & jd_words) / len(jd_words)


def compute_quality_score(resume_text):
    length = len(resume_text.split())

    if length > 500:
        return 1
    elif length > 200:
        return 0.7
    elif length > 100:
        return 0.5
    return 0.3


# -------------------------------
# SMART ATS SCORE (COMBINED & EXPANDED)
# -------------------------------
def compute_smart_ats_score(
    jd_text, resume_text, jd_skills, resume_skills, priority_skills, 
    soft_skills=None, action_verbs=None
):
    soft_skills = soft_skills or []
    action_verbs = action_verbs or []
    
    # 1. Technical Skill Score (weighted) - 50%
    technical_score_raw = compute_weighted_skill_score(jd_skills, resume_skills, priority_skills)
    technical_score = technical_score_raw * 100
    
    # 2. Soft Skills Score - 15%
    # Expecting at least 3 soft skills for a perfect soft skill score
    soft_skills_score = min(len(soft_skills) / 3.0, 1.0) * 100
    
    # 3. Action Verbs Score - 15%
    # Expecting at least 5 action verbs for a perfect action verb score
    action_verbs_score = min(len(action_verbs) / 5.0, 1.0) * 100
    
    # 4. Keyword Match Score (contextual match) - 10%
    keyword_score = compute_keyword_score(resume_text, jd_text) * 100
    
    # 5. Quality Score - 10%
    quality_score = compute_quality_score(resume_text) * 100
    
    # Combine scores with new weights
    final_score = (
        (technical_score * 0.50) + 
        (soft_skills_score * 0.15) + 
        (action_verbs_score * 0.15) + 
        (keyword_score * 0.10) + 
        (quality_score * 0.10)
    )
    
    return {
        "overall_score": round(final_score, 2),
        "breakdown": {
            "technical_score": round(technical_score, 2),
            "soft_skills_score": round(soft_skills_score, 2),
            "action_verbs_score": round(action_verbs_score, 2),
            "keyword_score": round(keyword_score, 2),
            "quality_score": round(quality_score, 2)
        }
    }
