def compute_ats_score(resume_text, jd_text, resume_skills, jd_skills):

    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    if not jd_set:
        return {
            "ats_score": 0,
            "matched_skills": [],
            "missing_skills": []
        }

    matched = resume_set.intersection(jd_set)
    missing = jd_set - resume_set

    skill_score = len(matched) / len(jd_set)

    # keyword match
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())

    keyword_score = len(resume_words.intersection(jd_words)) / len(jd_words) if jd_words else 0

    # quality score
    word_count = len(resume_text.split())
    quality_score = 1 if word_count > 200 else 0.5

    final_score = (0.6 * skill_score + 0.2 * keyword_score + 0.2 * quality_score) * 100

    return {
        "ats_score": round(final_score, 2),
        "skill_score": round(skill_score * 100, 2),
        "keyword_score": round(keyword_score * 100, 2),
        "quality_score": round(quality_score * 100, 2),
        "matched_skills": list(matched),
        "missing_skills": list(missing)
    }