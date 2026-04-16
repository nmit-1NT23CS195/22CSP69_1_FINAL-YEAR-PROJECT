import re

# -------------------------------
# CLEAN TEXT
# -------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+.# ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------
# GENERATE VARIATIONS
# -------------------------------
def generate_variations(skill):
    skill = skill.lower()
    variations = set()

    variations.add(skill)
    variations.add(skill.replace(" ", ""))

    # Short forms
    if "machine learning" in skill:
        variations.add("ml")
    if "deep learning" in skill:
        variations.add("dl")
    if "javascript" in skill:
        variations.add("js")
    if "typescript" in skill:
        variations.add("ts")

    # DB shortcuts
    if "database" in skill or "db" in skill:
        variations.add("sql")

    variations.add(skill.replace("-", " "))
    variations.add(skill.replace(".", ""))

    return variations


# -------------------------------
# BUILD SKILL MAP
# -------------------------------
def build_skill_map(skill_list):
    skill_map = {}

    for skill in skill_list:
        variations = generate_variations(skill)

        for var in variations:
            skill_map[var] = skill

    return skill_map


# -------------------------------
# EXTRACT SKILLS
# -------------------------------
def extract_skills(text, skill_map):
    text = clean_text(text)
    words = text.split()

    found_skills = set()

    for i in range(len(words)):

        # single word
        if words[i] in skill_map:
            found_skills.add(skill_map[words[i]])

        # bigram
        if i < len(words) - 1:
            phrase = words[i] + " " + words[i + 1]
            if phrase in skill_map:
                found_skills.add(skill_map[phrase])

        # trigram
        if i < len(words) - 2:
            phrase = words[i] + " " + words[i + 1] + " " + words[i + 2]
            if phrase in skill_map:
                found_skills.add(skill_map[phrase])

    return sorted(found_skills)


# -------------------------------
# SIMILARITY
# -------------------------------
def compute_similarity(resume_skills, jd_skills):
    r = set(resume_skills)
    j = set(jd_skills)

    if not j:
        return 0

    return len(r.intersection(j)) / len(r.union(j))