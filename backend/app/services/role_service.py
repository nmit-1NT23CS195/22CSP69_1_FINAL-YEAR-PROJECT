import pandas as pd
from rapidfuzz import process, fuzz

def load_roles():
    return pd.read_csv("app/data/IT_Job_Roles_Skills.csv")


# -------------------------------
# GET ALL ROLES (USED BY API)
# -------------------------------
def get_all_roles():
    df = load_roles()

    role_col = "Job Title"

    return df[role_col].dropna().unique().tolist()


# -------------------------------
# GET ROLE REQUIREMENTS (USED IN ATS)
# -------------------------------
def get_role_requirements(role_name):
    """
    Returns a combined list of skills and certifications required for a given role.
    """
    df = load_roles()

    role_col = "Job Title"
    skills_col = "Skills"
    certs_col = "Certifications"

    # Exact match
    row = df[df[role_col].str.lower() == role_name.lower()]

    if row.empty:
        # Fuzzy match
        roles = df[role_col].dropna().unique().tolist()
        match = process.extractOne(role_name, roles, scorer=fuzz.WRatio)

        # Threshold of 70 (out of 100)
        if match and match[1] >= 70:
            matched_role = match[0]
            row = df[df[role_col] == matched_role]

    if not row.empty:
        skills_str = str(row.iloc[0][skills_col]) if pd.notna(row.iloc[0][skills_col]) else ""
        certs_str = str(row.iloc[0].get(certs_col, "")) if pd.notna(row.iloc[0].get(certs_col, "")) else ""
        
        # Parse into a combined list
        combined = []
        if skills_str:
            combined.extend([s.strip() for s in skills_str.split(',') if s.strip()])
        if certs_str:
            combined.extend([c.strip() for c in certs_str.split(',') if c.strip()])
            
        return combined

    return []