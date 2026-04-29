import csv
import pandas as pd


# -------------------------------
# ORIGINAL (WORKING EXTRACTION)
# -------------------------------
def load_skills():
    skills = []

    with open("app/data/skills.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue

            skill = row[1].strip().lower()
            skills.append(skill)

    return list(set(skills))


# -------------------------------
# NEW (WEIGHTS)
# -------------------------------
def load_priority_skills():
    df = pd.read_csv("app/data/priority_skills.csv")

    return set(
        df["skill"]
        .dropna()
        .astype(str)
        .str.lower()
        .str.strip()
        .unique()
    )