import csv

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