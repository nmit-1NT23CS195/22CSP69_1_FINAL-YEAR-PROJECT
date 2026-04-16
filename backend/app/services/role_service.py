import csv

ROLES_FILE = "app/data/roles.csv"


# -------------------------------
# GET ALL ROLES
# -------------------------------
def get_all_roles():
    roles = []

    with open(ROLES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 2:
                continue

            role = row[1].strip()
            roles.append(role)

    return sorted(list(set(roles)))


# -------------------------------
# GET RAW TEXT FROM ROLE
# -------------------------------
def get_role_text(role_name):
    role_name = role_name.lower()

    with open(ROLES_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            role = row[1].strip().lower()

            if role == role_name:
                raw_text = row[2]
                return raw_text.lower()

    return ""