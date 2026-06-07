"""
enrich_roles.py
===============
Standalone script to enrich roles_dictionary.json by cross-referencing
every role's context against the full skill set in skills_dictionary.json
using fast semantic similarity (all-MiniLM-L6-v2).

Usage:
    # Install deps (run once inside your venv):
    pip install sentence-transformers tqdm

    # Run from the backend directory:
    python app/data/enrich_roles.py

Output:
    enriched_roles_dictionary.json  (saved in app/data/)
"""

import json
import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # app/data/
ROLES_FILE = SCRIPT_DIR / "roles_dictionary.json"
SKILLS_FILE = SCRIPT_DIR / "skills_dictionary.json"
OUTPUT_FILE = SCRIPT_DIR / "enriched_roles_dictionary.json"

SIMILARITY_THRESHOLD = 0.45   # Adjust up (stricter) or down (more permissive)
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def flatten_skills(skills_dict: dict) -> list:
    """Flatten all categories into a single deduplicated list."""
    seen = set()
    flat = []
    for category_skills in skills_dict.values():
        for skill in category_skills:
            key = skill.strip().lower()
            if key not in seen:
                seen.add(key)
                flat.append(skill.strip())
    print("  [OK] Flattened skill pool: {:,} unique skills".format(len(flat)))
    return flat


def cosine_similarity_matrix(a, b):
    """
    Compute cosine similarity between every row of `a` and every row of `b`.
    Returns a matrix of shape (len(a), len(b)).
    Both arrays must be L2-normalised (SentenceTransformer normalize_embeddings=True).
    """
    return a @ b.T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n-- PlaceBuddy Role Enrichment ------------------------------------------")

    # 1. Load data
    print("\n[1/5] Loading JSON files ...")
    roles_dict = load_json(ROLES_FILE)
    skills_dict = load_json(SKILLS_FILE)
    print("  [OK] Roles loaded   : {:,} roles".format(len(roles_dict)))
    print("  [OK] Skill categories: {:,} categories".format(len(skills_dict)))

    # 2. Flatten the skill pool
    print("\n[2/5] Flattening skill pool ...")
    all_skills = flatten_skills(skills_dict)

    # 3. Load the embedding model
    print("\n[3/5] Loading sentence-transformer model ({}) ...".format(MODEL_NAME))
    model = SentenceTransformer(MODEL_NAME)

    # 4. Encode all skills once (normalised -> cosine == dot product)
    print("\n[4/5] Encoding skill pool ...")
    skill_embeddings = model.encode(
        all_skills,
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=True,
    )  # shape: (n_skills, embedding_dim)

    # 5. For each role, compute similarity and inject matching skills
    print("\n[5/5] Enriching roles ...")
    enriched_roles = {}

    for role_key, role_data in tqdm(roles_dict.items(), desc="Roles", unit="role"):
        display_name = role_data.get("display_name", role_key)
        existing_skills = list(role_data.get("skills", []))

        # Build a rich context string from the role's name + existing broad skills
        context_str = ", ".join([display_name] + existing_skills)

        # Encode the context (single query vector)
        role_vec = model.encode(
            context_str,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).reshape(1, -1)   # shape: (1, embedding_dim)

        # Compute similarity against all skills at once
        sims = cosine_similarity_matrix(role_vec, skill_embeddings)[0]  # (n_skills,)

        # Collect skills above threshold
        existing_lower = set(s.lower() for s in existing_skills)
        added = []

        for idx, sim_score in enumerate(sims):
            if sim_score >= SIMILARITY_THRESHOLD:
                skill_raw = all_skills[idx]
                skill_title = skill_raw.title()
                if skill_raw.lower() not in existing_lower:
                    existing_lower.add(skill_raw.lower())
                    added.append(skill_title)

        # Merge: keep original broad skills + append new specifics (sorted)
        final_skills = existing_skills + sorted(added)

        enriched_roles[role_key] = {
            **role_data,
            "skills": final_skills,
        }

    # 6. Summarise
    total_added = sum(
        len(enriched_roles[k]["skills"]) - len(roles_dict[k].get("skills", []))
        for k in roles_dict
    )
    print("\n  Total new skill assignments injected: {:,}".format(total_added))

    # 7. Write output
    print("\nSaving enriched dictionary -> {} ...".format(OUTPUT_FILE.name))
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(enriched_roles, fh, indent=2, ensure_ascii=False)

    print("  Done! Written to: {}".format(OUTPUT_FILE))
    print("------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
