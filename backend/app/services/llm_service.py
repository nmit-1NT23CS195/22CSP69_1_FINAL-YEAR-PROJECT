import json
import logging
import os
import httpx
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LLM_API_URL: Optional[str] = os.environ.get("LLM_API_URL", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent")

def get_api_key() -> Optional[str]:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key: return key
    
    # Fallback to manual .env parsing
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY=") or line.startswith("LLM_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return None

_LLM_API_KEY = get_api_key()

_SYSTEM_PROMPT = """You are a panel consisting of a Senior Technical Lead, an HR Director, and a System Architect.
Analyze the candidate's resume against the tech industry standards.

1. THE FORENSIC PROFICIENCY ENGINE
Cross-reference every stated technical skill against the listed projects and experience. Provide a skill_matrix array of objects (skill_name, proficiency_score, estimated_yoe, context_proof). Flag skills with zero project evidence in the bullshit_detector array. Identify implicit_ghost_skills (skills not explicitly stated but clearly implied by the experience).

2. THE CAREER CONSTELLATION
Determine the best_fit_roles (array of objects with role, match_percentage, and rationale). Suggest pivot_opportunities.

3. THE BRUTAL INTERVIEWER
Generate 3-5 targeted_questions testing their weakest or unverified skills. Create a dsa_bridge mapping their project logic to core DSA concepts.

4. THE ACTIONABLE ROADMAP
Generate a micro_project_suggestion (2 sentences) to bridge their exact gap.

5. THE STAR BULLET REWRITER
Select one weak bullet point from their experience and provide a star_bullet_rewrite object with original_bullet and rewritten_bullet to a high-impact STAR-formatted version.

Finally, output an llm_diagnosis_score (0.0 to 1.0) representing overall ATS and job readiness.

Return ONLY valid JSON matching the exact schema provided.
"""

_USER_PROMPT_TEMPLATE = """RESUME TEXT:
{resume_text}

JOB DESCRIPTION/CONTEXT:
{jd_text}
"""

_GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "skill_matrix": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "proficiency_score": {"type": "integer"},
                    "estimated_yoe": {"type": "number"},
                    "context_proof": {"type": "string"}
                },
                "required": ["skill_name", "proficiency_score", "estimated_yoe", "context_proof"]
            }
        },
        "bullshit_detector": {
            "type": "array",
            "items": {"type": "string"}
        },
        "best_fit_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "match_percentage": {"type": "integer"},
                    "rationale": {"type": "string"}
                },
                "required": ["role", "match_percentage", "rationale"]
            }
        },
        "pivot_opportunities": {
            "type": "array",
            "items": {"type": "string"}
        },
        "targeted_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dsa_bridge": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "project_logic": {"type": "string"},
                    "dsa_concept": {"type": "string"}
                },
                "required": ["project_logic", "dsa_concept"]
            }
        },
        "micro_project_suggestion": {"type": "string"},
        "llm_diagnosis_score": {"type": "number"},
        "implicit_ghost_skills": {
            "type": "array",
            "items": {"type": "string"}
        },
        "star_bullet_rewrite": {
            "type": "object",
            "properties": {
                "original_bullet": {"type": "string"},
                "rewritten_bullet": {"type": "string"}
            },
            "required": ["original_bullet", "rewritten_bullet"]
        }
    },
    "required": [
        "skill_matrix", "bullshit_detector", "best_fit_roles", 
        "pivot_opportunities", "targeted_questions", "dsa_bridge", 
        "micro_project_suggestion", "llm_diagnosis_score",
        "implicit_ghost_skills", "star_bullet_rewrite"
    ]
}

def run_cognitive_analysis(resume_text: str, jd_text: str) -> Dict[str, Any]:
    if not _LLM_API_KEY:
        logger.info("LLM_API_KEY not configured. Skipping cognitive analysis.")
        return {}

    truncated_resume = resume_text[:8000]
    truncated_jd = jd_text[:2000]

    url = f"{_LLM_API_URL}?key={_LLM_API_KEY}"
    if "?" in _LLM_API_URL and "key=" not in _LLM_API_URL:
        url = f"{_LLM_API_URL}&key={_LLM_API_KEY}"
    elif "key=" in _LLM_API_URL:
        url = _LLM_API_URL

    payload = {
        "systemInstruction": {
            "parts": [{"text": _SYSTEM_PROMPT}]
        },
        "contents": [{
            "parts": [{"text": _USER_PROMPT_TEMPLATE.format(resume_text=truncated_resume, jd_text=truncated_jd)}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_SCHEMA
        }
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()

        data = response.json()
        content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if not content:
            # Fallback for OpenAI-compat
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return json.loads(content)
    except Exception as exc:
        logger.error(f"LLM Cognitive analysis request failed: {exc}")
        return {}
