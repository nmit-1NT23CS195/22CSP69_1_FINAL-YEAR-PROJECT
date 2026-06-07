"""
llm_service.py
==============
LLM Cognitive Analysis — Concurrent Dual-Call Architecture.

ARCHITECTURE:
    Instead of one massive monolithic schema call (which saturates Gemini's
    output token budget and stalls the event loop for 90-120s), we split the
    original schema into two smaller, focused schemas and fire them concurrently
    using asyncio.gather().

    CoreScoringSchema  →  skill_matrix, roles, ghost skills, star rewrite
    DeepAnalysisSchema →  targeted_questions, dsa_bridge, micro_project

    Expected wall-clock improvement: ~90-120s → ~35-55s (limited by the
    slower of the two parallel calls, not their sum).

CRITICAL RULES (do not alter):
    - The merged return dict must preserve ALL original keys consumed by
      ats_service.py  (skill_matrix, bullshit_detector, best_fit_roles,
      pivot_opportunities, targeted_questions, dsa_bridge,
      micro_project_suggestion, llm_diagnosis_score,
      implicit_ghost_skills, star_bullet_rewrite).
    - Each Gemini call uses httpx.AsyncClient (non-blocking) so FastAPI's
      event loop is never stalled.
    - Each call is independently wrapped in try/except — failure of one
      does not abort the other (graceful degradation).
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_LLM_API_URL: str = os.environ.get(
    "LLM_API_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
)


def get_api_key() -> Optional[str]:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Fallback: manual .env file parsing
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY=") or line.startswith("LLM_API_KEY="):
                    return line.strip().split("=", 1)[1]
    return None


_LLM_API_KEY = get_api_key()

# ---------------------------------------------------------------------------
# CALL 1: CoreScoringSchema
# Fields: skill_matrix, bullshit_detector, implicit_ghost_skills,
#         best_fit_roles, pivot_opportunities, llm_diagnosis_score,
#         star_bullet_rewrite
#
# These fields are data-dense but SHORT per-item. Gemini generates them fast.
# They are needed immediately to render the score and skill matrix UI.
# ---------------------------------------------------------------------------
_CORE_SYSTEM_PROMPT = """\
You are a Senior Technical Lead and HR Director auditing a candidate's resume.

Your job is to forensically analyse the resume and output ONLY a valid JSON object.

1. SKILL FORENSICS
   Build a skill_matrix: for each stated technical skill, assign a proficiency_score (0-100),
   an estimated_yoe (years, float), and a context_proof (one sentence from the resume that
   proves usage). List skills with ZERO project evidence in bullshit_detector.
   List skills implied but not stated in implicit_ghost_skills.

2. CAREER CONSTELLATION
   Identify best_fit_roles (role, match_percentage 0-100, rationale) and pivot_opportunities.

3. DIAGNOSIS
   Output llm_diagnosis_score (0.0-1.0) for overall ATS readiness.

4. STAR REWRITE
   Pick the weakest bullet point. Return star_bullet_rewrite with original_bullet and
   rewritten_bullet (high-impact STAR format).

Return ONLY valid JSON. No markdown. No explanation.
"""

_CORE_SCHEMA: Dict[str, Any] = {
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
                    "context_proof": {"type": "string"},
                },
                "required": ["skill_name", "proficiency_score", "estimated_yoe", "context_proof"],
            },
        },
        "bullshit_detector": {
            "type": "array",
            "items": {"type": "string"},
        },
        "implicit_ghost_skills": {
            "type": "array",
            "items": {"type": "string"},
        },
        "best_fit_roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "match_percentage": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["role", "match_percentage", "rationale"],
            },
        },
        "pivot_opportunities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "llm_diagnosis_score": {"type": "number"},
        "star_bullet_rewrite": {
            "type": "object",
            "properties": {
                "original_bullet": {"type": "string"},
                "rewritten_bullet": {"type": "string"},
            },
            "required": ["original_bullet", "rewritten_bullet"],
        },
        "matched_certifications": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "skill_matrix",
        "bullshit_detector",
        "implicit_ghost_skills",
        "best_fit_roles",
        "pivot_opportunities",
        "llm_diagnosis_score",
        "star_bullet_rewrite",
        "matched_certifications",
    ],
}

# ---------------------------------------------------------------------------
# CALL 2: DeepAnalysisSchema
# Fields: targeted_questions, dsa_bridge, micro_project_suggestion
#
# These are long-form prose fields — the heaviest output token consumers.
# Running them concurrently with Call 1 means we no longer wait serially.
# ---------------------------------------------------------------------------
_DEEP_SYSTEM_PROMPT = """\
You are a ruthless Senior Engineer conducting a focused technical interview.
You are evaluating a candidate based on a pre-parsed JSON profile below.
Output ONLY a valid JSON object. No markdown. No explanation.

CRITICAL OUTPUT LIMITS — STRICTLY ENFORCED:
- targeted_questions: Generate EXACTLY 2 questions. Each must be hostile and highly
  specific, targeting a concrete weakness or unverified claim in the candidate's
  skill_matrix or project_sections. Reference a specific skill or project by name.
  For each question, provide an expected_answer that is EXACTLY 1-2 concise sentences
  summarising the key criteria that would constitute a strong answer.
- dsa_bridge: For each project in the candidate's project sections, identify the
  underlying DSA concept. Each dsa_concept MUST be specific (e.g. "sliding window",
  not just "array"). The project_logic explanation must be under 3 sentences.
- micro_project_suggestion: EXACTLY 2 sentences bridging the candidate's exact skill
  gap to the JD requirements.
"""

_DEEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "targeted_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "expected_answer": {"type": "string"},
                },
                "required": ["question", "expected_answer"],
            },
        },
        "dsa_bridge": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "project_logic": {"type": "string"},
                    "dsa_concept": {"type": "string"},
                },
                "required": ["project_logic", "dsa_concept"],
            },
        },
        "micro_project_suggestion": {"type": "string"},
    },
    "required": ["targeted_questions", "dsa_bridge", "micro_project_suggestion"],
}

# Shared user prompt template for CORE call (raw text)
_USER_PROMPT_TEMPLATE = """\
RESUME TEXT:
{resume_text}

JOB DESCRIPTION:
{jd_text}
"""

# Structured user prompt for DEEP call (pre-parsed JSON — compact, low token count)
_DEEP_USER_PROMPT_TEMPLATE = """\
COMPACT CANDIDATE PROFILE (pre-parsed JSON — use this instead of raw resume):
{structured_profile}

JOB DESCRIPTION (truncated):
{jd_text}
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_url() -> str:
    """Construct the Gemini API URL with the API key appended."""
    if "key=" in _LLM_API_URL:
        return _LLM_API_URL
    sep = "&" if "?" in _LLM_API_URL else "?"
    return f"{_LLM_API_URL}{sep}key={_LLM_API_KEY}"


def _build_payload(system_prompt: str, schema: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    """Construct the Gemini generateContent request body."""
    return {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }


# ---------------------------------------------------------------------------
# Typed fallback dicts — returned when a Gemini call fails.
# Every key the downstream merger expects must be present so Pydantic
# never sees a missing field and never raises a 422.
# ---------------------------------------------------------------------------
_CORE_FALLBACK: Dict[str, Any] = {
    "skill_matrix": [],
    "bullshit_detector": [],
    "implicit_ghost_skills": [],
    "best_fit_roles": [],
    "pivot_opportunities": [],
    "llm_diagnosis_score": 0.0,
    "star_bullet_rewrite": None,
    "matched_certifications": [],
}

_DEEP_FALLBACK: Dict[str, Any] = {
    "targeted_questions": [],
    "dsa_bridge": [],
    "micro_project_suggestion": "",
}


async def _call_gemini_async(
    client: httpx.AsyncClient,
    system_prompt: str,
    schema: Dict[str, Any],
    user_text: str,
    call_name: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fire a single async Gemini API call and return the parsed JSON dict.

    On failure the typed `fallback` dict is returned so the merge step always
    receives every expected key — preventing Pydantic 422 / KeyError crashes.

    Error handling priority:
        1. httpx.HTTPStatusError with status 429 → log "Rate Limit Exceeded"
        2. Any other httpx.HTTPStatusError            → log HTTP error detail
        3. General Exception                          → log raw error
    No retries are attempted (fail-fast + graceful degradation).
    """
    url = _build_url()
    payload = _build_payload(system_prompt, schema, user_text)
    try:
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        if not content:
            # OpenAI-compat fallback
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        result = json.loads(content)
        logger.info("LLM [%s] completed successfully.", call_name)
        return result

    except httpx.HTTPStatusError as http_err:
        # Fix 3 — explicit 429 detection before the generic HTTP error handler
        if http_err.response.status_code == 429:
            logger.error(
                "LLM [%s] Rate Limit Exceeded (429). Returning typed fallback.",
                call_name,
            )
        else:
            logger.error(
                "LLM [%s] HTTP error %s: %s. Returning typed fallback.",
                call_name,
                http_err.response.status_code,
                http_err,
            )
        return fallback

    except Exception as exc:
        logger.error("LLM [%s] failed: %s. Returning typed fallback.", call_name, exc)
        return fallback


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_cognitive_analysis_concurrent(
    resume_text: str,
    jd_text: str,
) -> Dict[str, Any]:
    """
    Fire CoreScoringSchema and DeepAnalysisSchema calls concurrently via
    asyncio.gather(). Merge the results into one dict that is backward-
    compatible with the original monolithic CognitiveAnalysis schema.

    Wall-clock time ≈ max(core_time, deep_time) instead of core_time + deep_time.

    Args:
        resume_text: Extracted resume text (will be truncated internally).
        jd_text:     Job description text (will be truncated internally).

    Returns:
        Merged dict with all 10 original cognitive_analysis keys populated.
    """
    if not _LLM_API_KEY:
        logger.info("LLM_API_KEY not configured. Skipping cognitive analysis.")
        return {}

    # Truncate to stay well within Gemini's input token budget
    truncated_resume = resume_text[:8000]
    truncated_jd = jd_text[:2000]
    user_text = _USER_PROMPT_TEMPLATE.format(
        resume_text=truncated_resume, jd_text=truncated_jd
    )

    # Fix 1 — async with guarantees the connection pool is closed even on error.
    # Fix 2 — typed fallback dicts passed per-call so failures never return {}.
    async with httpx.AsyncClient(timeout=120.0) as client:
        core_result, deep_result = await asyncio.gather(
            _call_gemini_async(
                client, _CORE_SYSTEM_PROMPT, _CORE_SCHEMA, user_text,
                "CoreScoring", fallback=_CORE_FALLBACK,
            ),
            _call_gemini_async(
                client, _DEEP_SYSTEM_PROMPT, _DEEP_SCHEMA, user_text,
                "DeepAnalysis", fallback=_DEEP_FALLBACK,
            ),
        )

    # Merge both results into one backward-compatible dict
    merged: Dict[str, Any] = {
        # --- From CoreScoringSchema ---
        "skill_matrix":          core_result.get("skill_matrix", []),
        "bullshit_detector":     core_result.get("bullshit_detector", []),
        "implicit_ghost_skills": core_result.get("implicit_ghost_skills", []),
        "best_fit_roles":        core_result.get("best_fit_roles", []),
        "pivot_opportunities":   core_result.get("pivot_opportunities", []),
        "llm_diagnosis_score":   core_result.get("llm_diagnosis_score", 0.0),
        "star_bullet_rewrite":   core_result.get("star_bullet_rewrite", None),

        # --- From DeepAnalysisSchema ---
        "targeted_questions":       deep_result.get("targeted_questions", []),
        "dsa_bridge":               deep_result.get("dsa_bridge", []),
        "micro_project_suggestion": deep_result.get("micro_project_suggestion", ""),
    }

    return merged


# ---------------------------------------------------------------------------
# Legacy synchronous fallback (preserved for compatibility / CLI scripts)
# ---------------------------------------------------------------------------

def run_cognitive_analysis(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """
    Synchronous wrapper around the concurrent async function.
    Used only by legacy scripts or pytest fixtures that cannot await.
    For production FastAPI routes, call run_cognitive_analysis_concurrent() directly.
    """
    try:
        return asyncio.run(run_cognitive_analysis_concurrent(resume_text, jd_text))
    except RuntimeError:
        # Already inside a running event loop (e.g. Jupyter / nested async context)
        logger.warning(
            "run_cognitive_analysis() called inside a running event loop. "
            "Use 'await run_cognitive_analysis_concurrent()' instead."
        )
        return {}


# ---------------------------------------------------------------------------
# PUBLIC STAGE-SPLIT API
# Used by the two-stage loading pipeline: /ats/score/core and /ats/score/deep
# Each function owns its own AsyncClient — no shared state between stages.
# ---------------------------------------------------------------------------

async def run_core_analysis(
    resume_text: str,
    jd_text: str,
    role_mode_context: str = "",
) -> Dict[str, Any]:
    """
    Stage 1 — Core Scoring Call.

    Fires only the CoreScoringSchema against Gemini and returns the result.
    Called by run_core_pipeline() which also runs all deterministic stages.

    Args:
        resume_text:        Raw resume text.
        jd_text:            Job description or synthetic JD text.
        role_mode_context:  If non-empty, injected as a CRITICAL RULE to shift
                            the LLM's scoring weight toward role-specific certs.

    Returns a fully populated dict (or _CORE_FALLBACK on any failure):
        skill_matrix, bullshit_detector, implicit_ghost_skills,
        best_fit_roles, pivot_opportunities, llm_diagnosis_score,
        star_bullet_rewrite, matched_certifications
    """
    if not _LLM_API_KEY:
        logger.info("LLM_API_KEY not configured. Returning core fallback.")
        return _CORE_FALLBACK

    # Build the system prompt — inject role-mode rule if supplied
    system_prompt = _CORE_SYSTEM_PROMPT
    if role_mode_context:
        system_prompt = (
            _CORE_SYSTEM_PROMPT.rstrip()
            + "\n\nCRITICAL RULE (ROLE MODE ACTIVE): You have been provided structured "
            "Role Data instead of a raw JD. You MUST heavily weight the "
            "llm_diagnosis_score based on the explicit mathematical overlap of skills "
            "and the presence of matched_certifications listed below. Give massive "
            "bonus points to candidates possessing these semantically matched certs.\n"
            + role_mode_context
        )

    user_text = _USER_PROMPT_TEMPLATE.format(
        resume_text=resume_text[:8000],
        jd_text=jd_text[:2000],
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        return await _call_gemini_async(
            client, system_prompt, _CORE_SCHEMA,
            user_text, "CoreScoring", fallback=_CORE_FALLBACK,
        )


async def run_deep_analysis(
    resume_text: str,
    jd_text: str,
    skill_matrix: list = None,
    resume_sections: dict = None,
) -> Dict[str, Any]:
    """
    Stage 2 — Deep Analysis Call.

    Accepts a pre-parsed structured profile (skill_matrix + resume_sections)
    to dramatically reduce input token count vs. sending raw resume_text.
    Falls back to raw resume_text if structured data is not provided.

    Returns a fully populated dict (or _DEEP_FALLBACK on any failure):
        targeted_questions (list of {question, expected_answer} objects),
        dsa_bridge, micro_project_suggestion
    """
    if not _LLM_API_KEY:
        logger.info("LLM_API_KEY not configured. Returning deep fallback.")
        return _DEEP_FALLBACK

    # Build a compact structured profile if pre-parsed data is available
    if skill_matrix is not None and resume_sections is not None:
        # Only send the sections that have dense project/experience content
        key_sections = {k: v[:600] for k, v in resume_sections.items()
                        if k in ("experience", "projects", "education", "summary") and v}
        structured_profile = json.dumps({
            "skill_matrix": skill_matrix,       # LLM-parsed skills with YoE + context_proof
            "project_sections": key_sections,   # structured resume sections (truncated)
        }, indent=None, separators=(',', ':'))  # compact JSON = fewer tokens
        user_text = _DEEP_USER_PROMPT_TEMPLATE.format(
            structured_profile=structured_profile[:6000],  # hard cap: ~1500 tokens
            jd_text=jd_text[:1500],
        )
    else:
        # Fallback: raw text path (legacy, higher token cost)
        user_text = _USER_PROMPT_TEMPLATE.format(
            resume_text=resume_text[:8000],
            jd_text=jd_text[:2000],
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        return await _call_gemini_async(
            client, _DEEP_SYSTEM_PROMPT, _DEEP_SCHEMA,
            user_text, "DeepAnalysis", fallback=_DEEP_FALLBACK,
        )
