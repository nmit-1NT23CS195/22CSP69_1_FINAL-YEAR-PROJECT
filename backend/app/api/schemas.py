from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# 1. Strict Nested Models (No Dicts allowed)
class SkillNode(BaseModel):
    skill_name: str
    proficiency_score: int
    estimated_yoe: float
    context_proof: str

class RoleMatch(BaseModel):
    role: str
    match_percentage: int
    rationale: str

class DSABridgeNode(BaseModel):
    project_logic: str
    dsa_concept: str

class StarRewrite(BaseModel):
    original_bullet: str
    rewritten_bullet: str

# 2. The Main Payload
class CognitiveAnalysis(BaseModel):
    skill_matrix: Optional[List[SkillNode]] = Field(default_factory=list)
    bullshit_detector: Optional[List[str]] = Field(default_factory=list)
    best_fit_roles: Optional[List[RoleMatch]] = Field(default_factory=list)
    pivot_opportunities: Optional[List[str]] = Field(default_factory=list)
    targeted_questions: Optional[List[str]] = Field(default_factory=list)
    dsa_bridge: Optional[List[DSABridgeNode]] = Field(default_factory=list)
    micro_project_suggestion: Optional[str] = ""
    llm_diagnosis_score: Optional[float] = 0.0
    implicit_ghost_skills: Optional[List[str]] = Field(default_factory=list)
    star_bullet_rewrite: Optional[StarRewrite] = None

class ATSAnalysisResponse(BaseModel):
    ats_score: float
    keyword_metrics: Dict[str, Any]
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    contextual_skill_weights: Dict[str, float]
    estimated_experience: Dict[str, Any]
    soft_skills_found: List[str]
    action_verbs_found: List[str]
    llm_enriched_skills: Dict[str, float]
    cognitive_analysis: Optional[CognitiveAnalysis] = None

class ResumeUploadResponse(BaseModel):
    resume_text: str
