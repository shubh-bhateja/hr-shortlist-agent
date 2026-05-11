"""
schemas.py – Pydantic models for every structured output in the pipeline.
Using strict schemas eliminates hallucination risk and makes every agent's
output machine-readable from day one.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────
# JD Parser Output
# ─────────────────────────────────────────────

class JobRequirements(BaseModel):
    """Structured extraction from a Job Description."""
    role_title: str = Field(description="Exact job title")
    required_skills: list[str] = Field(description="Must-have technical / functional skills")
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have skills")
    min_experience_years: float = Field(description="Minimum years of relevant experience required")
    education_requirement: str = Field(description="Minimum education qualification")
    preferred_certifications: list[str] = Field(default_factory=list)
    domain: str = Field(description="Industry / functional domain (e.g. FinTech, Healthcare AI)")
    key_responsibilities: list[str] = Field(description="Top 5 responsibilities extracted from JD")
    seniority_level: str = Field(description="One of: Intern, Junior, Mid, Senior, Lead, Manager")


# ─────────────────────────────────────────────
# Resume / Profile Parser Output
# ─────────────────────────────────────────────

class CandidateProfile(BaseModel):
    """Structured candidate data extracted from resume or LinkedIn JSON."""
    candidate_id: str = Field(description="Unique ID assigned during ingestion")
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    current_title: Optional[str] = None
    total_experience_years: float = Field(default=0.0)
    education: list[str] = Field(default_factory=list, description="Degrees / institutions")
    certifications: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    domains_worked_in: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list, description="Project titles or short descriptions")
    raw_text_snippet: str = Field(default="", description="First 500 chars of raw text for communication quality scoring")


# ─────────────────────────────────────────────
# Scoring Output
# ─────────────────────────────────────────────

class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""
    dimension: str
    weight: float = Field(description="Weight as a decimal e.g. 0.30")
    raw_score: float = Field(ge=0, le=10, description="Score from 0–10")
    weighted_score: float = Field(description="raw_score * weight")
    justification: str = Field(description="One-sentence explanation for this score")

    @field_validator("weighted_score", mode="before")
    @classmethod
    def compute_weighted(cls, v, info):
        # Allow explicit value; computed in agent if not set
        return v


class CandidateScorecard(BaseModel):
    """Full rubric scorecard for one candidate."""
    candidate_id: str
    full_name: str
    dimensions: list[DimensionScore]
    total_weighted_score: float = Field(description="Sum of all weighted_scores, max 10.0")
    recommendation: str = Field(description="One of: STRONG HIRE | HIRE | CONSIDER | NO HIRE")
    summary: str = Field(description="2–3 sentence overall assessment")
    skill_gaps: list[str] = Field(default_factory=list, description="Key skills from JD missing in candidate")
    semantic_similarity_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity between candidate embedding and JD embedding (0–1)"
    )

    @field_validator("recommendation")
    @classmethod
    def valid_recommendation(cls, v):
        allowed = {"STRONG HIRE", "HIRE", "CONSIDER", "NO HIRE"}
        if v not in allowed:
            raise ValueError(f"recommendation must be one of {allowed}")
        return v


# ─────────────────────────────────────────────
# Override Log
# ─────────────────────────────────────────────

class HROverride(BaseModel):
    """Human-in-the-loop score override record."""
    candidate_id: str
    overridden_by: str = Field(description="HR reviewer name or ID")
    original_score: float
    adjusted_score: float
    reason: str
    timestamp: str
