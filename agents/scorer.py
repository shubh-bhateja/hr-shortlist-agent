"""
agents/scorer.py
─────────────────
LangGraph Node 3 — Score Agent

Dual-mode scoring (what makes this stand out vs. a basic LangChain chain):
  1. Semantic similarity: SentenceTransformer embeds the JD and each candidate
     profile, computing cosine similarity as a quantitative signal.
  2. LLM reasoning: GPT-4o evaluates each rubric dimension with structured JSON
     output (Pydantic-validated). The LLM receives the cosine sim score as context.

The combination produces scores that are both semantically grounded AND
contextually justified — reducing hallucination risk significantly.
"""

from __future__ import annotations
import logging
import json
from typing import Optional

import numpy as np
from utils.llm_factory import get_chat_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from schemas import (
    JobRequirements,
    CandidateProfile,
    CandidateScorecard,
    DimensionScore,
)

logger = logging.getLogger(__name__)

# ── Rubric definition (mirrors the brief exactly) ─────────────────────────────
RUBRIC = [
    {"dimension": "Skills Match",        "weight": 0.30},
    {"dimension": "Experience Relevance","weight": 0.25},
    {"dimension": "Education & Certs",   "weight": 0.15},
    {"dimension": "Project / Portfolio", "weight": 0.20},
    {"dimension": "Communication Quality","weight": 0.10},
]

# ── Semantic similarity ────────────────────────────────────────────────────────

_embedder = None  # lazy-loaded to avoid heavy import at startup

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, fast
    return _embedder


def _jd_to_text(jd: JobRequirements) -> str:
    parts = [
        jd.role_title,
        jd.domain,
        " ".join(jd.required_skills),
        " ".join(jd.preferred_skills),
        " ".join(jd.key_responsibilities),
    ]
    return " ".join(parts)


def _profile_to_text(profile: CandidateProfile) -> str:
    parts = [
        profile.current_title or "",
        " ".join(profile.skills),
        " ".join(profile.domains_worked_in),
        " ".join(profile.projects),
        " ".join(profile.education),
        profile.raw_text_snippet,
    ]
    return " ".join(parts)


def compute_semantic_similarity(jd: JobRequirements, profile: CandidateProfile) -> float:
    """Cosine similarity between JD embedding and candidate profile embedding."""
    embedder = _get_embedder()
    jd_text = _jd_to_text(jd)
    prof_text = _profile_to_text(profile)
    embeddings = embedder.encode([jd_text, prof_text], normalize_embeddings=True)
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return round(similarity, 4)


# ── LLM scoring prompt ────────────────────────────────────────────────────────

_SYSTEM = """You are a rigorous, unbiased HR evaluation expert.
Score a candidate against a Job Description using the provided rubric.

Scoring scale: 0 = Poor, 5 = Average, 10 = Excellent (as per brief).
Be strict. Reserve 9–10 for exceptional evidence, not assumptions.

Return a JSON object with this exact structure:
{
  "dimensions": [
    {
      "dimension": "<name>",
      "raw_score": <0-10 float>,
      "justification": "<one sentence, specific>"
    },
    ...
  ],
  "skill_gaps": ["<skill1>", "<skill2>"],
  "summary": "<2-3 sentence overall assessment>",
  "recommendation": "<STRONG HIRE | HIRE | CONSIDER | NO HIRE>"
}

recommendation thresholds (use weighted total as guidance):
  8.0–10.0 → STRONG HIRE
  6.5–7.9  → HIRE
  5.0–6.4  → CONSIDER
  0–4.9    → NO HIRE

Return ONLY valid JSON. No markdown. No preamble.
"""

_HUMAN = """RUBRIC DIMENSIONS (with weights):
{rubric_str}

JOB DESCRIPTION REQUIREMENTS:
Role: {role_title}
Domain: {domain}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Min Experience: {min_exp} years
Education Required: {education_req}
Preferred Certs: {preferred_certs}
Seniority: {seniority}
Key Responsibilities: {responsibilities}

CANDIDATE PROFILE:
Name: {name}
Current Title: {current_title}
Total Experience: {experience} years
Skills: {skills}
Education: {education}
Certifications: {certs}
Domains Worked In: {domains}
Projects / Portfolio: {projects}

SEMANTIC SIMILARITY SCORE (embedding cosine similarity, 0–1): {similarity}
(Use this as a quantitative signal to calibrate Skills Match and Experience Relevance, 
but use your own reasoning for the final scores.)

Score this candidate across all 5 dimensions. Return ONLY the JSON."""


def score_candidate(
    jd: JobRequirements,
    profile: CandidateProfile,
    model: str | None = None,
    semantic_sim: Optional[float] = None,
) -> CandidateScorecard:
    """Score one candidate against a JD. Returns a validated CandidateScorecard."""

    if semantic_sim is None:
        semantic_sim = compute_semantic_similarity(jd, profile)

    rubric_str = "\n".join(
        f"  • {r['dimension']} (weight {int(r['weight']*100)}%)" for r in RUBRIC
    )

    llm = get_chat_llm(model=model, temperature=0.0, json_mode=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser

    raw = chain.invoke({
        "rubric_str": rubric_str,
        "role_title": jd.role_title,
        "domain": jd.domain,
        "required_skills": ", ".join(jd.required_skills),
        "preferred_skills": ", ".join(jd.preferred_skills) or "None specified",
        "min_exp": jd.min_experience_years,
        "education_req": jd.education_requirement,
        "preferred_certs": ", ".join(jd.preferred_certifications) or "None",
        "seniority": jd.seniority_level,
        "responsibilities": "; ".join(jd.key_responsibilities),
        "name": profile.full_name,
        "current_title": profile.current_title or "Not specified",
        "experience": profile.total_experience_years,
        "skills": ", ".join(profile.skills) or "Not specified",
        "education": "; ".join(profile.education) or "Not specified",
        "certs": ", ".join(profile.certifications) or "None",
        "domains": ", ".join(profile.domains_worked_in) or "Not specified",
        "projects": "; ".join(profile.projects) or "None listed",
        "similarity": f"{semantic_sim:.3f}",
    })

    # Build validated DimensionScore objects
    weight_map = {r["dimension"]: r["weight"] for r in RUBRIC}
    dimensions = []
    for d in raw["dimensions"]:
        w = weight_map.get(d["dimension"], 0.0)
        raw_score = round(float(d["raw_score"]), 2)
        dimensions.append(DimensionScore(
            dimension=d["dimension"],
            weight=w,
            raw_score=raw_score,
            weighted_score=round(raw_score * w, 3),
            justification=d["justification"],
        ))

    total = round(sum(d.weighted_score for d in dimensions), 3)

    scorecard = CandidateScorecard(
        candidate_id=profile.candidate_id,
        full_name=profile.full_name,
        dimensions=dimensions,
        total_weighted_score=total,
        recommendation=raw["recommendation"],
        summary=raw["summary"],
        skill_gaps=raw.get("skill_gaps", []),
        semantic_similarity_score=semantic_sim,
    )

    logger.info(
        f"[{profile.candidate_id}] {profile.full_name}: "
        f"score={total:.2f}, rec={scorecard.recommendation}, sim={semantic_sim:.3f}"
    )
    return scorecard


# ── LangGraph node ─────────────────────────────────────────────────────────────

def score_all_candidates(state: dict) -> dict:
    """
    LangGraph node function.
    Input state keys:  job_requirements, candidate_profiles, model
    Output state keys: scorecards (sorted by score desc), scoring_errors
    """
    jd: JobRequirements = state["job_requirements"]
    profiles: list[CandidateProfile] = state.get("candidate_profiles", [])
    model: str | None = state.get("model")

    scorecards = []
    errors = []

    for profile in profiles:
        try:
            sim = compute_semantic_similarity(jd, profile)
            sc = score_candidate(jd, profile, model=model, semantic_sim=sim)
            scorecards.append(sc)
        except Exception as exc:
            logger.error(f"Scoring failed for {profile.candidate_id}: {exc}")
            errors.append({"candidate_id": profile.candidate_id, "error": str(exc)})

    # Sort descending by total weighted score
    scorecards.sort(key=lambda s: s.total_weighted_score, reverse=True)

    return {**state, "scorecards": scorecards, "scoring_errors": errors}
