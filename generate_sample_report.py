"""
generate_sample_report.py
──────────────────────────
Generates a sample shortlist report PDF using realistic mock data.
This demonstrates the report output format WITHOUT requiring an API key.

Run:  python generate_sample_report.py
"""

from schemas import (
    JobRequirements,
    CandidateProfile,
    CandidateScorecard,
    DimensionScore,
    HROverride,
)
from agents.report_generator import generate_pdf_report
from pathlib import Path


def main():
    # ── Mock JD ────────────────────────────────────────────────────────────
    jd = JobRequirements(
        role_title="Senior Machine Learning Engineer",
        required_skills=[
            "Python", "PyTorch", "TensorFlow", "LLM fine-tuning",
            "RAG", "LangChain", "MLOps", "Docker", "Kubernetes",
            "SQL", "NoSQL",
        ],
        preferred_skills=[
            "AWS SageMaker", "GCP Vertex AI", "FinTech domain",
            "Open-source contributions",
        ],
        min_experience_years=4.0,
        education_requirement="B.Tech/M.Tech in CS or related field",
        preferred_certifications=["AWS ML Specialty", "GCP Professional Data Engineer"],
        domain="FinTech / AI",
        key_responsibilities=[
            "Build production ML pipelines for credit risk and fraud detection",
            "Fine-tune and deploy LLMs for document understanding",
            "Implement MLOps best practices (MLflow, CI/CD)",
            "Design scalable inference APIs",
            "Mentor junior engineers",
        ],
        seniority_level="Senior",
    )

    # ── Mock Candidates ────────────────────────────────────────────────────
    scorecards = [
        CandidateScorecard(
            candidate_id="cand_001",
            full_name="Arjun Sharma",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=9.0, weighted_score=2.70, justification="Covers Python, PyTorch, TensorFlow, LangChain, Docker, K8s — 90%+ skills match."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=8.5, weighted_score=2.125, justification="5 years in ML at FinTech companies; directly relevant domain and seniority."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=9.0, weighted_score=1.35, justification="M.Tech from IIT; AWS ML Specialty certification — exceeds requirements."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=8.0, weighted_score=1.20, justification="Strong portfolio: fraud detection system, RAG pipeline, open-source contribution."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=8.0, weighted_score=1.20, justification="Resume is well-structured with clear impact metrics and action verbs."),
            ],
            total_weighted_score=8.575,
            recommendation="STRONG HIRE",
            summary="Exceptional candidate with 5 years of directly relevant ML experience in FinTech. Strong technical depth across the full stack — from model training to production deployment. AWS-certified with publications. Top pick for this role.",
            skill_gaps=["GCP Vertex AI"],
            semantic_similarity_score=0.823,
        ),
        CandidateScorecard(
            candidate_id="cand_002",
            full_name="Priya Nair",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=8.5, weighted_score=2.55, justification="Strong Python, PyTorch, and LangChain skills. Missing Kubernetes experience."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=7.5, weighted_score=1.875, justification="4 years ML experience but in Healthcare domain, not FinTech."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=8.0, weighted_score=1.20, justification="M.Tech in Data Science; GCP certification — meets requirements."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=7.0, weighted_score=1.05, justification="Good portfolio with NLP projects, but no FinTech-specific work."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=7.5, weighted_score=1.125, justification="Clear and structured resume, good use of quantitative achievements."),
            ],
            total_weighted_score=7.775,
            recommendation="HIRE",
            summary="Strong ML engineer with 4 years of experience and solid technical skills. Domain is Healthcare rather than FinTech, but skills are highly transferable. Would ramp up quickly.",
            skill_gaps=["Kubernetes", "FinTech domain", "MLflow"],
            semantic_similarity_score=0.712,
        ),
        CandidateScorecard(
            candidate_id="cand_003",
            full_name="Ananya Das",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=6.0, weighted_score=1.80, justification="Has Python and TensorFlow but lacks LangChain, RAG, and MLOps skills."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=5.5, weighted_score=1.375, justification="2.5 years experience — below the 4-year minimum. Adjacent domain."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=7.0, weighted_score=1.05, justification="B.Tech in CS from a reputed university. No certifications yet."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=6.5, weighted_score=0.975, justification="2 academic ML projects. No production or industry portfolio."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=6.0, weighted_score=0.90, justification="Resume is functional but lacks impact metrics and specificity."),
            ],
            total_weighted_score=6.125,
            recommendation="CONSIDER",
            summary="Junior candidate with potential but falls short on experience (2.5 years vs 4 required) and lacks several required skills. Could be a fit for a mid-level role.",
            skill_gaps=["LangChain", "RAG", "Docker", "Kubernetes", "MLflow", "LLM fine-tuning"],
            semantic_similarity_score=0.534,
        ),
        CandidateScorecard(
            candidate_id="cand_004",
            full_name="Rahul Verma",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=5.5, weighted_score=1.65, justification="Knows Python and basic ML but lacks deep learning, LLM, and MLOps skills."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=5.0, weighted_score=1.25, justification="3 years in data analytics, not ML engineering. Different role type."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=6.0, weighted_score=0.90, justification="B.Tech in IT — meets minimum. No relevant certifications."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=4.5, weighted_score=0.675, justification="Dashboard and BI projects — not ML model development."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=5.5, weighted_score=0.825, justification="Adequate resume but lacks technical depth in descriptions."),
            ],
            total_weighted_score=5.25,
            recommendation="CONSIDER",
            summary="Data analytics background with some ML exposure. Significant skill gaps in deep learning, LLMs, and MLOps. Better suited for a data analyst or junior ML role.",
            skill_gaps=["PyTorch", "TensorFlow", "LangChain", "LLM fine-tuning", "Docker", "Kubernetes", "MLflow", "RAG"],
            semantic_similarity_score=0.398,
        ),
        CandidateScorecard(
            candidate_id="cand_005",
            full_name="Deepak Mehta",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=3.0, weighted_score=0.90, justification="Frontend developer — Python is secondary. No ML/DL/LLM skills."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=2.0, weighted_score=0.50, justification="4 years of web development — completely different domain and role."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=5.0, weighted_score=0.75, justification="B.Tech in CS meets minimum but no ML-related coursework."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=2.5, weighted_score=0.375, justification="React/Node.js projects — no ML or data science portfolio."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=6.0, weighted_score=0.90, justification="Well-written resume but for a completely different role."),
            ],
            total_weighted_score=3.25,
            recommendation="NO HIRE",
            summary="Experienced frontend developer but completely misaligned with the ML Engineer role. No relevant ML skills, experience, or portfolio.",
            skill_gaps=["Python (advanced)", "PyTorch", "TensorFlow", "LangChain", "RAG", "LLM", "MLOps", "Docker", "Kubernetes"],
            semantic_similarity_score=0.187,
        ),
        CandidateScorecard(
            candidate_id="cand_006",
            full_name="Sneha Kulkarni",
            dimensions=[
                DimensionScore(dimension="Skills Match", weight=0.30, raw_score=2.5, weighted_score=0.75, justification="Marketing background — no technical ML skills listed."),
                DimensionScore(dimension="Experience Relevance", weight=0.25, raw_score=1.5, weighted_score=0.375, justification="3 years in digital marketing — zero ML engineering overlap."),
                DimensionScore(dimension="Education & Certs", weight=0.15, raw_score=3.0, weighted_score=0.45, justification="MBA in Marketing — does not meet technical education requirement."),
                DimensionScore(dimension="Project / Portfolio", weight=0.15, raw_score=1.5, weighted_score=0.225, justification="Marketing campaigns and SEO projects — no ML relevance."),
                DimensionScore(dimension="Communication Quality", weight=0.15, raw_score=7.0, weighted_score=1.05, justification="Excellent writing quality but wrong domain entirely."),
            ],
            total_weighted_score=2.575,
            recommendation="NO HIRE",
            summary="Strong marketing professional but a complete mismatch for an ML Engineer role. No technical skills, no CS education, no relevant projects.",
            skill_gaps=["Python", "PyTorch", "TensorFlow", "LangChain", "RAG", "LLM", "MLOps", "Docker", "Kubernetes", "SQL"],
            semantic_similarity_score=0.098,
        ),
    ]

    # ── Mock Override ──────────────────────────────────────────────────────
    overrides = [
        HROverride(
            candidate_id="cand_003",
            overridden_by="Priya Sharma – HR Lead",
            original_score=6.125,
            adjusted_score=6.8,
            reason="Candidate showed strong culture fit and learning agility in preliminary phone screen. Portfolio links not captured in resume — GitHub shows 3 additional ML projects.",
            timestamp="2026-05-10 14:23:45",
        ),
    ]

    # ── Generate ───────────────────────────────────────────────────────────
    out_path = Path("sample_output") / "sample_shortlist_report.pdf"
    result = generate_pdf_report(
        jd=jd,
        scorecards=scorecards,
        overrides=overrides,
        output_path=out_path,
        model_used="gpt-4o (2024-08-06)",
    )
    print(f"✅ Sample report generated: {result}")
    print(f"   Open: file://{Path(result).resolve()}")


if __name__ == "__main__":
    main()
