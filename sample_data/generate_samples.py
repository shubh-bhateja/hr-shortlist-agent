#!/usr/bin/env python3
"""
sample_data/generate_samples.py
─────────────────────────────────
Generates 6 realistic sample resume text files for testing the agent.
Mix: 2 strong matches · 2 partial matches · 2 no-matches

Run: python sample_data/generate_samples.py
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm

OUT = Path(__file__).parent
styles = getSampleStyleSheet()

SAMPLE_JD = """Senior Machine Learning Engineer – FinTech AI Division

We are hiring a Senior ML Engineer to build production-grade AI systems for our credit risk and fraud detection platforms.

Requirements:
- 4+ years of experience in applied Machine Learning / Deep Learning
- Proficiency in Python, PyTorch or TensorFlow
- Experience with LLM fine-tuning, RAG systems, or agentic AI frameworks (LangChain, LlamaIndex)
- Strong knowledge of MLOps: MLflow, Docker, Kubernetes, CI/CD pipelines
- Experience with SQL and NoSQL databases (PostgreSQL, MongoDB)
- Exposure to cloud platforms: AWS (SageMaker) or GCP (Vertex AI)
- Strong problem-solving and communication skills
- B.Tech/M.Tech in Computer Science, Data Science, or related field

Preferred:
- Publications or open-source contributions in ML
- AWS Certified ML Specialty or GCP Professional Data Engineer
- Experience in FinTech domain (credit scoring, fraud detection, risk modelling)

Seniority: Senior
Domain: FinTech / AI
"""

RESUMES = [
    {
        "filename": "resume_arjun_sharma_strong.pdf",
        "text": """ARJUN SHARMA
arjun.sharma@email.com | +91-9876543210 | github.com/arjunsharma | linkedin.com/in/arjunsharma

SUMMARY
Senior ML Engineer with 5+ years of experience building production AI systems in FinTech. Expert in PyTorch, LLM fine-tuning, and MLOps on AWS. Led fraud detection systems processing 10M+ transactions/day.

EXPERIENCE
Senior ML Engineer — PayTech Solutions (2020–Present, 4 years)
• Built end-to-end credit risk model (XGBoost + neural net ensemble) reducing default prediction error by 23%
• Fine-tuned LLaMA-2 for financial document summarisation using LoRA; deployed via SageMaker endpoints
• Implemented RAG pipeline with LangChain + FAISS for compliance Q&A chatbot
• Designed MLflow-based experiment tracking and model versioning across 6-engineer team
• Deployed models on Kubernetes (EKS); built CI/CD with GitHub Actions + Docker

ML Engineer — DataCore Analytics (2018–2020, 2 years)
• Developed fraud detection classifier (F1=0.94) using Gradient Boosting on PostgreSQL data
• Built feature engineering pipeline (Python, pandas, scikit-learn) for real-time inference

EDUCATION
M.Tech — Computer Science (AI specialisation) — IIT Delhi — 2018 (CGPA: 8.9/10)
B.Tech — Computer Science — NIT Trichy — 2016

SKILLS
Python · PyTorch · TensorFlow · LangChain · LlamaIndex · LLM fine-tuning · RAG · MLflow · Docker · Kubernetes · AWS SageMaker · GCP Vertex AI · PostgreSQL · MongoDB · FastAPI · Git

CERTIFICATIONS
AWS Certified Machine Learning – Specialty (2022)
GCP Professional Data Engineer (2023)

PROJECTS / PUBLICATIONS
• "Adaptive RAG for Financial Compliance" — arXiv preprint, 2023
• Open-source: langchain-fintools (820 GitHub stars)
• Credit Scoring Explainability Dashboard — Streamlit + SHAP
"""
    },
    {
        "filename": "resume_priya_nair_strong.pdf",
        "text": """PRIYA NAIR
priya.nair@techmail.com | +91-8765432109 | github.com/priyanair

PROFESSIONAL SUMMARY
ML Engineer with 4.5 years in applied deep learning and MLOps. Delivered production credit scoring and anomaly detection systems for two leading fintech firms. Published researcher in NLP.

WORK EXPERIENCE
ML Engineer — CreditAI Pvt. Ltd (2021–Present, 3.5 years)
• Designed transformer-based credit risk scoring model (AUROC 0.96) deployed on GCP Vertex AI
• Built LangGraph multi-agent pipeline for loan document verification, cutting review time 40%
• Implemented MLOps stack: MLflow + DVC + Docker + Kubernetes on GCP GKE
• Maintained PostgreSQL + MongoDB data warehouse feeding real-time model inference

Junior Data Scientist — Analytics House (2019–2021, 2 years)
• Built fraud detection models using Isolation Forest and LSTM; Python, scikit-learn, TensorFlow
• Automated data pipelines with Apache Airflow + AWS S3

EDUCATION
B.Tech — Computer Science — BITS Pilani — 2019 (CGPA: 9.1/10)

CERTIFICATIONS
GCP Professional Data Engineer
Deep Learning Specialisation — Coursera (Andrew Ng)

SKILLS
Python · PyTorch · TensorFlow · LangGraph · LangChain · MLflow · DVC · Docker · Kubernetes · GCP Vertex AI · AWS SageMaker · PostgreSQL · MongoDB · SQL · Airflow

PUBLICATIONS
"Multi-modal Fraud Detection using Graph Neural Networks" — AAAI Workshop 2023

PROJECTS
• Real-time fraud scoring API (FastAPI + Redis + Kafka)
• LLM-powered loan document classifier (LangChain + GPT-4)
• Explainable AI dashboard for credit decisions (SHAP + Streamlit)
"""
    },
    {
        "filename": "resume_rahul_verma_partial.pdf",
        "text": """RAHUL VERMA
rahul.verma@gmail.com | +91-9988776655

ABOUT ME
Software engineer with 3 years of experience in Python development and some exposure to data science and ML. Looking to transition fully into ML engineering.

EXPERIENCE
Backend Software Engineer — TechStartup India (2021–Present, 3 years)
• Built REST APIs with Python (FastAPI, Flask) and PostgreSQL
• Wrote automated tests; CI/CD pipelines with GitHub Actions and Docker
• Some exposure to pandas and scikit-learn for internal analytics dashboards

Intern — DataWorks (2020, 6 months)
• Data cleaning and EDA using pandas and matplotlib
• Built a logistic regression model for customer churn (accuracy 78%)

EDUCATION
B.Tech — Information Technology — VIT Vellore — 2021 (CGPA: 7.8/10)

SKILLS
Python · FastAPI · Flask · PostgreSQL · Docker · Git · GitHub Actions · pandas · scikit-learn · NumPy · JavaScript · React

PROJECTS
• E-commerce recommendation engine (collaborative filtering, scikit-learn)
• Sentiment analysis on Twitter data (NLTK, basic classifier)
• Personal finance tracker app (FastAPI + React)

CERTIFICATIONS
None

NOTES
Currently self-studying PyTorch through fast.ai course. Interested in LLMs and RAG.
"""
    },
    {
        "filename": "resume_ananya_das_partial.pdf",
        "text": """ANANYA DAS
ananya.das@outlook.com | 9123456780

SUMMARY
Data analyst with 2.5 years of experience in SQL, Python, and business intelligence. Transitioning into data science / ML with recent upskilling.

EXPERIENCE
Data Analyst — RetailCorp Analytics (2021–Present, 2.5 years)
• SQL queries on PostgreSQL for business reporting and KPI dashboards
• Python scripts for ETL automation (pandas, openpyxl)
• Power BI dashboards for sales and inventory management
• Collaborated with ML team to validate model outputs — basic understanding of ML lifecycle

Data Intern — StartupX (2020, 3 months)
• Data entry, cleaning, and basic EDA

EDUCATION
B.Sc — Statistics — Jadavpur University — 2021

CERTIFICATIONS
IBM Data Science Professional Certificate (Coursera, 2023)
— includes intro modules on scikit-learn, regression, classification

SKILLS
SQL · PostgreSQL · Python · pandas · NumPy · Power BI · Excel · scikit-learn (basic) · matplotlib

PROJECTS
• Customer segmentation using K-Means (side project)
• Sales forecasting with ARIMA (capstone project)

NOTES
Starting an online course on PyTorch. No production ML or deep learning experience yet.
"""
    },
    {
        "filename": "resume_deepak_mehta_nomatch.pdf",
        "text": """DEEPAK MEHTA
deepak.mehta@yahoo.com | 9001234567

OBJECTIVE
Experienced Java developer seeking a senior software engineering role in enterprise application development.

EXPERIENCE
Senior Java Developer — BankingSoft Ltd (2017–Present, 7 years)
• Developed core banking modules using Java Spring Boot, Hibernate, Oracle DB
• Led a 5-member team delivering customer-facing banking portals
• Performance optimisation of batch jobs (Spring Batch) processing 500K transactions/day
• REST API design and microservices architecture on AWS EC2

Software Engineer — Infosys (2014–2017, 3 years)
• Java EE application development for insurance domain
• Worked on SOAP/REST web services, Oracle PL/SQL

EDUCATION
B.Tech — Computer Science — RGPV — 2014

SKILLS
Java · Spring Boot · Spring Batch · Hibernate · Oracle DB · AWS EC2 · Microservices · REST APIs · Maven · Jenkins · Git

CERTIFICATIONS
Oracle Certified Java Programmer (2015)
AWS Solutions Architect – Associate (2021)

PROJECTS
• Core banking portal migration (monolith → microservices)
• Batch payment processing engine (Spring Batch + Oracle)

NOTE
No experience in Python, machine learning, data science, or AI/LLM frameworks.
"""
    },
    {
        "filename": "resume_sneha_kulkarni_nomatch.pdf",
        "text": """SNEHA KULKARNI
sneha.k@gmail.com | 8899001122

PROFILE
Recent MBA graduate with a marketing and product management background. Seeking a product manager or growth hacker role in a tech startup.

EXPERIENCE
Marketing Executive — FMCG Brand India (2021–2023, 2 years)
• Managed social media campaigns (Instagram, Facebook, LinkedIn)
• Ran A/B tests on ad copies using Meta Business Suite
• Coordinated with design team for campaign creatives
• Tracked campaign KPIs in Google Sheets and Tableau dashboards

Intern — Digital Agency (2020, 3 months)
• Content writing, SEO keyword research, Google Ads basic setup

EDUCATION
MBA — Marketing — Symbiosis Institute — 2021
BBA — Christ University Bangalore — 2019

SKILLS
Marketing strategy · Social media · SEO · Google Ads · Tableau · Excel · PowerPoint · Communication · Team management

CERTIFICATIONS
Google Digital Marketing Certificate (2020)
HubSpot Inbound Marketing (2022)

NOTE
No technical background. No programming or ML experience.
"""
    },
]

JD_PATH = OUT / "sample_jd.txt"
JD_PATH.write_text(SAMPLE_JD, encoding="utf-8")
print(f"Saved JD: {JD_PATH}")


def make_resume_pdf(filename: str, text: str):
    """Create a simple PDF resume."""
    path = OUT / filename
    doc = SimpleDocTemplate(str(path), pagesize=A4, topMargin=2*cm, bottomMargin=2*cm, leftMargin=2.5*cm, rightMargin=2.5*cm)
    story = []
    lines = text.strip().split("\n")
    for line in lines:
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 0.2*cm))
            continue
        if line == lines[0]:  # Name
            story.append(Paragraph(f"<b><font size=16>{stripped}</font></b>", styles["Normal"]))
        elif stripped.isupper() or stripped.endswith(":"):
            story.append(Spacer(1, 0.15*cm))
            story.append(Paragraph(f"<b>{stripped}</b>", styles["Normal"]))
        else:
            story.append(Paragraph(stripped, styles["Normal"]))
    doc.build(story)
    print(f"  Created: {path.name}")


print("\nGenerating sample resumes…")
for r in RESUMES:
    make_resume_pdf(r["filename"], r["text"])

print(f"\n✅ Done! {len(RESUMES)} resumes + 1 JD created in {OUT}/")
print("   Use sample_jd.txt as the JD and upload the PDF resumes in the app.")
