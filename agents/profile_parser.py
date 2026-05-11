"""
agents/profile_parser.py
─────────────────────────
LangGraph Node 2 — Profile Agent

Responsibility:
  • Accept uploaded PDF / DOCX resume files OR LinkedIn JSON
  • Extract raw text from each file
  • Run LLM extraction to produce a structured CandidateProfile per resume
  • Sanitises PII before sending to cloud LLM (logs remain masked)

Security mitigations applied here:
  • Phone numbers and emails are stripped from the prompt sent to the LLM;
    they are re-injected from the raw parse (regex) AFTER LLM response.
  • File type is validated before any parsing attempt.
"""

from __future__ import annotations
import re
import uuid
import logging
import json
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from utils.llm_factory import get_chat_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import CandidateProfile

logger = logging.getLogger(__name__)

# ── Regex helpers ──────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-().]{7,}\d)")

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """You are an expert resume parser.
Extract structured candidate information from the resume text.

Rules:
1. Be accurate — only extract what is explicitly written.
2. For total_experience_years: sum all relevant work experience durations. Use 0.0 if unclear.
3. For skills: include both technical and soft skills mentioned.
4. For projects: include project names or brief descriptions (max 10 words each).
5. raw_text_snippet: copy the FIRST 400 characters of the resume text exactly.
6. Do NOT include email or phone in your JSON — those fields will be filled separately.
7. Return ONLY valid JSON matching the schema — no markdown, no preamble.
"""

_HUMAN = """Parse this resume and return a JSON object matching the schema:

{format_instructions}

RESUME TEXT (PII redacted):
───────────────────────────────────────────
{resume_text}
───────────────────────────────────────────

Return ONLY the JSON."""


# ── Text extractors ────────────────────────────────────────────────────────────

def _extract_text_pdf(path: Union[str, Path]) -> str:
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def _extract_text_docx(path: Union[str, Path]) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text_json(path: Union[str, Path]) -> str:
    """LinkedIn JSON export → flat text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k}: {', '.join(str(i) for i in v)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def extract_raw_text(file_path: Union[str, Path]) -> str:
    """Route to correct extractor based on file extension."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        return _extract_text_pdf(p)
    elif ext in (".docx", ".doc"):
        return _extract_text_docx(p)
    elif ext == ".json":
        return _extract_text_json(p)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Accepted: PDF, DOCX, JSON.")


def _mask_pii(text: str) -> tuple[str, str, str]:
    """
    Remove email and phone from text before sending to LLM.
    Returns (masked_text, email_found, phone_found).
    """
    email = next(iter(_EMAIL_RE.findall(text)), None)
    phone = next(iter(_PHONE_RE.findall(text)), None)
    masked = _EMAIL_RE.sub("[EMAIL]", text)
    masked = _PHONE_RE.sub("[PHONE]", masked)
    return masked, email or "", phone or ""


# ── LLM chain ─────────────────────────────────────────────────────────────────

def _build_profile_chain(model: str | None = None):
    llm = get_chat_llm(model=model, temperature=0.0)
    parser = PydanticOutputParser(pydantic_object=CandidateProfile)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])
    return prompt | llm | parser, parser


def parse_single_resume(
    file_path: Union[str, Path],
    model: str | None = None,
    candidate_id: str | None = None,
) -> CandidateProfile:
    """Parse one resume file → CandidateProfile."""
    cid = candidate_id or f"cand_{uuid.uuid4().hex[:8]}"
    raw_text = extract_raw_text(file_path)

    # Security: mask PII before LLM call
    masked_text, email, phone = _mask_pii(raw_text)
    # Truncate to 6000 chars to stay within token limits
    truncated = masked_text[:6000]

    chain, parser = _build_profile_chain(model=model)
    profile: CandidateProfile = chain.invoke({
        "resume_text": truncated,
        "format_instructions": parser.get_format_instructions(),
    })

    # Re-inject PII extracted via regex (stays local, never sent to LLM)
    profile.candidate_id = cid
    profile.email = email or profile.email
    profile.phone = phone or profile.phone

    logger.info(f"[{cid}] Parsed: {profile.full_name} | {profile.total_experience_years}yrs | {len(profile.skills)} skills")
    return profile


# ── LangGraph node ─────────────────────────────────────────────────────────────

def parse_all_profiles(state: dict) -> dict:
    """
    LangGraph node function.
    Input state keys:  file_paths (list[str]), model (optional)
    Output state keys: candidate_profiles (list[CandidateProfile]), profile_errors
    """
    file_paths: list[str] = state.get("file_paths", [])
    model: str | None = state.get("model")

    profiles = []
    errors = []

    for i, fp in enumerate(file_paths):
        cid = f"cand_{i+1:03d}"
        try:
            profile = parse_single_resume(fp, model=model, candidate_id=cid)
            profiles.append(profile)
        except Exception as exc:
            logger.error(f"Failed to parse {fp}: {exc}")
            errors.append({"file": str(fp), "error": str(exc)})

    return {**state, "candidate_profiles": profiles, "profile_errors": errors}
