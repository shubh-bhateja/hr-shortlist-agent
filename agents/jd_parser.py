"""
agents/jd_parser.py
────────────────────
LangGraph Node 1 — JD Parser Agent

Responsibility: Extract a fully-structured JobRequirements object from raw JD text.
Uses JSON-mode / structured output to guarantee schema compliance.
"""

from __future__ import annotations
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from schemas import JobRequirements

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM = """You are a senior HR analyst and talent acquisition expert.
Your task is to parse a Job Description (JD) and extract structured requirements.

Rules:
1. Extract ONLY information that is explicitly stated or clearly implied by the JD.
2. Do NOT invent skills or requirements not present in the JD.
3. For min_experience_years: if a range is given (e.g. 3-5 years), use the lower bound.
4. If a field cannot be determined, use a sensible default ("Not specified" for strings, [] for lists, 0 for numbers).
5. Return ONLY valid JSON matching the schema — no markdown fences, no extra text.
"""

_HUMAN = """Parse the following Job Description and return a JSON object matching this schema:

{format_instructions}

JD TEXT:
───────────────────────────────────────────
{jd_text}
───────────────────────────────────────────

Return ONLY the JSON. No explanation."""


def build_jd_parser_chain(model: str = "gpt-4o", temperature: float = 0.0):
    """Build and return the JD parser chain."""
    llm = ChatOpenAI(model=model, temperature=temperature)
    parser = PydanticOutputParser(pydantic_object=JobRequirements)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        ("human", _HUMAN),
    ])
    return prompt | llm | parser, parser


def parse_jd(state: dict) -> dict:
    """
    LangGraph node function.
    Input state keys:  jd_text, model (optional)
    Output state keys: job_requirements, jd_parse_error (on failure)
    """
    jd_text: str = state.get("jd_text", "")
    model: str = state.get("model", "gpt-4o")

    if not jd_text.strip():
        logger.error("JD text is empty.")
        return {**state, "jd_parse_error": "JD text is empty."}

    try:
        chain, parser = build_jd_parser_chain(model=model)
        result: JobRequirements = chain.invoke({
            "jd_text": jd_text,
            "format_instructions": parser.get_format_instructions(),
        })
        logger.info(f"JD parsed successfully: role='{result.role_title}', domain='{result.domain}'")
        return {**state, "job_requirements": result, "jd_parse_error": None}

    except Exception as exc:
        logger.exception("JD parsing failed.")
        return {**state, "jd_parse_error": str(exc)}
