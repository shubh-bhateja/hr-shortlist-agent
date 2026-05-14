"""
utils/llm_factory.py
────────────────────
Central LLM factory — supports Groq.
"""

from __future__ import annotations
import os
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Valid providers
GROQ = "groq"

def get_provider() -> str:
    """Return the active LLM provider from env."""
    return GROQ

def get_chat_llm(model: str | None = None, temperature: float = 0.0, json_mode: bool = False):
    """
    Return a LangChain ChatModel for Groq.

    Args:
        model: Model name override. If None, uses a sensible default.
        temperature: Sampling temperature.
        json_mode: If True, enable JSON-structured output mode.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    kwargs = dict(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    llm = ChatOpenAI(**kwargs)
    logger.info(f"LLM: Groq ({model})")
    return llm

def get_model_display_name() -> str:
    """Human-readable name of the active model for reports."""
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
