"""
utils/llm_factory.py
────────────────────
Central LLM factory — supports OpenAI (paid) and Google Gemini (FREE tier).

Switch between providers by setting LLM_PROVIDER in your .env:
  - LLM_PROVIDER=openai   → uses OPENAI_API_KEY
  - LLM_PROVIDER=gemini   → uses GOOGLE_API_KEY  (FREE: 15 RPM, 1M tokens/day)
"""

from __future__ import annotations
import os
import logging

logger = logging.getLogger(__name__)

# Valid providers
OPENAI = "openai"
GEMINI = "gemini"


def get_provider() -> str:
    """Return the active LLM provider from env."""
    return os.getenv("LLM_PROVIDER", OPENAI).lower().strip()


def get_chat_llm(model: str | None = None, temperature: float = 0.0, json_mode: bool = False):
    """
    Return a LangChain ChatModel for the configured provider.

    Args:
        model: Model name override. If None, uses a sensible default.
        temperature: Sampling temperature.
        json_mode: If True, enable JSON-structured output mode.
    """
    provider = get_provider()

    if provider == GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        kwargs = dict(model=model, temperature=temperature)
        llm = ChatGoogleGenerativeAI(**kwargs)
        logger.info(f"LLM: Google Gemini ({model}) — FREE tier")
        return llm

    else:  # default: openai
        from langchain_openai import ChatOpenAI

        model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        kwargs = dict(model=model, temperature=temperature)
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        llm = ChatOpenAI(**kwargs)
        logger.info(f"LLM: OpenAI ({model})")
        return llm


def get_model_display_name() -> str:
    """Human-readable name of the active model for reports."""
    provider = get_provider()
    if provider == GEMINI:
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return os.getenv("OPENAI_MODEL", "gpt-4o")
