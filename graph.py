"""
graph.py
─────────
LangGraph Pipeline Orchestrator

Defines the multi-agent graph:

    [START]
       │
    [jd_parser]          ← Node 1: Extract structured JobRequirements from JD text
       │
    [profile_parser]     ← Node 2: Parse all resumes → CandidateProfile list
       │
    [scorer]             ← Node 3: Dual-mode scoring (embeddings + LLM rubric)
       │
    [report_generator]   ← Node 4: Generate PDF shortlist report
       │
    [END]

Each node is a pure function: (state: dict) → dict.
State is passed and merged through the graph automatically.
"""

from __future__ import annotations
import logging
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

from agents.jd_parser import parse_jd
from agents.profile_parser import parse_all_profiles
from agents.scorer import score_all_candidates
from agents.report_generator import generate_report

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM Caching ──────────────────────────────────────────────────────────────
# SQLite cache saves LLM responses locally — avoids redundant API calls during
# development and re-runs with the same resumes (Pro Tip from brief).
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
logger.info("LLM SQLite cache enabled (.langchain_cache.db)")



def build_graph():
    """Construct and compile the LangGraph agent pipeline."""
    builder = StateGraph(dict)

    # Register nodes
    builder.add_node("jd_parser", parse_jd)
    builder.add_node("profile_parser", parse_all_profiles)
    builder.add_node("scorer", score_all_candidates)
    builder.add_node("report_generator", generate_report)

    # Define edges (linear pipeline)
    builder.add_edge(START, "jd_parser")
    builder.add_edge("jd_parser", "profile_parser")
    builder.add_edge("profile_parser", "scorer")
    builder.add_edge("scorer", "report_generator")
    builder.add_edge("report_generator", END)

    return builder.compile()


def run_pipeline(
    jd_text: str,
    file_paths: list[str],
    model: str = "gpt-4o",
    output_dir: str = "output",
) -> dict:
    """
    High-level entry point.
    Returns the final state dict with keys:
      job_requirements, candidate_profiles, scorecards, report_path, ...
    """
    graph = build_graph()

    initial_state = {
        "jd_text": jd_text,
        "file_paths": file_paths,
        "model": model,
        "output_dir": output_dir,
        "overrides": [],
    }

    logger.info(f"Starting pipeline: {len(file_paths)} resumes, model={model}")
    final_state = graph.invoke(initial_state)
    logger.info("Pipeline complete.")
    return final_state
