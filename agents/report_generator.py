"""
agents/report_generator.py
──────────────────────────
LangGraph Node 4 — Report Agent

Generates a professional PDF shortlist report using ReportLab.
Report sections:
  1. Cover page (role, date, total candidates screened)
  2. Executive Summary table (ranked candidates at a glance)
  3. Per-candidate detail pages (dimension scores + bar chart simulation + gaps)
  4. Override log (if any HR overrides were applied)
  5. Methodology appendix (rubric weights, model used, semantic sim explanation)
"""

from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF

from schemas import CandidateScorecard, JobRequirements, HROverride

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#0A1628")
TEAL      = colors.HexColor("#0D9488")
LIGHT_TEAL= colors.HexColor("#CCFBF1")
AMBER     = colors.HexColor("#D97706")
RED_SOFT  = colors.HexColor("#DC2626")
GREY_BG   = colors.HexColor("#F8FAFC")
GREY_LINE = colors.HexColor("#CBD5E1")
WHITE     = colors.white

REC_COLORS = {
    "STRONG HIRE": colors.HexColor("#166534"),
    "HIRE":        colors.HexColor("#1D4ED8"),
    "CONSIDER":    colors.HexColor("#92400E"),
    "NO HIRE":     colors.HexColor("#991B1B"),
}
REC_BG = {
    "STRONG HIRE": colors.HexColor("#DCFCE7"),
    "HIRE":        colors.HexColor("#DBEAFE"),
    "CONSIDER":    colors.HexColor("#FEF3C7"),
    "NO HIRE":     colors.HexColor("#FEE2E2"),
}

# ── Styles ─────────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    custom = {}
    custom["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER, spaceAfter=8
    )
    custom["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, fontName="Helvetica",
        textColor=LIGHT_TEAL, alignment=TA_CENTER, spaceAfter=4
    )
    custom["section_header"] = ParagraphStyle(
        "section_header", fontSize=14, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=14, spaceAfter=6, borderPad=4
    )
    custom["body"] = ParagraphStyle(
        "body", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#1E293B"), spaceAfter=3, leading=13
    )
    custom["caption"] = ParagraphStyle(
        "caption", fontSize=7.5, fontName="Helvetica",
        textColor=colors.HexColor("#64748B"), alignment=TA_CENTER
    )
    custom["dim_label"] = ParagraphStyle(
        "dim_label", fontSize=8.5, fontName="Helvetica-Bold",
        textColor=NAVY
    )
    return custom


# ── Score bar drawing ─────────────────────────────────────────────────────────

def _score_bar(score: float, max_score: float = 10.0, width: float = 120, height: float = 10) -> Drawing:
    """Horizontal bar visualising a 0-10 score."""
    d = Drawing(width, height + 4)
    # Background bar
    d.add(Rect(0, 2, width, height, fillColor=GREY_LINE, strokeColor=None))
    # Filled portion
    filled_w = (score / max_score) * width
    bar_color = TEAL if score >= 7 else (AMBER if score >= 5 else RED_SOFT)
    d.add(Rect(0, 2, filled_w, height, fillColor=bar_color, strokeColor=None))
    return d


# ── Main report builder ────────────────────────────────────────────────────────

def generate_pdf_report(
    jd: JobRequirements,
    scorecards: list[CandidateScorecard],
    overrides: list[HROverride] | None = None,
    output_path: str | Path = "output/shortlist_report.pdf",
    model_used: str = "gpt-4o",
) -> str:
    """Generate the full PDF report. Returns the output path string."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overrides = overrides or []
    st = _styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.8*cm, bottomMargin=1.8*cm,
        leftMargin=2*cm, rightMargin=2*cm,
    )

    story = []

    # ── COVER PAGE ───────────────────────────────────────────────────────────
    # Navy banner background table
    cover_data = [[
        Paragraph(f"<b>Candidate Shortlist Report</b>", st["cover_title"])
    ]]
    cover_table = Table(cover_data, colWidths=[doc.width])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING", (0,0), (-1,-1), 40),
        ("BOTTOMPADDING", (0,0), (-1,-1), 40),
        ("LEFTPADDING", (0,0), (-1,-1), 20),
        ("RIGHTPADDING", (0,0), (-1,-1), 20),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.4*cm))

    meta = [
        ["Role", jd.role_title],
        ["Domain", jd.domain],
        ["Seniority", jd.seniority_level],
        ["Generated", datetime.now().strftime("%d %b %Y, %H:%M")],
        ["Candidates Screened", str(len(scorecards))],
        ["Model Used", model_used],
    ]
    meta_table = Table(meta, colWidths=[4*cm, doc.width - 4*cm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME", (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), NAVY),
        ("TEXTCOLOR", (1,0), (1,-1), colors.HexColor("#334155")),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [GREY_BG, WHITE]),
        ("GRID", (0,0), (-1,-1), 0.25, GREY_LINE),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
    story.append(Spacer(1, 0.3*cm))

    # ── EXECUTIVE SUMMARY TABLE ───────────────────────────────────────────────
    story.append(Paragraph("Executive Summary — Ranked Shortlist", st["section_header"]))

    headers = ["#", "Name", "Score /10", "Semantic Sim", "Recommendation"]
    rows = [headers]
    for i, sc in enumerate(scorecards, 1):
        rows.append([
            str(i),
            sc.full_name,
            f"{sc.total_weighted_score:.2f}",
            f"{sc.semantic_similarity_score:.3f}" if sc.semantic_similarity_score is not None else "—",
            sc.recommendation,
        ])

    exec_table = Table(rows, colWidths=[1*cm, 5.5*cm, 2.5*cm, 2.5*cm, 4.5*cm])
    exec_style = [
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("TEXTCOLOR", (0,0), (-1,0), WHITE),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, GREY_BG]),
        ("GRID", (0,0), (-1,-1), 0.3, GREY_LINE),
    ]
    # Colour-code recommendation cells
    for i, sc in enumerate(scorecards, 1):
        bg = REC_BG.get(sc.recommendation, WHITE)
        fg = REC_COLORS.get(sc.recommendation, NAVY)
        exec_style.append(("BACKGROUND", (4, i), (4, i), bg))
        exec_style.append(("TEXTCOLOR", (4, i), (4, i), fg))
        exec_style.append(("FONTNAME", (4, i), (4, i), "Helvetica-Bold"))

    exec_table.setStyle(TableStyle(exec_style))
    story.append(exec_table)
    story.append(PageBreak())

    # ── PER-CANDIDATE DETAIL PAGES ────────────────────────────────────────────
    story.append(Paragraph("Candidate Detail Scorecards", st["section_header"]))
    story.append(Spacer(1, 0.2*cm))

    for rank, sc in enumerate(scorecards, 1):
        rec_color = REC_COLORS.get(sc.recommendation, NAVY)
        rec_bg    = REC_BG.get(sc.recommendation, GREY_BG)

        # Header row for this candidate
        cand_header_data = [[
            Paragraph(f"#{rank} — {sc.full_name}", ParagraphStyle(
                "ch", fontSize=11, fontName="Helvetica-Bold", textColor=WHITE)),
            Paragraph(f"Total: {sc.total_weighted_score:.2f} / 10 &nbsp;&nbsp; {sc.recommendation}", ParagraphStyle(
                "cr", fontSize=10, fontName="Helvetica-Bold", textColor=WHITE, alignment=TA_RIGHT)),
        ]]
        cand_header = Table(cand_header_data, colWidths=[doc.width*0.6, doc.width*0.4])
        cand_header.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), NAVY),
            ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (0,-1), 10),
            ("RIGHTPADDING", (-1,0), (-1,-1), 10),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(cand_header)
        story.append(Spacer(1, 0.15*cm))

        # Summary box
        story.append(Paragraph(sc.summary, st["body"]))
        story.append(Spacer(1, 0.2*cm))

        # Dimension scores table with inline bars
        dim_rows = [["Dimension", "Weight", "Score", "Visual", "Justification"]]
        for d in sc.dimensions:
            dim_rows.append([
                Paragraph(d.dimension, st["dim_label"]),
                f"{int(d.weight*100)}%",
                f"{d.raw_score:.1f}",
                _score_bar(d.raw_score),
                Paragraph(d.justification, st["body"]),
            ])
        # Totals row
        dim_rows.append([
            Paragraph("<b>WEIGHTED TOTAL</b>", st["dim_label"]),
            "", f"<b>{sc.total_weighted_score:.2f}</b>", "", ""
        ])

        dim_table = Table(dim_rows, colWidths=[3.5*cm, 1.3*cm, 1.3*cm, 3*cm, 7.4*cm])
        dim_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("ALIGN", (1,0), (2,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("ROWBACKGROUNDS", (0,1), (-1,-2), [WHITE, GREY_BG]),
            ("GRID", (0,0), (-1,-1), 0.3, GREY_LINE),
            # Totals row
            ("BACKGROUND", (0,-1), (-1,-1), LIGHT_TEAL),
            ("FONTNAME", (0,-1), (-1,-1), "Helvetica-Bold"),
            ("TEXTCOLOR", (2,-1), (2,-1), NAVY),
        ]))
        story.append(dim_table)
        story.append(Spacer(1, 0.2*cm))

        # Skill gaps
        if sc.skill_gaps:
            gaps_str = " · ".join(sc.skill_gaps)
            story.append(Paragraph(f"<b>Skill Gaps:</b> {gaps_str}", st["body"]))

        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_LINE))
        story.append(Spacer(1, 0.3*cm))

    # ── HR OVERRIDE LOG ───────────────────────────────────────────────────────
    if overrides:
        story.append(PageBreak())
        story.append(Paragraph("HR Override Log", st["section_header"]))
        ov_rows = [["Candidate ID", "Overridden By", "Original", "Adjusted", "Reason", "Timestamp"]]
        for ov in overrides:
            ov_rows.append([
                ov.candidate_id, ov.overridden_by,
                f"{ov.original_score:.2f}", f"{ov.adjusted_score:.2f}",
                ov.reason, ov.timestamp,
            ])
        ov_table = Table(ov_rows, colWidths=[2.5*cm, 2.5*cm, 1.5*cm, 1.5*cm, 5.5*cm, 3*cm])
        ov_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), AMBER),
            ("TEXTCOLOR", (0,0), (-1,0), WHITE),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 0.3, GREY_LINE),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, GREY_BG]),
        ]))
        story.append(ov_table)

    # ── METHODOLOGY APPENDIX ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Methodology & Transparency Appendix", st["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
    story.append(Spacer(1, 0.3*cm))

    method_text = (
        f"<b>LLM Model:</b> {model_used}<br/>"
        "<b>Agent Framework:</b> LangGraph (multi-agent graph with 4 specialised nodes)<br/>"
        "<b>Scoring Mode:</b> Dual — Semantic similarity (SentenceTransformers all-MiniLM-L6-v2 cosine) "
        "combined with LLM rubric reasoning in JSON mode.<br/>"
        "<b>Rubric Weights:</b> Skills Match 30% · Experience Relevance 25% · "
        "Project / Portfolio 15% · Education & Certs 15% · Communication Quality 15%<br/>"
        "<b>Hallucination Mitigation:</b> Pydantic-validated structured output at every agent node; "
        "LLM operates in JSON mode; dimension scores are cross-checked against embedding similarity.<br/>"
        "<b>PII Handling:</b> Email and phone are extracted via regex locally and masked before "
        "being sent to the cloud LLM; re-injected post-response.<br/>"
        "<b>Human-in-the-Loop:</b> HR may override any score via the UI; all overrides are logged "
        "with reason and timestamp in this report.<br/>"
        "<b>Disclaimer:</b> This report is a decision-support tool. All final hiring decisions "
        "must be reviewed and approved by a qualified HR professional."
    )
    story.append(Paragraph(method_text, st["body"]))

    doc.build(story)
    logger.info(f"PDF report generated: {output_path}")
    return str(output_path)


# ── LangGraph node ─────────────────────────────────────────────────────────────

def generate_report(state: dict) -> dict:
    """
    LangGraph node function.
    Input state keys:  job_requirements, scorecards, overrides (optional), model, output_dir
    Output state keys: report_path, report_error
    """
    try:
        output_dir = state.get("output_dir", "output")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / f"shortlist_report_{ts}.pdf"

        path = generate_pdf_report(
            jd=state["job_requirements"],
            scorecards=state["scorecards"],
            overrides=state.get("overrides", []),
            output_path=out_path,
            model_used=state.get("model", "gpt-4o"),
        )
        return {**state, "report_path": path, "report_error": None}
    except Exception as exc:
        logger.exception("Report generation failed.")
        return {**state, "report_path": None, "report_error": str(exc)}
