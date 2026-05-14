"""
app.py – Streamlit UI for the HR Shortlisting Agent
─────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

from __future__ import annotations
import os
import json
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from dotenv import load_dotenv

# Force load .env from the absolute path of this folder
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Debug print to terminal (helps confirm key is loaded)
g_key = os.getenv("GOOGLE_API_KEY", "")
if g_key:
    print(f"DEBUG: Loaded Google API Key starting with: {g_key[:4]}...")
else:
    print("DEBUG: NO GOOGLE API KEY FOUND IN ENV")

logging.basicConfig(level=logging.INFO)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Shortlisting Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0A1628 0%, #0f2447 100%);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(13,148,136,0.4) !important;
    color: #f1f5f9 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
}
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 12px !important; }

/* Main area */
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }

/* Score card */
.scorecard {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    transition: box-shadow 0.2s;
}
.scorecard:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }

/* Recommendation badges */
.badge-STRONG-HIRE  { background:#DCFCE7; color:#166534; border:1px solid #86EFAC; }
.badge-HIRE         { background:#DBEAFE; color:#1D4ED8; border:1px solid #93C5FD; }
.badge-CONSIDER     { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
.badge-NO-HIRE      { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
.badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:12px; font-weight:600; letter-spacing:0.5px;
}

/* Metric tiles */
.metric-tile {
    background: linear-gradient(135deg, #0A1628, #0f2447);
    border-radius: 12px; padding: 16px 20px; text-align: center; color: white;
}
.metric-tile .val { font-size: 32px; font-weight: 700; color: #0D9488; }
.metric-tile .lbl { font-size: 12px; color: #94a3b8; margin-top: 4px; }

/* Progress bar override */
.stProgress > div > div { background-color: #0D9488 !important; }

/* Override section */
.override-box {
    background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px; padding: 14px 18px; margin-top: 10px;
}

/* Divider */
.section-divider { border-top: 2px solid #0D9488; margin: 24px 0 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
for key in ("results", "overrides", "report_path", "jd_requirements"):
    if key not in st.session_state:
        st.session_state[key] = None
if "overrides" not in st.session_state or st.session_state.overrides is None:
    st.session_state.overrides = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 HR Shortlisting Agent")
    st.markdown("<p style='color:#64748b;font-size:12px;'>Powered by LangGraph · SentenceTransformers</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ LLM Configuration")
    os.environ["LLM_PROVIDER"] = "groq"
    api_key = st.text_input(
        "Groq API Key (free)",
        type="password",
        value=os.getenv("GROQ_API_KEY", "").strip(),
        help="Get your FREE key at https://console.groq.com/keys",
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key.strip()
    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
    )
    os.environ["GROQ_MODEL"] = model_choice

    st.markdown("---")
    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        "Paste JD here",
        height=280,
        placeholder="We are looking for a Senior ML Engineer with 4+ years experience in Python, PyTorch, and LLM fine-tuning...",
    )

    st.markdown("---")
    st.markdown("### 📁 Upload Resumes")
    uploaded_files = st.file_uploader(
        "PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload up to 20 resumes.",
    )

    st.markdown("---")
    run_btn = st.button("🚀 Run Shortlisting Agent", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#475569; line-height:1.7;'>
    <b>Agent Pipeline</b><br/>
    1️⃣ JD Parser → JobRequirements<br/>
    2️⃣ Profile Parser → CandidateProfile<br/>
    3️⃣ Scorer → Dual-mode rubric<br/>
    4️⃣ Report Generator → PDF<br/><br/>
    <b>Observability:</b> LangSmith tracing<br/>
    <b>Security:</b> PII masked before LLM
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────

st.markdown("# HR Resume Shortlisting Agent")
st.markdown(
    "Upload a Job Description and candidate resumes. The agent parses, scores, and ranks "
    "every candidate using a structured rubric — combining **LLM reasoning** and **semantic similarity**."
)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()

    if not jd_text.strip():
        st.error("⚠️ Please enter a Job Description.")
        st.stop()
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one resume.")
    else:
        # Save uploaded files to temp dir
        tmp_dir = tempfile.mkdtemp()
        file_paths = []
        for uf in uploaded_files:
            dest = Path(tmp_dir) / uf.name
            dest.write_bytes(uf.read())
            file_paths.append(str(dest))

        with st.spinner("🔄 Agent pipeline running…"):
            progress = st.progress(0, text="Step 1/4 — Parsing Job Description…")
            try:
                from graph import run_pipeline

                # Step indicators via callback isn't native in LangGraph invoke,
                # so we run node by node for UI feedback
                from agents.jd_parser import parse_jd
                from agents.profile_parser import parse_all_profiles
                from agents.scorer import score_all_candidates
                from agents.report_generator import generate_report

                state = {
                    "jd_text": jd_text,
                    "file_paths": file_paths,
                    "model": model_choice,
                    "output_dir": "output",
                    "overrides": [],
                }

                progress.progress(10, text="Step 1/4 — Parsing Job Description…")
                state = parse_jd(state)
                if state.get("jd_parse_error"):
                    st.error(f"JD parsing failed: {state['jd_parse_error']}")
                    st.stop()

                progress.progress(30, text=f"Step 2/4 — Parsing {len(file_paths)} resume(s)…")
                state = parse_all_profiles(state)

                progress.progress(60, text="Step 3/4 — Scoring candidates…")
                state = score_all_candidates(state)

                progress.progress(85, text="Step 4/4 — Generating PDF report…")
                state = generate_report(state)

                progress.progress(100, text="✅ Done!")

                st.session_state.results = state
                st.session_state.jd_requirements = state.get("job_requirements")
                st.session_state.report_path = state.get("report_path")
                st.session_state.overrides = []

            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# ── Display results ────────────────────────────────────────────────────────────
if st.session_state.results:
    state = st.session_state.results
    scorecards = state.get("scorecards", [])
    jd = state.get("job_requirements")
    profiles = {p.candidate_id: p for p in state.get("candidate_profiles", [])}

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Metrics row ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    rec_counts = {}
    for sc in scorecards:
        rec_counts[sc.recommendation] = rec_counts.get(sc.recommendation, 0) + 1

    with col1:
        st.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{len(scorecards)}</div>
            <div class='lbl'>Candidates Screened</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        hire_count = rec_counts.get("STRONG HIRE", 0) + rec_counts.get("HIRE", 0)
        st.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{hire_count}</div>
            <div class='lbl'>Hire / Strong Hire</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        avg_score = sum(s.total_weighted_score for s in scorecards) / len(scorecards) if scorecards else 0
        st.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{avg_score:.2f}</div>
            <div class='lbl'>Average Score /10</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        top = scorecards[0] if scorecards else None
        top_score = f"{top.total_weighted_score:.2f}" if top else "0.00"
        top_name = top.full_name.split()[0] if top else "—"
        st.markdown(f"""
        <div class='metric-tile'>
            <div class='val'>{top_score}</div>
            <div class='lbl'>Top Score — {top_name}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Ranked Shortlist", "🔍 Candidate Deep Dive", "✏️ HR Overrides", "📄 Download Report"])

    # ── TAB 1: Ranked shortlist ────────────────────────────────────────────────
    with tab1:
        st.subheader(f"Shortlist — {jd.role_title if jd else 'Role'}")

        # Summary dataframe
        rows = []
        for i, sc in enumerate(scorecards, 1):
            rows.append({
                "Rank": i,
                "Name": sc.full_name,
                "Score /10": sc.total_weighted_score,
                "Semantic Sim": sc.semantic_similarity_score,
                "Recommendation": sc.recommendation,
                "Skill Gaps": ", ".join(sc.skill_gaps[:3]),
            })
        df = pd.DataFrame(rows)

        def color_rec(val):
            colors_map = {
                "STRONG HIRE": "background-color:#DCFCE7;color:#166534;font-weight:600",
                "HIRE": "background-color:#DBEAFE;color:#1D4ED8;font-weight:600",
                "CONSIDER": "background-color:#FEF3C7;color:#92400E;font-weight:600",
                "NO HIRE": "background-color:#FEE2E2;color:#991B1B;font-weight:600",
            }
            return colors_map.get(val, "")

        if not df.empty:
            styled = df.style.map(color_rec, subset=["Recommendation"])\
                             .format({"Score /10": "{:.2f}", "Semantic Sim": "{:.3f}"})\
                             .background_gradient(subset=["Score /10"], cmap="RdYlGn", vmin=0, vmax=10)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.warning("No candidates were successfully scored. Please check the terminal logs.")


        # Score comparison bar chart
        st.markdown("#### Score Comparison")
        fig = go.Figure()
        dims = ["Skills Match", "Experience Relevance", "Education & Certs", "Project / Portfolio", "Communication Quality"]
        colors_list = ["#0D9488", "#0EA5E9", "#8B5CF6", "#F59E0B", "#EF4444"]

        for sc in scorecards[:8]:  # max 8 for readability
            dim_scores = {d.dimension: d.raw_score for d in sc.dimensions}
            fig.add_trace(go.Bar(
                name=sc.full_name,
                x=dims,
                y=[dim_scores.get(d, 0) for d in dims],
                text=[f"{dim_scores.get(d, 0):.1f}" for d in dims],
                textposition="outside",
            ))

        fig.update_layout(
            barmode="group",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 11], gridcolor="#e2e8f0"),
            xaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Candidate deep dive ─────────────────────────────────────────────
    with tab2:
        selected_name = st.selectbox(
            "Select candidate",
            [sc.full_name for sc in scorecards],
            format_func=lambda n: f"{n} — {next(s.total_weighted_score for s in scorecards if s.full_name == n):.2f}/10"
        )
        sc = next(s for s in scorecards if s.full_name == selected_name)
        profile = profiles.get(sc.candidate_id)

        left, right = st.columns([1.2, 1])

        with left:
            # Header
            rec_css = sc.recommendation.replace(" ", "-")
            st.markdown(f"""
            <div class='scorecard'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <h3 style='margin:0; color:#0A1628;'>{sc.full_name}</h3>
                    <span class='badge badge-{rec_css}'>{sc.recommendation}</span>
                </div>
                <p style='color:#64748b; font-size:13px; margin:6px 0 12px 0;'>
                    {profile.current_title or "—"} · {profile.total_experience_years:.1f} yrs exp
                </p>
                <p style='font-size:13px; color:#334155; line-height:1.6;'>{sc.summary}</p>
            </div>""", unsafe_allow_html=True)

            # Dimension breakdown
            st.markdown("**Rubric Scores**")
            for d in sc.dimensions:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    pct = int(d.raw_score / 10 * 100)
                    bar_color = "#0D9488" if d.raw_score >= 7 else ("#F59E0B" if d.raw_score >= 5 else "#EF4444")
                    st.markdown(f"""
                    <div style='margin-bottom:8px;'>
                        <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>
                            <span style='font-weight:600;color:#0A1628;'>{d.dimension}</span>
                            <span style='color:#64748b;'>weight {int(d.weight*100)}% · <b style='color:{bar_color};'>{d.raw_score:.1f}/10</b></span>
                        </div>
                        <div style='background:rgba(255,255,255,0.1);border-radius:4px;height:7px;'>
                            <div style='width:{pct}%;background:{bar_color};border-radius:4px;height:7px;'></div>
                        </div>
                        <div style='font-size:11px;color:#64748b;margin-top:3px;'>{d.justification}</div>
                    </div>""", unsafe_allow_html=True)
                with col_b:
                    pass

            # Skill gaps
            if sc.skill_gaps:
                st.markdown("**Skill Gaps**")
                gaps_html = " ".join(
                    f"<span style='background:#FEE2E2;color:#991B1B;border-radius:4px;padding:2px 8px;font-size:11px;margin:2px;display:inline-block;'>{g}</span>"
                    for g in sc.skill_gaps
                )
                st.markdown(gaps_html, unsafe_allow_html=True)

        with right:
            # Radar chart
            categories = [d.dimension for d in sc.dimensions]
            values = [d.raw_score for d in sc.dimensions]
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            radar_fig = go.Figure(data=go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(13,148,136,0.15)",
                line=dict(color="#0D9488", width=2),
                marker=dict(size=6, color="#0D9488"),
            ))
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=9)),
                    angularaxis=dict(tickfont=dict(size=10, family="DM Sans")),
                    bgcolor="rgba(0,0,0,0)",
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                height=320,
                margin=dict(t=30, b=30, l=30, r=30),
                font=dict(family="DM Sans"),
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            # Stats
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:14px;font-size:12px;'>
            <b>Total Score:</b> <span style='color:#0D9488;font-size:18px;font-weight:700;'>{sc.total_weighted_score:.2f}</span> / 10<br/>
            <b>Semantic Similarity:</b> {sc.semantic_similarity_score:.3f}<br/>
            <b>Skills:</b> {", ".join((profile.skills or [])[:6]) or "—"}<br/>
            <b>Education:</b> {"; ".join((profile.education or [])[:2]) or "—"}<br/>
            <b>Certifications:</b> {", ".join((profile.certifications or [])[:4]) or "None listed"}
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: HR Overrides ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Human-in-the-Loop Score Override")
        st.markdown(
            "HR can adjust any candidate's total score with a justification reason. "
            "All overrides are logged and included in the final PDF report."
        )

        with st.form("override_form"):
            ov_candidate = st.selectbox("Candidate to Override", [sc.full_name for sc in scorecards])
            ov_reviewer = st.text_input("Your Name / HR ID", placeholder="e.g. Priya Sharma – HR Lead")
            ov_score = st.slider("Adjusted Score (override total)", 0.0, 10.0, 5.0, 0.1)
            ov_reason = st.text_area("Reason for Override", placeholder="Candidate showed strong culture fit in interview; portfolio links not captured in resume.")
            submit_ov = st.form_submit_button("Log Override", type="primary")

        if submit_ov:
            if not ov_reviewer or not ov_reason:
                st.warning("Please fill in reviewer name and reason.")
            else:
                sc_obj = next(s for s in scorecards if s.full_name == ov_candidate)
                from schemas import HROverride
                override = HROverride(
                    candidate_id=sc_obj.candidate_id,
                    overridden_by=ov_reviewer,
                    original_score=sc_obj.total_weighted_score,
                    adjusted_score=ov_score,
                    reason=ov_reason,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                st.session_state.overrides.append(override)
                # Update score in scorecard
                sc_obj.total_weighted_score = ov_score
                # Regenerate report with override
                from agents.report_generator import generate_report as gen_report
                new_state = {**state, "overrides": st.session_state.overrides}
                new_state = gen_report(new_state)
                st.session_state.report_path = new_state.get("report_path")
                st.success(f"✅ Override logged for {ov_candidate}. Report regenerated.")

        if st.session_state.overrides:
            st.markdown("#### Override Log")
            ov_rows = [{
                "Candidate": o.candidate_id,
                "Reviewer": o.overridden_by,
                "Original": o.original_score,
                "Adjusted": o.adjusted_score,
                "Reason": o.reason,
                "Timestamp": o.timestamp,
            } for o in st.session_state.overrides]
            st.dataframe(pd.DataFrame(ov_rows), use_container_width=True, hide_index=True)

    # ── TAB 4: Download ────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Download Shortlist Report")
        report_path = st.session_state.report_path

        if report_path and Path(report_path).exists():
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=Path(report_path).name,
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
            st.markdown(f"*Generated: {Path(report_path).name}*")
            st.markdown("""
            **Report contains:**
            - Cover page with role metadata
            - Executive summary ranked table
            - Per-candidate scorecard with dimension bars
            - Skill gap analysis
            - HR override log
            - Methodology & transparency appendix
            """)
        else:
            st.info("Run the agent first to generate the PDF report.")

        # JSON export
        if scorecards:
            st.markdown("---")
            st.markdown("#### Raw JSON Export")
            json_data = [sc.model_dump() for sc in scorecards]
            st.download_button(
                label="📥 Download JSON (all scorecards)",
                data=json.dumps(json_data, indent=2),
                file_name="scorecards.json",
                mime="application/json",
                use_container_width=True,
            )

# ── Empty state ────────────────────────────────────────────────────────────────
else:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style='background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:20px;'>
        <h4 style='color:#166534;margin:0 0 8px 0;'>🧠 LLM Reasoning</h4>
        <p style='color:#374151;font-size:13px;margin:0;'>
        GPT-4o evaluates each candidate across 5 weighted dimensions with a structured JSON rubric — 
        not a generic summary.
        </p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#eff6ff;border:1px solid #93c5fd;border-radius:10px;padding:20px;'>
        <h4 style='color:#1d4ed8;margin:0 0 8px 0;'>📐 Semantic Similarity</h4>
        <p style='color:#374151;font-size:13px;margin:0;'>
        SentenceTransformer embeddings compute cosine similarity between the JD and each candidate — 
        quantitative signal alongside LLM reasoning.
        </p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background:#fdf4ff;border:1px solid #d8b4fe;border-radius:10px;padding:20px;'>
        <h4 style='color:#7e22ce;margin:0 0 8px 0;'>🔒 Privacy-first</h4>
        <p style='color:#374151;font-size:13px;margin:0;'>
        PII (email, phone) is masked before being sent to the cloud LLM and re-injected locally — 
        keeping personal data off third-party servers.
        </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.info("👈 Paste a Job Description, upload resumes, and click **Run Shortlisting Agent** to begin.")
