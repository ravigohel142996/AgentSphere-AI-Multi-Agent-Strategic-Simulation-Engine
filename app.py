"""
AgentSphere AI – Multi-Agent Strategic Simulation Engine
Main Streamlit application entry point.

Run locally:
    streamlit run app.py

Deploy: Push to GitHub and connect to Streamlit Cloud, selecting app.py.
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on the Python path so `agentsphere` is importable
# when Streamlit Cloud clones the repo root (not a sub-directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from agentsphere.config import (
    APP_TITLE,
    DEFAULT_ROUNDS,
    ENV_DEFAULTS,
    MAX_ROUNDS,
    CHART_TEMPLATE,
)
from agentsphere.simulation.simulator import Simulator
from agentsphere.analytics.metrics import MetricsEngine

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark professional theme) ─────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Root & background ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.4rem; }
    [data-testid="stMetricDelta"] { font-size: 0.85rem; }
    /* ── Expander ── */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #c9d1d9 !important;
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    /* ── DataFrames ── */
    [data-testid="stDataFrame"] { background: #161b22; }
    /* ── Divider ── */
    hr { border-color: #30363d; }
    /* ── Headers ── */
    h1 { color: #58a6ff !important; }
    h2, h3 { color: #c9d1d9 !important; }
    /* ── Sidebar text ── */
    [data-testid="stSidebar"] label { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] .stSlider { color: #c9d1d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session-state initialisation ──────────────────────────────────────────────

def _init_session() -> None:
    """Initialise all required session-state keys."""
    defaults: dict = {
        "simulation_result": None,
        "simulation_history": [],   # list of SimulationResult objects
        "run_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Simulation Settings")
    st.markdown("---")

    scenario_name = st.text_input("Scenario Name", value="Baseline Scenario")

    n_rounds = st.slider(
        "Simulation Rounds",
        min_value=1,
        max_value=MAX_ROUNDS,
        value=DEFAULT_ROUNDS,
        step=1,
        help="Number of agent negotiation rounds to run.",
    )

    stochastic = st.toggle(
        "Enable Market Noise",
        value=True,
        help="Add stochastic noise scaled by current volatility.",
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
        help="Set for reproducible results.",
    )

    st.markdown("---")
    st.markdown("### 🏢 Initial Environment")

    revenue = st.number_input(
        "Revenue ($)",
        min_value=10_000.0,
        max_value=100_000_000.0,
        value=ENV_DEFAULTS["revenue"],
        step=10_000.0,
        format="%.0f",
    )
    cost = st.number_input(
        "Cost ($)",
        min_value=1_000.0,
        max_value=100_000_000.0,
        value=ENV_DEFAULTS["cost"],
        step=10_000.0,
        format="%.0f",
    )
    risk_score = st.slider(
        "Risk Score", 0.0, 1.0, float(ENV_DEFAULTS["risk_score"]), 0.01
    )
    churn = st.slider(
        "Churn Rate", 0.0, 1.0, float(ENV_DEFAULTS["churn"]), 0.01
    )
    marketing_budget = st.number_input(
        "Marketing Budget ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=ENV_DEFAULTS["marketing_budget"],
        step=5_000.0,
        format="%.0f",
    )
    growth_rate = st.slider(
        "Growth Rate", -0.5, 0.5, float(ENV_DEFAULTS["growth_rate"]), 0.01
    )
    volatility = st.slider(
        "Volatility", 0.0, 1.0, float(ENV_DEFAULTS["volatility"]), 0.01
    )

    st.markdown("---")

    run_btn = st.button("▶ Run Scenario", type="primary", use_container_width=True)
    clear_btn = st.button("🗑 Clear History", use_container_width=True)

# ── Run simulation ─────────────────────────────────────────────────────────────

if run_btn:
    initial_state = {
        "revenue": revenue,
        "cost": cost,
        "risk_score": risk_score,
        "churn": churn,
        "marketing_budget": marketing_budget,
        "growth_rate": growth_rate,
        "volatility": volatility,
    }

    with st.spinner("Running multi-agent simulation…"):
        simulator = Simulator(initial_state=initial_state, seed=int(seed))
        result = simulator.run(
            n_rounds=n_rounds,
            scenario_name=scenario_name,
            stochastic=stochastic,
        )

    st.session_state["simulation_result"] = result
    st.session_state["simulation_history"].append(result)
    st.session_state["run_count"] += 1

if clear_btn:
    st.session_state["simulation_result"] = None
    st.session_state["simulation_history"] = []
    st.session_state["run_count"] = 0
    st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────

st.markdown(f"# 🤖 {APP_TITLE}")

result = st.session_state.get("simulation_result")

if result is None:
    st.info(
        "👈 Configure the simulation settings in the sidebar and click "
        "**▶ Run Scenario** to start.",
        icon="ℹ️",
    )
    st.stop()

# ── Compute analytics ─────────────────────────────────────────────────────────

kpis = MetricsEngine.executive_kpis(result)
rev_proj = MetricsEngine.revenue_projection(result)
risk_tl = MetricsEngine.risk_timeline(result)
radar_data = MetricsEngine.agent_radar(result)
roi_data = MetricsEngine.roi_comparison(result)
timeline = MetricsEngine.simulation_timeline(result)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_agents, tab_charts, tab_history = st.tabs(
    ["📊 Executive Overview", "🤖 Agent Recommendations", "📈 Analytics", "📋 History"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 – EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown(f"### Scenario: **{result.scenario_name}**  ·  {result.n_rounds} rounds")
    st.markdown("---")

    # Row 1: Revenue / Profit / ROI
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Final Revenue",
        f"${kpis.revenue_final:,.0f}",
        delta=f"{kpis.revenue_change_pct:+.1%}",
    )
    c2.metric(
        "Final Profit",
        f"${kpis.profit_final:,.0f}",
        delta=f"Margin {kpis.profit_margin:.1%}",
    )
    c3.metric(
        "ROI",
        f"{kpis.roi:.2%}",
        delta=f"Cost ratio {kpis.cost_ratio:.1%}",
    )
    c4.metric(
        "Consensus Confidence",
        f"{kpis.consensus_confidence:.1%}",
        delta=f"{result.n_rounds} rounds",
    )

    # Row 2: Risk / Churn / Growth
    c5, c6, c7, c8 = st.columns(4)
    risk_delta_label = f"{kpis.risk_score_change:+.3f}"
    c5.metric("Risk Score", f"{kpis.risk_score_final:.3f}", delta=risk_delta_label,
              delta_color="inverse")
    c6.metric("Churn Rate", f"{kpis.churn_final:.2%}")
    c7.metric("Growth Rate", f"{kpis.growth_rate_final:.2%}")
    c8.metric("Simulation Runs", st.session_state["run_count"])

    st.markdown("---")

    # Risk gauge
    st.markdown("#### 🎯 Risk Gauge")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=kpis.risk_score_final,
            delta={"reference": result.initial_state.risk_score, "valueformat": ".3f"},
            number={"valueformat": ".3f"},
            title={"text": "Current Risk Score", "font": {"color": "#c9d1d9"}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#8b949e"},
                "bar": {"color": "#388bfd"},
                "bgcolor": "#161b22",
                "steps": [
                    {"range": [0.0, 0.25], "color": "#1a3a1a"},
                    {"range": [0.25, 0.50], "color": "#2d3a10"},
                    {"range": [0.50, 0.75], "color": "#3a2a10"},
                    {"range": [0.75, 1.00], "color": "#3a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "#f85149", "width": 4},
                    "thickness": 0.75,
                    "value": 0.65,
                },
            },
        )
    )
    gauge.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#c9d1d9",
        height=280,
        margin=dict(t=40, b=10, l=40, r=40),
    )
    st.plotly_chart(gauge, use_container_width=True)

    # Consensus summary
    st.markdown("#### 🤝 Final Consensus")
    last_round = result.rounds[-1]
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;"
        f"border-radius:8px;padding:14px 18px;color:#c9d1d9;'>"
        f"{last_round.consensus.summary.replace(chr(10), '<br>')}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 – AGENT RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════

with tab_agents:
    st.markdown("### Agent Proposals – Final Round")
    st.markdown("---")

    last_proposals = result.rounds[-1].proposals
    last_votes = result.rounds[-1].consensus.agent_votes

    agent_cols = st.columns(len(last_proposals))
    for col, proposal in zip(agent_cols, last_proposals):
        vote_pct = last_votes.get(proposal.agent_name, 0.0)
        conf_colour = (
            "#3fb950" if proposal.confidence >= 0.75
            else "#d29922" if proposal.confidence >= 0.50
            else "#f85149"
        )
        col.markdown(
            f"""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:8px;padding:14px 16px;">
                <div style="color:#58a6ff;font-weight:700;font-size:1rem;">
                    {proposal.agent_name}
                </div>
                <div style="color:#8b949e;font-size:0.78rem;margin-bottom:8px;">
                    Vote share: {vote_pct:.1%}
                </div>
                <div style="font-size:0.9rem;color:#c9d1d9;margin-bottom:6px;">
                    <b>Action:</b> {proposal.action}
                </div>
                <div style="font-size:0.82rem;color:{conf_colour};margin-bottom:6px;">
                    Confidence: {proposal.confidence:.1%}
                </div>
                <div style="font-size:0.78rem;color:#8b949e;">
                    {proposal.rationale}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Proposed Deltas (Final Round)")

    delta_rows = []
    for proposal in last_proposals:
        for metric, delta in proposal.deltas.items():
            delta_rows.append(
                {
                    "Agent": proposal.agent_name,
                    "Metric": metric,
                    "Delta": f"{delta:+.2%}",
                    "Confidence": f"{proposal.confidence:.1%}",
                }
            )

    if delta_rows:
        df_deltas = pd.DataFrame(delta_rows)
        st.dataframe(
            df_deltas,
            use_container_width=True,
            hide_index=True,
        )

    # Consensus deltas
    st.markdown("### Consensus Deltas (Final Round)")
    consensus_deltas = result.rounds[-1].consensus.final_deltas
    if consensus_deltas:
        df_consensus = pd.DataFrame(
            [
                {"Metric": k, "Weighted Delta": f"{v:+.2%}"}
                for k, v in sorted(consensus_deltas.items())
            ]
        )
        st.dataframe(df_consensus, use_container_width=True, hide_index=True)

    # Conflicts
    conflicts = result.rounds[-1].consensus.conflicts
    if conflicts:
        st.markdown("### ⚠️ Detected Conflicts (Final Round)")
        for c in conflicts:
            st.warning(
                f"**{c.agent_a}** ↔ **{c.agent_b}**  |  "
                f"Metrics: {', '.join(c.conflicting_keys)}  |  "
                f"Severity: {c.severity:.0%}"
            )
    else:
        st.success("No conflicts detected in the final round. ✅")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 – ANALYTICS CHARTS
# ════════════════════════════════════════════════════════════════════════════════

with tab_charts:
    # ── Revenue projection ────────────────────────────────────────────────────
    st.markdown("### 💰 Revenue & Cost Projection")
    fig_rev = go.Figure()
    fig_rev.add_trace(
        go.Scatter(
            x=rev_proj["rounds"],
            y=rev_proj["revenue"],
            name="Revenue",
            line=dict(color="#58a6ff", width=2),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.08)",
        )
    )
    fig_rev.add_trace(
        go.Scatter(
            x=rev_proj["rounds"],
            y=rev_proj["cost"],
            name="Cost",
            line=dict(color="#f85149", width=2, dash="dot"),
        )
    )
    fig_rev.add_trace(
        go.Scatter(
            x=rev_proj["rounds"],
            y=rev_proj["profit"],
            name="Profit",
            line=dict(color="#3fb950", width=2),
        )
    )
    fig_rev.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_title="Round",
        yaxis_title="USD ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(t=40, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Radar chart ───────────────────────────────────────────────────────────
    with col_l:
        st.markdown("### 🕸 Agent Alignment Radar")
        agents_r = radar_data["agents"]
        scores_r = radar_data["scores"]
        # Close the polygon
        agents_closed = agents_r + [agents_r[0]] if agents_r else agents_r
        scores_closed = scores_r + [scores_r[0]] if scores_r else scores_r
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=scores_closed,
                theta=agents_closed,
                fill="toself",
                fillcolor="rgba(56,139,253,0.20)",
                line=dict(color="#388bfd", width=2),
                name="Alignment",
            )
        )
        fig_radar.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor="#0d1117",
            polar=dict(
                bgcolor="#161b22",
                radialaxis=dict(visible=True, range=[0, 1], color="#8b949e"),
                angularaxis=dict(color="#8b949e"),
            ),
            showlegend=False,
            height=360,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── ROI comparison ────────────────────────────────────────────────────────
    with col_r:
        st.markdown("### 📊 ROI vs Growth Rate")
        fig_roi = go.Figure()
        fig_roi.add_trace(
            go.Bar(
                x=roi_data["rounds"],
                y=roi_data["roi"],
                name="ROI",
                marker_color="#388bfd",
                opacity=0.85,
            )
        )
        fig_roi.add_trace(
            go.Scatter(
                x=roi_data["rounds"],
                y=roi_data["growth_rate"],
                name="Growth Rate",
                line=dict(color="#3fb950", width=2),
                yaxis="y2",
            )
        )
        fig_roi.update_layout(
            template=CHART_TEMPLATE,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            xaxis_title="Round",
            yaxis=dict(title="ROI", tickformat=".0%"),
            yaxis2=dict(
                title="Growth Rate",
                overlaying="y",
                side="right",
                tickformat=".1%",
                color="#3fb950",
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=360,
            margin=dict(t=20, b=40, l=10, r=60),
        )
        st.plotly_chart(fig_roi, use_container_width=True)

    # ── Risk timeline ─────────────────────────────────────────────────────────
    st.markdown("### 🔴 Risk & Volatility Timeline")
    fig_risk = go.Figure()
    fig_risk.add_trace(
        go.Scatter(
            x=risk_tl["rounds"],
            y=risk_tl["risk_score"],
            name="Risk Score",
            line=dict(color="#f85149", width=2),
            fill="tozeroy",
            fillcolor="rgba(248,81,73,0.10)",
        )
    )
    fig_risk.add_trace(
        go.Scatter(
            x=risk_tl["rounds"],
            y=risk_tl["volatility"],
            name="Volatility",
            line=dict(color="#d29922", width=2, dash="dot"),
        )
    )
    fig_risk.update_layout(
        template=CHART_TEMPLATE,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        xaxis_title="Round",
        yaxis_title="Score (0–1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=300,
        margin=dict(t=20, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 – SIMULATION HISTORY
# ════════════════════════════════════════════════════════════════════════════════

with tab_history:
    st.markdown("### 📋 Simulation History Timeline")
    st.markdown("---")

    if not timeline:
        st.info("No timeline data available.")
    else:
        # Timeline table
        df_timeline = pd.DataFrame(timeline)
        df_timeline["revenue"] = df_timeline["revenue"].map("${:,.0f}".format)
        df_timeline["risk_score"] = df_timeline["risk_score"].map("{:.3f}".format)
        df_timeline["growth_rate"] = df_timeline["growth_rate"].map("{:.2%}".format)
        df_timeline["confidence"] = df_timeline["confidence"].map("{:.1%}".format)
        df_timeline["consensus_confidence"] = df_timeline["consensus_confidence"].map(
            "{:.1%}".format
        )
        df_timeline["noise"] = df_timeline["noise"].map("{:+.4f}".format)
        df_timeline.rename(
            columns={
                "round": "Round",
                "top_agent": "Top Agent",
                "action": "Action",
                "confidence": "Agent Conf.",
                "consensus_confidence": "Consensus Conf.",
                "conflicts": "Conflicts",
                "revenue": "Revenue",
                "risk_score": "Risk Score",
                "growth_rate": "Growth",
                "noise": "Noise",
            },
            inplace=True,
        )
        st.dataframe(df_timeline, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📚 All Simulation Runs")

    history = st.session_state["simulation_history"]
    if not history:
        st.info("No previous runs recorded.")
    else:
        summary_rows = []
        for idx, hist_result in enumerate(history, start=1):
            h_kpis = MetricsEngine.executive_kpis(hist_result)
            summary_rows.append(
                {
                    "Run #": idx,
                    "Scenario": hist_result.scenario_name,
                    "Rounds": hist_result.n_rounds,
                    "Final Revenue": f"${h_kpis.revenue_final:,.0f}",
                    "Revenue Δ": f"{h_kpis.revenue_change_pct:+.1%}",
                    "Final Risk": f"{h_kpis.risk_score_final:.3f}",
                    "ROI": f"{h_kpis.roi:.2%}",
                    "Avg Confidence": f"{h_kpis.consensus_confidence:.1%}",
                }
            )
        df_history = pd.DataFrame(summary_rows)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
