# AgentSphere AI – Multi-Agent Strategic Simulation Engine

> Multiple AI agents interacting, negotiating, competing, and optimising decisions in a simulated business environment.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## Overview

AgentSphere AI is a **production-ready multi-agent simulation system** built with Python and Streamlit.  Four specialised AI agents continuously observe a `BusinessEnvironment`, propose strategy changes, negotiate a weighted consensus, and update the simulated environment – all in real time inside the browser.

## Project Structure

```
agentsphere/
├── __init__.py
├── config.py                   # Global constants & defaults
│
├── agents/
│   ├── base_agent.py           # Abstract BaseAgent + AgentProposal dataclass
│   ├── revenue_agent.py        # Revenue maximisation & churn reduction
│   ├── risk_agent.py           # Risk assessment & mitigation
│   ├── cost_agent.py           # Cost optimisation & margin management
│   └── growth_agent.py         # Market expansion & growth strategy
│
├── environment/
│   └── business_env.py         # Mutable BusinessEnvironment + EnvironmentState
│
├── negotiation/
│   └── engine.py               # Weighted-voting NegotiationEngine + conflict detection
│
├── simulation/
│   └── simulator.py            # Multi-round Simulator orchestration
│
└── analytics/
    └── metrics.py              # MetricsEngine – KPIs, charts, timeline data

app.py                          # Streamlit UI entry point
requirements.txt
```

## Agents

| Agent | Responsibility | Weight |
|-------|---------------|--------|
| **RevenueAgent** | Maximise revenue growth, reduce churn | 30 % |
| **RiskAgent** | Assess & mitigate business risk | 25 % |
| **CostAgent** | Optimise cost structure & profit margin | 25 % |
| **GrowthAgent** | Drive market expansion | 20 % |

Each agent:
1. **Evaluates** the current environment state.
2. **Proposes** relative metric deltas with a confidence score.
3. Participates in **weighted-voting negotiation** – conflicting agents are penalised.

## Simulation Engine

- Runs **1–12 configurable rounds**.
- Applies **stochastic market noise** scaled by current *volatility*.
- Stores full per-round history with proposals, consensus, and environment snapshots.

## Streamlit UI

- **Executive KPIs** – Revenue, Profit, ROI, Risk Score, Churn, Growth Rate.
- **Agent Recommendations** – Per-agent cards, delta tables, conflict reports.
- **Analytics Charts** – Revenue projection, Risk timeline, Radar alignment, ROI comparison.
- **Simulation History** – Round-by-round event log and multi-run comparison table.
- **Professional dark theme** throughout.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repository, branch `main`, and entry point `app.py`.
4. Click **Deploy** – no backend required.
