"""
AgentSphere AI – Global configuration constants and defaults.
"""

from __future__ import annotations

# ── Environment defaults ──────────────────────────────────────────────────────

ENV_DEFAULTS: dict[str, float] = {
    "revenue": 1_000_000.0,      # $1 M baseline annual revenue
    "cost": 600_000.0,            # $600 K baseline annual cost
    "risk_score": 0.35,           # 0–1 scale (lower is better)
    "churn": 0.08,                # 8 % monthly churn rate
    "marketing_budget": 80_000.0, # $80 K marketing budget
    "growth_rate": 0.05,          # 5 % quarterly growth rate
    "volatility": 0.15,           # 15 % market volatility
}

# ── Simulation parameters ─────────────────────────────────────────────────────

MAX_ROUNDS: int = 12          # Maximum simulation rounds per run
DEFAULT_ROUNDS: int = 6       # Default number of rounds
RANDOM_SEED: int = 42

# ── Agent negotiation weights ─────────────────────────────────────────────────
# Must sum to 1.0

AGENT_WEIGHTS: dict[str, float] = {
    "RevenueAgent": 0.30,
    "RiskAgent":    0.25,
    "CostAgent":    0.25,
    "GrowthAgent":  0.20,
}

# ── Conflict detection threshold ──────────────────────────────────────────────
# Two proposals conflict when their recommended delta directions oppose each
# other across this fraction of shared metrics.

CONFLICT_THRESHOLD: float = 0.50

# ── UI constants ──────────────────────────────────────────────────────────────

APP_TITLE: str = "AgentSphere AI – Multi-Agent Strategic Simulation Engine"
CHART_TEMPLATE: str = "plotly_dark"
