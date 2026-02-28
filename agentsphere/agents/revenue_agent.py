"""
RevenueAgent – maximises revenue growth while managing churn.

Strategy:
- If revenue growth is below target, recommend increasing marketing budget.
- If churn is high, recommend retention investment (reduces churn, boosts revenue).
- Confidence scales with the magnitude of the opportunity detected.
"""

from __future__ import annotations

import math

from agentsphere.agents.base_agent import AgentProposal, BaseAgent
from agentsphere.environment.business_env import EnvironmentState


class RevenueAgent(BaseAgent):
    """Agent focused on revenue optimisation and customer retention."""

    TARGET_GROWTH_RATE: float = 0.08   # 8 % desired quarterly growth
    HIGH_CHURN_THRESHOLD: float = 0.10 # 10 % churn is considered critical

    def __init__(self, weight: float = 0.30) -> None:
        super().__init__(name="RevenueAgent", weight=weight)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def evaluate(self, state: EnvironmentState) -> dict[str, float]:
        """Return revenue-specific KPI scores.

        Args:
            state: Current environment state.

        Returns:
            Dict with ``revenue_gap``, ``churn_severity``, and
            ``marketing_efficiency`` metrics.
        """
        revenue_gap = self.TARGET_GROWTH_RATE - state.growth_rate
        churn_severity = max(0.0, state.churn - self.HIGH_CHURN_THRESHOLD)
        # Marketing efficiency: revenue generated per dollar of marketing spend
        marketing_efficiency = (
            state.revenue / state.marketing_budget
            if state.marketing_budget > 0
            else 0.0
        )
        return {
            "revenue_gap": revenue_gap,
            "churn_severity": churn_severity,
            "marketing_efficiency": marketing_efficiency,
        }

    def propose(self, state: EnvironmentState) -> AgentProposal:
        """Recommend revenue-maximising actions.

        Logic:
        - Low growth  → increase marketing budget (+15 %) and revenue (+4 %).
        - High churn  → reduce churn (-20 %) with retention spend
                        (marketing -5 %, cost +2 %).
        - Both issues → combine adjustments.
        - Healthy     → maintain trajectory (small positive nudge).

        Args:
            state: Current environment state.

        Returns:
            AgentProposal with recommended deltas and confidence score.
        """
        metrics = self.evaluate(state)
        revenue_gap = metrics["revenue_gap"]
        churn_severity = metrics["churn_severity"]

        deltas: dict[str, float] = {}
        reasons: list[str] = []
        confidence_factors: list[float] = []

        if revenue_gap > 0:
            # Boost marketing and project revenue gain
            boost = min(0.20, revenue_gap * 2.0)
            deltas["marketing_budget"] = boost
            deltas["revenue"] = revenue_gap * 0.80
            deltas["growth_rate"] = revenue_gap * 0.50
            reasons.append(
                f"Growth rate {state.growth_rate:.1%} is below target "
                f"{self.TARGET_GROWTH_RATE:.1%}; increasing marketing by "
                f"{boost:.1%}."
            )
            confidence_factors.append(min(1.0, revenue_gap / self.TARGET_GROWTH_RATE))

        if churn_severity > 0:
            # Invest in retention
            churn_reduction = min(0.25, churn_severity * 3.0)
            deltas["churn"] = -churn_reduction
            deltas["cost"] = deltas.get("cost", 0.0) + 0.02  # retention cost
            deltas["revenue"] = deltas.get("revenue", 0.0) + churn_severity * 1.5
            reasons.append(
                f"Churn {state.churn:.1%} exceeds threshold; deploying "
                f"retention programme (churn ↓{churn_reduction:.1%})."
            )
            confidence_factors.append(
                min(1.0, churn_severity / self.HIGH_CHURN_THRESHOLD)
            )

        if not deltas:
            # Healthy state – maintain with a small revenue nudge
            deltas = {"revenue": 0.02, "growth_rate": 0.005}
            reasons.append("Revenue metrics healthy; maintaining trajectory.")
            confidence_factors.append(0.70)

        confidence = float(
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.65
        )
        # Normalise confidence using a logistic-style curve so it stays in (0,1)
        confidence = 1.0 / (1.0 + math.exp(-6.0 * (confidence - 0.5)))

        return AgentProposal(
            agent_name=self.name,
            action=(
                "Boost marketing & reduce churn"
                if len(reasons) > 1
                else reasons[0].split(";")[0]
            ),
            deltas=deltas,
            confidence=round(confidence, 4),
            rationale=" | ".join(reasons),
            priority=1,
        )
