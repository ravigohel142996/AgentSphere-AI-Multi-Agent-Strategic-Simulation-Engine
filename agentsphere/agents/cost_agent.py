"""
CostAgent – optimises operational expenditure and profit margins.

Strategy:
- If cost/revenue ratio is high, recommend cost reduction initiatives.
- If margin is healthy, allow moderate reinvestment.
- Targets a sustainable cost ratio and tracks efficiency improvements.
"""

from __future__ import annotations

import math

from agentsphere.agents.base_agent import AgentProposal, BaseAgent
from agentsphere.environment.business_env import EnvironmentState


class CostAgent(BaseAgent):
    """Agent focused on cost optimisation and margin improvement."""

    TARGET_COST_RATIO: float = 0.55  # Target cost-to-revenue ratio
    CRITICAL_COST_RATIO: float = 0.80
    MIN_COST_RATIO: float = 0.40     # Below this, cost cuts may harm quality

    def __init__(self, weight: float = 0.25) -> None:
        super().__init__(name="CostAgent", weight=weight)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def evaluate(self, state: EnvironmentState) -> dict[str, float]:
        """Return cost-specific evaluation metrics.

        Args:
            state: Current environment state.

        Returns:
            Dict with ``cost_ratio``, ``margin``, and ``cost_gap`` metrics.
        """
        cost_ratio = state.cost / state.revenue if state.revenue > 0 else 1.0
        margin = 1.0 - cost_ratio
        cost_gap = cost_ratio - self.TARGET_COST_RATIO
        return {
            "cost_ratio": cost_ratio,
            "margin": margin,
            "cost_gap": cost_gap,
        }

    def propose(self, state: EnvironmentState) -> AgentProposal:
        """Recommend cost-reduction or reinvestment actions.

        Logic:
        - Critical cost ratio (≥ 0.80) → aggressive cost reduction.
        - High cost ratio               → moderate reduction.
        - Target cost ratio achieved    → allow reinvestment.
        - Below minimum cost ratio      → flag over-cutting risk.

        Args:
            state: Current environment state.

        Returns:
            AgentProposal with recommended deltas and confidence score.
        """
        metrics = self.evaluate(state)
        cost_ratio = metrics["cost_ratio"]
        cost_gap = metrics["cost_gap"]

        deltas: dict[str, float] = {}
        reasons: list[str] = []
        confidence_raw: float

        if cost_ratio >= self.CRITICAL_COST_RATIO:
            # Aggressive cost-cutting
            reduction = min(0.15, cost_gap * 0.70)
            deltas = {
                "cost": -reduction,
                "marketing_budget": -0.05,  # trim non-critical spend
                "risk_score": 0.03,          # cost pressure increases risk
            }
            reasons.append(
                f"Critical cost ratio {cost_ratio:.2f}; aggressive reduction "
                f"targeted (cost ↓{reduction:.1%})."
            )
            confidence_raw = 0.88

        elif cost_gap > 0:
            # Moderate reduction
            reduction = min(0.08, cost_gap * 0.50)
            deltas = {
                "cost": -reduction,
                "marketing_budget": -0.02,
            }
            reasons.append(
                f"Cost ratio {cost_ratio:.2f} above target {self.TARGET_COST_RATIO}; "
                f"moderate reduction (cost ↓{reduction:.1%})."
            )
            confidence_raw = 0.75

        elif cost_ratio < self.MIN_COST_RATIO:
            # Over-cut warning – allow some cost increase for quality
            deltas = {
                "cost": 0.03,
                "revenue": 0.02,   # quality reinvestment drives revenue
                "risk_score": -0.02,
            }
            reasons.append(
                f"Cost ratio {cost_ratio:.2f} below minimum; reinvesting "
                "to sustain quality and reduce operational risk."
            )
            confidence_raw = 0.68

        else:
            # On-target: minor optimisation
            deltas = {"cost": -0.01, "revenue": 0.01}
            reasons.append(
                f"Cost ratio {cost_ratio:.2f} near target; maintaining "
                "steady-state efficiency."
            )
            confidence_raw = 0.70

        confidence = 1.0 / (1.0 + math.exp(-6.0 * (confidence_raw - 0.5)))

        return AgentProposal(
            agent_name=self.name,
            action=reasons[0].split(";")[0],
            deltas=deltas,
            confidence=round(confidence, 4),
            rationale=" | ".join(reasons),
            priority=2,
        )
