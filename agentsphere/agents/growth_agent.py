"""
GrowthAgent – drives market expansion and strategic growth initiatives.

Strategy:
- Identifies growth opportunities based on current growth_rate and market conditions.
- Balances aggressive expansion against risk and cost constraints.
- Recommends marketing investment to capture market share.
"""

from __future__ import annotations

import math

from agentsphere.agents.base_agent import AgentProposal, BaseAgent
from agentsphere.environment.business_env import EnvironmentState


class GrowthAgent(BaseAgent):
    """Agent focused on strategic growth and market expansion."""

    AGGRESSIVE_GROWTH: float = 0.12  # Target when conditions are favourable
    CONSERVATIVE_GROWTH: float = 0.04
    RISK_TOLERANCE: float = 0.40      # Max risk_score before pulling back

    def __init__(self, weight: float = 0.20) -> None:
        super().__init__(name="GrowthAgent", weight=weight)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def evaluate(self, state: EnvironmentState) -> dict[str, float]:
        """Return growth-specific evaluation metrics.

        Args:
            state: Current environment state.

        Returns:
            Dict with ``growth_potential``, ``market_saturation``, and
            ``risk_adjusted_potential`` metrics.
        """
        # Growth potential: how much room is there above the conservative floor?
        growth_potential = max(
            0.0, self.AGGRESSIVE_GROWTH - state.growth_rate
        )
        # Market saturation proxy: high marketing spend relative to revenue
        market_saturation = min(
            1.0,
            state.marketing_budget / (state.revenue * 0.15)
            if state.revenue > 0
            else 1.0,
        )
        # Risk-adjusted potential
        risk_factor = max(0.0, 1.0 - state.risk_score / self.RISK_TOLERANCE)
        risk_adjusted_potential = growth_potential * risk_factor
        return {
            "growth_potential": growth_potential,
            "market_saturation": market_saturation,
            "risk_adjusted_potential": risk_adjusted_potential,
        }

    def propose(self, state: EnvironmentState) -> AgentProposal:
        """Recommend growth strategy actions.

        Logic:
        - High risk        → conservative growth, reduce marketing exposure.
        - Unsaturated market + low risk → aggressive expansion.
        - Stagnant growth  → moderate marketing investment.
        - Already growing well          → sustain and optimise.

        Args:
            state: Current environment state.

        Returns:
            AgentProposal with recommended deltas and confidence score.
        """
        metrics = self.evaluate(state)
        growth_potential = metrics["growth_potential"]
        market_saturation = metrics["market_saturation"]
        risk_adjusted_potential = metrics["risk_adjusted_potential"]

        deltas: dict[str, float] = {}
        reasons: list[str] = []
        confidence_raw: float

        if state.risk_score > self.RISK_TOLERANCE:
            # Pull back growth plans due to risk
            deltas = {
                "growth_rate": -0.01,
                "marketing_budget": -0.05,
                "revenue": -0.01,
            }
            reasons.append(
                f"Risk {state.risk_score:.2f} exceeds tolerance "
                f"{self.RISK_TOLERANCE}; scaling back growth strategy."
            )
            confidence_raw = 0.80

        elif risk_adjusted_potential > 0.04 and market_saturation < 0.70:
            # Aggressive expansion
            mkt_boost = min(0.20, risk_adjusted_potential * 1.5)
            rev_boost = risk_adjusted_potential * 1.2
            deltas = {
                "growth_rate": risk_adjusted_potential * 0.80,
                "marketing_budget": mkt_boost,
                "revenue": rev_boost,
                "churn": -0.01,  # better brand drives retention
            }
            reasons.append(
                f"Favourable conditions (risk {state.risk_score:.2f}, "
                f"saturation {market_saturation:.0%}); aggressive expansion "
                f"(growth ↑{risk_adjusted_potential * 0.80:.1%})."
            )
            confidence_raw = 0.85

        elif state.growth_rate < self.CONSERVATIVE_GROWTH:
            # Stagnation – moderate push
            deltas = {
                "growth_rate": 0.02,
                "marketing_budget": 0.08,
                "revenue": 0.03,
            }
            reasons.append(
                f"Stagnant growth {state.growth_rate:.1%}; moderate marketing "
                "push to reignite expansion."
            )
            confidence_raw = 0.72

        else:
            # Sustain and optimise
            deltas = {
                "growth_rate": 0.005,
                "revenue": 0.015,
            }
            reasons.append(
                f"Growth {state.growth_rate:.1%} on track; sustaining momentum "
                "with incremental optimisation."
            )
            confidence_raw = 0.68

        confidence = 1.0 / (1.0 + math.exp(-6.0 * (confidence_raw - 0.5)))

        return AgentProposal(
            agent_name=self.name,
            action=reasons[0].split(";")[0],
            deltas=deltas,
            confidence=round(confidence, 4),
            rationale=" | ".join(reasons),
            priority=3,
        )
