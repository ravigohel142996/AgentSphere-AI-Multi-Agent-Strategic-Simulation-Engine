"""
RiskAgent – monitors, predicts, and mitigates business risk.

Strategy:
- If risk_score is high, recommend risk-mitigation spend and lower volatility.
- If volatility is extreme, dampen growth targets.
- If risk is low, allow more aggressive growth with a modest risk allowance.
"""

from __future__ import annotations

import math

from agentsphere.agents.base_agent import AgentProposal, BaseAgent
from agentsphere.environment.business_env import EnvironmentState


class RiskAgent(BaseAgent):
    """Agent focused on risk assessment and mitigation."""

    CRITICAL_RISK: float = 0.65   # risk_score threshold for critical state
    HIGH_RISK: float = 0.45       # risk_score threshold for elevated state
    SAFE_RISK: float = 0.25       # risk_score threshold for safe state
    HIGH_VOLATILITY: float = 0.30 # volatility threshold

    def __init__(self, weight: float = 0.25) -> None:
        super().__init__(name="RiskAgent", weight=weight)

    # ── BaseAgent interface ───────────────────────────────────────────────────

    def evaluate(self, state: EnvironmentState) -> dict[str, float]:
        """Return risk-specific evaluation metrics.

        Args:
            state: Current environment state.

        Returns:
            Dict with ``risk_level``, ``volatility_excess``, and
            ``risk_adjusted_growth`` metrics.
        """
        risk_excess = max(0.0, state.risk_score - self.SAFE_RISK)
        volatility_excess = max(0.0, state.volatility - self.HIGH_VOLATILITY)
        risk_adjusted_growth = state.growth_rate * (1.0 - state.risk_score)
        return {
            "risk_excess": risk_excess,
            "volatility_excess": volatility_excess,
            "risk_adjusted_growth": risk_adjusted_growth,
        }

    def propose(self, state: EnvironmentState) -> AgentProposal:
        """Recommend risk-mitigation or risk-taking actions.

        Logic:
        - Critical risk → significant mitigation (cost +5 %, revenue growth capped).
        - High risk     → moderate mitigation.
        - High volatility → dampen growth exposure.
        - Low/safe risk → permit growth acceleration.

        Args:
            state: Current environment state.

        Returns:
            AgentProposal with recommended deltas and confidence score.
        """
        metrics = self.evaluate(state)
        risk_excess = metrics["risk_excess"]
        volatility_excess = metrics["volatility_excess"]

        deltas: dict[str, float] = {}
        reasons: list[str] = []
        confidence_raw: float

        if state.risk_score >= self.CRITICAL_RISK:
            # Emergency mitigation
            mitigation = risk_excess * 0.40
            deltas = {
                "risk_score": -mitigation,
                "cost": 0.05,               # risk-mitigation spend
                "revenue": -0.02,           # short-term revenue hit
                "volatility": -0.05,
                "growth_rate": -0.02,
            }
            reasons.append(
                f"CRITICAL risk {state.risk_score:.2f}; emergency mitigation "
                f"deployed (risk ↓{mitigation:.2f})."
            )
            confidence_raw = 0.90

        elif state.risk_score >= self.HIGH_RISK:
            mitigation = risk_excess * 0.25
            deltas = {
                "risk_score": -mitigation,
                "cost": 0.02,
                "volatility": -0.03,
            }
            reasons.append(
                f"Elevated risk {state.risk_score:.2f}; moderate mitigation "
                f"(risk ↓{mitigation:.2f})."
            )
            confidence_raw = 0.78

        elif state.risk_score <= self.SAFE_RISK:
            # Safe zone: allow growth headroom
            deltas = {
                "risk_score": 0.02,          # slight risk allowance for growth
                "growth_rate": 0.01,
                "revenue": 0.03,
            }
            reasons.append(
                f"Risk at safe level {state.risk_score:.2f}; allocating "
                "headroom for growth acceleration."
            )
            confidence_raw = 0.72

        else:
            # Moderate risk – hold steady
            deltas = {"risk_score": -0.02, "cost": 0.01}
            reasons.append(
                f"Moderate risk {state.risk_score:.2f}; holding steady with "
                "minor mitigation."
            )
            confidence_raw = 0.65

        if volatility_excess > 0:
            vol_reduction = min(0.10, volatility_excess * 0.50)
            deltas["volatility"] = deltas.get("volatility", 0.0) - vol_reduction
            deltas["growth_rate"] = deltas.get("growth_rate", 0.0) - volatility_excess * 0.15
            reasons.append(
                f"High volatility {state.volatility:.2f}; dampening growth "
                f"exposure (volatility ↓{vol_reduction:.2f})."
            )
            confidence_raw = min(1.0, confidence_raw + 0.05)

        confidence = 1.0 / (1.0 + math.exp(-6.0 * (confidence_raw - 0.5)))

        return AgentProposal(
            agent_name=self.name,
            action=reasons[0].split(";")[0],
            deltas=deltas,
            confidence=round(confidence, 4),
            rationale=" | ".join(reasons),
            priority=2 if state.risk_score < self.HIGH_RISK else 1,
        )
