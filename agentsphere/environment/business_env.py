"""
BusinessEnvironment – mutable simulation environment for AgentSphere AI.

The environment holds the current state of the business and exposes methods to
apply agent-proposed deltas, snapshot the state, and reset to initial values.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EnvironmentState:
    """Immutable snapshot of the business environment at a point in time.

    All monetary values are in USD.  Rates and scores are in [0, 1].

    Attributes:
        revenue:          Annual revenue.
        cost:             Annual operational cost.
        risk_score:       Composite risk score (0 = no risk, 1 = maximum risk).
        churn:            Monthly customer churn rate.
        marketing_budget: Monthly marketing expenditure.
        growth_rate:      Quarterly revenue growth rate.
        volatility:       Market volatility index.
        round_number:     The simulation round this snapshot was captured in.
    """

    revenue: float
    cost: float
    risk_score: float
    churn: float
    marketing_budget: float
    growth_rate: float
    volatility: float
    round_number: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this state."""
        return asdict(self)


class BusinessEnvironment:
    """Mutable business environment that agents observe and act upon.

    The environment applies agent-proposed *relative* deltas to its state
    while enforcing domain-specific bounds (e.g. revenue > 0, risk_score ∈ [0,1]).

    Args:
        initial_state: Initial values for all environment metrics.  Defaults
                       to the project-level ``ENV_DEFAULTS``.
    """

    # Hard bounds for each metric
    _BOUNDS: dict[str, tuple[float, float]] = {
        "revenue":          (1.0,    1e9),
        "cost":             (1.0,    1e9),
        "risk_score":       (0.0,    1.0),
        "churn":            (0.0,    1.0),
        "marketing_budget": (0.0,    1e8),
        "growth_rate":      (-0.50,  2.0),
        "volatility":       (0.0,    1.0),
    }

    def __init__(self, initial_state: dict[str, float] | None = None) -> None:
        from agentsphere.config import ENV_DEFAULTS

        defaults = ENV_DEFAULTS.copy()
        if initial_state:
            defaults.update(initial_state)

        self._initial: dict[str, float] = defaults.copy()
        self._state: EnvironmentState = EnvironmentState(**defaults)
        self._round: int = 0
        self._history: list[EnvironmentState] = []

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> EnvironmentState:
        """Current environment state (read-only snapshot)."""
        return copy.copy(self._state)

    @property
    def round(self) -> int:
        """Current simulation round number."""
        return self._round

    @property
    def history(self) -> list[EnvironmentState]:
        """All snapshots captured so far (one per round)."""
        return list(self._history)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def apply_deltas(
        self, deltas: dict[str, float], noise: float = 0.0
    ) -> EnvironmentState:
        """Apply relative deltas to the current state and advance the round.

        Each delta is interpreted as a *fractional change*:
        ``new_value = old_value * (1 + delta + noise)``.

        Unknown metric keys are silently ignored.

        Args:
            deltas: Mapping of metric name → fractional change (e.g. 0.05 = +5 %).
            noise:  Optional random noise factor already sampled by the simulator.

        Returns:
            The new ``EnvironmentState`` after applying all deltas.
        """
        current = asdict(self._state)
        new_values: dict[str, float] = {}

        for key, value in current.items():
            if key == "round_number":
                continue
            delta = deltas.get(key, 0.0)
            if isinstance(value, (int, float)):
                raw = value * (1.0 + delta + noise * (1 if delta >= 0 else -1))
                lo, hi = self._BOUNDS.get(key, (-1e12, 1e12))
                new_values[key] = max(lo, min(hi, raw))
            else:
                new_values[key] = value

        self._round += 1
        new_values["round_number"] = self._round
        self._state = EnvironmentState(**new_values)
        self._history.append(copy.copy(self._state))
        return self._state

    def snapshot(self) -> EnvironmentState:
        """Capture the current state without advancing the round.

        Returns:
            An immutable copy of the current ``EnvironmentState``.
        """
        snapshot = copy.copy(self._state)
        snapshot = EnvironmentState(**{**asdict(snapshot), "round_number": self._round})
        return snapshot

    def reset(self, initial_state: dict[str, float] | None = None) -> None:
        """Reset the environment to its initial (or provided) state.

        Args:
            initial_state: Optional override; if omitted the original defaults
                           passed to ``__init__`` are used.
        """
        values = self._initial.copy()
        if initial_state:
            values.update(initial_state)
        self._state = EnvironmentState(**values)
        self._round = 0
        self._history.clear()

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def profit(self) -> float:
        """Current period profit (revenue − cost)."""
        return self._state.revenue - self._state.cost

    @property
    def profit_margin(self) -> float:
        """Current profit margin as a fraction of revenue."""
        if self._state.revenue == 0:
            return 0.0
        return self.profit / self._state.revenue

    @property
    def roi(self) -> float:
        """Return on investment: profit / cost."""
        if self._state.cost == 0:
            return 0.0
        return self.profit / self._state.cost

    def __repr__(self) -> str:
        s = self._state
        return (
            f"BusinessEnvironment(round={self._round}, "
            f"revenue={s.revenue:,.0f}, cost={s.cost:,.0f}, "
            f"risk={s.risk_score:.2f})"
        )
