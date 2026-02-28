"""
Simulator – orchestrates multi-round agent interactions and state updates.

Each round:
1. Agents observe the current environment state.
2. Each agent produces an ``AgentProposal``.
3. The ``NegotiationEngine`` reaches consensus on a merged delta set.
4. Optional stochastic noise (driven by the environment's *volatility*) is added.
5. The ``BusinessEnvironment`` applies the final deltas and advances one round.
6. The round result is stored in history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from agentsphere.agents import (
    BaseAgent,
    CostAgent,
    GrowthAgent,
    RevenueAgent,
    RiskAgent,
)
from agentsphere.agents.base_agent import AgentProposal
from agentsphere.config import AGENT_WEIGHTS, DEFAULT_ROUNDS, RANDOM_SEED
from agentsphere.environment.business_env import BusinessEnvironment, EnvironmentState
from agentsphere.negotiation.engine import ConsensusResult, NegotiationEngine


@dataclass
class RoundResult:
    """Outcome of a single simulation round.

    Attributes:
        round_number:  1-based round index.
        proposals:     One ``AgentProposal`` per agent.
        consensus:     The ``ConsensusResult`` from the negotiation engine.
        state_before:  Environment snapshot before applying deltas.
        state_after:   Environment snapshot after applying deltas.
        noise:         Stochastic noise factor applied this round.
    """

    round_number: int
    proposals: list[AgentProposal]
    consensus: ConsensusResult
    state_before: EnvironmentState
    state_after: EnvironmentState
    noise: float


@dataclass
class SimulationResult:
    """Aggregated result of a complete simulation run.

    Attributes:
        rounds:         Ordered list of ``RoundResult`` objects.
        initial_state:  Environment snapshot before the first round.
        final_state:    Environment snapshot after the last round.
        scenario_name:  Optional label for the scenario.
        metadata:       Arbitrary extra data stored by the caller.
    """

    rounds: list[RoundResult]
    initial_state: EnvironmentState
    final_state: EnvironmentState
    scenario_name: str = "Default Scenario"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rounds(self) -> int:
        """Number of rounds executed."""
        return len(self.rounds)

    @property
    def revenue_delta(self) -> float:
        """Absolute revenue change over the full simulation."""
        return self.final_state.revenue - self.initial_state.revenue

    @property
    def revenue_growth_pct(self) -> float:
        """Percentage revenue change over the full simulation."""
        if self.initial_state.revenue == 0:
            return 0.0
        return self.revenue_delta / self.initial_state.revenue

    @property
    def risk_delta(self) -> float:
        """Change in risk_score over the full simulation (negative is good)."""
        return self.final_state.risk_score - self.initial_state.risk_score


class Simulator:
    """Runs multi-round agent simulations against a ``BusinessEnvironment``.

    Args:
        initial_state: Optional dict of initial environment values to override
                       project defaults.
        seed:          Random seed for reproducible stochastic noise.
        agents:        Optional custom agent list. Defaults to the four standard
                       agents with weights from ``config.AGENT_WEIGHTS``.
    """

    def __init__(
        self,
        initial_state: dict[str, float] | None = None,
        seed: int = RANDOM_SEED,
        agents: list[BaseAgent] | None = None,
    ) -> None:
        self._env = BusinessEnvironment(initial_state)
        self._rng = random.Random(seed)
        self._agents: list[BaseAgent] = agents or self._default_agents()
        self._engine = NegotiationEngine(AGENT_WEIGHTS)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        n_rounds: int = DEFAULT_ROUNDS,
        scenario_name: str = "Default Scenario",
        stochastic: bool = True,
    ) -> SimulationResult:
        """Execute *n_rounds* of the simulation and return aggregated results.

        Args:
            n_rounds:      Number of simulation rounds to run.
            scenario_name: Human-readable label stored in the result.
            stochastic:    If ``True``, environmental noise is sampled each
                           round using the current *volatility* value.

        Returns:
            A :class:`SimulationResult` containing round-by-round data.
        """
        # Reset state
        self._env.reset()
        for agent in self._agents:
            agent.reset()

        initial_state = self._env.snapshot()
        rounds: list[RoundResult] = []

        for round_idx in range(1, n_rounds + 1):
            state_before = self._env.snapshot()

            # Each agent observes and proposes
            proposals: list[AgentProposal] = [
                agent.act(state_before) for agent in self._agents
            ]

            # Negotiate consensus
            consensus: ConsensusResult = self._engine.negotiate(proposals)

            # Sample noise proportional to current volatility
            noise = (
                self._rng.gauss(0.0, state_before.volatility * 0.1)
                if stochastic
                else 0.0
            )

            # Apply consensus deltas + noise to environment
            state_after = self._env.apply_deltas(consensus.final_deltas, noise=noise)

            rounds.append(
                RoundResult(
                    round_number=round_idx,
                    proposals=proposals,
                    consensus=consensus,
                    state_before=state_before,
                    state_after=state_after,
                    noise=round(noise, 6),
                )
            )

        return SimulationResult(
            rounds=rounds,
            initial_state=initial_state,
            final_state=self._env.snapshot(),
            scenario_name=scenario_name,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_agents() -> list[BaseAgent]:
        """Instantiate the four standard agents with default weights."""
        return [
            RevenueAgent(weight=AGENT_WEIGHTS["RevenueAgent"]),
            RiskAgent(weight=AGENT_WEIGHTS["RiskAgent"]),
            CostAgent(weight=AGENT_WEIGHTS["CostAgent"]),
            GrowthAgent(weight=AGENT_WEIGHTS["GrowthAgent"]),
        ]
