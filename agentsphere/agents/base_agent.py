"""
Abstract base class that all AgentSphere agents must inherit from.

Every agent follows a read-evaluate-propose lifecycle:
  1. ``evaluate(state)``  – inspect the current ``BusinessEnvironment`` state.
  2. ``propose(state)``   – return an ``AgentProposal`` with recommended deltas,
                            rationale, and a confidence score in [0, 1].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsphere.environment.business_env import EnvironmentState


@dataclass
class AgentProposal:
    """Structured output produced by a single agent for one simulation round.

    Attributes:
        agent_name: Identifier of the agent that produced this proposal.
        action: Short human-readable description of the recommended action.
        deltas: Mapping of environment metric names to proposed *relative*
                changes (e.g. ``{"revenue": 0.05}`` means +5 % revenue).
        confidence: Confidence score in the interval [0, 1].
        rationale: Free-text explanation of the proposal logic.
        priority: Urgency rank (1 = highest) used during conflict resolution.
    """

    agent_name: str
    action: str
    deltas: dict[str, float]
    confidence: float
    rationale: str
    priority: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


class BaseAgent(ABC):
    """Abstract agent that operates inside a ``BusinessEnvironment``.

    Sub-classes must implement :meth:`evaluate` and :meth:`propose`.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        """Initialise the agent.

        Args:
            name:   Human-readable identifier for the agent.
            weight: Negotiation weight used by the consensus engine (0–1).
        """
        self.name = name
        self.weight = weight
        self._history: list[AgentProposal] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def evaluate(self, state: "EnvironmentState") -> dict[str, float]:
        """Analyse the current environment state.

        Args:
            state: Snapshot of the current ``BusinessEnvironment``.

        Returns:
            A dictionary of scalar evaluation metrics relevant to this agent.
        """

    @abstractmethod
    def propose(self, state: "EnvironmentState") -> AgentProposal:
        """Generate an action proposal for the current state.

        Args:
            state: Snapshot of the current ``BusinessEnvironment``.

        Returns:
            An ``AgentProposal`` describing the recommended action.
        """

    # ── Concrete helpers ──────────────────────────────────────────────────────

    def act(self, state: "EnvironmentState") -> AgentProposal:
        """Evaluate the state and return a proposal (records history).

        Args:
            state: Current environment state.

        Returns:
            The ``AgentProposal`` for this round.
        """
        proposal = self.propose(state)
        self._history.append(proposal)
        return proposal

    @property
    def history(self) -> list[AgentProposal]:
        """All proposals made by this agent across rounds."""
        return list(self._history)

    def reset(self) -> None:
        """Clear proposal history (called at the start of a new simulation)."""
        self._history.clear()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"
