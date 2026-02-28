"""
NegotiationEngine – resolves competing agent proposals into a single consensus.

Algorithm:
1. Collect all ``AgentProposal`` objects from active agents.
2. Detect conflicts between proposal pairs (opposing delta directions on the
   same metrics).
3. Compute a weighted-average delta for each metric, weighting by agent weight
   × proposal confidence.
4. Produce a ``ConsensusResult`` that includes the final deltas, conflict
   report, and an overall confidence index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentsphere.agents.base_agent import AgentProposal
from agentsphere.config import CONFLICT_THRESHOLD


@dataclass
class ConflictReport:
    """Details of a detected negotiation conflict.

    Attributes:
        agent_a:          Name of the first conflicting agent.
        agent_b:          Name of the second conflicting agent.
        conflicting_keys: Metrics on which the two agents disagree.
        severity:         Fraction of shared metrics that conflict (0–1).
    """

    agent_a: str
    agent_b: str
    conflicting_keys: list[str]
    severity: float


@dataclass
class ConsensusResult:
    """Output of one negotiation round.

    Attributes:
        final_deltas:     Weighted-averaged delta for every metric.
        conflicts:        List of detected conflicts between agent pairs.
        confidence_index: Overall confidence of the consensus (0–1).
        agent_votes:      Per-agent effective weight × confidence contributions.
        summary:          Human-readable summary of the consensus decision.
    """

    final_deltas: dict[str, float]
    conflicts: list[ConflictReport]
    confidence_index: float
    agent_votes: dict[str, float]
    summary: str
    proposals: list[AgentProposal] = field(default_factory=list)


class NegotiationEngine:
    """Weighted-voting negotiation engine for multi-agent consensus.

    Args:
        agent_weights: Mapping of agent name → base negotiation weight.
    """

    def __init__(self, agent_weights: dict[str, float]) -> None:
        self.agent_weights = agent_weights

    # ── Public API ────────────────────────────────────────────────────────────

    def negotiate(self, proposals: list[AgentProposal]) -> ConsensusResult:
        """Run the full negotiation pipeline for a set of proposals.

        Steps:
        1. Detect conflicts between every pair of proposals.
        2. Penalise conflicting agents' effective weights.
        3. Compute weighted-average deltas.
        4. Build and return the ``ConsensusResult``.

        Args:
            proposals: One ``AgentProposal`` per active agent.

        Returns:
            A ``ConsensusResult`` encapsulating the consensus decision.
        """
        if not proposals:
            return ConsensusResult(
                final_deltas={},
                conflicts=[],
                confidence_index=0.0,
                agent_votes={},
                summary="No proposals received.",
            )

        conflicts = self._detect_conflicts(proposals)
        conflict_agents: set[str] = set()
        for c in conflicts:
            conflict_agents.add(c.agent_a)
            conflict_agents.add(c.agent_b)

        # Effective weight = base_weight × confidence × conflict_penalty
        effective_weights: dict[str, float] = {}
        for proposal in proposals:
            base = self.agent_weights.get(proposal.agent_name, 1.0)
            penalty = 0.80 if proposal.agent_name in conflict_agents else 1.0
            effective_weights[proposal.agent_name] = (
                base * proposal.confidence * penalty
            )

        total_weight = sum(effective_weights.values())
        if total_weight == 0:
            total_weight = 1.0

        # Weighted-average deltas
        final_deltas: dict[str, float] = {}
        all_keys: set[str] = set()
        for p in proposals:
            all_keys.update(p.deltas.keys())

        for key in all_keys:
            weighted_sum = 0.0
            key_weight_sum = 0.0
            for proposal in proposals:
                if key in proposal.deltas:
                    w = effective_weights[proposal.agent_name]
                    weighted_sum += proposal.deltas[key] * w
                    key_weight_sum += w
            if key_weight_sum > 0:
                final_deltas[key] = weighted_sum / key_weight_sum

        # Confidence index: weighted mean of individual confidences
        confidence_index = (
            sum(
                effective_weights[p.agent_name] * p.confidence
                for p in proposals
            )
            / total_weight
        )

        agent_votes = {
            p.agent_name: round(effective_weights[p.agent_name] / total_weight, 4)
            for p in proposals
        }

        summary = self._build_summary(proposals, conflicts, confidence_index)

        return ConsensusResult(
            final_deltas=final_deltas,
            conflicts=conflicts,
            confidence_index=round(confidence_index, 4),
            agent_votes=agent_votes,
            summary=summary,
            proposals=proposals,
        )

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_conflicts(proposals: list[AgentProposal]) -> list[ConflictReport]:
        """Identify conflicting proposal pairs.

        Two proposals conflict when their recommended delta directions (positive
        vs negative) differ on at least ``CONFLICT_THRESHOLD`` fraction of their
        shared metric keys.

        Args:
            proposals: All agent proposals for this round.

        Returns:
            List of ``ConflictReport`` objects (may be empty).
        """
        conflicts: list[ConflictReport] = []
        n = len(proposals)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = proposals[i], proposals[j]
                shared = set(a.deltas.keys()) & set(b.deltas.keys())
                if not shared:
                    continue
                conflicting = [
                    k
                    for k in shared
                    if a.deltas[k] * b.deltas[k] < 0  # opposite signs
                ]
                severity = len(conflicting) / len(shared)
                if severity >= CONFLICT_THRESHOLD:
                    conflicts.append(
                        ConflictReport(
                            agent_a=a.agent_name,
                            agent_b=b.agent_name,
                            conflicting_keys=conflicting,
                            severity=round(severity, 4),
                        )
                    )
        return conflicts

    @staticmethod
    def _build_summary(
        proposals: list[AgentProposal],
        conflicts: list[ConflictReport],
        confidence_index: float,
    ) -> str:
        """Build a human-readable summary of the negotiation round.

        Args:
            proposals:        All agent proposals.
            conflicts:        Detected conflicts.
            confidence_index: Final consensus confidence.

        Returns:
            Multi-line summary string.
        """
        lines: list[str] = [
            f"Consensus reached with confidence {confidence_index:.1%}.",
            f"{len(proposals)} agents participated; "
            f"{len(conflicts)} conflict(s) detected.",
        ]
        if conflicts:
            for c in conflicts:
                lines.append(
                    f"  Conflict [{c.agent_a} ↔ {c.agent_b}] on "
                    f"{', '.join(c.conflicting_keys)} "
                    f"(severity {c.severity:.0%})."
                )
        # Highlight the highest-confidence proposal
        top = max(proposals, key=lambda p: p.confidence)
        lines.append(
            f"Highest-confidence agent: {top.agent_name} "
            f"({top.confidence:.1%}) – {top.action}."
        )
        return "\n".join(lines)
