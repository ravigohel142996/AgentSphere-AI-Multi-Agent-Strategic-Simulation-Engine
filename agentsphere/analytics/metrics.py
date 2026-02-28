"""
MetricsEngine – derives KPIs, chart datasets, and analytics from simulation results.

All methods are pure functions operating on ``SimulationResult`` data; they
have no side-effects and produce serialisable outputs suitable for Streamlit
chart components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentsphere.simulation.simulator import SimulationResult


@dataclass
class ExecutiveKPIs:
    """Top-level KPIs displayed in the executive dashboard.

    All monetary values are USD.  Rates are fractional (e.g. 0.05 = 5 %).
    """

    revenue_final: float
    revenue_change_pct: float
    profit_final: float
    profit_margin: float
    roi: float
    risk_score_final: float
    risk_score_change: float
    churn_final: float
    growth_rate_final: float
    cost_ratio: float
    consensus_confidence: float   # average across all rounds
    total_rounds: int


class MetricsEngine:
    """Derives analytics artefacts from a completed ``SimulationResult``."""

    # ── Executive KPIs ────────────────────────────────────────────────────────

    @staticmethod
    def executive_kpis(result: SimulationResult) -> ExecutiveKPIs:
        """Compute top-level executive KPIs.

        Args:
            result: Completed simulation result.

        Returns:
            ``ExecutiveKPIs`` dataclass instance.
        """
        fs = result.final_state
        i_s = result.initial_state

        profit = fs.revenue - fs.cost
        profit_margin = profit / fs.revenue if fs.revenue > 0 else 0.0
        roi = profit / fs.cost if fs.cost > 0 else 0.0
        cost_ratio = fs.cost / fs.revenue if fs.revenue > 0 else 1.0
        revenue_change_pct = (
            (fs.revenue - i_s.revenue) / i_s.revenue if i_s.revenue > 0 else 0.0
        )
        risk_change = fs.risk_score - i_s.risk_score

        avg_confidence = (
            sum(r.consensus.confidence_index for r in result.rounds) / result.n_rounds
            if result.n_rounds > 0
            else 0.0
        )

        return ExecutiveKPIs(
            revenue_final=fs.revenue,
            revenue_change_pct=revenue_change_pct,
            profit_final=profit,
            profit_margin=profit_margin,
            roi=roi,
            risk_score_final=fs.risk_score,
            risk_score_change=risk_change,
            churn_final=fs.churn,
            growth_rate_final=fs.growth_rate,
            cost_ratio=cost_ratio,
            consensus_confidence=avg_confidence,
            total_rounds=result.n_rounds,
        )

    # ── Time-series data ──────────────────────────────────────────────────────

    @staticmethod
    def revenue_projection(result: SimulationResult) -> dict[str, list[Any]]:
        """Build a round-by-round revenue and profit projection dataset.

        Args:
            result: Completed simulation result.

        Returns:
            Dict with ``rounds``, ``revenue``, ``cost``, and ``profit`` lists.
        """
        rounds, revenue, cost, profit = [], [], [], []
        # Include the initial state as round 0
        i = result.initial_state
        rounds.append(0)
        revenue.append(i.revenue)
        cost.append(i.cost)
        profit.append(i.revenue - i.cost)

        for r in result.rounds:
            s = r.state_after
            rounds.append(r.round_number)
            revenue.append(s.revenue)
            cost.append(s.cost)
            profit.append(s.revenue - s.cost)

        return {"rounds": rounds, "revenue": revenue, "cost": cost, "profit": profit}

    @staticmethod
    def risk_timeline(result: SimulationResult) -> dict[str, list[Any]]:
        """Build a round-by-round risk and volatility timeline.

        Args:
            result: Completed simulation result.

        Returns:
            Dict with ``rounds``, ``risk_score``, and ``volatility`` lists.
        """
        rounds, risk, volatility = [], [], []
        i = result.initial_state
        rounds.append(0)
        risk.append(i.risk_score)
        volatility.append(i.volatility)

        for r in result.rounds:
            s = r.state_after
            rounds.append(r.round_number)
            risk.append(s.risk_score)
            volatility.append(s.volatility)

        return {"rounds": rounds, "risk_score": risk, "volatility": volatility}

    @staticmethod
    def agent_radar(result: SimulationResult) -> dict[str, Any]:
        """Compute per-agent alignment scores for a radar chart.

        Alignment score = average confidence × average vote share across rounds.

        Args:
            result: Completed simulation result.

        Returns:
            Dict with ``agents`` (list of names) and ``scores`` (list of floats).
        """
        agent_confidence: dict[str, list[float]] = {}
        agent_vote_share: dict[str, list[float]] = {}

        for rnd in result.rounds:
            for proposal in rnd.consensus.proposals:
                name = proposal.agent_name
                agent_confidence.setdefault(name, []).append(proposal.confidence)
            for name, vote in rnd.consensus.agent_votes.items():
                agent_vote_share.setdefault(name, []).append(vote)

        agents: list[str] = sorted(agent_confidence.keys())
        scores: list[float] = []
        for name in agents:
            avg_conf = (
                sum(agent_confidence[name]) / len(agent_confidence[name])
                if agent_confidence[name]
                else 0.0
            )
            vote_list = agent_vote_share.get(name, [])
            avg_vote = (
                sum(vote_list) / len(vote_list)
                if vote_list
                else 0.0
            )
            # Multiply by n_agents so that a perfectly aligned agent scores ~1.0
            n_agents = max(1, len(agents))
            scores.append(round(avg_conf * avg_vote * n_agents, 4))

        return {"agents": agents, "scores": scores}

    @staticmethod
    def roi_comparison(result: SimulationResult) -> dict[str, list[Any]]:
        """Compute per-round ROI for a bar/comparison chart.

        Args:
            result: Completed simulation result.

        Returns:
            Dict with ``rounds``, ``roi``, and ``growth_rate`` lists.
        """
        rounds, roi, growth = [], [], []
        i = result.initial_state
        initial_roi = (
            (i.revenue - i.cost) / i.cost if i.cost > 0 else 0.0
        )
        rounds.append(0)
        roi.append(initial_roi)
        growth.append(i.growth_rate)

        for r in result.rounds:
            s = r.state_after
            round_roi = (s.revenue - s.cost) / s.cost if s.cost > 0 else 0.0
            rounds.append(r.round_number)
            roi.append(round_roi)
            growth.append(s.growth_rate)

        return {"rounds": rounds, "roi": roi, "growth_rate": growth}

    @staticmethod
    def simulation_timeline(result: SimulationResult) -> list[dict[str, Any]]:
        """Build a structured timeline of key events per round.

        Args:
            result: Completed simulation result.

        Returns:
            List of dicts (one per round) with summary event data.
        """
        timeline: list[dict[str, Any]] = []
        for rnd in result.rounds:
            top_agent = max(rnd.proposals, key=lambda p: p.confidence)
            timeline.append(
                {
                    "round": rnd.round_number,
                    "top_agent": top_agent.agent_name,
                    "action": top_agent.action,
                    "confidence": top_agent.confidence,
                    "consensus_confidence": rnd.consensus.confidence_index,
                    "conflicts": len(rnd.consensus.conflicts),
                    "revenue": rnd.state_after.revenue,
                    "risk_score": rnd.state_after.risk_score,
                    "growth_rate": rnd.state_after.growth_rate,
                    "noise": rnd.noise,
                }
            )
        return timeline
