"""Risk engine: turns a Decision into an order size, or refuses to.

Everything that can stop the agent from doing something stupid lives here. The default
posture is a M$10 nibble. Large size requires every conviction gate to pass at once:
a big liquid market, a big edge, high stated confidence, and a near resolution date.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .config import RiskConfig
from .models import Decision, Market, Position, Sizing, now_ms

log = logging.getLogger(__name__)

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Fraction of the theoretical edge we are willing to give up to the limit price.
EDGE_GIVEBACK = 0.4
MAX_LIMIT_ORDER_HOURS = 24


def _no(reason: str) -> Sizing:
    return Sizing(amount=0.0, outcome="", limit_prob=None, expires_ms=None, reason=reason)


class RiskEngine:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def size(
        self,
        *,
        decision: Decision,
        market: Market,
        position: Position | None,
        net_worth: float,
        balance: float,
        budget_spent: float,
        open_position_count: int,
    ) -> Sizing:
        cfg = self.cfg

        if not market.is_binary:
            return _no("not a cpmm binary market")

        p = decision.probability
        q = market.probability

        # The side follows from the forecast, not from the model's own trade call.
        # Letting the model choose the action meant it answered "hold" on every small
        # edge and the risk engine never got to weigh in. The model forecasts; this
        # decides whether the forecast is worth a position and how big.
        outcome = "YES" if p >= q else "NO"
        edge = abs(p - q)
        if edge < cfg.min_edge:
            return _no(f"edge {edge:.3f} below minimum {cfg.min_edge}")

        # Kelly fraction of bankroll for a binary contract bought at the market price.
        price = q if outcome == "YES" else (1.0 - q)
        if price <= 0.01 or price >= 0.99:
            return _no("price is pinned at an extreme")
        kelly = edge / (1.0 - price)
        target = cfg.kelly_fraction * kelly * net_worth

        # Far-dated markets lock up mana, so shrink with time to resolution.
        days = max(0.5, market.days_to_close)
        decay = min(1.0, cfg.time_decay_days / days)
        target *= decay

        conviction = self._is_conviction(decision, market, edge)
        # An ordinary trade is capped as a share of the bankroll, not at a flat M$10.
        # A fixed cap meant every position stayed a rounding error as the account grew,
        # and Kelly never got to express the difference between a 5pp and a 15pp edge.
        cap = (
            cfg.conviction_max_fraction * net_worth
            if conviction
            else max(cfg.default_max_bet, cfg.default_max_fraction * net_worth)
        )

        if position is None and open_position_count >= cfg.max_open_positions:
            return _no(f"already holding {open_position_count} positions")

        # Soft limits express appetite and can be overridden by the min_bet floor.
        # Hard limits protect the bankroll and the market, and never are.
        soft: list[tuple[str, float]] = [
            ("kelly target", target),
            ("conviction cap" if conviction else "default cap", cap),
        ]
        hard: list[tuple[str, float]] = [
            ("share of volume", cfg.max_share_of_volume * market.volume),
            ("daily budget", cfg.daily_mana_budget - budget_spent),
            ("balance reserve", balance - cfg.min_balance_reserve),
        ]

        soft_name, soft_cap = min(soft, key=lambda item: item[1])
        hard_name, hard_cap = min(hard, key=lambda item: item[1])

        binding, amount = (
            (soft_name, soft_cap) if soft_cap <= hard_cap else (hard_name, hard_cap)
        )
        if amount < cfg.min_bet:
            # A weak edge should still take a real position, just a small one. Rounding
            # it to zero is what left the agent flat for a whole day.
            amount = min(cfg.min_bet, hard_cap)
            binding = "min bet floor" if amount >= cfg.min_bet else hard_name

        amount = float(int(amount))  # whole mana only
        if amount < 1:
            return _no(f"size floored to zero by {binding}")

        reason = (
            f"edge {edge:.3f}, kelly {kelly:.2f}, decay {decay:.2f}, "
            f"bound by {binding}"
        )

        limit_prob, expires = self._limit_terms(
            outcome=outcome, p=p, q=q, market=market, conviction=conviction
        )
        return Sizing(
            amount=amount,
            outcome=outcome,
            limit_prob=limit_prob,
            expires_ms=expires,
            reason=reason,
            conviction=conviction,
        )

    def _is_conviction(self, decision: Decision, market: Market, edge: float) -> bool:
        cfg = self.cfg
        needed = CONFIDENCE_RANK.get(cfg.conviction_min_confidence, 2)
        got = CONFIDENCE_RANK.get(decision.confidence, 0)
        return (
            market.volume >= cfg.conviction_min_volume
            and market.days_to_close <= cfg.conviction_max_days
            and edge >= cfg.conviction_min_edge
            and got >= needed
        )

    def _limit_terms(
        self, *, outcome: str, p: float, q: float, market: Market, conviction: bool
    ) -> tuple[float | None, int | None]:
        """Conviction trades cross the spread. Everything else rests as a limit order.

        The limit price is set so a fill still leaves us most of the estimated edge,
        which means a market that has already moved past us simply does not fill.
        """
        if conviction:
            return None, None

        if outcome == "YES":
            limit = q + EDGE_GIVEBACK * (p - q)
        else:
            limit = q - EDGE_GIVEBACK * (q - p)
        limit = min(0.99, max(0.01, limit))

        horizon = now_ms() + MAX_LIMIT_ORDER_HOURS * 3_600_000
        if market.close_time:
            horizon = min(horizon, market.close_time - 3_600_000)
        return round(limit, 2), int(horizon)

    async def fit_to_impact(
        self,
        sizing: Sizing,
        *,
        market: Market,
        probe: Callable[[float], Awaitable[float | None]],
    ) -> Sizing:
        """Shrink an order until its price impact is acceptable.

        Uses Manifold's dryRun bet to measure impact exactly rather than modelling the
        AMM ourselves. Manifold's markets are thin, so a large order can move a price
        several points on its own and then we are just trading against ourselves.
        """
        if sizing.limit_prob is not None:
            # Resting orders cannot move the price past their own limit.
            return sizing

        amount = sizing.amount
        for _ in range(6):
            after = await probe(amount)
            if after is None:
                return sizing
            impact = abs(after - market.probability)
            if impact <= self.cfg.max_market_impact:
                if amount != sizing.amount:
                    log.info(
                        "Trimmed order on %s from M$%.0f to M$%.0f to cap impact at %.1fpp",
                        market.slug, sizing.amount, amount, self.cfg.max_market_impact * 100,
                    )
                    sizing.reason += f", trimmed from M${sizing.amount:.0f} for impact"
                sizing.amount = amount
                return sizing
            amount = float(int(amount * 0.6))
            if amount < 1:
                return _no("any executable size would move the market too far")

        sizing.amount = amount
        return sizing
