"""Risk engine: turns a Decision into an order size, or refuses to.

Everything that can stop the agent from doing something stupid lives here. The default
posture is a M$10 nibble. Large size requires every conviction gate to pass at once:
a big liquid market, a big edge, high stated confidence, and a near resolution date.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from .config import RiskConfig
from .models import (
    Decision, Market, Position, Sizing, backs_favourite, logodds_gap, now_ms,
)

log = logging.getLogger(__name__)

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Fraction of the theoretical edge we are willing to give up to the limit price.
EDGE_GIVEBACK = 0.4
# Conviction trades are the ones worth getting filled, so they reach further.
CONVICTION_GIVEBACK = 0.65
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

        if not market.tradeable:
            return _no("not a binary market or a multiple-choice answer")

        p = decision.probability
        q = market.probability

        # The side follows from the forecast, not from the model's own trade call.
        # Letting the model choose the action meant it answered "hold" on every small
        # edge and the risk engine never got to weigh in. The model forecasts; this
        # decides whether the forecast is worth a position and how big.
        outcome = "YES" if p >= q else "NO"
        edge = abs(p - q)
        if cfg.favourites_only and not backs_favourite(p, q):
            return _no(f"would be a longshot bet against the {q:.0%} favourite")
        gap = logodds_gap(p, q)
        if gap < cfg.min_edge_logodds:
            return _no(f"edge {gap:.2f} log-odds below minimum {cfg.min_edge_logodds}")

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
        # No share-of-volume cap: every order is a limit order now, and a limit
        # cannot move the price past itself, which is the thing that cap was for.
        hard: list[tuple[str, float]] = [
            ("daily budget", cfg.daily_mana_budget - budget_spent),
            ("balance reserve", balance - cfg.min_balance_reserve),
        ]

        # Whatever is already riding on this side counts against the new order. Sizing
        # used to ignore the existing position, so every re-analysis of a held market
        # bought another full-size chunk on top of the last one.
        if position is not None and position.side == outcome:
            already = max(position.invested, position.value)
            soft = [(name, value - already) for name, value in soft]
            if min(value for _, value in soft) < cfg.min_bet:
                return _no(f"already holding M${already:.0f} {outcome}, which is the full size")

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
        """Every order is a limit order.

        A limit order fills immediately against the market up to its limit price and
        rests the remainder, so its price impact is capped by construction. That is
        what lets it take a real position in a thin market, where a market order of
        the same size would mostly be paying itself. Conviction trades used to cross
        the spread as market orders, and the impact trimmer then cut several from
        M$100-200 down to a fraction; now they simply accept a worse limit, giving up
        more of the edge in exchange for filling more of the order.
        """
        giveback = CONVICTION_GIVEBACK if conviction else EDGE_GIVEBACK
        if outcome == "YES":
            limit = q + giveback * (p - q)
        else:
            limit = q - giveback * (q - p)
        limit = min(0.99, max(0.01, limit))

        horizon = now_ms() + MAX_LIMIT_ORDER_HOURS * 3_600_000
        if market.close_time:
            horizon = min(horizon, market.close_time - 3_600_000)
        return round(limit, 2), int(horizon)

