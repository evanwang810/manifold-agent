"""Finds markets worth spending an LLM call on.

Every filter here is cheap and runs before any model is invoked, because the scan is
the only thing standing between a free API tier and a rate limit.
"""

from __future__ import annotations

import logging

from .config import ScanConfig
from .manifold import ManifoldClient
from .memory import Memory
from .models import Market

log = logging.getLogger(__name__)


class Scanner:
    def __init__(self, cfg: ScanConfig, client: ManifoldClient, memory: Memory) -> None:
        self.cfg = cfg
        self.client = client
        self.memory = memory

    def qualifies(self, market: Market) -> tuple[bool, str]:
        cfg = self.cfg
        if not market.is_binary:
            return False, "not a binary YES/NO market"
        if market.is_resolved:
            return False, "resolved"
        if market.unique_bettors < cfg.min_unique_bettors:
            return False, f"{market.unique_bettors} traders < {cfg.min_unique_bettors}"
        if market.volume < cfg.min_volume:
            return False, f"volume M${market.volume:.0f} < M${cfg.min_volume:.0f}"

        days = market.days_to_close
        if days > cfg.max_days_to_resolve:
            return False, f"closes in {days:.0f}d > {cfg.max_days_to_resolve}d"
        if days * 24 < cfg.min_hours_to_resolve:
            return False, f"closes in {days * 24:.1f}h, too soon to matter"

        # A price already at the edge has almost no room to be wrong in our favor.
        if market.probability <= 0.03 or market.probability >= 0.97:
            return False, "price pinned at an extreme"
        return True, "ok"

    async def find_candidates(self, limit: int = 5) -> list[Market]:
        """Pull soonest-closing open binary markets and filter them down.

        Sorting by close date means the API is already doing most of the
        "resolves within a month" work for us.
        """
        seen_ids: set[str] = set()
        candidates: list[Market] = []
        rejects: dict[str, int] = {}

        for offset in (0, 100, 200):
            batch = await self.client.search_markets(
                sort="close-date", filter="open", contract_type="BINARY",
                limit=100, offset=offset,
            )
            if not batch:
                break

            for market in batch:
                if market.id in seen_ids:
                    continue
                seen_ids.add(market.id)

                ok, why = self.qualifies(market)
                if not ok:
                    rejects[why.split(",")[0]] = rejects.get(why.split(",")[0], 0) + 1
                    continue
                if self.memory.seen_within(market.id, self.cfg.skip_if_seen_within_hours):
                    continue
                candidates.append(market)

            # Once the batch runs past our horizon there is nothing left to find.
            if batch and batch[-1].days_to_close > self.cfg.max_days_to_resolve:
                break

        # Prefer more liquid markets, they tolerate real size and are less noisy.
        candidates.sort(key=lambda m: m.volume, reverse=True)
        log.info(
            "Scan: %d candidates from %d markets (top rejections: %s)",
            len(candidates), len(seen_ids),
            ", ".join(f"{k} x{v}" for k, v in sorted(
                rejects.items(), key=lambda kv: -kv[1])[:3]) or "none",
        )
        return candidates[:limit]
