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
        if not (market.is_binary or market.is_multiple_choice):
            return False, "not a binary or multiple-choice market"
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
        # (A multiple-choice market has no single price; its answers are checked when
        # it is split.)
        if market.is_binary and (market.probability <= 0.03 or market.probability >= 0.97):
            return False, "price pinned at an extreme"
        return True, "ok"

    async def _answers(self, held: set[str], rejects: dict[str, int]) -> list[Market]:
        """Answers of open multiple-choice markets, as yes/no sub-markets.

        A market where any answer is already held is skipped whole, so the agent ends
        up with at most one position per multiple-choice question rather than several
        correlated ones.
        """
        try:
            batch = await self.client.search_markets(
                sort="close-date", filter="open", contract_type="MULTIPLE_CHOICE", limit=100,
            )
        except Exception as exc:  # noqa: BLE001 - a failed extra search is not fatal
            log.info("Multiple-choice search failed: %s", exc)
            return []
        out: list[Market] = []
        fetched = 0
        for market in batch:
            if market.id in held:
                continue
            ok, why = self.qualifies(market)
            if not ok:
                rejects[why.split(",")[0]] = rejects.get(why.split(",")[0], 0) + 1
                continue
            if fetched >= self.cfg.max_multiple_choice_markets:
                break
            fetched += 1
            full = await self.client.market(market.id)  # search results carry no answers
            for answer in full.split(self.cfg.max_answers_per_market):
                if self.memory.worth_another_look(
                    answer.contract_key, answer.probability,
                    min_hours=self.cfg.skip_if_seen_within_hours,
                    max_days=self.cfg.rescan_after_days, move=self.cfg.rescan_on_move,
                ):
                    out.append(answer)
        return out

    async def find_candidates(self, limit: int = 5, held: set[str] | None = None) -> list[Market]:
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
                # Held markets are watched by the price-move check and the desk. Letting
                # them back into the scan is how one favourite got bought eight times.
                if held and market.id in held:
                    continue
                if not self.memory.worth_another_look(
                    market.contract_key, market.probability,
                    min_hours=self.cfg.skip_if_seen_within_hours,
                    max_days=self.cfg.rescan_after_days, move=self.cfg.rescan_on_move,
                ):
                    continue
                candidates.append(market)

            # Once the batch runs past our horizon there is nothing left to find.
            if batch and batch[-1].days_to_close > self.cfg.max_days_to_resolve:
                break

        if self.cfg.multiple_choice:
            candidates += await self._answers(held or set(), rejects)

        # Prefer more liquid markets, they tolerate real size and are less noisy.
        candidates.sort(key=lambda m: m.volume, reverse=True)
        log.info(
            "Scan: %d candidates from %d markets (top rejections: %s)",
            len(candidates), len(seen_ids),
            ", ".join(f"{k} x{v}" for k, v in sorted(
                rejects.items(), key=lambda kv: -kv[1])[:3]) or "none",
        )
        return candidates[:limit]
