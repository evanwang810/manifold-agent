"""One tick.

The whole agent is a single pass with no background loops, so it can be driven by a
cron job that has no memory of the last run. Priority is fixed and descending: react to
things that already happened before going looking for new ones.

  1. limit orders that filled
  2. held markets whose price moved
  3. people who replied to the bot
  4. new markets worth a look
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .brain import Brain, CallBudget
from .config import Config
from .llm import build_llm
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Market
from .scanner import Scanner
from .sizing import RiskEngine
from .social import Social

log = logging.getLogger(__name__)


@dataclass
class TickReport:
    username: str = ""
    balance: float = 0.0
    net_worth: float = 0.0
    positions: int = 0
    evaluated: list[str] = field(default_factory=list)
    bets: int = 0
    replies: int = 0
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"@{self.username}  balance M${self.balance:,.0f}  "
            f"net worth M${self.net_worth:,.0f}  positions {self.positions}",
            f"evaluated {len(self.evaluated)}  bets {self.bets}  "
            f"replies {self.replies}  llm calls {self.llm_calls}",
        ]
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = ManifoldClient(cfg.manifold)
        self.llm = build_llm(cfg.llm)
        self.memory = Memory(cfg.memory, cfg.state_dir, self.llm)
        self.budget = CallBudget(cfg.budget.max_llm_calls_per_tick)
        self.risk = RiskEngine(cfg.risk)
        self.scanner = Scanner(cfg.scan, self.client, self.memory)
        self.report = TickReport()

        self.user_id = ""
        self.brain: Brain | None = None
        self.social: Social | None = None
        self._evaluations = 0

    async def aclose(self) -> None:
        await self.client.aclose()
        await self.llm.aclose()

    async def tick(self) -> TickReport:
        me = await self.client.me()
        self.user_id = me["id"]
        username = me.get("username", "?")

        self.brain = Brain(
            self.cfg, client=self.client, llm=self.llm, memory=self.memory,
            risk=self.risk, budget=self.budget, user_id=self.user_id,
        )
        self.social = Social(
            self.cfg, client=self.client, llm=self.llm, memory=self.memory,
            budget=self.budget, user_id=self.user_id, username=username,
        )

        portfolio = await self.client.portfolio(self.user_id)
        positions = await self.client.positions(self.user_id)
        self.report.username = username
        self.report.balance = float(portfolio.get("balance") or 0.0)
        self.report.net_worth = (
            self.report.balance
            + float(portfolio.get("investmentValue") or 0.0)
            - float(portfolio.get("loanTotal") or 0.0)
        )
        self.report.positions = len(positions)

        if self.cfg.manifold.dry_run:
            self.report.notes.append("DRY RUN, nothing was sent")
        if self.cfg.one_off_order:
            self.report.notes.append(f"one-off order: {self.cfg.one_off_order[:120]}")

        await self._check_fills()
        await self._check_moves(positions)
        self.report.replies = await self.social.run()
        await self._scan()

        if not self.budget.spent:
            await self.memory.maybe_compress()

        self.report.llm_calls = self.budget.used
        return self.report

    # -- stages -----------------------------------------------------------

    @property
    def _can_evaluate(self) -> bool:
        return (
            self._evaluations < self.cfg.budget.max_evaluations_per_tick
            and not self.budget.spent
        )

    async def _evaluate(self, market: Market, trigger: str) -> None:
        assert self.brain is not None
        self._evaluations += 1
        try:
            result = await self.brain.evaluate(market, trigger=trigger)
        except ManifoldError as exc:
            log.error("Evaluation failed on %s: %s", market.slug, exc)
            self.report.notes.append(f"error on {market.slug}: {exc}")
            return
        self.report.evaluated.append(market.slug)
        if result.executed:
            self.report.bets += 1
        self.report.notes.append(f"{market.slug}: {result.detail}")

    async def _check_fills(self) -> None:
        """A resting order that vanished either filled or expired. Both are news."""
        tracked = dict(self.memory.tracked_orders)
        if not tracked:
            return
        open_ids = {b["id"] for b in await self.client.open_limit_orders(self.user_id)}

        for bet_id, info in tracked.items():
            if bet_id in open_ids:
                continue
            record = self.memory.untrack_order(bet_id)
            if record is None:
                continue

            filled = await self._was_filled(bet_id, info["contract_id"])
            self.memory.log_event("order_closed", bet_id=bet_id, filled=filled, **record)
            if not filled:
                self.report.notes.append(f"limit order on {info['slug']} expired unfilled")
                continue

            self.report.notes.append(f"limit order FILLED on {info['slug']}")
            if not self._can_evaluate:
                continue
            market = await self.client.market(info["contract_id"])
            await self._evaluate(
                market,
                f"Your resting limit order for M${info['amount']:.0f} {info['outcome']} at "
                f"{info['limit_prob']:.2f} just filled. Someone traded against you. "
                "Reconsider whether the thesis still holds at the current price.",
            )

    async def _was_filled(self, bet_id: str, contract_id: str) -> bool:
        bets = await self.client.my_bets(self.user_id, contract_id=contract_id, limit=200)
        for bet in bets:
            if bet.get("id") == bet_id:
                return bool(bet.get("isFilled")) or bool(bet.get("fills"))
        return False

    async def _check_moves(self, positions: list[Any]) -> None:
        if not positions:
            return
        probs = await self.client.market_probs([p.contract_id for p in positions])
        baseline = self.memory.state.setdefault("watch_probs", {})

        moved = []
        for position in positions:
            current = probs.get(position.contract_id, position.last_prob)
            previous = baseline.get(position.contract_id)
            baseline[position.contract_id] = current
            if previous is None:
                continue
            move = current - previous
            if abs(move) >= self.cfg.watch.move_threshold:
                moved.append((abs(move), move, previous, current, position))
        self.memory.save()

        # Biggest mover first, since the tick may only afford one evaluation.
        for _, move, previous, current, position in sorted(moved, key=lambda m: -m[0]):
            if not self._can_evaluate:
                self.report.notes.append(
                    f"{position.slug} moved {move:+.0%} but the tick was out of budget"
                )
                continue
            market = await self.client.market(position.contract_id)
            await self._evaluate(
                market,
                f"The price moved {move:+.0%} since you last looked (from {previous:.0%} "
                f"to {current:.0%}) and you hold {position.shares:.0f} {position.side} "
                "shares. Decide whether the move reflects information you missed.",
            )

    async def _scan(self) -> None:
        if not self._can_evaluate:
            return
        due = self.memory.minutes_since_scan()
        if due < self.cfg.scan.min_minutes_between_scans:
            self.report.notes.append(
                f"scan not due for another {self.cfg.scan.min_minutes_between_scans - due:.0f}m"
            )
            return

        candidates = await self.scanner.find_candidates(
            limit=self.cfg.budget.max_evaluations_per_tick
        )
        self.memory.mark_scanned()
        if not candidates:
            self.report.notes.append("scan found nothing that passed the filters")
            return

        for market in candidates:
            if not self._can_evaluate:
                break
            await self._evaluate(
                market,
                "Routine scan. This market passed the size and timing filters and has "
                "not been looked at recently.",
            )


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
