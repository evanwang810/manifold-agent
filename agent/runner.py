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

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .brain import Brain, Budgets, CallBudget
from .config import Config
from .inbox import Inbox
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
    model: str = ""
    balance: float = 0.0
    net_worth: float = 0.0
    positions: int = 0
    evaluated: list[str] = field(default_factory=list)
    screened: list[str] = field(default_factory=list)
    bets: int = 0
    replies: int = 0
    llm_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"@{self.username}  via {self.model}",
            f"balance M${self.balance:,.0f}  "
            f"net worth M${self.net_worth:,.0f}  positions {self.positions}",
            f"screened {len(self.screened)}  evaluated {len(self.evaluated)}  "
            f"bets {self.bets}  replies {self.replies}  llm calls {self.llm_calls}",
        ]
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = ManifoldClient(cfg.manifold)
        # One client per distinct tier config, so the common case of two tiers naming
        # the same model does not open two connection pools for no reason.
        clients: dict[tuple, Any] = {}

        def client_for(tier_cfg):
            key = (tier_cfg.provider, tier_cfg.model, tuple(tier_cfg.fallbacks),
                   tier_cfg.base_url, tier_cfg.key_env)
            if key not in clients:
                clients[key] = build_llm(tier_cfg)
            return clients[key]

        self.fast = client_for(cfg.llm.fast)
        self.chat = client_for(cfg.llm.chat)
        self.deep = client_for(cfg.llm.deep)
        self._clients = list(clients.values())

        self.memory = Memory(cfg.memory, cfg.state_dir, self.chat)
        b = cfg.budget
        self.budget = Budgets(
            fast=self._budget("fast", b.max_fast_calls_per_tick, b.max_fast_calls_per_day),
            chat=self._budget("chat", b.max_chat_calls_per_tick, b.max_chat_calls_per_day),
            deep=self._budget("deep", b.max_deep_calls_per_tick, b.max_deep_calls_per_day),
        )
        self.risk = RiskEngine(cfg.risk)
        self.scanner = Scanner(cfg.scan, self.client, self.memory)
        self.report = TickReport()

        self.user_id = ""
        self.brain: Brain | None = None
        self.social: Social | None = None
        self._evaluations = 0

    def _budget(self, tier: str, per_tick: int, per_day: int) -> CallBudget:
        now = time.gmtime()
        elapsed = (now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec) / 86400
        return CallBudget(
            per_tick,
            daily_limit=per_day,
            used_today=self.memory.llm_used_today(tier),
            day_fraction=elapsed,
            pace_burst=self.cfg.budget.pace_burst,
            on_take=lambda: self.memory.record_llm_call(tier),
        )

    async def aclose(self) -> None:
        await self.client.aclose()
        for client in self._clients:
            await client.aclose()

    async def tick(self) -> TickReport:
        me = await self.client.me()
        self.user_id = me["id"]
        username = me.get("username", "?")

        self.brain = Brain(
            self.cfg, client=self.client, fast=self.fast, deep=self.deep,
            memory=self.memory, risk=self.risk, budget=self.budget,
            user_id=self.user_id,
        )
        self.social = Social(
            self.cfg, client=self.client, llm=self.chat, memory=self.memory,
            budget=self.budget, user_id=self.user_id, username=username,
        )

        portfolio = await self.client.portfolio(self.user_id)
        positions = await self.client.positions(self.user_id)
        self.report.username = username
        # The headline model is the one that actually decides trades, and it is the
        # active one rather than the configured one so a quota fallback shows up.
        self.report.model = f"{self.cfg.llm.deep.provider}/{self.deep.active_model}"
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

        self._observe_portfolio(positions)
        await self._check_fills()
        await self._check_moves(positions)
        self.report.replies = await self.social.run()
        self.report.replies += await Inbox(
            self.cfg, llm=self.chat, memory=self.memory, budget=self.budget,
            portfolio_line=(
                f"balance M${self.report.balance:,.0f}, net worth "
                f"M${self.report.net_worth:,.0f}, {len(positions)} open positions"
            ),
        ).run()
        await self._scan()

        if not self.budget.chat.spent:
            await self.memory.maybe_compress()

        self.report.llm_calls = self.budget.used
        await self._write_snapshot()
        # Daily call counts are only held in memory during the tick, so the tick has to
        # write them back or a restart would hand the quota straight back.
        self.memory.save()
        return self.report

    async def _write_snapshot(self) -> None:
        """Dump a public view of the agent for the showcase site to fetch.

        The site is static and reads this straight off the state branch, so everything
        it needs has to be in one file. Keep it small: it is fetched on every page load.
        """
        try:
            positions = await self.client.positions(self.user_id)
        except ManifoldError:
            positions = []

        kinds = ("bet", "no_trade", "sell")
        recent = [e for e in self.memory.recent_events(300) if e.get("kind") in kinds]

        snapshot = {
            "generated_ms": int(time.time() * 1000),
            "username": self.report.username,
            "profile_url": f"https://manifold.markets/{self.report.username}",
            "model": self.report.model,
            "models": {
                name: {
                    "configured": tier.model,
                    "active": client.active_model,
                    "fallbacks": list(tier.fallbacks),
                }
                for name, tier, client in (
                    ("fast", self.cfg.llm.fast, self.fast),
                    ("chat", self.cfg.llm.chat, self.chat),
                    ("deep", self.cfg.llm.deep, self.deep),
                )
            },
            "usage": {
                tier: {
                    "today": self.memory.llm_used_today(tier),
                    "cap": cap,
                }
                for tier, cap in (
                    ("fast", self.cfg.budget.max_fast_calls_per_day),
                    ("chat", self.cfg.budget.max_chat_calls_per_day),
                    ("deep", self.cfg.budget.max_deep_calls_per_day),
                )
            },
            "dry_run": self.cfg.manifold.dry_run,
            "balance": round(self.report.balance, 2),
            "net_worth": round(self.report.net_worth, 2),
            "summary": self.memory.state.get("summary", ""),
            "journal": [
                {"ts": e.get("ts"), "kind": e.get("kind"), "text": e.get("text", "")}
                for e in self.memory.recent_journal(40)[::-1]
            ],
            "lessons": [
                {
                    "text": le.get("text", ""),
                    "source": le.get("source", ""),
                    "ts": le.get("ts"),
                }
                for le in self.memory.state.get("lessons", [])
            ],
            "conversations": [
                {
                    "channel": convo.get("channel", ""),
                    "who": convo.get("who", ""),
                    "title": convo.get("title", ""),
                    "url": convo.get("url", ""),
                    "updated_ms": convo.get("updated_ms"),
                    "messages": [
                        {"role": m.get("role"), "text": (m.get("text") or "")[:400]}
                        for m in (convo.get("messages") or [])[-6:]
                    ],
                }
                for convo in self.memory.recent_conversations(6)
            ],
            "notes": [
                {
                    "note": n.get("note", ""),
                    "question": n.get("question", ""),
                    "url": n.get("url", ""),
                    "ts": n.get("ts"),
                }
                for n in self.memory.recent_notes(12)
            ],
            "positions": sorted(
                (
                    {
                        "question": p.question,
                        "url": f"https://manifold.markets/{self.report.username}/{p.slug}"
                        if p.slug else "",
                        "side": p.side,
                        "shares": round(p.shares, 1),
                        "invested": round(p.invested, 1),
                        "profit": round(p.profit, 1),
                        "prob": round(p.last_prob, 3),
                        "days_to_close": None
                        if p.days_to_close == float("inf")
                        else round(p.days_to_close, 1),
                    }
                    for p in positions
                ),
                key=lambda row: -abs(row["invested"]),
            )[:20],
            "decisions": [
                {
                    "ts": e.get("ts"),
                    "kind": e.get("kind"),
                    "question": e.get("question"),
                    "url": e.get("url"),
                    "market_prob": e.get("market_prob"),
                    "model_prob": e.get("model_prob"),
                    "confidence": e.get("confidence"),
                    "amount": e.get("amount"),
                    "outcome": e.get("outcome"),
                    "thinking": (e.get("thinking") or "")[:900],
                    "uncertainty": (e.get("uncertainty") or "")[:300],
                    "resolution_risk": (e.get("resolution_risk") or "")[:300],
                    "evidence_for": (e.get("evidence_for") or [])[:4],
                    "evidence_against": (e.get("evidence_against") or [])[:4],
                    "reason": e.get("reason"),
                }
                for e in recent[-15:][::-1]
            ],
        }
        path = self.cfg.state_dir / "public.json"
        path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    # -- observation ------------------------------------------------------

    def _observe_portfolio(self, positions: list[Any]) -> None:
        """Write down what changed since the last tick. No LLM, so it is free.

        Most ticks have nothing to decide, and those ticks used to leave no trace at
        all. The agent could not tell you a position had been bleeding for an hour
        because nothing had written it down. Thresholds are here so a market that ticks
        one point back and forth does not fill the journal with nothing.
        """
        mark = self.memory.state["portfolio_mark"]
        old_net = float(mark.get("net_worth") or 0.0)
        if old_net and abs(self.report.net_worth - old_net) >= max(5.0, old_net * 0.02):
            direction = "up" if self.report.net_worth > old_net else "down"
            self.memory.observe(
                "portfolio",
                f"Net worth {direction} from M${old_net:,.0f} to "
                f"M${self.report.net_worth:,.0f}, balance M${self.report.balance:,.0f}.",
            )
        mark["net_worth"] = self.report.net_worth
        mark["balance"] = self.report.balance

        seen = self.memory.state["position_mark"]
        live = {p.contract_id for p in positions}

        for position in positions:
            before = seen.get(position.contract_id)
            if before is None:
                self.memory.observe(
                    "position",
                    f"Opened {position.shares:.0f} {position.side} on "
                    f"\"{position.question[:80]}\" at {position.last_prob:.0%}, "
                    f"M${position.invested:.0f} in.",
                )
            else:
                moved = position.last_prob - float(before.get("prob") or 0.0)
                pnl = position.profit - float(before.get("profit") or 0.0)
                if abs(moved) >= 0.03:
                    self.memory.observe(
                        "move",
                        f"\"{position.question[:70]}\" moved {moved:+.0%} to "
                        f"{position.last_prob:.0%}. Holding {position.side}, "
                        f"P/L M${position.profit:+.0f}.",
                    )
                elif abs(pnl) >= max(10.0, abs(position.invested) * 0.15):
                    self.memory.observe(
                        "pnl",
                        f"\"{position.question[:70]}\" P/L moved M${pnl:+.0f} to "
                        f"M${position.profit:+.0f} without much price change.",
                    )
            seen[position.contract_id] = {
                "prob": position.last_prob,
                "profit": position.profit,
                "shares": position.shares,
                "side": position.side,
                "question": position.question[:120],
            }

        # A position that vanished either resolved or was sold. Either way it is the
        # single most informative thing that happens to a forecaster, so say so.
        for contract_id in [k for k in seen if k not in live]:
            gone = seen.pop(contract_id)
            self.memory.observe(
                "closed",
                f"No longer holding \"{gone.get('question', contract_id)[:70]}\". "
                f"Last seen {gone.get('side')} at {float(gone.get('prob') or 0):.0%}, "
                f"P/L M${float(gone.get('profit') or 0):+.0f}. Resolved or sold.",
            )

    # -- stages -----------------------------------------------------------

    @property
    def _can_evaluate(self) -> bool:
        return (
            self._evaluations < self.cfg.budget.max_evaluations_per_tick
            and not self.budget.deep.spent
        )

    async def _evaluate(self, market: Market, trigger: str, *, screen: bool = False) -> None:
        assert self.brain is not None
        self._evaluations += 1
        try:
            result = await self.brain.evaluate(market, trigger=trigger, screen=screen)
        except ManifoldError as exc:
            log.error("Evaluation failed on %s: %s", market.slug, exc)
            self.report.notes.append(f"error on {market.slug}: {exc}")
            return
        if result.screened_out:
            self._evaluations -= 1
            self.report.screened.append(market.slug)
        else:
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
            price = "" if self.cfg.forecast.blind else f" at {info['limit_prob']:.2f}"
            await self._evaluate(
                market,
                f"Your resting limit order for M${info['amount']:.0f} "
                f"{info['outcome']}{price} just filled. Someone traded against you. "
                "Re-forecast the question from scratch and see whether you still "
                "believe what you believed when you placed it.",
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
        # positions() already carries each contract's current probability, so there is
        # no second price call to make here.
        baseline = self.memory.state.setdefault("watch_probs", {})

        moved = []
        for position in positions:
            current = position.last_prob
            previous = baseline.get(position.contract_id)
            baseline[position.contract_id] = current
            if previous is None:
                continue
            move = current - previous
            if abs(move) >= self.cfg.watch.move_threshold:
                moved.append((abs(move), move, previous, current, position))
        self.memory.save()

        # Biggest mover first, since the tick may only afford one evaluation.
        cooldown_hours = self.cfg.watch.reevaluate_cooldown_minutes / 60
        for _, move, previous, current, position in sorted(moved, key=lambda m: -m[0]):
            if self.memory.seen_within(position.contract_id, cooldown_hours):
                continue
            if not self._can_evaluate:
                self.report.notes.append(
                    f"{position.slug} moved {move:+.0%} but the tick was out of budget"
                )
                continue
            market = await self.client.market(position.contract_id)
            detail = (
                "The market has moved notably since you last looked. The size and "
                "direction are withheld so they cannot anchor you."
                if self.cfg.forecast.blind
                else f"The price moved {move:+.0%} since you last looked "
                     f"(from {previous:.0%} to {current:.0%})."
            )
            await self._evaluate(
                market,
                f"{detail} You hold {position.shares:.0f} {position.side} shares. "
                "Re-forecast from scratch and decide whether the move reflects "
                "information you missed.",
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

        # More candidates than the tick can afford to analyse: the screen throws most
        # of them away for one cheap call each, and only survivors spend a deep call.
        candidates = await self.scanner.find_candidates(
            limit=self.cfg.scan.candidates_per_scan
            if self.cfg.screen.enabled
            else self.cfg.budget.max_evaluations_per_tick
        )
        self.memory.mark_scanned()
        if not candidates:
            self.report.notes.append("scan found nothing that passed the filters")
            return

        for market in candidates:
            if not self._can_evaluate or self.budget.fast.spent:
                break
            await self._evaluate(
                market,
                "Routine scan. This market passed the size and timing filters and has "
                "not been looked at recently.",
                screen=True,
            )


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())
