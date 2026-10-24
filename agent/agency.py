"""The desk: the agent's own turn, run by the deep model.

Everything else the agent does is a reaction: a scan finds a market, a price moves, an
order fills, somebody asks something. The desk is where it manages its own book. It
sees every position, the cash it has, the markets it forecast but could not afford, who
it owes, and its own track record, and it decides directly what to do with its money.

This used to be two cheap-model calls (an own turn and a book review) whose trades a
deep model then approved or vetoed. The cheap model spent its turns writing the same
"liquidity is frozen" note every hour and selling favourites the deep model had just
bought. Now the deep model decides in one call, and only when something has changed.
Code still enforces the hard limits after it speaks: the cash reserve, the daily
budget, the per-position cap, favourites only, and what is actually owed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .brain import Budgets
from .config import Config
from .llm import LLMClient, extract_json
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Decision, Market, Position, backs_favourite
from .prompts import DESK_SCHEMA, DESK_SYSTEM, build_desk_prompt, system_with_orders
from .sizing import RiskEngine

log = logging.getLogger(__name__)

MANAGRAM_MINIMUM = 10  # the API rejects anything smaller
# Journal entries that mean the book or the agent's situation actually changed. A turn
# with none of these since the last one, no cash to deploy and no long gap is skipped:
# that was the hourly "still frozen, no new information" loop.
CHANGES = {"position", "closed", "move", "pnl", "trade", "conversation", "nudge"}
# Reasons a forecast with an edge did not become a trade that cash could fix later.
UNFUNDED = ("balance reserve", "daily budget", "min bet floor")


@dataclass
class Action:
    action: str
    market: str = ""
    amount: float = 0.0
    recipient: str = ""
    text: str = ""
    reasoning: str = ""

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Action:
        try:
            amount = float(data.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0.0
        return cls(
            action=str(data.get("action", "")).strip().lower(),
            market=str(data.get("market", "") or data.get("market_id", "")).strip(),
            amount=max(0.0, amount),
            recipient=str(data.get("recipient", "")).strip().lstrip("@"),
            text=str(data.get("text", "")).strip(),
            reasoning=str(data.get("reasoning", "")).strip()[:900],
        )


class Desk:
    def __init__(
        self,
        cfg: Config,
        *,
        client: ManifoldClient,
        deep: LLMClient,
        memory: Memory,
        budget: Budgets,
        risk: RiskEngine,
        user_id: str,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.deep = deep
        self.memory = memory
        self.budget = budget
        self.risk = risk
        self.user_id = user_id
        # Usernames, lowercased, to the mana of theirs still held: the ceiling on what a
        # repayment may send, and the only thing allowed to dip into the reserve.
        self._owed: dict[str, float] = {}

    # -- the turn ------------------------------------------------------------

    async def run(self, positions: list[Position], balance: float, net_worth: float) -> str:
        cfg = self.cfg.agency
        if not cfg.enabled:
            return ""
        waited = self.memory.minutes_since_own_action()
        if waited < cfg.min_minutes_between_actions:
            return ""
        spendable = max(0.0, balance - self.cfg.risk.min_balance_reserve)
        if not self._worth_a_turn(waited, spendable):
            return ""
        if not self.budget.deep.take():
            return ""
        # Marked before anything is attempted, so a crash cannot retry every tick.
        self.memory.mark_own_action()

        watchlist = self._watchlist(positions)
        prompt = build_desk_prompt(
            today=time.strftime("%Y-%m-%d %H:%MZ", time.gmtime()),
            portfolio=self._portfolio_line(positions, balance, net_worth),
            positions=render_positions(positions),
            watchlist=_render_watchlist(watchlist),
            ledger=await self._ledger(),
            lessons=self.memory.lessons_block(),
            todos=self.memory.todos_block(),
            memory=self.memory.context_block(),
            last_turns=self._last_turns(3),
        )
        try:
            response = await self.deep.generate(
                prompt,
                system=system_with_orders(DESK_SYSTEM, self.cfg.owner_block()),
                json_schema=DESK_SCHEMA,
            )
            data = extract_json(response.text)
        except Exception as exc:  # noqa: BLE001 - a bad turn must not kill the tick
            log.warning("Desk turn failed: %s", exc)
            self.memory.log_event("desk", thinking="", actions=[], error=str(exc)[:300])
            return ""

        thinking = str(data.get("thinking", "")).strip()[:1500]
        raw = data.get("actions")
        actions = [Action.parse(a) for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []
        if thinking:
            self.memory.set_status(thinking)
            self.memory.observe("desk", f"Desk: {thinking}")

        results = []
        held = {p.key: p for p in positions}
        for action in actions[: cfg.max_actions_per_turn]:
            ok, detail = await self._do(action, held, watchlist, balance, net_worth, positions)
            # Cash moves within the turn, so a second buy sees what the first spent.
            if ok and action.action in ("buy", "add", "send_mana"):
                balance -= action.amount
            results.append({
                "action": action.action, "market": action.market, "amount": action.amount,
                "recipient": action.recipient, "text": action.text[:300],
                "reasoning": action.reasoning, "ok": ok, "result": detail,
            })

        # One event per turn with everything in it, which is what the site shows: the
        # whole of its thinking and every action with whether it happened and why not.
        self.memory.log_event("desk", thinking=thinking, actions=results)
        done = [r["result"] for r in results if r["ok"]]
        return "; ".join(done) if done else ("took a turn, no actions" if not results else "took a turn, nothing went through")

    def _worth_a_turn(self, waited_minutes: float, spendable: float) -> bool:
        if waited_minutes >= self.cfg.agency.idle_hours * 60:
            return True
        if spendable >= self.cfg.risk.min_bet:
            return True
        since = int(time.time() * 1000 - waited_minutes * 60_000)
        return any(
            e.get("kind") in CHANGES and int(e.get("ts", 0)) > since
            for e in self.memory.recent_journal(80)
        )

    # -- doing things --------------------------------------------------------

    async def _do(
        self,
        action: Action,
        held: dict[str, Position],
        watchlist: dict[str, dict[str, Any]],
        balance: float,
        net_worth: float,
        positions: list[Position],
    ) -> tuple[bool, str]:
        """Clamp, then execute. Returns (done, what happened or why it did not)."""
        risk = self.cfg.risk
        spendable = max(0.0, balance - risk.min_balance_reserve)
        cap = min(
            max(risk.default_max_bet, risk.default_max_fraction * net_worth),
            spendable,
            risk.daily_mana_budget - self.memory.budget_spent(),
        )
        dry = self.cfg.manifold.dry_run
        try:
            if action.action == "sell":
                p = held.get(action.market)
                if p is None:
                    return False, "not a market it holds"
                if not p.tradable:
                    return False, "market is closed, it can only be waited out"
                await self.client.sell_shares(
                    market_id=p.contract_id, outcome=p.side, answer_id=p.answer_id
                )
                self._record("sell", action, p.question, p.url, p.side, profit=p.profit)
                return True, f"sold {p.side} on \"{p.question[:60]}\" (P/L M${p.profit:+.0f})"

            if action.action == "add":
                p = held.get(action.market)
                if p is None:
                    return False, "not a market it holds"
                if not p.tradable:
                    return False, "market is closed"
                price = p.last_prob if p.side == "YES" else 1.0 - p.last_prob
                if risk.favourites_only and price < 0.5:
                    return False, f"{p.side} is at {price:.0%}, a longshot; favourites only"
                amount = float(int(min(action.amount or risk.min_bet, cap)))
                if amount < 1:
                    return False, "no spendable cash"
                await self.client.place_bet(
                    contract_id=p.contract_id, amount=amount, outcome=p.side,
                    answer_id=p.answer_id,
                )
                if not dry:
                    self.memory.record_spend(amount)
                action.amount = amount
                self._record("add", action, p.question, p.url, p.side)
                return True, f"added M${amount:.0f} {p.side} to \"{p.question[:60]}\""

            if action.action == "buy":
                return await self._buy(action, watchlist, cap, net_worth, balance, positions)

            if action.action == "send_mana":
                if not self.cfg.agency.allow_send_mana or not action.recipient:
                    return False, "no recipient"
                limit = spendable
                owed = self._owed.get(action.recipient.lower(), 0.0)
                if owed >= MANAGRAM_MINIMUM:
                    # Repaying someone may dip into the reserve, up to what they sent
                    # and have not had back. Nothing else can.
                    limit = max(limit, min(owed, balance))
                if action.amount < MANAGRAM_MINIMUM:
                    return False, f"under the M${MANAGRAM_MINIMUM} minimum"
                if action.amount > limit:
                    return False, f"only M${limit:.0f} available for that"
                user = await self.client.user_by_username(action.recipient)
                if not user or not user.get("id"):
                    return False, f"no user @{action.recipient}"
                await self.client.send_managram(
                    to_ids=[user["id"]], amount=action.amount,
                    message=action.text[:200] or "Thanks.",
                )
                if not dry:
                    self.memory.record_spend(action.amount)
                self._record("send_mana", action, "", "", "")
                return True, f"sent M${action.amount:.0f} to @{action.recipient}"

            if action.action == "note_add":
                ok = self.memory.add_lesson(action.text, source="self")
                return ok, "wrote a standing note" if ok else "note refused (status, duplicate or too short)"
            if action.action == "note_remove":
                ok = self.memory.remove_lesson(action.text)
                return ok, "retired a note" if ok else "no such note"
            if action.action == "todo_add":
                ok = self.memory.add_todo(action.text)
                return ok, "added a to-do" if ok else "to-do refused (duplicate or too short)"
            if action.action == "todo_done":
                ok = self.memory.complete_todo(action.text)
                return ok, "ticked off a to-do" if ok else "no such to-do"
            return False, f"unknown action {action.action!r}"
        except ManifoldError as exc:
            log.warning("Desk %s failed: %s", action.action, exc)
            return False, f"Manifold refused: {str(exc)[:160]}"

    async def _buy(
        self,
        action: Action,
        watchlist: dict[str, dict[str, Any]],
        cap: float,
        net_worth: float,
        balance: float,
        positions: list[Position],
    ) -> tuple[bool, str]:
        """Open a position on a market from the watchlist, at today's price.

        The forecast is the deep model's own from when it analysed the market. The
        risk engine re-checks it against the current price, so a market that has moved
        past the edge, or flipped favourite, is refused rather than bought stale.
        """
        w = watchlist.get(action.market)
        if w is None:
            return False, "only markets on the watchlist can be bought from the desk"
        slug, _, answer_id = w["slug"].partition("::")
        market = Market.parse(await self.client.market_by_slug(slug))
        if answer_id:
            try:
                market = market.for_answer(answer_id)
            except ValueError:
                return False, "that answer no longer exists"
        if market.is_resolved or market.days_to_close <= 0:
            return False, "market has closed"
        decision = Decision(
            probability=w["model_prob"], confidence=w.get("confidence") or "medium",
            action="buy", evidence_for=[], evidence_against=[], key_uncertainty="",
            resolution_risk="", comment="", memory_note="",
        )
        sizing = self.risk.size(
            decision=decision, market=market, position=None, net_worth=net_worth,
            balance=balance, budget_spent=self.memory.budget_spent(),
            open_position_count=len(positions),
        )
        if not sizing.is_trade:
            return False, f"no longer a trade at {market.probability:.0%}: {sizing.reason}"
        # The desk chooses the size, within the hard cap. Kelly's suggestion is the
        # default when it does not say.
        amount = float(int(min(action.amount or sizing.amount, cap)))
        if amount < 1:
            return False, "no spendable cash"
        result = await self.client.place_bet(
            contract_id=market.id, amount=amount, outcome=sizing.outcome,
            limit_prob=sizing.limit_prob, expires_at=sizing.expires_ms,
            answer_id=market.answer_id,
        )
        if not self.cfg.manifold.dry_run:
            self.memory.record_spend(amount)
        self.memory.log_event(
            "bet", source="desk", market=market.key, question=market.question,
            url=market.url, outcome=sizing.outcome, amount=amount,
            limit_prob=sizing.limit_prob, market_prob=market.probability,
            model_prob=w["model_prob"], confidence=w.get("confidence"),
            conviction=False, reason=f"desk buy; {sizing.reason}",
            bet_id=(result or {}).get("id"), dry_run=self.cfg.manifold.dry_run,
            thinking=action.reasoning, research=w.get("research", ""),
        )
        self.memory.observe(
            "trade",
            f"Desk bought M${amount:.0f} {sizing.outcome} on \"{market.question[:70]}\" "
            f"at {market.probability:.0%} YES, forecast {w['model_prob']:.0%}. "
            f"Because: {action.reasoning[:300]}",
        )
        return True, f"bought M${amount:.0f} {sizing.outcome} on \"{market.question[:60]}\""

    def _record(self, kind: str, action: Action, question: str, url: str, side: str,
                profit: float | None = None) -> None:
        self.memory.record_own_action(kind, f"{kind} {question[:60]}", action.reasoning)
        self.memory.observe("trade" if kind != "send_mana" else "conversation",
                            f"Desk {kind}: {question[:70] or action.recipient}. "
                            f"Because: {action.reasoning[:300]}")
        self.memory.log_event(
            "own_action", action=kind, approved=True, amount=action.amount,
            reasoning=action.reasoning, dry_run=self.cfg.manifold.dry_run,
            question=question, url=url, side=side, recipient=action.recipient,
            profit=None if profit is None else round(profit, 1),
            detail=f"{kind} {question[:80] or '@' + action.recipient}",
        )

    # -- what it sees -----------------------------------------------------------

    def _portfolio_line(self, positions: list[Position], balance: float, net_worth: float) -> str:
        risk = self.cfg.risk
        spendable = max(0.0, balance - risk.min_balance_reserve)
        stuck = sum(1 for p in positions if not p.tradable)
        line = (
            f"Cash M${balance:,.0f}, of which M${spendable:,.0f} is spendable after the "
            f"M${risk.min_balance_reserve:,.0f} reserve. Net worth M${net_worth:,.0f} "
            f"across {len(positions)} positions"
            + (f", {stuck} of them closed and waiting on resolution." if stuck else ".")
            + f" Most you can put into one position: "
            f"M${max(risk.default_max_bet, risk.default_max_fraction * net_worth):,.0f}."
        )
        if spendable < risk.min_bet:
            line += " Not enough cash to buy anything right now; resolutions will free some."
        return line

    def _watchlist(self, positions: list[Position]) -> dict[str, dict[str, Any]]:
        """Markets it forecast with an edge on the favourite but could not afford.

        The scan finds a market, the deep model forecasts it, and then the trade dies
        on the cash reserve. Until now that forecast was simply lost; with no cash, the
        agent had nothing to do but wait, and when cash came back it bought whatever
        the next scan happened to turn up. Kept for a few days, best edge first.
        """
        # Event keys: slug, plus the answer for a multiple-choice holding.
        held_slugs = {p.slug + (f"::{p.answer_id}" if p.answer_id else "") for p in positions}
        held_markets = {p.slug for p in positions if p.answer_id}
        horizon = time.time() * 1000 - self.cfg.agency.watchlist_days * 86_400_000
        latest: dict[str, dict[str, Any]] = {}
        for e in self.memory.recent_events(3000):
            if e.get("kind") in ("bet", "no_trade") and e.get("market"):
                latest[e["market"]] = e
        out = {}
        for slug, e in latest.items():
            if e["kind"] != "no_trade" or int(e.get("ts", 0)) < horizon or slug in held_slugs:
                continue
            if "::" in slug and slug.partition("::")[0] in held_markets:
                continue  # one answer per multiple-choice market
            if not any(r in str(e.get("reason") or "") for r in UNFUNDED):
                continue
            p, q = e.get("model_prob"), e.get("market_prob")
            if p is None or q is None or not backs_favourite(float(p), float(q)):
                continue
            out[slug] = {
                "slug": slug, "question": e.get("question", ""), "url": e.get("url", ""),
                "model_prob": float(p), "market_prob": float(q), "ts": e.get("ts"),
                "confidence": e.get("confidence"), "research": e.get("research", ""),
            }
        ranked = sorted(out.values(), key=lambda w: -abs(w["model_prob"] - w["market_prob"]))
        return {w["slug"]: w for w in ranked[:10]}

    def _last_turns(self, n: int) -> str:
        turns = [e for e in self.memory.recent_events(400) if e.get("kind") == "desk"][-n:]
        if not turns:
            return "(this is the first desk turn)"
        lines = []
        for t in turns:
            stamp = time.strftime("%m-%d %H:%MZ", time.gmtime(int(t["ts"]) / 1000))
            acts = ", ".join(
                f"{a['action']}{' ' + a['market'][:30] if a.get('market') else ''}"
                f"{'' if a.get('ok') else ' (did not happen: ' + str(a.get('result'))[:60] + ')'}"
                for a in t.get("actions") or []
            ) or "no actions"
            lines.append(f"- {stamp}: {str(t.get('thinking'))[:300]} -> {acts}")
        return "\n".join(lines)

    async def _ledger(self) -> str:
        """Everyone it has exchanged mana with, by username, netted, with messages.

        Computed from the API every time, so its own notes are never the ledger.
        """
        try:
            received, sent = await self.client.managrams(self.user_id)
        except ManifoldError:
            return "(could not read transactions)"
        if not received and not sent:
            return "(nobody has sent you any, and you have sent none)"
        names: dict[str, str] = {}
        people: dict[str, dict[str, Any]] = {}
        for txn, direction in [(t, "in") for t in received] + [(t, "out") for t in sent]:
            other = txn.get("fromId") if direction == "in" else txn.get("toId")
            if not other:
                continue
            if other not in names:
                names[other] = await self.client.username_for(other)
            row = people.setdefault(names[other], {"in": 0.0, "out": 0.0, "notes": []})
            row[direction] += float(txn.get("amount") or 0)
            message = str((txn.get("data") or {}).get("message") or "").strip()
            arrow = "they sent" if direction == "in" else "you sent"
            row["notes"].append(f"{arrow} M${float(txn.get('amount') or 0):.0f}: "
                                f"\"{message[:140] or '(no message)'}\"")
        lines = []
        for name, row in people.items():
            net = row["in"] - row["out"]
            self._owed[name.lower()] = max(0.0, net)
            lines.append(f"- @{name}: received M${row['in']:.0f} from them, returned "
                         f"M${row['out']:.0f}.")
            lines += [f"    {note}" for note in row["notes"][-4:]]
        lines.append(
            "Only a loan is owed: something the message asks you to give back. A gift, a "
            "tip or a donation is not, and returning one just costs you the mana. It "
            "already returned the owner's gift once by mistake."
        )
        return "\n".join(lines)


def render_positions(positions: list[Position]) -> str:
    """One line per position, worst first, with the price of the side actually held.

    "60 NO at 20%" read as buying NO at 20%, a longshot, when it meant NO shares in a
    market at 20% YES, the favourite, and the agent sold favourites because of it.
    """
    if not positions:
        return "(none)"
    out = []
    for p in sorted(positions, key=lambda p: (p.profit, -abs(p.invested)))[:25]:
        pct = (p.profit / p.invested * 100) if p.invested else 0.0
        if not p.tradable:
            state = "CLOSED, waiting on resolution"
        elif p.days_to_close < 1:
            state = "closes within a day"
        else:
            state = f"{p.days_to_close:.0f}d to close"
        side_price = p.last_prob if p.side == "YES" else 1.0 - p.last_prob
        kind = "the favourite" if side_price >= 0.5 else "a longshot"
        out.append(
            f"- {p.key} | \"{p.question[:100]}\" | {p.shares:.0f} {p.side} shares, "
            f"{p.side} trading at {side_price:.0%} ({kind}) | invested M${p.invested:.0f}, "
            f"worth M${p.value:.0f}, P/L M${p.profit:+.0f} ({pct:+.0f}%) | {state}"
        )
    return "\n".join(out)


def _render_watchlist(watchlist: dict[str, dict[str, Any]]) -> str:
    if not watchlist:
        return "(nothing: no recent forecast with an edge went unbought for lack of cash)"
    out = []
    for w in watchlist.values():
        side = "YES" if w["model_prob"] > w["market_prob"] else "NO"
        mine = w["model_prob"] if side == "YES" else 1 - w["model_prob"]
        then = w["market_prob"] if side == "YES" else 1 - w["market_prob"]
        stamp = time.strftime("%m-%d", time.gmtime(int(w["ts"] or 0) / 1000))
        out.append(
            f"- {w['slug']} | \"{w['question'][:80]}\" | on {stamp} you put {side} at "
            f"{mine:.0%} when it traded at {then:.0%} ({w.get('confidence') or '?'} "
            f"confidence); not bought for lack of cash"
        )
    return "\n".join(out)
