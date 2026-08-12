"""The agent's own turn.

Everything else the agent does is a reaction: an order filled, a price moved, somebody
asked something. This is the one place it acts unprompted, and it may take several
actions at once: trim positions, add to them, send mana back to somebody who sent it
some, and write or retire its own standing notes.

Only outgoing mana is reviewed by the deep model. Selling reduces exposure and notes
touch nothing, so gating those on a scarce deep call mostly just gave the agent a reason
to keep holding things it had stopped believing in. Amounts are clamped in code either
way, by the balance reserve and the daily mana budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .brain import Budgets
from .config import Config
from .llm import LLMClient, extract_json
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Position
from .prompts import (
    AGENCY_SCHEMA,
    AGENCY_SYSTEM,
    PORTFOLIO_SCHEMA,
    PORTFOLIO_SYSTEM,
    REVIEW_SCHEMA,
    REVIEW_SYSTEM,
    build_agency_prompt,
    build_portfolio_prompt,
    build_review_prompt,
    system_with_orders,
)

log = logging.getLogger(__name__)

MANAGRAM_MINIMUM = 10  # the API rejects anything smaller
# Only outgoing mana is reviewed. Selling reduces exposure and notes touch nothing.
REVIEWED_ACTIONS = {"add", "send_mana"}


@dataclass
class Action:
    action: str
    market_id: str = ""
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
            action=str(data.get("action", "nothing")).strip().lower(),
            market_id=str(data.get("market_id", "")).strip(),
            amount=max(0.0, amount),
            recipient=str(data.get("recipient", "")).strip().lstrip("@"),
            text=str(data.get("text", "")).strip(),
            reasoning=str(data.get("reasoning", "")).strip(),
        )

    def describe(self) -> str:
        bits = [f"action: {self.action}"]
        if self.market_id:
            bits.append(f"market: {self.market_id}")
        if self.amount:
            bits.append(f"amount: M${self.amount:.0f}")
        if self.recipient:
            bits.append(f"recipient: @{self.recipient}")
        if self.text:
            bits.append(f"text: {self.text[:300]}")
        bits.append(f"reasoning: {self.reasoning[:400]}")
        return "\n".join(bits)


class Agency:
    def __init__(
        self,
        cfg: Config,
        *,
        client: ManifoldClient,
        chat: LLMClient,
        deep: LLMClient,
        memory: Memory,
        budget: Budgets,
        user_id: str,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.chat = chat
        self.deep = deep
        self.memory = memory
        self.budget = budget
        self.user_id = user_id

    async def run(self, positions: list[Position], balance: float, net_worth: float) -> str:
        cfg = self.cfg.agency
        if not cfg.enabled:
            return ""
        waited = self.memory.minutes_since_own_action()
        if waited < cfg.min_minutes_between_actions:
            return ""
        if not self.budget.chat.take():
            return ""

        # The turn is marked as taken before anything is attempted. A crash partway
        # through must not leave it retrying every tick.
        self.memory.mark_own_action()

        thinking, proposals = await self._propose(positions, balance, net_worth)
        if thinking:
            self.memory.observe("agency", f"Own turn: {thinking}")
        if not proposals:
            self.memory.log_event("own_action", action="nothing", reasoning=thinking)
            return "took a turn, nothing worth doing"

        done = []
        for action in proposals[: cfg.max_actions_per_turn]:
            result = await self._consider(action, positions, balance, net_worth)
            if result:
                done.append(result)
        return "; ".join(done) if done else "took a turn, nothing survived"

    async def _consider(
        self, action: Action, positions: list[Position], balance: float, net_worth: float
    ) -> str:
        """One proposed action: clamp it, review it if it moves mana, then do it."""
        capped = self._clamp(action, positions, balance)
        if capped is None:
            return ""

        # Only outgoing mana is reviewed. Selling reduces exposure and note-keeping
        # touches nothing, so making those wait on a scarce deep call was just a reason
        # for the agent to sit on positions it had already stopped believing in.
        if capped.action in REVIEWED_ACTIONS:
            verdict = await self._review(capped, positions, balance, net_worth)
            if verdict is None or not verdict.get("approved"):
                reason = (verdict or {}).get("verdict", "review unavailable")
                self.memory.observe(
                    "agency", f"Wanted to {capped.action}, vetoed by review: {reason}"
                )
                self.memory.log_event(
                    "own_action", action=capped.action, approved=False,
                    reasoning=capped.reasoning, verdict=reason,
                )
                return f"{capped.action} vetoed"
            # The reviewer may cut the amount but never raise it.
            approved = float(verdict.get("amount") or capped.amount)
            capped.amount = min(capped.amount, max(0.0, approved))
            capped = self._clamp(capped, positions, balance)
            if capped is None:
                return ""

        return await self._execute(capped, positions)

    # -- steps ------------------------------------------------------------

    def _portfolio_line(
        self, positions: list[Position], balance: float, net_worth: float
    ) -> str:
        """Spendable cash is stated outright because it is what actually gates buying.

        Given only a balance, the agent kept proposing buys it could not afford and
        reading the silent rejection as the market being unattractive.
        """
        spendable = max(0.0, balance - self.cfg.risk.min_balance_reserve)
        stuck = sum(1 for p in positions if not p.tradable)
        line = (
            f"Balance M${balance:,.0f}, of which M${spendable:,.0f} is spendable after "
            f"the M${self.cfg.risk.min_balance_reserve:,.0f} reserve. Net worth "
            f"M${net_worth:,.0f} across {len(positions)} open positions."
        )
        if spendable < self.cfg.risk.min_bet:
            line += (
                " There is not enough free mana to open or add to anything right now, "
                "so selling is the only way to free some up."
            )
        if stuck:
            line += f" {stuck} of them are closed and cannot be traded."
        return line

    async def _propose(
        self, positions: list[Position], balance: float, net_worth: float
    ) -> tuple[str, list[Action]]:
        prompt = build_agency_prompt(
            today=_today(),
            portfolio=self._portfolio_line(positions, balance, net_worth),
            positions=_render_positions(positions),
            lessons=self.memory.lessons_block(),
            memory=self.memory.context_block(),
            recent_actions=self.memory.own_actions_block(5),
            owed=await self._incoming_mana(),
            todos=self.memory.todos_block(),
        )
        try:
            response = await self.chat.generate(
                prompt,
                system=system_with_orders(AGENCY_SYSTEM, self.cfg.owner_block()),
                json_schema=AGENCY_SCHEMA,
            )
            data = extract_json(response.text)
            raw = data.get("actions")
            actions = [
                Action.parse(item) for item in raw if isinstance(item, dict)
            ] if isinstance(raw, list) else []
            return str(data.get("thinking", ""))[:500], actions
        except Exception as exc:  # noqa: BLE001 - a bad turn must not kill the tick
            log.warning("Own-turn proposal failed: %s", exc)
            return "", []

    async def _review(
        self, action: Action, positions: list[Position], balance: float, net_worth: float
    ) -> dict[str, Any] | None:
        if not self.cfg.agency.require_review:
            return {"approved": True, "amount": action.amount, "verdict": "review disabled"}
        if not self.budget.deep.take():
            log.info("No deep budget to review %s, so it does not happen", action.action)
            return None
        try:
            response = await self.deep.generate(
                build_review_prompt(
                    proposal=action.describe(),
                    portfolio=(
                        self._portfolio_line(positions, balance, net_worth)
                    ),
                    positions=_render_positions(positions),
                    owed=await self._incoming_mana(),
                ),
                system=REVIEW_SYSTEM,
                json_schema=REVIEW_SCHEMA,
            )
            return extract_json(response.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Review failed, so the action does not happen: %s", exc)
            return None

    def _clamp(
        self, action: Action, positions: list[Position], balance: float
    ) -> Action | None:
        """Everything the models cannot be trusted to respect, enforced in code."""
        cfg = self.cfg.agency
        held = {p.contract_id: p for p in positions}

        if action.action in ("sell", "add"):
            if action.market_id not in held:
                log.info("Own turn named a market it does not hold, dropping")
                return None
            # A closed market still appears in the book but the API answers 403 to any
            # order on it. Without this the agent proposes the same doomed sell every
            # turn, burns the turn on it, and learns nothing from the error.
            if not held[action.market_id].tradable:
                self.memory.observe(
                    "agency",
                    f"Wanted to {action.action} \"{held[action.market_id].question[:60]}\" "
                    f"but that market is closed; it can only be waited out.",
                )
                return None

        if action.action == "add":
            spendable = max(0.0, balance - self.cfg.risk.min_balance_reserve)
            action.amount = min(action.amount, spendable)
            if action.amount < 1:
                return None

        if action.action == "send_mana":
            if not cfg.allow_send_mana or not action.recipient:
                return None
            spendable = max(0.0, balance - self.cfg.risk.min_balance_reserve)
            if action.amount > spendable or action.amount < MANAGRAM_MINIMUM:
                log.info("Own turn wanted to send M$%.0f, which is not affordable or "
                         "under the M$%d minimum", action.amount, MANAGRAM_MINIMUM)
                return None

        if action.action == "note_add" and len(action.text) < 8:
            return None
        return action

    async def _execute(self, action: Action, positions: list[Position]) -> str:
        held = {p.contract_id: p for p in positions}
        dry = self.cfg.manifold.dry_run
        detail = ""

        try:
            if action.action == "sell":
                position = held[action.market_id]
                await self.client.sell_shares(
                    market_id=action.market_id, outcome=position.side
                )
                detail = (
                    f"sold out of \"{position.question[:60]}\" ({position.side}, "
                    f"P/L M${position.profit:+.0f})"
                )
            elif action.action == "add":
                position = held[action.market_id]
                await self.client.place_bet(
                    contract_id=action.market_id,
                    amount=action.amount,
                    outcome=position.side,
                )
                if not dry:
                    self.memory.record_spend(action.amount)
                detail = (
                    f"added M${action.amount:.0f} {position.side} to "
                    f"\"{position.question[:60]}\""
                )
            elif action.action == "send_mana":
                user = await self.client.user_by_username(action.recipient)
                if not user or not user.get("id"):
                    log.info("Own turn named an unknown user @%s", action.recipient)
                    return ""
                await self.client.send_managram(
                    to_ids=[user["id"]],
                    amount=action.amount,
                    message=action.text[:200] or "Thanks.",
                )
                if not dry:
                    self.memory.record_spend(action.amount)
                detail = f"sent M${action.amount:.0f} to @{action.recipient}"
            elif action.action == "note_add":
                if not self.memory.add_lesson(action.text, source="self"):
                    return ""
                detail = f"wrote a standing note: {action.text[:120]}"
            elif action.action == "note_remove":
                if not self.memory.remove_lesson(action.text):
                    return ""
                detail = f"retired a standing note: {action.text[:120]}"
            elif action.action == "todo_add":
                if not self.memory.add_todo(action.text):
                    return ""
                detail = f"added to its to-do list: {action.text[:120]}"
            elif action.action == "todo_done":
                if not self.memory.complete_todo(action.text):
                    return ""
                detail = f"ticked off: {action.text[:120]}"
            else:
                return ""
        except ManifoldError as exc:
            log.warning("Own action %s failed: %s", action.action, exc)
            self.memory.log_event("own_action", action=action.action, error=str(exc))
            return f"{action.action} rejected by Manifold"

        log.info("Own turn: %s%s", detail, " [dry-run]" if dry else "")
        self.memory.record_own_action(action.action, detail, action.reasoning)
        self.memory.observe("agency", f"Own turn, {detail}. Because: {action.reasoning[:450]}")
        self.memory.log_event(
            "own_action", action=action.action, approved=True, detail=detail,
            amount=action.amount, reasoning=action.reasoning, dry_run=dry,
        )
        return detail

    async def review_book(
        self, positions: list[Position], balance: float, net_worth: float
    ) -> str:
        """Look over every open position and act on the ones whose case has changed.

        Separate from the free turn above, and deliberately narrower: this may only sell
        or add on markets already held. It runs on the cheap model, and each resulting
        trade still goes through the deep reviewer before any mana moves.
        """
        cfg = self.cfg.agency
        if not cfg.review_positions or not positions:
            return ""
        if self.memory.minutes_since_book_review() < cfg.min_minutes_between_book_reviews:
            return ""
        if not self.budget.chat.take():
            return ""
        self.memory.mark_book_review()

        try:
            response = await self.chat.generate(
                build_portfolio_prompt(
                    today=_today(),
                    portfolio=(
                        self._portfolio_line(positions, balance, net_worth)
                    ),
                    positions=_render_positions(positions),
                    lessons=self.memory.lessons_block(),
                    memory=self.memory.context_block(),
                ),
                system=system_with_orders(PORTFOLIO_SYSTEM, self.cfg.owner_block()),
                json_schema=PORTFOLIO_SCHEMA,
            )
            data = extract_json(response.text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Book review failed: %s", exc)
            return ""

        assessment = str(data.get("assessment", ""))[:400]
        changes = data.get("changes") or []
        self.memory.observe(
            "book",
            f"Looked over {len(positions)} positions: {assessment} "
            f"({len(changes)} change(s) proposed)",
        )
        self.memory.log_event(
            "book_review", positions=len(positions), assessment=assessment,
            proposed=len(changes),
        )
        if not isinstance(changes, list) or not changes:
            return "reviewed the book, no changes"

        done = []
        for change in changes[:2]:
            if not isinstance(change, dict):
                continue
            action = Action(
                action=str(change.get("action", "")).strip().lower(),
                market_id=str(change.get("market_id", "")).strip(),
                amount=float(change.get("amount") or 0),
                reasoning=str(change.get("why", ""))[:700],
            )
            if action.action not in ("sell", "add"):
                continue
            capped = self._clamp(action, positions, balance)
            if capped is None:
                continue

            verdict = await self._review(capped, positions, balance, net_worth)
            if verdict is None or not verdict.get("approved"):
                reason = (verdict or {}).get("verdict", "review unavailable")
                self.memory.observe(
                    "book", f"Book review wanted to {capped.action}, vetoed: {reason}"
                )
                continue
            capped.amount = min(capped.amount, max(0.0, float(verdict.get("amount") or 0)))
            capped = self._clamp(capped, positions, balance)
            if capped is None:
                continue
            result = await self._execute(capped, positions)
            if result:
                done.append(result)

        return "; ".join(done) if done else "reviewed the book, nothing survived review"

    async def _incoming_mana(self) -> str:
        """Who has sent it mana lately, so it can pay people back."""
        try:
            txns = await self.client.incoming_managrams(self.user_id, 0)
        except ManifoldError:
            return "(could not read transactions)"
        if not txns:
            return "(nobody has sent you any)"
        lines = []
        for txn in sorted(txns, key=lambda t: int(t.get("createdTime") or 0))[-8:]:
            message = (txn.get("data") or {}).get("message") or "(no message)"
            lines.append(
                f"- M${float(txn.get('amount') or 0):.0f} from user {txn.get('fromId')}: "
                f"{str(message)[:160]}"
            )
        return "\n".join(lines)


def _render_positions(positions: list[Position]) -> str:
    """One line per position, worst first.

    Ordering by loss rather than size is deliberate: the thing most likely to need a
    decision should be the first thing read, not buried under the biggest holdings.
    """
    if not positions:
        return "(none)"
    rows = sorted(positions, key=lambda p: (p.profit, -abs(p.invested)))[:20]
    out = []
    for p in rows:
        pct = (p.profit / p.invested * 100) if p.invested else 0.0
        if not p.tradable:
            state = "CLOSED, cannot be sold, waiting on resolution"
        elif p.days_to_close < 1:
            state = "closes within a day"
        else:
            state = f"{p.days_to_close:.0f}d to close"
        out.append(
            f"- {p.contract_id} | \"{p.question[:70]}\" | {p.shares:.0f} {p.side} at "
            f"{p.last_prob:.0%} | invested M${p.invested:.0f}, now worth M${p.value:.0f}, "
            f"P/L M${p.profit:+.0f} ({pct:+.0f}%) | {state}"
        )
    return "\n".join(out)


def _today() -> str:
    import time

    return time.strftime("%Y-%m-%d", time.gmtime())
