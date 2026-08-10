"""Durable state: an append-only event log plus a compressed narrative the model reads.

Raw events are the audit trail and are never rewritten. The narrative summary is what
gets fed back into prompts, and it is periodically recompressed by the model so the
context stays bounded no matter how long the agent runs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .config import MemoryConfig
from .llm import LLMClient

log = logging.getLogger(__name__)

COMPRESS_SYSTEM = (
    "You maintain the long-term memory of an autonomous prediction market trader. "
    "You are given the existing summary plus recent events. Produce an updated summary "
    "that a future instance of the trader would find useful."
)


class Memory:
    def __init__(self, cfg: MemoryConfig, state_dir: Path, llm: LLMClient) -> None:
        self.cfg = cfg
        self.llm = llm
        self.dir = state_dir
        self.events_path = self.dir / "events.jsonl"
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = self._load_state()

    # -- persistence ------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        state = self._defaults()
        if self.state_path.exists():
            try:
                # Merge over defaults so a state file written by an older version
                # picks up new keys instead of raising KeyError everywhere.
                state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                log.error("state.json is corrupt, starting fresh (old file kept as .bak)")
                self.state_path.rename(self.state_path.with_suffix(".json.bak"))
        return state

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "summary": "",
            "market_notes": {},        # market_id -> one line the model wrote about it
            "seen": {},                # market_id -> ms timestamp of last evaluation
            "budget": {"day": "", "spent": 0.0},
            "last_scan_ms": 0,
            "last_managram_ms": 0,
            "my_comments": [],         # [{id, contract_id, ts}], newest last
            "commented_markets": [],   # markets we have already introduced ourselves on
            "answered_issues": [],
            "replied_to": [],
            "tracked_orders": {},      # bet_id -> {contract_id, question, outcome, amount}
            "events_since_compress": 0,
        }

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def log_event(self, kind: str, **payload: Any) -> None:
        event = {"ts": int(time.time() * 1000), "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, default=str) + "\n")
        self.state["events_since_compress"] = self.state.get("events_since_compress", 0) + 1
        self.save()

    def recent_events(self, limit: int) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    # -- market bookkeeping -----------------------------------------------

    def note_for(self, market_id: str) -> str:
        return self.state["market_notes"].get(market_id, "")

    def set_note(self, market_id: str, note: str) -> None:
        if note.strip():
            self.state["market_notes"][market_id] = note.strip()[:400]
            self.save()

    def mark_seen(self, market_id: str) -> None:
        self.state["seen"][market_id] = int(time.time() * 1000)
        self.save()

    def seen_within(self, market_id: str, hours: float) -> bool:
        ts = self.state["seen"].get(market_id)
        if not ts:
            return False
        return (time.time() * 1000 - ts) < hours * 3_600_000

    def minutes_since_scan(self) -> float:
        last = float(self.state.get("last_scan_ms") or 0)
        if last <= 0:
            return float("inf")
        return (time.time() * 1000 - last) / 60_000

    def mark_scanned(self) -> None:
        self.state["last_scan_ms"] = int(time.time() * 1000)
        self.save()

    # -- budget -----------------------------------------------------------

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def budget_spent(self) -> float:
        budget = self.state["budget"]
        if budget.get("day") != self._today():
            return 0.0
        return float(budget.get("spent", 0.0))

    def record_spend(self, amount: float) -> None:
        today = self._today()
        budget = self.state["budget"]
        if budget.get("day") != today:
            budget["day"] = today
            budget["spent"] = 0.0
        budget["spent"] = float(budget.get("spent", 0.0)) + amount
        self.save()

    # -- social bookkeeping -----------------------------------------------

    def remember_comment(self, comment_id: str, contract_id: str) -> None:
        """Track our own comments so replies to them can be found later."""
        comments = self.state["my_comments"]
        comments.append(
            {"id": comment_id, "contract_id": contract_id, "ts": int(time.time() * 1000)}
        )
        self.state["my_comments"] = comments[-50:]
        self.save()

    def recent_comments(self, n: int) -> list[dict[str, Any]]:
        return self.state["my_comments"][-n:]

    def has_commented_on(self, market_id: str) -> bool:
        return market_id in self.state["commented_markets"]

    def mark_commented_on(self, market_id: str) -> None:
        markets = self.state["commented_markets"]
        if market_id not in markets:
            markets.append(market_id)
            self.state["commented_markets"] = markets[-500:]
            self.save()

    def has_answered_issue(self, number: int) -> bool:
        return number in self.state["answered_issues"]

    def mark_issue_answered(self, number: int) -> None:
        issues = self.state["answered_issues"]
        issues.append(number)
        self.state["answered_issues"] = issues[-200:]
        self.save()

    def has_replied(self, comment_id: str) -> bool:
        return comment_id in self.state["replied_to"]

    def mark_replied(self, comment_id: str) -> None:
        replied = self.state["replied_to"]
        replied.append(comment_id)
        self.state["replied_to"] = replied[-500:]
        self.save()

    # -- order tracking ---------------------------------------------------

    def track_order(self, bet_id: str, info: dict[str, Any]) -> None:
        self.state["tracked_orders"][bet_id] = info
        self.save()

    def untrack_order(self, bet_id: str) -> dict[str, Any] | None:
        info = self.state["tracked_orders"].pop(bet_id, None)
        self.save()
        return info

    @property
    def tracked_orders(self) -> dict[str, Any]:
        return self.state["tracked_orders"]

    # -- compression ------------------------------------------------------

    def context_block(self) -> str:
        """The memory the model sees on every decision."""
        summary = self.state.get("summary", "").strip()
        return summary or "No prior history yet. This is an early trade."

    async def maybe_compress(self) -> None:
        if self.state.get("events_since_compress", 0) < self.cfg.compress_after_events:
            return
        await self.compress()

    async def compress(self) -> None:
        events = self.recent_events(self.cfg.compress_after_events * 2)
        if not events:
            return

        rendered = "\n".join(
            json.dumps({k: v for k, v in e.items() if k != "raw"}, default=str)[:500]
            for e in events
        )
        prompt = (
            f"Existing summary:\n{self.state.get('summary') or '(none)'}\n\n"
            f"Recent events:\n{rendered}\n\n"
            "Write an updated summary under 500 words covering:\n"
            "- Running performance: how many trades, roughly how they went, current exposure.\n"
            "- Mistakes worth not repeating, stated concretely.\n"
            "- Categories or question styles where this agent has been well or badly calibrated.\n"
            "- Anything about specific still-open markets that matters.\n"
            "Write plainly. No headers, no bullet decoration, just tight prose."
        )
        try:
            response = await self.llm.generate(prompt, system=COMPRESS_SYSTEM)
        except Exception as exc:  # noqa: BLE001 - compression is not critical
            log.warning("Memory compression failed: %s", exc)
            return

        self.state["summary"] = response.text.strip()
        self.state["events_since_compress"] = 0
        self.save()
        log.info("Compressed memory to %d chars", len(self.state["summary"]))
