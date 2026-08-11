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
            "journal": [],             # cheap running observations: [{ts, kind, text}]
            "lessons": [],             # standing notes: [{text, source, ts}]
            "conversations": {},       # key -> {channel, who, title, url, messages[]}
            "market_notes": {},        # market_id -> one line the model wrote about it
            "seen": {},                # market_id -> ms timestamp of last evaluation
            "budget": {"day": "", "spent": 0.0},
            "llm_usage": {"day": ""},   # day -> per-tier call counts, reset daily
            "last_scan_ms": 0,
            "last_compress_ms": 0,
            "portfolio_mark": {},      # last seen balance/net worth, to spot changes
            "position_mark": {},       # contract_id -> last seen shares/profit/prob
            "last_managram_ms": 0,
            "my_comments": [],         # [{id, contract_id, ts}], newest last
            "commented_markets": [],   # markets we have already introduced ourselves on
            "answered_issues": [],     # legacy, superseded by issue_threads
            "issue_threads": {},       # issue number -> {last_id} of the last reply handled
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

    # -- journal ----------------------------------------------------------

    def observe(self, kind: str, text: str) -> None:
        """Write down something that happened. Costs nothing.

        The narrative summary is only rewritten every so often, because rewriting it
        costs an LLM call. Between rewrites the agent used to remember almost nothing:
        it could not tell you that a position moved against it an hour ago, because
        nothing wrote that down. Observations are recorded by plain code on every tick
        and folded into the summary later, so noticing is free and only remembering
        long-term costs anything.
        """
        text = " ".join(text.split())[:300]
        if not text:
            return
        journal = self.state["journal"]
        # Same observation twice in a row is noise: a price that has not moved since the
        # last tick does not need a second line saying so.
        if journal and journal[-1]["kind"] == kind and journal[-1]["text"] == text:
            return
        journal.append({"ts": int(time.time() * 1000), "kind": kind, "text": text})
        self.state["journal"] = journal[-self.cfg.max_journal :]

    def recent_journal(self, n: int) -> list[dict[str, Any]]:
        return self.state["journal"][-n:]

    def journal_block(self, n: int) -> str:
        entries = self.recent_journal(n)
        if not entries:
            return "(nothing noted yet)"
        return "\n".join(f"- {_stamp(e['ts'])} {e['text']}" for e in entries)

    # -- market bookkeeping -----------------------------------------------

    def note_for(self, market_id: str) -> str:
        entry = self.state["market_notes"].get(market_id, "")
        # Notes used to be bare strings; older state files still hold them that way.
        return entry.get("note", "") if isinstance(entry, dict) else entry

    def set_note(self, market_id: str, note: str, *, question: str = "", url: str = "") -> None:
        if not note.strip():
            return
        self.state["market_notes"][market_id] = {
            "note": note.strip()[:400],
            "question": question,
            "url": url,
            "ts": int(time.time() * 1000),
        }
        self.save()

    def recent_notes(self, n: int) -> list[dict[str, Any]]:
        """Newest market notes, for showing what it is carrying around."""
        out = []
        for market_id, entry in self.state["market_notes"].items():
            if isinstance(entry, dict):
                out.append({**entry, "id": market_id})
            else:
                out.append({"note": entry, "question": "", "url": "", "ts": 0, "id": market_id})
        return sorted(out, key=lambda e: e.get("ts") or 0, reverse=True)[:n]

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

    def issue_watermark(self, number: int) -> int:
        """Id of the newest comment on this issue the agent has already answered.

        Zero means the issue body itself is still unanswered. Threads are left open,
        so this is what stops the agent from answering the same message twice while
        still letting a follow-up comment pull it back into the conversation.
        """
        thread = self.state["issue_threads"].get(str(number))
        if thread:
            return int(thread.get("last_id") or 0)
        # Migration: issues answered before threads were tracked stay answered.
        return -1 if number in self.state["answered_issues"] else 0

    def minutes_since_issue_reply(self, number: int) -> float:
        thread = self.state["issue_threads"].get(str(number))
        if not thread or not thread.get("ts"):
            return float("inf")
        return (time.time() * 1000 - float(thread["ts"])) / 60_000

    def mark_issue_handled(self, number: int, last_id: int) -> None:
        threads = self.state["issue_threads"]
        threads[str(number)] = {"last_id": int(last_id), "ts": int(time.time() * 1000)}
        if len(threads) > 200:
            for key, _ in sorted(threads.items(), key=lambda kv: kv[1].get("ts", 0))[:50]:
                threads.pop(key, None)
        self.save()

    # -- llm usage --------------------------------------------------------

    def llm_used_today(self, tier: str) -> int:
        usage = self.state["llm_usage"]
        if usage.get("day") != self._today():
            return 0
        return int(usage.get(tier) or 0)

    def record_llm_call(self, tier: str) -> None:
        usage = self.state["llm_usage"]
        if usage.get("day") != self._today():
            usage.clear()
            usage["day"] = self._today()
        usage[tier] = int(usage.get(tier) or 0) + 1
        # Deliberately not saving here: the tick saves state on its way out, and one
        # write per LLM call would mean a git commit's worth of churn for nothing.

    def has_replied(self, comment_id: str) -> bool:
        return comment_id in self.state["replied_to"]

    def mark_replied(self, comment_id: str) -> None:
        replied = self.state["replied_to"]
        replied.append(comment_id)
        self.state["replied_to"] = replied[-500:]
        self.save()

    # -- conversations ----------------------------------------------------

    def conversation(self, key: str) -> dict[str, Any]:
        return self.state["conversations"].get(key, {})

    def record_message(
        self,
        key: str,
        role: str,
        text: str,
        *,
        channel: str = "",
        who: str = "",
        title: str = "",
        url: str = "",
    ) -> None:
        """Append one turn to a conversation. `role` is "them" or "me"."""
        convos = self.state["conversations"]
        entry = convos.setdefault(
            key,
            {"channel": channel, "who": who, "title": title, "url": url, "messages": []},
        )
        for field_name, value in (("channel", channel), ("who", who), ("title", title), ("url", url)):
            if value:
                entry[field_name] = value
        entry["messages"].append(
            {"role": role, "text": text.strip()[:1200], "ts": int(time.time() * 1000)}
        )
        entry["messages"] = entry["messages"][-self.cfg.conversation_depth :]
        entry["updated_ms"] = int(time.time() * 1000)

        # Drop the least recently touched threads rather than growing without bound.
        if len(convos) > self.cfg.max_conversations:
            oldest = sorted(convos.items(), key=lambda kv: kv[1].get("updated_ms", 0))
            for stale_key, _ in oldest[: len(convos) - self.cfg.max_conversations]:
                convos.pop(stale_key, None)
        self.save()

    def conversation_block(self, key: str) -> str:
        entry = self.conversation(key)
        messages = entry.get("messages") or []
        if not messages:
            return "(you have not spoken with them before)"
        who = entry.get("who") or "them"
        return "\n".join(
            f"{'you' if m['role'] == 'me' else who}: {m['text'][:600]}" for m in messages
        )

    def recent_conversations(self, n: int) -> list[dict[str, Any]]:
        convos = self.state["conversations"].values()
        return sorted(convos, key=lambda c: c.get("updated_ms", 0), reverse=True)[:n]

    # -- lessons ----------------------------------------------------------

    def add_lesson(self, text: str, source: str) -> bool:
        """Record a standing note. Returns False if it was empty or already known.

        Lessons come from three places: the agent writing one after a trade, someone
        giving it advice in a thread, and the owner's instructions file. Only the last
        is authoritative, so the source travels with the text and the prompt says so.
        """
        text = " ".join(text.split())[:280]
        if len(text) < 8:
            return False
        lessons = self.state["lessons"]
        if any(existing["text"].lower() == text.lower() for existing in lessons):
            return False

        lessons.append({"text": text, "source": source, "ts": int(time.time() * 1000)})
        if len(lessons) > self.cfg.max_lessons:
            # Owner instructions outlast everything else; otherwise oldest goes first.
            expendable = [i for i, le in enumerate(lessons) if le["source"] != "owner"]
            lessons.pop(expendable[0] if expendable else 0)
        self.state["lessons"] = lessons
        self.save()
        log.info("New lesson from %s: %s", source, text[:120])
        return True

    def lessons_block(self) -> str:
        lessons = self.state["lessons"]
        if not lessons:
            return "(nothing yet)"
        return "\n".join(f"- [{le['source']}] {le['text']}" for le in lessons)

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
        """The memory the model sees on every decision.

        The compressed summary is the long term. The journal tail is the last few hours,
        which the summary will not have absorbed yet and which is usually the part that
        matters for what to do right now.
        """
        summary = self.state.get("summary", "").strip()
        parts = [summary or "No compressed history yet. This is still early."]
        recent = self.journal_block(self.cfg.journal_in_context)
        if recent != "(nothing noted yet)":
            parts.append(f"Since that was written:\n{recent}")
        return "\n\n".join(parts)

    def _hours_since_compress(self) -> float:
        last = float(self.state.get("last_compress_ms") or 0)
        if last <= 0:
            return float("inf")
        return (time.time() * 1000 - last) / 3_600_000

    async def maybe_compress(self) -> None:
        """Compress on whichever comes first: enough events, or enough time.

        Event count alone is the wrong trigger now that the journal fills up on quiet
        ticks. A slow day would otherwise never compress and the context would just grow.
        """
        due_by_events = (
            self.state.get("events_since_compress", 0) >= self.cfg.compress_after_events
        )
        due_by_time = self._hours_since_compress() >= self.cfg.compress_after_hours
        due_by_journal = len(self.state["journal"]) >= self.cfg.max_journal * 0.8
        if due_by_events or due_by_time or due_by_journal:
            await self.compress()

    async def compress(self) -> None:
        events = self.recent_events(self.cfg.compress_after_events * 2)
        journal = self.state["journal"]
        if not events and not journal:
            return

        rendered = "\n".join(
            json.dumps({k: v for k, v in e.items() if k != "raw"}, default=str)[:500]
            for e in events
        )
        observed = "\n".join(f"{_stamp(e['ts'])} [{e['kind']}] {e['text']}" for e in journal)
        prompt = (
            f"Existing summary:\n{self.state.get('summary') or '(none)'}\n\n"
            f"Decisions and actions since then:\n{rendered}\n\n"
            f"Running observations since then:\n{observed or '(none)'}\n\n"
            "Write an updated summary under 500 words covering:\n"
            "- Running performance: how many trades, roughly how they went, current exposure.\n"
            "- What has happened to the open positions, including which are moving against us.\n"
            "- Mistakes worth not repeating, stated concretely.\n"
            "- Categories or question styles where this agent has been well or badly calibrated.\n"
            "- Anything about specific still-open markets that matters.\n"
            "Keep concrete details worth keeping and drop the rest: this replaces the "
            "observations above, which are deleted once you have written it. Write "
            "plainly. No headers, no bullet decoration, just tight prose."
        )
        try:
            response = await self.llm.generate(prompt, system=COMPRESS_SYSTEM)
        except Exception as exc:  # noqa: BLE001 - compression is not critical
            log.warning("Memory compression failed: %s", exc)
            return

        self.state["summary"] = response.text.strip()
        self.state["events_since_compress"] = 0
        self.state["last_compress_ms"] = int(time.time() * 1000)
        # The tail stays so the next few ticks still have immediate context; everything
        # older is now represented in the summary.
        self.state["journal"] = journal[-10:]
        self.save()
        log.info("Compressed %d observations into %d chars of memory",
                 len(journal), len(self.state["summary"]))


def _stamp(ms: int) -> str:
    return time.strftime("%m-%d %H:%MZ", time.gmtime(ms / 1000))
