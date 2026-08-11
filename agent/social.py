"""Answering people who talk to the bot.

Manifold has no public notifications endpoint, so this looks in the only place the bot
is likely to be addressed: the threads under its own last few comments.

Nobody here is the owner. Manifold accounts are anonymous and every one of them holds
positions, so this channel can argue with the agent and change its mind about a fact,
but it cannot write anything into the standing notes that steer future trades.
"""

from __future__ import annotations

import logging
import time

from .brain import Budgets
from .config import Config
from .llm import LLMClient
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Comment, Market
from .prompts import (
    MANAGRAM_SYSTEM,
    REPLY_SYSTEM,
    STRANGER_NOTE,
    build_reply_prompt,
    system_with_orders,
)

log = logging.getLogger(__name__)

MANAGRAM_REPLY_AMOUNT = 10  # the API minimum, and the only channel that exists


class Social:
    def __init__(
        self,
        cfg: Config,
        *,
        client: ManifoldClient,
        llm: LLMClient,
        memory: Memory,
        budget: Budgets,
        user_id: str,
        username: str,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.llm = llm
        self.memory = memory
        self.budget = budget
        self.user_id = user_id
        self.username = username

    async def run(self) -> int:
        handled = 0
        if self.cfg.social.reply_to_comments:
            handled += await self._handle_comments()
        if self.cfg.social.reply_to_managrams:
            handled += await self._handle_managrams()
        return handled

    # -- comments ---------------------------------------------------------

    def _is_for_us(self, comment: Comment, my_ids: set[str]) -> bool:
        if comment.user_id == self.user_id:
            return False
        if comment.reply_to_id and comment.reply_to_id in my_ids:
            return True
        return f"@{self.username}".lower() in comment.text.lower()

    async def _handle_comments(self) -> int:
        recent = self.memory.recent_comments(self.cfg.social.watch_last_n_comments)
        if not recent:
            return 0

        my_ids = {c["id"] for c in recent}
        targets: list[tuple[str, Comment, list[Comment]]] = []

        for contract_id in dict.fromkeys(c["contract_id"] for c in recent):
            try:
                thread = await self.client.comments(contract_id)
            except ManifoldError as exc:
                log.warning("Could not load comments for %s: %s", contract_id, exc)
                continue
            for comment in thread:
                if self._is_for_us(comment, my_ids) and not self.memory.has_replied(comment.id):
                    targets.append((contract_id, comment, thread))

        if not targets:
            return 0

        # Oldest first, so a conversation is answered in the order it happened.
        targets.sort(key=lambda t: t[1].created_time)

        handled = 0
        for contract_id, comment, thread in targets[: self.cfg.social.max_replies_per_tick]:
            if self.budget.chat.spent:
                break
            try:
                market = await self.client.market(contract_id)
            except ManifoldError:
                continue
            if await self._reply(market, comment, thread):
                handled += 1
        return handled

    async def _reply(self, market: Market, comment: Comment, thread: list[Comment]) -> bool:
        if not self.budget.chat.take():
            return False

        positions = await self.client.positions(self.user_id)
        position = next((p for p in positions if p.contract_id == market.id), None)

        # Keyed by person, not by market, so a regular is still recognised the next
        # time they turn up somewhere else.
        key = f"manifold:{comment.username}"
        self.memory.record_message(
            key, "them", comment.text,
            channel="manifold", who=comment.username,
            title=market.question, url=market.url,
        )

        prompt = build_reply_prompt(
            market=market,
            thread=self._render_thread(comment, thread),
            memory=self.memory.context_block(),
            market_note=self.memory.note_for(market.id),
            position=position,
            who=comment.username,
            lessons=self.memory.lessons_block(),
            history=self.memory.conversation_block(key),
        )
        try:
            # No structured lesson field on this channel at all. A Manifold account is
            # anonymous and holds positions, so it does not get to write standing rules
            # into something that trades. Advice worth keeping comes from the owner.
            response = await self.llm.generate(
                prompt,
                system=system_with_orders(
                    f"{REPLY_SYSTEM}\n\n{STRANGER_NOTE}", self.cfg.owner_block()
                ),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Reply generation failed: %s", exc)
            return False

        text = response.text.strip()[:900]
        if not text:
            return False

        try:
            result = await self.client.post_comment(
                contract_id=market.id, content=text, reply_to=comment.id
            )
        except ManifoldError as exc:
            log.warning("Reply failed on %s: %s", market.slug, exc)
            return False

        self.memory.mark_replied(comment.id)
        self.memory.record_message(key, "me", text)
        if result and result.get("id"):
            self.memory.remember_comment(result["id"], market.id)
        self.memory.log_event(
            "reply", market=market.slug, to=comment.username, text=text[:300]
        )
        self.memory.observe(
            "conversation",
            f"@{comment.username} said \"{comment.text[:90]}\" on "
            f"\"{market.question[:50]}\". I replied: {text[:120]}",
        )
        log.info("Replied to @%s on %s", comment.username, market.slug)
        return True

    def _render_thread(self, target: Comment, thread: list[Comment], depth: int = 4) -> str:
        """Walk up the reply chain so the model sees what it is responding to."""
        by_id = {c.id: c for c in thread}
        chain = [target]
        cursor = target
        while cursor.reply_to_id and len(chain) < depth:
            parent = by_id.get(cursor.reply_to_id)
            if parent is None:
                break
            chain.append(parent)
            cursor = parent
        return "\n\n".join(f"@{c.username}: {c.text[:600]}" for c in reversed(chain))

    # -- managrams --------------------------------------------------------

    async def _handle_managrams(self) -> int:
        since = int(self.memory.state.get("last_managram_ms") or 0)
        if since == 0:
            # First run: set the watermark instead of replying to all of history.
            self.memory.state["last_managram_ms"] = int(time.time() * 1000)
            self.memory.save()
            return 0

        try:
            incoming = await self.client.incoming_managrams(self.user_id, since)
        except ManifoldError as exc:
            log.warning("Could not read transactions: %s", exc)
            return 0

        handled, newest = 0, since
        for txn in sorted(incoming, key=lambda t: int(t.get("createdTime") or 0)):
            newest = max(newest, int(txn.get("createdTime") or 0))
            from_id = txn.get("fromId")
            amount = float(txn.get("amount") or 0)
            if not from_id or amount < MANAGRAM_REPLY_AMOUNT or self.budget.chat.spent:
                continue
            if not self.budget.chat.take():
                break

            message = (txn.get("data") or {}).get("message") or "(no message)"
            try:
                response = await self.llm.generate(
                    f"Someone sent you M${amount:.0f} with the message: {message}\n\n"
                    f"Your memory:\n{self.memory.context_block()}\n\nWrite your reply.",
                    system=system_with_orders(MANAGRAM_SYSTEM, self.cfg.owner_block()),
                )
                await self.client.send_managram(
                    to_ids=[from_id],
                    amount=MANAGRAM_REPLY_AMOUNT,
                    message=response.text.strip()[:400],
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Managram reply failed: %s", exc)
                continue

            self.memory.log_event("managram_reply", to=from_id, received=amount)
            handled += 1

        self.memory.state["last_managram_ms"] = newest
        self.memory.save()
        return handled
