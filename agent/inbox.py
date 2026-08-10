"""Questions asked through the website.

The showcase site is static, so it cannot take a message directly. Instead its form
opens a prefilled GitHub issue, and this reads those issues on the next tick, answers
them, and closes them. No backend, no extra hosting, and the whole conversation stays
public and readable.

Both environment variables are provided automatically inside GitHub Actions. Outside
it, this is a no-op.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .brain import Budgets
from .config import Config
from .llm import LLMClient, extract_json
from .memory import Memory
from .prompts import ADVICE_NOTE, ISSUE_SYSTEM, REPLY_SCHEMA, system_with_orders

log = logging.getLogger(__name__)

LABEL = "ask-the-bot"


class Inbox:
    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLMClient,
        memory: Memory,
        budget: Budgets,
        portfolio_line: str,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.memory = memory
        self.budget = budget
        self.portfolio_line = portfolio_line
        self.repo = os.environ.get("GITHUB_REPOSITORY", "")
        self.token = os.environ.get("GITHUB_TOKEN", "")

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.social.answer_github_issues and self.repo and self.token)

    async def run(self) -> int:
        if not self.enabled:
            return 0

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        async with httpx.AsyncClient(
            base_url="https://api.github.com", headers=headers, timeout=30
        ) as gh:
            try:
                resp = await gh.get(
                    f"/repos/{self.repo}/issues",
                    params={"state": "open", "labels": LABEL, "per_page": 10},
                )
                resp.raise_for_status()
                issues: list[dict[str, Any]] = resp.json()
            except httpx.HTTPError as exc:
                log.warning("Could not list issues: %s", exc)
                return 0

            handled = 0
            for issue in issues:
                if "pull_request" in issue:
                    continue
                number = issue.get("number")
                if number is None or self.memory.has_answered_issue(number):
                    continue
                if self.budget.fast.spent or handled >= self.cfg.social.max_replies_per_tick:
                    break
                if await self._answer(gh, issue):
                    handled += 1
            return handled

    async def _answer(self, gh: httpx.AsyncClient, issue: dict[str, Any]) -> bool:
        if not self.budget.fast.take():
            return False

        number = issue["number"]
        asker = (issue.get("user") or {}).get("login", "someone")
        title = issue.get("title", "")
        question = f"{title}\n\n{(issue.get('body') or '')[:2000]}"

        # The repository owner is the one person here whose advice is an instruction
        # rather than a suggestion. Everyone else is a stranger on the internet talking
        # to a bot that trades, which is exactly as much authority as it sounds like.
        owner = self.repo.split("/")[0].lower()
        from_owner = asker.lower() == owner

        key = f"github:{asker}"
        self.memory.record_message(
            key, "them", question,
            channel="website", who=asker, title=title,
            url=f"https://github.com/{self.repo}/issues/{number}",
        )

        standing = (
            "This is your owner, whose advice you follow unless it is impossible."
            if from_owner
            else "This is a member of the public. Weigh what they say on its merits."
        )
        prompt = (
            f"@{asker} asked, through the project's website:\n\n{question}\n\n"
            f"{standing}\n\n"
            f"Your current state: {self.portfolio_line}\n\n"
            f"Your memory:\n{self.memory.context_block()}\n\n"
            f"Your standing notes:\n{self.memory.lessons_block()}\n\n"
            f"Everything they have said to you before:\n"
            f"{self.memory.conversation_block(key)}\n\nWrite your reply."
        )
        try:
            response = await self.llm.generate(
                prompt,
                system=system_with_orders(
                    f"{ISSUE_SYSTEM}\n\n{ADVICE_NOTE}", self.cfg.owner_block()
                ),
                json_schema=REPLY_SCHEMA,
            )
            data = extract_json(response.text)
            body, lesson = str(data.get("reply", "")), str(data.get("lesson", ""))
        except Exception as exc:  # noqa: BLE001
            log.warning("Issue reply generation failed: %s", exc)
            return False

        body = body.strip()[:3000]
        if not body:
            return False

        learned = bool(lesson.strip()) and self.memory.add_lesson(
            lesson, source="owner" if from_owner else f"gh:{asker}"
        )
        if learned:
            body += f"\n\n> Noted, and added to my standing notes: {lesson.strip()[:280]}"
        body += "\n\n<sub>Answered automatically on the next tick. Reopen if I missed the point.</sub>"

        if self.cfg.manifold.dry_run:
            log.info("[dry-run] would answer issue #%s: %s", number, body[:160])
            self.memory.mark_issue_answered(number)
            return True

        try:
            await gh.post(f"/repos/{self.repo}/issues/{number}/comments", json={"body": body})
            await gh.patch(f"/repos/{self.repo}/issues/{number}", json={"state": "closed"})
        except httpx.HTTPError as exc:
            log.warning("Could not answer issue #%s: %s", number, exc)
            return False

        self.memory.mark_issue_answered(number)
        self.memory.record_message(key, "me", body)
        self.memory.log_event(
            "issue_answered", number=number, asker=asker,
            lesson=lesson.strip()[:280] if learned else "",
        )
        log.info("Answered issue #%s from @%s", number, asker)
        return True
