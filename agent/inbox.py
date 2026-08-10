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

from .brain import CallBudget
from .config import Config
from .llm import LLMClient
from .memory import Memory
from .prompts import ISSUE_SYSTEM, system_with_orders

log = logging.getLogger(__name__)

LABEL = "ask-the-bot"


class Inbox:
    def __init__(
        self,
        cfg: Config,
        *,
        llm: LLMClient,
        memory: Memory,
        budget: CallBudget,
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
                if self.budget.spent or handled >= self.cfg.social.max_replies_per_tick:
                    break
                if await self._answer(gh, issue):
                    handled += 1
            return handled

    async def _answer(self, gh: httpx.AsyncClient, issue: dict[str, Any]) -> bool:
        if not self.budget.take():
            return False

        number = issue["number"]
        asker = (issue.get("user") or {}).get("login", "someone")
        question = f"{issue.get('title', '')}\n\n{(issue.get('body') or '')[:2000]}"

        prompt = (
            f"@{asker} asked, through the project's website:\n\n{question}\n\n"
            f"Your current state: {self.portfolio_line}\n\n"
            f"Your memory:\n{self.memory.context_block()}\n\nWrite your reply."
        )
        try:
            response = await self.llm.generate(
                prompt, system=system_with_orders(ISSUE_SYSTEM, self.cfg.owner_block())
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Issue reply generation failed: %s", exc)
            return False

        body = response.text.strip()[:3000]
        if not body:
            return False
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
        self.memory.log_event("issue_answered", number=number, asker=asker)
        log.info("Answered issue #%s from @%s", number, asker)
        return True
