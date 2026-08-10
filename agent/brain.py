"""Evaluate one market and act on it: research, decide, size, execute, explain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .config import Config
from .llm import LLMClient, extract_json
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Comment, Decision, Market, Position, Sizing
from .prompts import (
    DECISION_SCHEMA,
    RESEARCH_SYSTEM,
    TRADER_SYSTEM,
    build_decision_prompt,
    build_research_prompt,
    system_with_orders,
)
from . import websearch
from .sizing import RiskEngine

log = logging.getLogger(__name__)

NO_RESEARCH = (
    "No live search was available, so you are working from training data that may be "
    "months stale. Treat anything time-sensitive as unknown."
)


class CallBudget:
    """Hard cap on LLM calls per tick. A 5-minute cron makes 288 ticks a day."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def take(self, n: int = 1) -> bool:
        if self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def spent(self) -> bool:
        return self.used >= self.limit


@dataclass
class Evaluation:
    market: Market
    decision: Decision | None
    sizing: Sizing | None
    executed: bool
    detail: str


class Brain:
    def __init__(
        self,
        cfg: Config,
        *,
        client: ManifoldClient,
        llm: LLMClient,
        memory: Memory,
        risk: RiskEngine,
        budget: CallBudget,
        user_id: str,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.llm = llm
        self.memory = memory
        self.risk = risk
        self.budget = budget
        self.user_id = user_id

    async def evaluate(self, market: Market, *, trigger: str) -> Evaluation:
        market = await self.client.market(market.id)
        self.memory.mark_seen(market.id)
        if market.is_resolved:
            return Evaluation(market, None, None, False, "resolved while queued")

        comments = await self._safe_comments(market.id)
        positions = await self.client.positions(self.user_id)
        position = next((p for p in positions if p.contract_id == market.id), None)

        research = await self._research(market)
        prompt = build_decision_prompt(
            market=market,
            comments=comments,
            research=research,
            memory=self.memory.context_block(),
            market_note=self.memory.note_for(market.id),
            position=position,
            today=_today(),
            trigger=trigger,
        )

        if not self.budget.take():
            return Evaluation(market, None, None, False, "out of LLM budget for this tick")

        try:
            response = await self.llm.generate(
                prompt,
                system=system_with_orders(TRADER_SYSTEM, self.cfg.owner_block()),
                json_schema=DECISION_SCHEMA,
            )
            decision = Decision.parse(extract_json(response.text))
        except Exception as exc:  # noqa: BLE001 - a bad response must not kill the tick
            log.warning("Decision failed on %s: %s", market.slug, exc)
            self.memory.log_event("decision_error", market=market.slug, error=str(exc))
            return Evaluation(market, None, None, False, f"model error: {exc}")

        self.memory.set_note(market.id, decision.memory_note)
        log.info(
            "%s | market %.0f%% | model %.0f%% (%s) | %s",
            market.question[:60], market.probability * 100,
            decision.probability * 100, decision.confidence, decision.action,
        )

        if decision.action == "sell":
            return await self._sell(market, decision, position)

        sizing = await self._size(market, decision, position, len(positions))
        if not sizing.is_trade:
            self.memory.log_event(
                "no_trade",
                market=market.slug, question=market.question, url=market.url,
                market_prob=market.probability, model_prob=decision.probability,
                confidence=decision.confidence, reason=sizing.reason,
                thinking=decision.comment, uncertainty=decision.key_uncertainty,
                evidence_for=decision.evidence_for[:4],
                evidence_against=decision.evidence_against[:4],
                resolution_risk=decision.resolution_risk,
            )
            return Evaluation(market, decision, sizing, False, sizing.reason)

        return await self._execute(market, decision, sizing)

    # -- steps ------------------------------------------------------------

    async def _research(self, market: Market) -> str:
        """Grounded generation where the provider supports it, keyless search otherwise.

        Either way this costs exactly one LLM call, so the budget maths does not change
        when you switch providers.
        """
        if not self.cfg.llm.use_search or not self.budget.take():
            return NO_RESEARCH

        prompt = build_research_prompt(
            question=market.question, description=market.description, today=_today()
        )

        # Native grounding first where it exists. Its free quota is far smaller than
        # plain generation's and runs out long before the model does, so a failure
        # here falls through to keyless search rather than giving up on research.
        if self.llm.supports_search:
            try:
                response = await self.llm.generate(
                    prompt, system=RESEARCH_SYSTEM, grounded=True, attempts=1
                )
                body = response.text.strip()
                if len(body) >= 40:
                    if response.citations:
                        body += "\n\nSources:\n" + "\n".join(
                            f"- {c}" for c in response.citations[:8]
                        )
                    return body
            except Exception as exc:  # noqa: BLE001
                log.info("Grounded search unavailable (%s), using web search", exc)

        snippets = await websearch.search(market.question, self.cfg.llm.search_results)
        if not snippets:
            return NO_RESEARCH
        try:
            response = await self.llm.generate(
                prompt
                + "\n\nSearch results follow. Use only these, and say so plainly if they"
                f" do not actually address the question.\n\n{snippets}",
                system=RESEARCH_SYSTEM,
            )
        except Exception as exc:  # noqa: BLE001 - research is best effort
            log.warning("Research failed on %s: %s", market.slug, exc)
            return NO_RESEARCH

        body = response.text.strip()
        return body if len(body) >= 40 else NO_RESEARCH

    async def _size(
        self,
        market: Market,
        decision: Decision,
        position: Position | None,
        open_positions: int,
    ) -> Sizing:
        portfolio = await self.client.portfolio(self.user_id)
        balance = float(portfolio.get("balance") or 0.0)
        net_worth = (
            balance
            + float(portfolio.get("investmentValue") or 0.0)
            - float(portfolio.get("loanTotal") or 0.0)
        )

        sizing = self.risk.size(
            decision=decision,
            market=market,
            position=position,
            net_worth=net_worth,
            balance=balance,
            budget_spent=self.memory.budget_spent(),
            open_position_count=open_positions,
        )
        if not sizing.is_trade:
            return sizing

        async def probe(amount: float) -> float | None:
            try:
                return await self.client.probe_impact(
                    contract_id=market.id, amount=amount, outcome=sizing.outcome
                )
            except ManifoldError as exc:
                log.warning("Impact probe failed on %s: %s", market.slug, exc)
                return None

        return await self.risk.fit_to_impact(sizing, market=market, probe=probe)

    async def _execute(
        self, market: Market, decision: Decision, sizing: Sizing
    ) -> Evaluation:
        kind = "market" if sizing.limit_prob is None else f"limit @ {sizing.limit_prob:.2f}"
        log.info(
            "BET M$%.0f %s on %s (%s) [%s]%s",
            sizing.amount, sizing.outcome, market.slug, kind, sizing.reason,
            " [dry-run]" if self.cfg.manifold.dry_run else "",
        )

        try:
            result = await self.client.place_bet(
                contract_id=market.id,
                amount=sizing.amount,
                outcome=sizing.outcome,
                limit_prob=sizing.limit_prob,
                expires_at=sizing.expires_ms,
            )
        except ManifoldError as exc:
            self.memory.log_event("bet_failed", market=market.slug, error=str(exc))
            return Evaluation(market, decision, sizing, False, f"bet rejected: {exc}")

        if not self.cfg.manifold.dry_run:
            self.memory.record_spend(sizing.amount)

        bet_id = (result or {}).get("id")
        if bet_id and sizing.limit_prob is not None:
            self.memory.track_order(
                bet_id,
                {
                    "contract_id": market.id,
                    "question": market.question,
                    "slug": market.slug,
                    "outcome": sizing.outcome,
                    "amount": sizing.amount,
                    "limit_prob": sizing.limit_prob,
                    "placed_ms": int(time.time() * 1000),
                },
            )

        self.memory.log_event(
            "bet",
            market=market.slug, question=market.question, url=market.url,
            outcome=sizing.outcome, amount=sizing.amount, limit_prob=sizing.limit_prob,
            market_prob=market.probability, model_prob=decision.probability,
            confidence=decision.confidence, conviction=sizing.conviction,
            reason=sizing.reason, bet_id=bet_id, dry_run=self.cfg.manifold.dry_run,
            thinking=decision.comment, uncertainty=decision.key_uncertainty,
            evidence_for=decision.evidence_for[:4],
            evidence_against=decision.evidence_against[:4],
            resolution_risk=decision.resolution_risk,
        )

        await self._maybe_comment(market, decision, sizing)
        return Evaluation(market, decision, sizing, True, "placed")

    async def _sell(
        self, market: Market, decision: Decision, position: Position | None
    ) -> Evaluation:
        if position is None:
            return Evaluation(market, decision, None, False, "sell with no position")

        log.info("SELL %s on %s (model %.0f%%)", position.side, market.slug,
                 decision.probability * 100)
        try:
            await self.client.sell_shares(market_id=market.id, outcome=position.side)
        except ManifoldError as exc:
            self.memory.log_event("sell_failed", market=market.slug, error=str(exc))
            return Evaluation(market, decision, None, False, f"sell rejected: {exc}")

        self.memory.log_event(
            "sell",
            market=market.slug, question=market.question, url=market.url,
            side=position.side, profit=position.profit,
            model_prob=decision.probability, market_prob=market.probability,
            thinking=decision.comment, dry_run=self.cfg.manifold.dry_run,
        )
        return Evaluation(market, decision, None, True, "sold")

    async def _maybe_comment(
        self, market: Market, decision: Decision, sizing: Sizing
    ) -> None:
        """Explain a position publicly, at most once per market.

        Comments cost M$1 each, so this is reserved for positions large enough that
        the fee is noise. Subsequent trades on the same market stay silent: the
        original comment thread is still there, and the agent still answers replies
        to it for free.
        """
        cfg = self.cfg.social
        if not cfg.comment_decisions:
            return
        if sizing.amount < cfg.comment_min_amount:
            return
        if self.memory.has_commented_on(market.id):
            return

        kind = "Limit order" if sizing.limit_prob is not None else "Bought"
        for_side = "; ".join(decision.evidence_for[:3]) or "nothing specific"
        against = "; ".join(decision.evidence_against[:3]) or "nothing specific"
        body = (
            f"**{kind} M${sizing.amount:.0f} {sizing.outcome}.** My estimate: "
            f"**{decision.probability:.0%}** vs market {market.probability:.0%}, "
            f"confidence {decision.confidence}.\n\n"
            f"{decision.comment.strip()[:600]}\n\n"
            f"For: {for_side[:400]}\n\n"
            f"Against: {against[:400]}\n\n"
            f"Biggest unknown: {decision.key_uncertainty[:300]}\n\n"
            f"_Automated. Reply and I will answer._"
        )
        try:
            result = await self.client.post_comment(contract_id=market.id, content=body)
        except ManifoldError as exc:
            log.warning("Comment failed on %s: %s", market.slug, exc)
            return
        self.memory.mark_commented_on(market.id)
        comment_id = (result or {}).get("id")
        if comment_id:
            self.memory.remember_comment(comment_id, market.id)

    async def _safe_comments(self, market_id: str) -> list[Comment]:
        try:
            return await self.client.comments(market_id)
        except ManifoldError as exc:
            log.warning("Could not load comments for %s: %s", market_id, exc)
            return []


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())
