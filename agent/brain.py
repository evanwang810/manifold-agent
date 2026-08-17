"""Evaluate one market and act on it: research, decide, size, execute, explain."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

from .config import Config
from .llm import LLMClient, QuotaError, extract_json
from .manifold import ManifoldClient, ManifoldError
from .memory import Memory
from .models import Comment, Decision, Market, Position, Sizing
from .prompts import (
    DECISION_SCHEMA,
    QUERY_SCHEMA,
    QUERY_SYSTEM,
    RESEARCH_SYSTEM,
    SCREEN_SCHEMA,
    SCREEN_SYSTEM,
    TRADER_SYSTEM,
    build_decision_prompt,
    build_query_prompt,
    build_research_prompt,
    build_screen_prompt,
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
    """Two ceilings on one tier's LLM calls: this tick, and this day.

    The per-tick cap is really a rate limit, since ticks run a minute apart. The daily
    cap is the one that keeps a free tier alive to the end of the day, and it has to
    survive across ticks, so it is read from and written back to durable state.
    """

    def __init__(
        self,
        limit: int,
        *,
        daily_limit: int = 0,
        used_today: int = 0,
        day_fraction: float = 1.0,
        pace_burst: int = 0,
        on_take: Callable[[], None] | None = None,
    ) -> None:
        self.limit = limit
        self.daily_limit = daily_limit
        self.used = 0
        self.used_today = used_today
        self.day_fraction = day_fraction
        self.pace_burst = pace_burst
        self._on_take = on_take

    @property
    def paced_allowance(self) -> float:
        """How many calls the day is far enough along to justify having spent.

        A daily cap with no pacing is spent in the first ten minutes and then the agent
        is dark until midnight. This tracks the clock instead, with a small burst so it
        can still react to two things at once rather than trickling one call an hour.
        """
        if not (self.daily_limit and self.pace_burst):
            return float("inf")
        return self.daily_limit * self.day_fraction + self.pace_burst

    def take(self, n: int = 1) -> bool:
        if self.used + n > self.limit:
            return False
        if self.daily_limit and self.used_today + n > self.daily_limit:
            log.warning("Daily cap of %d calls reached, holding off", self.daily_limit)
            return False
        if self.used_today + n > self.paced_allowance:
            log.info(
                "Pacing: %d of %d daily calls used and the day is %.0f%% gone, waiting",
                self.used_today, self.daily_limit, self.day_fraction * 100,
            )
            return False
        self.used += n
        self.used_today += n
        if self._on_take:
            for _ in range(n):
                self._on_take()
        return True

    @property
    def spent(self) -> bool:
        if self.used >= self.limit:
            return True
        if self.daily_limit and self.used_today >= self.daily_limit:
            return True
        return self.used_today >= self.paced_allowance


@dataclass
class Budgets:
    """Separate ceilings, because the tiers do not cost the same.

    The fast model can be spent freely on screening. The deep model is rationed, which
    is the whole reason the screen exists.
    """

    fast: CallBudget
    chat: CallBudget
    deep: CallBudget

    @property
    def used(self) -> int:
        return self.fast.used + self.chat.used + self.deep.used


@dataclass
class Evaluation:
    market: Market
    decision: Decision | None
    sizing: Sizing | None
    executed: bool
    detail: str
    # Rejected by the cheap pass, so it cost no deep call and should not count against
    # the tick's evaluation budget. Screening is meant to buy more looks, not fewer.
    screened_out: bool = False


class Brain:
    def __init__(
        self,
        cfg: Config,
        *,
        client: ManifoldClient,
        fast: LLMClient,
        deep: LLMClient,
        memory: Memory,
        risk: RiskEngine,
        budget: Budgets,
        user_id: str,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.fast = fast
        self.deep = deep
        self.memory = memory
        self.risk = risk
        self.budget = budget
        self.user_id = user_id

    async def evaluate(
        self, market: Market, *, trigger: str, screen: bool = False
    ) -> Evaluation:
        market = await self.client.market(market.id)
        self.memory.mark_seen(market.id)
        if market.is_resolved:
            return Evaluation(market, None, None, False, "resolved while queued")

        comments = await self._safe_comments(market.id)

        # Screening only makes sense on markets we went looking for. A filled order or
        # a position in freefall is already news, and gets the deep model regardless.
        if screen and self.cfg.screen.enabled:
            passed, detail, quick = await self._screen(market, comments)
            # Both outcomes are logged with the screener's own number, so the pass rate
            # and the threshold that produced it can be measured later rather than
            # guessed at. Calibrating this on the deep model's gaps is a proxy at best.
            self.memory.log_event(
                "screened_in" if passed else "screened_out",
                market=market.slug, question=market.question, url=market.url,
                market_prob=market.probability, quick_prob=quick,
                gap=None if quick is None else round(abs(quick - market.probability), 3),
                threshold=self.cfg.screen.escalate_edge, reason=detail,
            )
            self.memory.observe(
                "screen",
                f"{'Escalated' if passed else 'Passed on'} \"{market.question[:70]}\" "
                f"at {market.probability:.0%}: {detail}",
            )
            if not passed:
                return Evaluation(
                    market, None, None, False, f"screened out: {detail}",
                    screened_out=True,
                )

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
            lessons=self.memory.lessons_block(),
            show_price=not self.cfg.forecast.blind,
        )

        if not self.budget.deep.take():
            return Evaluation(market, None, None, False, "out of deep model budget")

        try:
            response = await self.deep.generate(
                prompt,
                system=system_with_orders(TRADER_SYSTEM, self.cfg.owner_block()),
                json_schema=DECISION_SCHEMA,
            )
            decision = Decision.parse(extract_json(response.text))
        except Exception as exc:  # noqa: BLE001 - a bad response must not kill the tick
            log.warning("Decision failed on %s: %s", market.slug, exc)
            self.memory.log_event("decision_error", market=market.slug, error=str(exc))
            return Evaluation(market, None, None, False, f"model error: {exc}")

        self.memory.set_note(
            market.id, decision.memory_note, question=market.question, url=market.url
        )
        if decision.lesson.strip():
            self.memory.add_lesson(decision.lesson, source="self")
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
            self.memory.observe(
                "analysis",
                f"Analysed \"{market.question[:70]}\": market {market.probability:.0%}, "
                f"my {decision.probability:.0%} ({decision.confidence}). No trade, "
                f"{sizing.reason}.",
            )
            return Evaluation(market, decision, sizing, False, sizing.reason)

        return await self._execute(market, decision, sizing)

    # -- steps ------------------------------------------------------------

    async def _screen(
        self, market: Market, comments: list[Comment]
    ) -> tuple[bool, str, float | None]:
        """Cheap first pass. True means the deep model should look at this properly.

        The screener forecasts blind like the trader does, so its number can be compared
        to the price the same way. Two independent reasons to escalate: it disagrees
        with the market, or it says the question has something in it worth reading
        carefully. Either is enough, because the cost of a wasted deep call is one call
        and the cost of skipping a mispriced market is the entire point of the project.
        """
        if not self.budget.fast.take():
            # The deep model is the scarce one. If the cheap pass cannot run, skipping
            # the market is right: escalating unscreened is how the quota disappears.
            return False, "screen skipped, no fast budget", None

        try:
            response = await self.fast.generate(
                build_screen_prompt(market=market, comments=comments, today=_today()),
                system=SCREEN_SYSTEM,
                json_schema=SCREEN_SCHEMA,
                attempts=2,
            )
            data = extract_json(response.text)
        except Exception as exc:  # noqa: BLE001
            log.info("Screen failed on %s (%s), skipping it", market.slug, exc)
            return False, "screen failed", None

        rough = min(0.99, max(0.01, float(data.get("probability", 0.5))))
        gap = abs(rough - market.probability)
        why = str(data.get("why", ""))[:200]
        interesting = bool(data.get("worth_a_look"))

        if gap >= self.cfg.screen.escalate_edge:
            return True, f"quick estimate {rough:.0%} vs market {market.probability:.0%}", rough
        if interesting:
            return True, f"flagged: {why}", rough
        return False, f"quick estimate {rough:.0%} agrees with market. {why}", rough

    async def _research(self, market: Market) -> str:
        """Grounded generation where the provider supports it, keyless search otherwise.

        Either way this costs exactly one LLM call, so the budget maths does not change
        when you switch providers.
        """
        if not self.cfg.llm.fast.use_search or not self.budget.fast.take():
            return NO_RESEARCH

        prompt = build_research_prompt(
            question=market.question, description=market.description, today=_today()
        )

        # Native grounding first where it exists. Its free quota is far smaller than
        # plain generation's and runs out long before the model does, so a failure
        # here falls through to keyless search rather than giving up on research.
        # Grounding quota is separate from generation quota and much smaller. Once it is
        # gone it is gone for the rest of the day, so remember that instead of
        # rediscovering it on every single research call: each rediscovery cost a wasted
        # request and a throttle wait before falling through to the search below anyway.
        if self.fast.supports_search and not self.memory.grounding_exhausted():
            try:
                response = await self.fast.generate(
                    prompt, system=RESEARCH_SYSTEM, grounded=True, attempts=1
                )
                body = response.text.strip()
                if len(body) >= 40:
                    if response.citations:
                        body += "\n\nSources:\n" + "\n".join(
                            f"- {c}" for c in response.citations[:8]
                        )
                    return body
            except QuotaError as exc:
                self.memory.mark_grounding_exhausted()
                log.info("Grounding quota gone for today (%s), using web search", exc)
            except Exception as exc:  # noqa: BLE001
                log.info("Grounded search unavailable (%s), using web search", exc)

        queries = await self._plan_queries(market)
        snippets = await websearch.search_many(queries, self.cfg.search)
        if not snippets:
            return NO_RESEARCH
        try:
            response = await self.fast.generate(
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

    async def _plan_queries(self, market: Market) -> list[str]:
        """Let the model write its own search queries.

        Only the fallback path needs this. A provider with native search already picks
        its own queries inside the grounded call, so doing it here as well would spend
        a request to duplicate work that already happened.

        The market question is kept as the last query regardless. The planner sometimes
        drops the one detail that actually identified the event, and a bad extra query
        costs nothing next to missing the obvious one.
        """
        plan = self.cfg.search.plan_queries
        if plan < 1 or not self.budget.fast.take():
            return [market.question]
        try:
            response = await self.fast.generate(
                build_query_prompt(
                    question=market.question,
                    description=market.description,
                    today=_today(),
                    count=plan,
                ),
                system=QUERY_SYSTEM,
                json_schema=QUERY_SCHEMA,
                attempts=1,
            )
            raw = extract_json(response.text).get("queries")
            queries = [str(q).strip() for q in raw if str(q).strip()][:plan] \
                if isinstance(raw, list) else []
        except Exception as exc:  # noqa: BLE001 - fall back to the question itself
            log.info("Query planning failed (%s), searching the question as written", exc)
            return [market.question]

        if queries:
            log.info("Searching %s for: %s", market.slug, "; ".join(queries))
        return [*queries, market.question]

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

        self.memory.observe(
            "trade",
            f"{'Limit order' if sizing.limit_prob else 'Bought'} M${sizing.amount:.0f} "
            f"{sizing.outcome} on \"{market.question[:70]}\" at "
            f"{market.probability:.0%}, my estimate {decision.probability:.0%} "
            f"({decision.confidence}). Because: {decision.comment[:450]}",
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
        self.memory.observe(
            "trade",
            f"Sold out of \"{market.question[:70]}\" ({position.side}) for "
            f"M${position.profit:+.0f}. Now think {decision.probability:.0%} against "
            f"market {market.probability:.0%}.",
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
