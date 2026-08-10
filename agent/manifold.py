"""Async client for the Manifold Markets v0 API.

Docs: https://docs.manifold.markets/api
Rate limit is 500 requests/minute per IP, which we stay far under by polling.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .config import ManifoldConfig
from .models import Comment, Market, Position, now_ms

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}


class ManifoldError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Manifold API {status}: {body[:400]}")
        self.status = status
        self.body = body


class ManifoldClient:
    def __init__(self, cfg: ManifoldConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/") + "/v0",
            headers={
                "Authorization": f"Key {cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(30.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        attempts: int = 4,
    ) -> Any:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        delay = 1.0
        last: Exception | None = None

        for attempt in range(attempts):
            try:
                resp = await self._client.request(method, path, params=clean, json=json)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if resp.status_code < 400:
                    return resp.json() if resp.content else None
                if resp.status_code not in RETRY_STATUSES:
                    raise ManifoldError(resp.status_code, resp.text)
                last = ManifoldError(resp.status_code, resp.text)

            if attempt < attempts - 1:
                log.warning("%s %s failed (%s), retrying in %.0fs", method, path, last, delay)
                await asyncio.sleep(delay)
                delay *= 2

        assert last is not None
        raise last

    # -- reads ------------------------------------------------------------

    async def me(self) -> dict[str, Any]:
        return await self._request("GET", "/me")

    async def portfolio(self, user_id: str) -> dict[str, Any]:
        return await self._request("GET", "/get-user-portfolio", params={"userId": user_id})

    async def net_worth(self, user_id: str) -> float:
        """Balance plus the current value of open positions."""
        data = await self.portfolio(user_id)
        balance = float(data.get("balance") or 0.0)
        invested = float(data.get("investmentValue") or 0.0)
        loans = float(data.get("loanTotal") or 0.0)
        return balance + invested - loans

    async def balance(self, user_id: str) -> float:
        data = await self.portfolio(user_id)
        return float(data.get("balance") or 0.0)

    async def search_markets(
        self,
        *,
        term: str = "",
        sort: str = "close-date",
        filter: str = "open",
        contract_type: str = "BINARY",
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        data = await self._request(
            "GET",
            "/search-markets",
            params={
                "term": term,
                "sort": sort,
                "filter": filter,
                "contractType": contract_type,
                "limit": limit,
                "offset": offset,
            },
        )
        return [Market.parse(d) for d in (data or [])]

    async def market(self, market_id: str) -> Market:
        return Market.parse(await self._request("GET", f"/market/{market_id}"))

    async def market_probs(self, ids: list[str]) -> dict[str, float]:
        """Cheap bulk probability read, up to 100 ids per call."""
        out: dict[str, float] = {}
        for i in range(0, len(ids), 100):
            chunk = ids[i : i + 100]
            data = await self._request("GET", "/market-probs", params={"ids": ",".join(chunk)})
            if isinstance(data, dict):
                for key, value in data.items():
                    prob = value.get("prob") if isinstance(value, dict) else value
                    if prob is not None:
                        out[key] = float(prob)
        return out

    async def comments(self, contract_id: str, limit: int = 200) -> list[Comment]:
        data = await self._request(
            "GET", "/comments", params={"contractId": contract_id, "limit": limit}
        )
        return [Comment.parse(d) for d in (data or [])]

    async def my_bets(
        self, user_id: str, *, contract_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        return await self._request(
            "GET",
            "/bets",
            params={"userId": user_id, "contractId": contract_id, "limit": limit},
        ) or []

    async def open_limit_orders(self, user_id: str) -> list[dict[str, Any]]:
        bets = await self.my_bets(user_id, limit=500)
        return [
            b
            for b in bets
            if b.get("limitProb") is not None
            and not b.get("isFilled")
            and not b.get("isCancelled")
        ]

    async def positions(self, user_id: str, limit: int = 200) -> list[Position]:
        data = await self._request(
            "GET",
            "/get-user-contract-metrics-with-contracts",
            params={"userId": user_id, "limit": limit, "offset": 0},
        )
        if not data:
            return []

        contracts = {c["id"]: Market.parse(c) for c in data.get("contracts", [])}
        metrics_by_contract = data.get("metricsByContract", {})

        out: list[Position] = []
        for contract_id, metrics in metrics_by_contract.items():
            entry = metrics[0] if isinstance(metrics, list) else metrics
            if not entry:
                continue
            total = entry.get("totalShares") or {}
            has_yes = float(total.get("YES") or 0.0)
            has_no = float(total.get("NO") or 0.0)
            if has_yes < 1e-6 and has_no < 1e-6:
                continue
            market = contracts.get(contract_id)
            out.append(
                Position(
                    contract_id=contract_id,
                    question=market.question if market else contract_id,
                    slug=market.slug if market else "",
                    has_yes=has_yes,
                    has_no=has_no,
                    invested=float(entry.get("invested") or 0.0),
                    payout=float(entry.get("payout") or 0.0),
                    profit=float(entry.get("profit") or 0.0),
                    last_prob=market.probability if market else 0.0,
                    days_to_close=market.days_to_close if market else float("inf"),
                )
            )
        return out

    async def incoming_managrams(self, user_id: str, after_ms: int) -> list[dict[str, Any]]:
        data = await self._request(
            "GET",
            "/txns",
            params={"toId": user_id, "category": "MANA_PAYMENT", "limit": 50},
        )
        return [t for t in (data or []) if int(t.get("createdTime") or 0) > after_ms]

    # -- writes -----------------------------------------------------------

    async def place_bet(
        self,
        *,
        contract_id: str,
        amount: float,
        outcome: str,
        limit_prob: float | None = None,
        expires_at: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Place a market or limit order.

        dry_run is also forced on whenever the agent is in global dry-run mode, so a
        misconfigured caller cannot spend mana by accident.
        """
        body: dict[str, Any] = {
            "contractId": contract_id,
            "amount": round(amount, 2),
            "outcome": outcome,
            "dryRun": bool(dry_run or self.cfg.dry_run),
        }
        if limit_prob is not None:
            # The API only accepts two decimal places on limitProb.
            body["limitProb"] = round(min(0.99, max(0.01, limit_prob)), 2)
        if expires_at is not None:
            body["expiresAt"] = int(expires_at)
        return await self._request("POST", "/bet", json=body)

    async def probe_impact(
        self, *, contract_id: str, amount: float, outcome: str
    ) -> float | None:
        """Return the probability this order would leave behind, without executing it."""
        result = await self.place_bet(
            contract_id=contract_id, amount=amount, outcome=outcome, dry_run=True
        )
        if not isinstance(result, dict):
            return None
        after = result.get("probAfter")
        return float(after) if after is not None else None

    async def cancel_bet(self, bet_id: str) -> Any:
        if self.cfg.dry_run:
            log.info("[dry-run] would cancel limit order %s", bet_id)
            return None
        return await self._request("POST", f"/bet/cancel/{bet_id}")

    async def sell_shares(
        self, *, market_id: str, outcome: str, shares: float | None = None
    ) -> Any:
        if self.cfg.dry_run:
            log.info("[dry-run] would sell %s %s on %s", shares or "all", outcome, market_id)
            return None
        body: dict[str, Any] = {"outcome": outcome}
        if shares is not None:
            body["shares"] = round(shares, 4)
        return await self._request("POST", f"/market/{market_id}/sell", json=body)

    async def post_comment(
        self, *, contract_id: str, content: str, reply_to: str | None = None
    ) -> Any:
        """Post a markdown comment. Costs M$1 per the API docs."""
        if self.cfg.dry_run:
            log.info("[dry-run] would comment on %s: %s", contract_id, content[:200])
            return None
        body: dict[str, Any] = {"contractId": contract_id, "content": content}
        if reply_to:
            body["replyToCommentId"] = reply_to
        return await self._request("POST", "/comment", json=body)

    async def send_managram(self, *, to_ids: list[str], amount: float, message: str) -> Any:
        """Minimum transfer is M$10 per the API docs."""
        if self.cfg.dry_run:
            log.info("[dry-run] would managram %s to %s: %s", amount, to_ids, message)
            return None
        return await self._request(
            "POST",
            "/managram",
            json={"toIds": to_ids, "amount": round(amount, 2), "message": message},
        )
