"""Domain objects. Manifold returns loose JSON, these are the parts we rely on."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

MS_PER_DAY = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


def tiptap_text(node: Any) -> str:
    """Flatten Manifold's TipTap rich-text JSON into plain text.

    Comments and market descriptions come back as either a plain string or a
    ProseMirror document, so callers get one shape regardless.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(tiptap_text(n) for n in node)
    if not isinstance(node, dict):
        return str(node)

    kind = node.get("type")
    if kind == "text":
        return node.get("text", "")
    if kind == "mention":
        return "@" + node.get("attrs", {}).get("label", "")
    if kind == "hardBreak":
        return "\n"

    inner = tiptap_text(node.get("content"))
    if kind in ("paragraph", "heading", "listItem", "blockquote"):
        return inner + "\n"
    return inner


@dataclass
class Market:
    id: str
    question: str
    slug: str
    url: str
    creator_username: str
    outcome_type: str
    mechanism: str
    probability: float
    volume: float
    liquidity: float
    unique_bettors: int
    close_time: int | None
    created_time: int
    is_resolved: bool
    description: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Market:
        description = data.get("textDescription") or tiptap_text(data.get("description"))
        return cls(
            id=data["id"],
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            url=data.get("url", ""),
            creator_username=data.get("creatorUsername", ""),
            outcome_type=data.get("outcomeType", ""),
            mechanism=data.get("mechanism", ""),
            # The markets and positions endpoints disagree on the name of the current
            # price: /markets says `probability`, the contract-metrics payload says
            # `prob`. Reading only the first left every held position looking like it
            # sat at 0%, which is exactly the field the agent uses to decide whether a
            # position has gone wrong, so it never saw one that had.
            probability=float(data.get("probability") or data.get("prob") or 0.0),
            volume=float(data.get("volume") or 0.0),
            liquidity=float(data.get("totalLiquidity") or 0.0),
            unique_bettors=int(data.get("uniqueBettorCount") or 0),
            close_time=data.get("closeTime"),
            created_time=int(data.get("createdTime") or 0),
            is_resolved=bool(data.get("isResolved")),
            description=description.strip(),
            raw=data,
        )

    @property
    def is_binary(self) -> bool:
        return self.outcome_type == "BINARY" and self.mechanism == "cpmm-1"

    @property
    def days_to_close(self) -> float:
        if not self.close_time:
            return float("inf")
        return max(0.0, (self.close_time - now_ms()) / MS_PER_DAY)


@dataclass
class Comment:
    id: str
    contract_id: str
    user_id: str
    username: str
    display_name: str
    text: str
    created_time: int
    reply_to_id: str | None

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Comment:
        return cls(
            id=data["id"],
            contract_id=data.get("contractId", ""),
            user_id=data.get("userId", ""),
            username=data.get("userUsername", ""),
            display_name=data.get("userName", ""),
            text=tiptap_text(data.get("content")).strip(),
            created_time=int(data.get("createdTime") or 0),
            reply_to_id=data.get("replyToCommentId"),
        )


@dataclass
class Position:
    contract_id: str
    question: str
    slug: str
    has_yes: float
    has_no: float
    invested: float
    payout: float
    profit: float
    last_prob: float
    days_to_close: float
    creator_username: str = ""
    is_closed: bool = False

    @property
    def side(self) -> str:
        return "YES" if self.has_yes >= self.has_no else "NO"

    @property
    def shares(self) -> float:
        return max(self.has_yes, self.has_no)

    @property
    def url(self) -> str:
        if not self.slug:
            return ""
        return f"https://manifold.markets/{self.creator_username or 'manifold'}/{self.slug}"

    @property
    def tradable(self) -> bool:
        """Closed markets still show as positions but the API refuses to sell them."""
        return not self.is_closed and self.days_to_close > 0

    @property
    def value(self) -> float:
        """What the position is worth now: shares pay out 1 mana each if they land."""
        price = self.last_prob if self.side == "YES" else 1.0 - self.last_prob
        return self.shares * price


@dataclass
class Decision:
    """What the model decided about one market."""

    probability: float
    confidence: str
    action: str  # buy_yes | buy_no | sell | hold
    evidence_for: list[str]
    evidence_against: list[str]
    key_uncertainty: str
    resolution_risk: str
    comment: str
    memory_note: str
    lesson: str = ""
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def parse(cls, data: dict[str, Any]) -> Decision:
        def as_list(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)] if value else []

        prob = float(data.get("probability", 0.5))
        return cls(
            probability=min(0.99, max(0.01, prob)),
            confidence=str(data.get("confidence", "low")).lower(),
            action=str(data.get("action", "hold")).lower(),
            evidence_for=as_list(data.get("evidence_for")),
            evidence_against=as_list(data.get("evidence_against")),
            key_uncertainty=str(data.get("key_uncertainty", "")),
            resolution_risk=str(data.get("resolution_risk", "")),
            comment=str(data.get("comment", "")),
            memory_note=str(data.get("memory_note", "")),
            lesson=str(data.get("lesson", "")),
            raw=data,
        )


@dataclass
class Sizing:
    """Result of running a Decision through the risk rules."""

    amount: float
    outcome: str  # YES | NO
    limit_prob: float | None
    expires_ms: int | None
    reason: str
    conviction: bool = False

    @property
    def is_trade(self) -> bool:
        return self.amount > 0
