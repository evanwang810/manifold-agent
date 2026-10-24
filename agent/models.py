"""Domain objects. Manifold returns loose JSON, these are the parts we rely on."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

MS_PER_DAY = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


def _answer_prob(answer: dict[str, Any]) -> float:
    """Full market payloads call it `probability`; the positions payload calls it `prob`."""
    return float(answer.get("probability") or answer.get("prob") or 0.0)


def logit(p: float) -> float:
    p = min(0.99, max(0.01, p))
    return math.log(p / (1.0 - p))


def logodds_gap(p: float, q: float) -> float:
    """Disagreement between two probabilities, measured in log-odds.

    Percentage points are the wrong unit at the extremes. 95% to 98% is the
    difference between one failure in twenty and one in fifty, and a 3-point gap
    there matters far more than 50% to 53%. Log-odds is looser at the tails and
    stricter in the middle, which also happens to be where the model's forecasts
    have been good and bad respectively.
    """
    return abs(logit(p) - logit(q))


def backs_favourite(p: float, q: float) -> bool:
    """True when the forecast agrees with the market's favourite and goes further.

    On the agent's resolved record to September 2026, bets on the favourite won 17 of
    20 and longshot bets against it won 13 of 70. Nearly all of the losses were the
    model talking itself into a longshot.
    """
    return (q >= 0.5 and p > q) or (q < 0.5 and p < q)


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
    # Set when this is one answer of a multiple-choice market, viewed as its own yes/no
    # question: "will this answer be the one?". Everything downstream treats it as a
    # binary market; only placing and closing orders need the answer id.
    answer_id: str = ""
    answer_text: str = ""

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
    def is_multiple_choice(self) -> bool:
        return self.mechanism == "cpmm-multi-1" and not self.answer_id

    @property
    def tradeable(self) -> bool:
        """A plain binary market, or one answer of a multiple-choice market."""
        return self.is_binary or (bool(self.answer_id) and self.mechanism == "cpmm-multi-1")

    @property
    def key(self) -> str:
        """Unique per tradeable thing. Answers of one market share a slug and an id."""
        return f"{self.slug}::{self.answer_id}" if self.answer_id else self.slug

    @property
    def contract_key(self) -> str:
        """Same shape as Position.key, for matching a market to what is held."""
        return f"{self.id}::{self.answer_id}" if self.answer_id else self.id

    def for_answer(self, answer_id: str) -> Market:
        """This market seen as the yes/no question "is it this answer?".

        The other answers are listed by name but not by price, so the forecast stays
        blind. Whether they are mutually exclusive is spelled out, because that changes
        what a probability for one of them means.
        """
        answers = self.raw.get("answers") or []
        answer = next((a for a in answers if a.get("id") == answer_id), None)
        if answer is None:
            raise ValueError(f"no answer {answer_id} on {self.slug}")
        exclusive = self.raw.get("shouldAnswersSumToOne", True) is not False
        others = [a.get("text", "") for a in answers if a.get("id") != answer_id][:25]
        text = str(answer.get("text", "")).strip()
        note = (
            f"\n\n[This is one answer of a multiple-choice market. You are forecasting "
            f"whether \"{text}\" is the answer. "
            + ("Exactly one answer resolves YES. " if exclusive
               else "Each answer resolves YES or NO on its own. ")
            + f"The other answers are: {'; '.join(o for o in others if o) or '(none)'}.]"
        )
        resolved = self.is_resolved or bool(answer.get("resolution"))
        return Market(
            id=self.id, question=f"{self.question} \u2192 {text}", slug=self.slug, url=self.url,
            creator_username=self.creator_username, outcome_type=self.outcome_type,
            mechanism=self.mechanism, probability=_answer_prob(answer),
            volume=float(answer.get("volume") or self.volume), liquidity=self.liquidity,
            unique_bettors=self.unique_bettors, close_time=self.close_time,
            created_time=self.created_time, is_resolved=resolved,
            description=self.description + note, raw=self.raw,
            answer_id=answer_id, answer_text=text,
        )

    def split(self, max_answers: int = 3, min_prob: float = 0.03) -> list[Market]:
        """The answers worth a look, most likely first, as yes/no sub-markets.

        "Other" is skipped (it resolves on whatever nobody listed), and so are answers
        pinned at the extremes, which have no room to be wrong in our favour.
        """
        answers = [
            a for a in (self.raw.get("answers") or [])
            if not a.get("isOther") and not a.get("resolution")
            and min_prob < _answer_prob(a) < 1 - min_prob
        ]
        answers.sort(key=lambda a: -_answer_prob(a))
        return [self.for_answer(a["id"]) for a in answers[:max_answers]]

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
    # Where the author stood. Manifold spells these "commentor".
    position_outcome: str = ""
    position_shares: float = 0.0
    bet_outcome: str = ""
    bet_amount: float = 0.0
    likes: int = 0
    is_api: bool = False  # posted through the API, which on Manifold means a bot

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
            position_outcome=str(data.get("commentorPositionOutcome") or ""),
            position_shares=float(data.get("commentorPositionShares") or 0.0),
            bet_outcome=str(data.get("betOutcome") or ""),
            bet_amount=float(data.get("betAmount") or 0.0),
            likes=int(data.get("likes") or 0),
            is_api=bool(data.get("isApi")),
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
    close_time: int | None = None
    answer_id: str = ""  # set for one answer of a multiple-choice market

    @property
    def key(self) -> str:
        """Unique per holding: two answers of one market share a contract id."""
        return f"{self.contract_id}::{self.answer_id}" if self.answer_id else self.contract_id

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
