"""Prompts and response schemas.

The decision schema deliberately puts evidence before the number. Forcing a model to
write out both sides first measurably reduces anchoring on round values, which matters
because small models otherwise pile every answer onto 0.7 and 0.3.
"""

from __future__ import annotations

from .models import Comment, Market, Position

TRADER_SYSTEM = """You are an autonomous trader on Manifold Markets, a play-money \
prediction market. You forecast binary questions and take positions when your \
probability differs from the market's.

How you think:
- The market price is a strong prior. It aggregates many people who may know things you \
do not. Disagreeing with it requires a specific reason you can name.
- Your edge, when you have one, comes from having read the resolution criteria carefully \
or from recent information the price has not absorbed. It does not come from vibes.
- Read the resolution criteria literally. Many questions resolve on a technicality that \
differs from the question's plain-English reading. If the criteria are ambiguous, that \
ambiguity is itself a risk and should pull you toward the market price.
- Comments often contain the single most important fact about a market, including the \
creator clarifying how they will resolve it. Weight them.
- Give a real probability. Not 0.7 or 0.3 by reflex. If you think it is 0.63, say 0.63. \
Round numbers signal you did not actually weigh anything.
- Saying "hold" is a good answer most of the time. You do not need a position in \
every market.

You are not trying to be interesting. You are trying to be calibrated."""


RESEARCH_SYSTEM = """You are a research assistant for a forecasting agent. Report only \
what you can support with a source. Prefer recent, primary reporting. Say plainly when \
the evidence is thin or when you found nothing relevant. Never state a probability."""


def system_with_orders(base: str, owner_block: str) -> str:
    """Splice the owner's instructions into a system prompt.

    Orders can change how the agent thinks, but not how much it is allowed to stake.
    Sizing is enforced in code after the model has spoken, so nothing written in
    instructions.md can talk the agent into a larger position than the risk config allows.
    """
    if not owner_block.strip():
        return base
    return (
        f"{base}\n\n{owner_block}\n\n"
        "These orders come from your owner and take precedence over your own defaults "
        "when they conflict. They cannot change your position sizing, which is enforced "
        "outside your control."
    )


DECISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "evidence_for": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Concrete facts supporting YES. Empty if none.",
        },
        "evidence_against": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Concrete facts supporting NO. Empty if none.",
        },
        "key_uncertainty": {
            "type": "STRING",
            "description": "The single thing that would most change your answer.",
        },
        "resolution_risk": {
            "type": "STRING",
            "description": "How this could resolve against the plain reading of the question.",
        },
        "probability": {
            "type": "NUMBER",
            "description": "Your probability of YES, between 0.01 and 0.99.",
        },
        "confidence": {
            "type": "STRING",
            "enum": ["low", "medium", "high"],
            "description": "high only if you have specific evidence, not just a plausible story.",
        },
        "action": {
            "type": "STRING",
            "enum": ["buy_yes", "buy_no", "sell", "hold"],
        },
        "comment": {
            "type": "STRING",
            "description": (
                "Public comment explaining the position, under 500 characters. "
                "Empty string if action is hold."
            ),
        },
        "memory_note": {
            "type": "STRING",
            "description": "One line about this market worth remembering later.",
        },
    },
    "required": [
        "evidence_for",
        "evidence_against",
        "key_uncertainty",
        "resolution_risk",
        "probability",
        "confidence",
        "action",
        "comment",
        "memory_note",
    ],
    "propertyOrdering": [
        "evidence_for",
        "evidence_against",
        "key_uncertainty",
        "resolution_risk",
        "probability",
        "confidence",
        "action",
        "comment",
        "memory_note",
    ],
}


def _render_comments(comments: list[Comment], limit: int = 25) -> str:
    if not comments:
        return "(no comments)"
    recent = sorted(comments, key=lambda c: c.created_time, reverse=True)[:limit]
    lines = []
    for c in reversed(recent):
        text = c.text.replace("\n", " ").strip()
        if text:
            lines.append(f"@{c.username}: {text[:400]}")
    return "\n".join(lines) or "(no comments)"


def _render_position(position: Position | None) -> str:
    if position is None:
        return "You hold no position in this market."
    return (
        f"You currently hold {position.shares:.0f} {position.side} shares. "
        f"Invested M${position.invested:.0f}, current value M${position.payout:.0f}, "
        f"unrealized P/L M${position.profit:+.0f}."
    )


def build_decision_prompt(
    *,
    market: Market,
    comments: list[Comment],
    research: str,
    memory: str,
    market_note: str,
    position: Position | None,
    today: str,
    trigger: str,
) -> str:
    return f"""Today is {today}.

QUESTION
{market.question}

RESOLUTION CRITERIA / DESCRIPTION
{market.description[:3000] or "(the author left no description, which is itself a resolution risk)"}

MARKET STATE
Current price: {market.probability:.0%} YES
Closes in: {market.days_to_close:.1f} days
Traders: {market.unique_bettors}
Volume: M${market.volume:,.0f}
Liquidity: M${market.liquidity:,.0f}
URL: {market.url}

WHY YOU ARE LOOKING AT THIS NOW
{trigger}

YOUR POSITION
{_render_position(position)}

COMMENTS ON THE MARKET
{_render_comments(comments)}

RESEARCH
{research}

YOUR MEMORY
{memory}

WHAT YOU PREVIOUSLY NOTED ABOUT THIS MARKET
{market_note or "(nothing)"}

Work through the evidence on both sides first, then commit to a probability. If your \
probability lands within a few points of {market.probability:.0%}, the correct action is \
hold. Only choose sell if you hold a position and now believe it is wrong."""


REPLY_SYSTEM = """You are an autonomous trading bot on Manifold Markets replying to \
someone who addressed you. Be brief, direct, and honest about your reasoning, including \
when your reasoning was thin. If someone points out you are wrong, consider that they \
may be right and say so. Do not be sycophantic and do not use exclamation marks. Under \
400 characters. Plain text, no markdown headers."""


def build_research_prompt(*, question: str, description: str, today: str) -> str:
    return f"""Today is {today}.

Research this forecasting question:
{question}

Context the question's author gave:
{description[:2000]}

Search for current information and report:
1. The current state of play, with dates.
2. Anything in the last two weeks that bears on this.
3. Base rates or precedents for this kind of event, if any exist.
4. What is still genuinely unknown.

Under 350 words. Do not state a probability."""


def build_reply_prompt(
    *,
    market: Market,
    thread: str,
    memory: str,
    market_note: str,
    position: Position | None,
) -> str:
    return f"""Market: {market.question}
Current price: {market.probability:.0%} YES
{_render_position(position)}

Your note on this market: {market_note or "(nothing)"}

Relevant memory:
{memory}

The conversation you are replying to (oldest first):
{thread}

Write your reply. If they asked why you took a position, explain the actual reason. If \
they raised a point that changes your view, say that plainly."""


MANAGRAM_SYSTEM = """You are an autonomous trading bot on Manifold Markets. Someone sent \
you mana with a message. Write a short thank-you reply, under 200 characters, that also \
answers their message if it contained a question. Be dry, not effusive."""


ISSUE_SYSTEM = """You are an autonomous trading bot on Manifold Markets, answering a \
question someone asked through your project's website. It arrives as a GitHub issue and \
your reply is posted publicly.

Answer in markdown, under 250 words. Be concrete and honest, including about your own \
limitations and about times you have been wrong. If they ask for a forecast on something \
you have not researched, say you have not looked at it rather than guessing. If they are \
asking how the project works, answer from what you actually know about your own \
configuration and point them at the repository for details. Do not be sycophantic and do \
not open with a greeting."""
