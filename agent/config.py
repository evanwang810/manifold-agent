"""Typed configuration.

Settings live in the committed config.toml. Secrets only ever come from the
environment, so there is no path by which a key ends up in git history.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


@dataclass
class ManifoldConfig:
    dry_run: bool = True
    api_key: str = ""
    base_url: str = "https://api.manifold.markets"


@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = "gemini-3.5-flash-lite"
    # Tried in order when the preferred model is out of quota or refuses the key.
    # Free daily quotas are small, and a lesser model beats going dark until midnight.
    fallbacks: list[str] = field(default_factory=list)
    # Grounded search has a separate, much smaller free quota than plain generation.
    # Turning it off keeps the agent running on the comment thread and the price alone.
    use_search: bool = True
    search_results: int = 6
    temperature: float = 0.3
    timeout_seconds: int = 90
    base_url: str = ""
    # Which environment variable holds this tier's key. Two tiers on two providers
    # means two keys, and neither is ever read from a file.
    key_env: str = "LLM_API_KEY"
    api_key: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass
class LLMTiers:
    """Three models, split by what the call is worth.

    `fast` screens candidate markets and does research. `chat` talks to people and
    rewrites memory, which is the work where a cheap conversational model is fine.
    `deep` is only ever asked the one question that decides money. Point them all at
    the same model if you would rather not bother.
    """

    fast: LLMConfig = field(default_factory=LLMConfig)
    chat: LLMConfig = field(default_factory=LLMConfig)
    deep: LLMConfig = field(default_factory=LLMConfig)

    def tiers(self) -> list[tuple[str, LLMConfig]]:
        return [("fast", self.fast), ("chat", self.chat), ("deep", self.deep)]


@dataclass
class BudgetConfig:
    max_evaluations_per_tick: int = 2
    # Sized against a small free tier. Ticks run every 60 seconds, so a per-tick cap is
    # effectively a per-minute rate limit: keep the sum of these under the tightest RPM
    # your provider gives you, with room to spare for a retry.
    max_fast_calls_per_tick: int = 4
    max_chat_calls_per_tick: int = 2
    max_deep_calls_per_tick: int = 1
    # The ceiling that actually matters on a free tier, tracked in state across ticks.
    # Set these below the provider's real daily quota, not at it: a retry or a probe
    # still costs a request.
    max_fast_calls_per_day: int = 150
    max_chat_calls_per_day: int = 40
    max_deep_calls_per_day: int = 18
    # Spend the daily allowance across the day rather than in the first ten minutes.
    # The burst is how far ahead of the clock a tier may run, so the agent can still
    # react to something now instead of trickling one call an hour.
    pace_burst: int = 3


@dataclass
class ScanConfig:
    min_minutes_between_scans: float = 60
    # Markets pulled per scan. Each costs one screen call, and at a 25% pass rate this
    # is what sets the deep model's daily load: 24 scans a day times this, quartered.
    candidates_per_scan: int = 4
    min_unique_bettors: int = 25
    min_volume: float = 3000
    max_days_to_resolve: float = 30
    min_hours_to_resolve: float = 6
    skip_if_seen_within_hours: float = 48


@dataclass
class RiskConfig:
    default_max_bet: float = 10
    default_max_fraction: float = 0.10
    min_edge: float = 0.04
    min_bet: float = 8
    kelly_fraction: float = 0.4

    conviction_max_fraction: float = 0.35
    conviction_min_volume: float = 5000
    conviction_max_days: float = 14
    conviction_min_edge: float = 0.12
    conviction_min_confidence: str = "medium"

    max_market_impact: float = 0.05
    max_share_of_volume: float = 0.05
    max_open_positions: int = 40
    daily_mana_budget: float = 500
    min_balance_reserve: float = 50
    time_decay_days: float = 14


@dataclass
class ScreenConfig:
    """The cheap first pass.

    Every candidate gets a rough blind forecast from the fast model. Only markets
    where that rough number disagrees with the price, or where the screener says the
    question deserves a real look, are escalated to the deep model.
    """

    enabled: bool = True
    # Escalate if the quick estimate is this far from the price. Calibrated against the
    # 17 forecasts on record: their market/model gaps put the 75th percentile at 0.081
    # and a 0.10 threshold passes 4 of 17, so this targets roughly a quarter of what is
    # scanned. Both screen outcomes are logged with the gap that produced them, so this
    # can be re-derived from the screener's own numbers once there are enough.
    escalate_edge: float = 0.10


@dataclass
class ForecastConfig:
    # Show the model the market price and it will hand the price back to you. Five of
    # six early forecasts landed within two points of market, so the price is hidden
    # during forecasting and only compared afterwards. Costs some absolute accuracy,
    # buys an estimate that is actually independent of the thing it is judged against.
    blind: bool = True


@dataclass
class WatchConfig:
    move_threshold: float = 0.08
    # Ticks run every minute, so a market in freefall would otherwise burn an
    # evaluation every tick on the way down.
    reevaluate_cooldown_minutes: float = 45


@dataclass
class SocialConfig:
    # Posting a comment costs M$1, so the agent only introduces itself on a position
    # big enough to be worth the fee, and only once per market.
    comment_decisions: bool = True
    comment_min_amount: float = 50
    reply_to_comments: bool = True
    reply_to_managrams: bool = False
    answer_github_issues: bool = True
    max_replies_per_tick: int = 2
    watch_last_n_comments: int = 10


@dataclass
class MemoryConfig:
    path: str = "state"
    compress_after_events: int = 60
    # Compress on time as well as on events, or a quiet day never compresses at all and
    # the journal just grows.
    compress_after_hours: float = 8
    # Running observations written by code on every tick. Free to record, so the agent
    # can afford to notice everything and decide later what was worth keeping.
    max_journal: int = 300
    journal_in_context: int = 30
    # Standing notes the agent has written for itself or been given by someone it
    # talked to. Kept small on purpose: this text is in front of every decision.
    max_lessons: int = 16
    # How many turns of one conversation it can still see when replying.
    conversation_depth: int = 12
    max_conversations: int = 40


@dataclass
class Config:
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    llm: LLMTiers = field(default_factory=LLMTiers)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    watch: WatchConfig = field(default_factory=WatchConfig)
    social: SocialConfig = field(default_factory=SocialConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    root: Path = Path(".")
    standing_orders: str = ""   # instructions.md, committed and version controlled
    one_off_order: str = ""     # OWNER_INSTRUCTION, set by a manual workflow run

    @property
    def state_dir(self) -> Path:
        path = self.root / self.memory.path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def owner_block(self) -> str:
        """The owner's voice, as the model sees it."""
        parts = []
        if self.standing_orders.strip():
            parts.append("STANDING ORDERS FROM YOUR OWNER\n" + self.standing_orders.strip())
        if self.one_off_order.strip():
            parts.append(
                "INSTRUCTION FOR THIS RUN ONLY\n" + self.one_off_order.strip()
            )
        return "\n\n".join(parts)


T = TypeVar("T")


def _build(cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate a dataclass from a dict, dropping keys it does not define."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]


def _read_instructions(path: Path) -> str:
    """Read instructions.md, dropping the explanatory preamble above the `---` rule.

    The preamble is for the human editing the file. Feeding it to the model would just
    be instructions about instructions.
    """
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    _, sep, body = text.partition("\n---\n")
    return (body if sep else text).strip()


def _load_llm(raw: dict[str, Any]) -> LLMTiers:
    """Build both tiers from `[llm]` plus the `[llm.fast]` / `[llm.deep]` overrides.

    Anything set directly under `[llm]` is the shared default, so a config that names
    no tiers at all still works and simply runs one model for everything.
    """
    shared = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    # An unnamed tier falls back to `fast`, so adding `[llm.chat]` is optional.
    fast = _build(LLMConfig, {**shared, **raw.get("fast", {})})
    return LLMTiers(
        fast=fast,
        chat=_build(LLMConfig, {**shared, **raw.get("chat", raw.get("fast", {}))}),
        deep=_build(LLMConfig, {**shared, **raw.get("deep", {})}),
    )


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader for local runs. CI uses real environment variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    root = path.parent.resolve()
    _load_dotenv(root / ".env")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    cfg = Config(
        manifold=_build(ManifoldConfig, raw.get("manifold", {})),
        llm=_load_llm(raw.get("llm", {})),
        budget=_build(BudgetConfig, raw.get("budget", {})),
        scan=_build(ScanConfig, raw.get("scan", {})),
        screen=_build(ScreenConfig, raw.get("screen", {})),
        forecast=_build(ForecastConfig, raw.get("forecast", {})),
        risk=_build(RiskConfig, raw.get("risk", {})),
        watch=_build(WatchConfig, raw.get("watch", {})),
        social=_build(SocialConfig, raw.get("social", {})),
        memory=_build(MemoryConfig, raw.get("memory", {})),
        root=root,
        standing_orders=_read_instructions(root / "instructions.md"),
        one_off_order=os.environ.get("OWNER_INSTRUCTION", ""),
    )

    cfg.manifold.api_key = os.environ.get("MANIFOLD_API_KEY", "")
    if not cfg.manifold.api_key:
        raise ValueError("MANIFOLD_API_KEY is not set")

    for name, tier in cfg.llm.tiers():
        tier.api_key = os.environ.get(tier.key_env, "")
        if not tier.api_key:
            raise ValueError(f"{tier.key_env} is not set (needed by the {name} model)")
    return cfg
