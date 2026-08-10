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
    model: str = "gemini-3.6-flash"
    # Grounded search has a separate, much smaller free quota than plain generation.
    # Turning it off keeps the agent running on the comment thread and the price alone.
    use_search: bool = True
    temperature: float = 0.3
    timeout_seconds: int = 90
    base_url: str = ""
    api_key: str = ""


@dataclass
class BudgetConfig:
    max_evaluations_per_tick: int = 1
    max_llm_calls_per_tick: int = 4


@dataclass
class ScanConfig:
    min_minutes_between_scans: float = 60
    min_unique_bettors: int = 25
    min_volume: float = 3000
    max_days_to_resolve: float = 30
    min_hours_to_resolve: float = 6
    skip_if_seen_within_hours: float = 48


@dataclass
class RiskConfig:
    default_max_bet: float = 10
    min_edge: float = 0.10
    kelly_fraction: float = 0.25

    conviction_max_fraction: float = 0.35
    conviction_min_volume: float = 10000
    conviction_max_days: float = 7
    conviction_min_edge: float = 0.18
    conviction_min_confidence: str = "high"

    max_market_impact: float = 0.03
    max_share_of_volume: float = 0.05
    max_open_positions: int = 40
    daily_mana_budget: float = 500
    min_balance_reserve: float = 50
    time_decay_days: float = 14


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


@dataclass
class Config:
    manifold: ManifoldConfig = field(default_factory=ManifoldConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    scan: ScanConfig = field(default_factory=ScanConfig)
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
        llm=_build(LLMConfig, raw.get("llm", {})),
        budget=_build(BudgetConfig, raw.get("budget", {})),
        scan=_build(ScanConfig, raw.get("scan", {})),
        risk=_build(RiskConfig, raw.get("risk", {})),
        watch=_build(WatchConfig, raw.get("watch", {})),
        social=_build(SocialConfig, raw.get("social", {})),
        memory=_build(MemoryConfig, raw.get("memory", {})),
        root=root,
        standing_orders=_read_instructions(root / "instructions.md"),
        one_off_order=os.environ.get("OWNER_INSTRUCTION", ""),
    )

    cfg.manifold.api_key = os.environ.get("MANIFOLD_API_KEY", "")
    cfg.llm.api_key = os.environ.get("LLM_API_KEY", "")
    if not cfg.manifold.api_key:
        raise ValueError("MANIFOLD_API_KEY is not set")
    if not cfg.llm.api_key:
        raise ValueError("LLM_API_KEY is not set")
    return cfg
