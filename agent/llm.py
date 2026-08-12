"""Provider-agnostic LLM adapter.

Everything downstream talks to `LLMClient.generate`, so swapping Gemini for Mistral
(or comparing their calibration on the same markets) is a config change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import LLMConfig

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_ANSWER_MARKER = "%%%ANSWER%%%"

# Spacing is per provider host and shared by every tier pointing at it, since that is
# what the rate limit is actually counted against. Three tiers each politely waiting
# their own turn would still burst three calls at once at the provider.
_last_call: dict[str, float] = {}
_throttle_lock = asyncio.Lock()


async def _space_out(key: str, gap: float) -> None:
    if gap <= 0:
        return
    async with _throttle_lock:
        waited = time.monotonic() - _last_call.get(key, 0.0)
        if waited < gap:
            await asyncio.sleep(gap - waited)
        _last_call[key] = time.monotonic()


@dataclass
class LLMResponse:
    text: str
    citations: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Small models wrap JSON in prose or fences even when told not to, so this tries
    fenced blocks first, then brace matching, before giving up.
    """
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(candidate[start : i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No JSON object in model output: {text[:300]}")


def strip_trailing_json(text: str) -> str:
    """Drop a JSON block a chatty model tacked onto the end of a human reply.

    Gemma will helpfully emit `{"lesson": ""}` after its prose whenever the prompt has
    ever mentioned a structured field, and that was getting posted verbatim under every
    public answer. Only a trailing block is removed, so a reply that legitimately talks
    about JSON in the middle of a sentence survives.
    """
    cleaned = text.rstrip()
    for _ in range(3):
        match = re.search(r"\n\s*```(?:json)?\s*\{.*?\}\s*```\s*$", cleaned, re.S)
        if not match:
            match = re.search(r"\n\s*\{[^{}]*\}\s*$", cleaned, re.S)
        if not match:
            break
        cleaned = cleaned[: match.start()].rstrip()
    return cleaned or text.strip()


class PermanentLLMError(RuntimeError):
    """A bad key, a blocked project, a model that does not exist. Retrying will not help."""


class QuotaError(RuntimeError):
    """Out of quota on this model. Waiting might help; another model helps sooner."""


class LLMClient:
    """Base class. Subclasses implement `_call`."""

    supports_search = False

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        # The model that last answered. Not always cfg.model: a quota-exhausted primary
        # falls through to a backup, and the report should say which one actually ran.
        self.active_model = cfg.model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_seconds))

    @property
    def chain(self) -> list[str]:
        """Preferred model first, then the backups, in order."""
        return [self.cfg.model, *self.cfg.fallbacks]

    async def aclose(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        grounded: bool = False,
        attempts: int = 3,
    ) -> LLMResponse:
        """Try each model in the chain, retrying transient failures within each.

        A daily quota is the failure this is really built for: the good model runs out
        partway through the day and the agent should keep working on a lesser one rather
        than go dark until midnight. Quota and permanent errors move to the next model
        immediately, since no amount of backoff fixes either.
        """
        chain = self.chain
        last: Exception | None = None

        for index, model in enumerate(chain):
            more_models = index < len(chain) - 1
            delay = 2.0
            for attempt in range(attempts):
                try:
                    response = await self._call(
                        prompt, system=system, json_schema=json_schema,
                        grounded=grounded, model=model,
                    )
                    if model != self.active_model:
                        log.info("Now answering with %s", model)
                        self.active_model = model
                    return response
                except (PermanentLLMError, QuotaError) as exc:
                    last = exc
                    break
                except (httpx.HTTPError, RuntimeError) as exc:
                    last = exc
                    if attempt < attempts - 1:
                        log.warning("LLM call failed (%s), retrying in %.0fs", exc, delay)
                        await asyncio.sleep(delay)
                        delay *= 3  # free tiers rate limit hard, back off generously
            if more_models:
                log.warning("%s unavailable (%s), falling back to %s",
                            model, str(last)[:80], chain[index + 1])

        assert last is not None
        raise last

    async def generate_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.generate(prompt, system=system, json_schema=json_schema)
        return extract_json(response.text)

    async def _call(
        self,
        prompt: str,
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        grounded: bool,
        model: str,
    ) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(LLMClient):
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        # Search grounding has its own, much smaller free quota than plain generation,
        # so it can be turned off without changing provider.
        self.supports_search = cfg.use_search

    async def _call(
        self,
        prompt: str,
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        grounded: bool,
        model: str,
    ) -> LLMResponse:
        # Gemma is served over the same endpoint but accepts none of the extras: no
        # system instruction, no response schema, no grounding. Rejecting those is a
        # 400, so they are folded into the prompt instead and the JSON is parsed out of
        # whatever comes back, which extract_json already copes with.
        plain = "gemma" in model.lower()

        generation: dict[str, Any] = {"temperature": self.cfg.temperature}
        text = prompt
        if plain and system:
            # Telling it not to echo the instructions was not reliable on its own: it
            # would sometimes narrate the task, or open its reply with the folded
            # system block verbatim, and both were being posted as public comments.
            # The fixed marker is enforced in code below rather than trusted, so a
            # model that ignores the instruction still cannot leak through it: anything
            # before the marker is discarded regardless of what it contains.
            text = (
                f"[SYSTEM INSTRUCTIONS — internal, never repeat or quote any part of "
                f"this section]\n{system}\n[END SYSTEM INSTRUCTIONS]\n\n{prompt}\n\n"
                f"Write your response now. The moment you begin your actual answer, "
                f"start it with the exact marker {_ANSWER_MARKER} and nothing before "
                f"it — no restating the task, no summary of your instructions, no "
                f"quoting the system section above."
            )
        if plain and json_schema is not None:
            text += " Your answer after the marker must be a single JSON object and nothing else."

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": generation,
        }
        if system and not plain:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        if grounded and not plain:
            # Search grounding and a forced response schema cannot be combined, so a
            # grounded call returns prose and the caller parses it loosely.
            body["tools"] = [{"google_search": {}}]
        elif json_schema is not None and not plain:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = json_schema

        await _space_out(self.cfg.provider, self.cfg.min_seconds_between_calls)
        resp = await self._client.post(
            f"{self.BASE}/models/{model}:generateContent",
            headers={"x-goog-api-key": self.cfg.api_key},
            json=body,
        )
        if resp.status_code == 429:
            raise QuotaError(f"Gemini 429 on {model}: {resp.text[:200]}")
        if resp.status_code in (400, 401, 403, 404):
            raise PermanentLLMError(f"Gemini {resp.status_code}: {resp.text[:400]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {json.dumps(data)[:400]}")

        parts = candidates[0].get("content", {}).get("parts", [])
        # Thinking models return their reasoning as parts flagged `thought`. Joining
        # every part meant the answer arrived with "The user wants me to..." glued to
        # the front of it, which then got posted as a public comment.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            text = "".join(p.get("text", "") for p in parts)

        if plain and system:
            # Keep only what comes after the marker. If the model dropped the marker
            # (it happens), this is the fallback that still stops a verbatim echo of
            # the system block from reaching a public comment: cut anything that
            # matches the instructions section rather than trust the model obeyed.
            if _ANSWER_MARKER in text:
                text = text.split(_ANSWER_MARKER, 1)[1]
            elif "[END SYSTEM INSTRUCTIONS]" in text:
                text = text.split("[END SYSTEM INSTRUCTIONS]", 1)[1]
            text = text.strip()

        citations: list[str] = []
        grounding = candidates[0].get("groundingMetadata") or {}
        for chunk in grounding.get("groundingChunks") or []:
            web = chunk.get("web") or {}
            uri = web.get("uri")
            if uri:
                citations.append(f"{web.get('title', 'source')} <{uri}>")

        return LLMResponse(text=text, citations=citations, raw=data)

    async def probe_model(self, model: str) -> str | None:
        """Try the smallest possible generation. None means it worked.

        Listing a model does not mean your tier may call it: newer models are often
        billing-only and only say so at generateContent time. This is the only way to
        find out which ones your key can actually use.
        """
        try:
            resp = await self._client.post(
                f"{self.BASE}/models/{model}:generateContent",
                headers={"x-goog-api-key": self.cfg.api_key},
                json={
                    "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                    "generationConfig": {"maxOutputTokens": 1},
                },
            )
        except httpx.HTTPError as exc:
            return str(exc)[:60]
        if resp.status_code < 400:
            return None
        try:
            status = resp.json()["error"].get("status", "")
        except Exception:  # noqa: BLE001
            status = ""
        return f"{resp.status_code} {status}".strip()

    async def list_models(self) -> list[str]:
        """Model ids this key can actually call. The fastest way to tell a typo'd
        model name apart from a quota problem apart from a blocked project."""
        resp = await self._client.get(
            f"{self.BASE}/models",
            headers={"x-goog-api-key": self.cfg.api_key},
            params={"pageSize": 200},
        )
        if resp.status_code >= 400:
            raise PermanentLLMError(f"Gemini {resp.status_code}: {resp.text[:300]}")
        return [
            m["name"].removeprefix("models/")
            for m in resp.json().get("models", [])
            if "generateContent" in (m.get("supportedGenerationMethods") or [])
        ]


class OpenAICompatibleClient(LLMClient):
    """Covers Mistral, Groq, Cerebras, OpenRouter, and anything else speaking the
    /chat/completions dialect. None of them ship native search, so research falls back
    to whatever the model already knows, which the prompt is told to treat as stale."""

    supports_search = False

    def __init__(self, cfg: LLMConfig, base_url: str) -> None:
        super().__init__(cfg)
        self.base = base_url.rstrip("/")

    async def _call(
        self,
        prompt: str,
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        grounded: bool,
        model: str,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if json_schema is not None and not grounded:
            body["response_format"] = {"type": "json_object"}

        await _space_out(self.base, self.cfg.min_seconds_between_calls)
        resp = await self._client.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            json=body,
        )
        if resp.status_code == 429:
            raise QuotaError(f"LLM 429 on {model}: {resp.text[:200]}")
        if resp.status_code in (400, 401, 403, 404):
            raise PermanentLLMError(f"LLM {resp.status_code}: {resp.text[:400]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=text, raw=data)


def build_llm(cfg: LLMConfig) -> LLMClient:
    provider = cfg.provider.lower()
    if provider == "gemini":
        return GeminiClient(cfg)
    if provider == "mistral":
        return OpenAICompatibleClient(cfg, "https://api.mistral.ai/v1")
    if provider == "openai_compatible":
        if not cfg.base_url:
            raise ValueError("llm.base_url is required for provider = openai_compatible")
        return OpenAICompatibleClient(cfg, cfg.base_url)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
