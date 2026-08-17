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


def describe_schema(schema: dict[str, Any]) -> str:
    """Render a response schema as instructions for a model that cannot be given one.

    Providers that reject a structured-output field still follow a shape described in
    the prompt. Without this they get told "answer in JSON" and nothing about which
    keys, so they invent their own and every field the caller reads comes back empty.
    """
    def one(name: str, spec: Any, required: bool) -> str:
        spec = spec if isinstance(spec, dict) else {}
        kind = str(spec.get("type", "STRING")).lower()
        note = str(spec.get("description", ""))
        if spec.get("enum"):
            note = f"{note} One of: {', '.join(str(v) for v in spec['enum'])}.".strip()
        if kind == "array":
            item = spec.get("items") or {}
            inner = (item.get("properties") or {}) if isinstance(item, dict) else {}
            if inner:
                fields = "; ".join(
                    one(k, v, True).lstrip("- ") for k, v in inner.items()
                )
                note = f"{note} Each entry is an object of: {fields}".strip()
        flag = "" if required else " (optional)"
        return f'- "{name}" ({kind}){flag}: {note}'.rstrip()

    props = schema.get("properties") or {}
    if not props:
        return ""
    required = set(schema.get("required") or props.keys())
    return "\n".join(one(n, s, n in required) for n, s in props.items())


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


_REASONING = re.compile(
    r"<(think|thinking|reasoning|scratchpad)>.*?</\1>", re.S | re.I
)


def strip_reasoning(text: str) -> str:
    """Remove inline reasoning blocks before the text is used as an answer.

    Gemini returns reasoning as separate parts flagged `thought`, which is easy to drop.
    Open models served over the OpenAI dialect inline it in the content as <think>...
    </think> instead, and that would go straight into a public comment. An unclosed
    opening tag means the reasoning ran to the end, so everything after it goes too.
    """
    cleaned = _REASONING.sub("", text)
    opening = re.search(r"<(think|thinking|reasoning|scratchpad)>", cleaned, re.I)
    if opening:
        cleaned = cleaned[: opening.start()]
    return cleaned.strip() or text.strip()


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
        # A grounded 429 means the search grounding quota is gone, which is an account
        # level thing: retrying the same call on a different model 429s identically and
        # just burns another request. Only plain generation is worth failing over.
        chain = self.chain[:1] if grounded else self.chain
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
        # Some models reject a system instruction, a response schema, or both, and
        # answer 400 rather than ignoring the field. Gemma is the common case so it is
        # detected by name, but any model can opt in with `prompt_only` in config.
        plain = (
            self.cfg.prompt_only
            if self.cfg.prompt_only is not None
            else "gemma" in model.lower()
        )

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
            fields = describe_schema(json_schema)
            text += (
                "\n\nYour answer after the marker must be a single JSON object and "
                "nothing else: no prose around it, no code fence."
            )
            if fields:
                text += f"\nUse exactly these keys:\n{fields}"

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
    """Covers OpenAI, OpenRouter, Mistral, Groq, Cerebras, DeepSeek and anything else
    speaking the /chat/completions dialect.

    Some of them have a first-party web search and each asks for it differently, which
    is what `search_style` selects. Where there is none, research falls back to the
    keyless search in websearch.py rather than to whatever the model remembers.
    """

    def __init__(self, cfg: LLMConfig, base_url: str, search_style: str = "") -> None:
        super().__init__(cfg)
        self.base = base_url.rstrip("/")
        self.search_style = search_style
        self.supports_search = bool(search_style) and cfg.use_search

    async def _call(
        self,
        prompt: str,
        *,
        system: str | None,
        json_schema: dict[str, Any] | None,
        grounded: bool,
        model: str,
    ) -> LLMResponse:
        if grounded and self.search_style == "responses":
            return await self._call_responses(prompt, system=system, model=model)
        if grounded and self.search_style == "conversations":
            return await self._call_conversations(prompt, system=system, model=model)

        # json_object mode only guarantees the reply parses, never that it has the keys
        # the caller reads, so the shape goes in the prompt the same way it does for a
        # model with no structured-output support at all.
        if json_schema is not None and not grounded:
            fields = describe_schema(json_schema)
            prompt += (
                "\n\nAnswer with a single JSON object and nothing else."
                + (f" Use exactly these keys:\n{fields}" if fields else "")
            )

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
        if grounded and self.search_style == "plugin":
            # OpenRouter runs search as a plugin in front of any model it serves, so
            # this works whatever `model` happens to be, including the fallbacks.
            body["plugins"] = [
                {"id": "web", "max_results": self.cfg.search_results}
            ]

        data = await self._post("/chat/completions", body, model)
        message = data["choices"][0]["message"]
        # Some providers put reasoning in its own field, others inline it in the content.
        text = strip_reasoning(message.get("content") or "")
        return LLMResponse(text=text, citations=_annotated_urls(message), raw=data)

    async def _call_responses(
        self, prompt: str, *, system: str | None, model: str
    ) -> LLMResponse:
        """OpenAI's hosted web search, which lives on /responses rather than on chat.

        Only grounded calls come here. Everything else stays on /chat/completions so a
        base_url pointed at some other implementation of that dialect keeps working.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": prompt,
            "tools": [{"type": "web_search"}],
        }
        if system:
            body["instructions"] = system

        data = await self._post("/responses", body, model)
        chunks, citations = [], []
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if part.get("type") != "output_text":
                    continue
                chunks.append(part.get("text") or "")
                for note in part.get("annotations") or []:
                    if note.get("url"):
                        citations.append(note["url"])
        return LLMResponse(
            text=strip_reasoning("".join(chunks)),
            citations=citations[:8],
            raw=data,
        )

    async def _call_conversations(
        self, prompt: str, *, system: str | None, model: str
    ) -> LLMResponse:
        """Mistral's web search, which is a connector on the conversations API.

        Their /chat/completions has no search at all, so grounded calls detour here and
        everything else stays on the normal endpoint.
        """
        body: dict[str, Any] = {
            "model": model,
            "inputs": prompt if not system else f"{system}\n\n{prompt}",
            "tools": [{"type": "web_search"}],
            "store": False,
        }

        data = await self._post("/conversations", body, model)
        chunks, citations = [], []
        for item in data.get("outputs") or []:
            if item.get("type") not in (None, "message.output"):
                continue
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            for part in content or []:
                if part.get("type") == "text":
                    chunks.append(part.get("text") or "")
                elif part.get("type") == "tool_reference" and part.get("url"):
                    citations.append(part["url"])
        return LLMResponse(
            text=strip_reasoning("".join(chunks)),
            citations=citations[:8],
            raw=data,
        )

    async def _post(self, path: str, body: dict[str, Any], model: str) -> dict[str, Any]:
        await _space_out(self.base, self.cfg.min_seconds_between_calls)
        resp = await self._client.post(
            f"{self.base}{path}",
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            json=body,
        )
        if resp.status_code == 429:
            raise QuotaError(f"LLM 429 on {model}: {resp.text[:200]}")
        if resp.status_code in (400, 401, 403, 404):
            raise PermanentLLMError(f"LLM {resp.status_code}: {resp.text[:400]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"LLM {resp.status_code}: {resp.text[:400]}")
        return resp.json()


def _annotated_urls(message: dict[str, Any]) -> list[str]:
    """Pull citation URLs out of a chat-completions message, if the provider adds any."""
    urls = []
    for note in message.get("annotations") or []:
        url = (note.get("url_citation") or {}).get("url") or note.get("url")
        if url:
            urls.append(url)
    return urls[:8]


class AnthropicClient(LLMClient):
    """Anthropic's Messages API, which is its own dialect: a different auth header, a
    required max_tokens, system as a top-level field, and search as a server-side tool
    the model calls on its own during the turn."""

    BASE = "https://api.anthropic.com/v1"
    MAX_TOKENS = 4096

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
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
        # No response-schema field on this API, so the shape goes in the prompt exactly
        # as it does for any other model that cannot be handed one.
        if json_schema is not None and not grounded:
            fields = describe_schema(json_schema)
            prompt += (
                "\n\nAnswer with a single JSON object and nothing else."
                + (f" Use exactly these keys:\n{fields}" if fields else "")
            )

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.cfg.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        if grounded:
            body["tools"] = [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": self.cfg.search_results,
            }]

        await _space_out("anthropic", self.cfg.min_seconds_between_calls)
        resp = await self._client.post(
            f"{self.BASE}/messages",
            headers={
                "x-api-key": self.cfg.api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
        )
        if resp.status_code == 429:
            raise QuotaError(f"Anthropic 429 on {model}: {resp.text[:200]}")
        if resp.status_code in (400, 401, 403, 404):
            raise PermanentLLMError(f"Anthropic {resp.status_code}: {resp.text[:400]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"Anthropic {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

        # Content is a list of blocks. Only the text ones are the answer: a grounded
        # turn also carries the model's search calls and their raw results, and joining
        # those in would put a wall of scraped page text into a public comment.
        chunks, citations = [], []
        for block in data.get("content") or []:
            if block.get("type") != "text":
                continue
            chunks.append(block.get("text") or "")
            for note in block.get("citations") or []:
                if note.get("url"):
                    citations.append(note["url"])
        return LLMResponse(
            text=strip_reasoning("".join(chunks)),
            citations=citations[:8],
            raw=data,
        )


# Named providers, so the common ones need only a key and a model name. The third
# field is how that provider exposes web search, empty where it has none on the plain
# completions API: those fall through to the keyless search in websearch.py.
#
#   "plugin"        search runs in front of the model, asked for in the request body
#   "responses"     hosted search tool, on OpenAI's /responses endpoint
#   "conversations" hosted search connector, on Mistral's /conversations endpoint
#
# DeepSeek, Groq and Cerebras have no first-party search to call. They are named here
# anyway so they need only a key and a model, and their research goes through the
# search backend in websearch.py instead.
PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("https://api.openai.com/v1", "responses"),
    "openrouter": ("https://openrouter.ai/api/v1", "plugin"),
    "mistral": ("https://api.mistral.ai/v1", "conversations"),
    "deepseek": ("https://api.deepseek.com/v1", ""),
    "groq": ("https://api.groq.com/openai/v1", ""),
    "cerebras": ("https://api.cerebras.ai/v1", ""),
}


def build_llm(cfg: LLMConfig) -> LLMClient:
    provider = cfg.provider.lower()
    if provider == "gemini":
        return GeminiClient(cfg)
    if provider == "anthropic":
        return AnthropicClient(cfg)
    if provider in PROVIDERS:
        base, style = PROVIDERS[provider]
        return OpenAICompatibleClient(cfg, cfg.base_url or base, style)
    if provider == "openai_compatible":
        if not cfg.base_url:
            raise ValueError("llm.base_url is required for provider = openai_compatible")
        # An unknown endpoint speaking this dialect. `search_style` can still be set in
        # config for something that is OpenRouter-shaped behind a different host.
        return OpenAICompatibleClient(cfg, cfg.base_url, cfg.search_style)
    known = ", ".join(["gemini", "anthropic", *PROVIDERS, "openai_compatible"])
    raise ValueError(f"Unknown LLM provider: {cfg.provider}. Known: {known}")
