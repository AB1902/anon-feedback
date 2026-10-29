"""Provider-agnostic LLM client.

Usage:
    from app.llm import generate, generate_json
    text = generate(system="You are...", user="...")
    obj  = generate_json(system=..., user=..., schema=Triage)

Speed choices:
- Clients are created once and reused, with a 30 s timeout.
- Gemini "thinking" is switched off (or set to low) because labelling and rewriting don't need it.
- On a rate limit (HTTP 429) we switch to the other provider immediately instead of
  waiting and retrying, so a busy free tier doesn't stall the user.
"""
import json
import re
import time
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from app import config

T = TypeVar("T", bound=BaseModel)
TIMEOUT_S = 30
LAST = {"provider": None, "seconds": 0.0}     # shown in the pipeline trace


class LLMError(RuntimeError):
    pass


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "resource_exhausted" in s or "rate limit" in s or "quota" in s


# ---------- provider calls ----------

@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai
    from google.genai import types
    return genai.Client(api_key=config.GEMINI_API_KEY,
                        http_options=types.HttpOptions(timeout=TIMEOUT_S * 1000))


_thinking_mode = {"i": 0}     # which thinking setting this model accepts (learned on first call)


def _thinking_options():
    from google.genai import types
    return [types.ThinkingConfig(thinking_budget=0), types.ThinkingConfig(thinking_level="low"), None]


def _gemini(system: str, user: str, temperature: float, json_mode: bool) -> str:
    from google.genai import types
    opts = _thinking_options()
    while True:
        think = opts[_thinking_mode["i"]]
        kwargs = {"thinking_config": think} if think is not None else {}
        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain", **kwargs)
        try:
            resp = _gemini_client().models.generate_content(model=config.GEMINI_MODEL, contents=user, config=cfg)
            return resp.text or ""
        except Exception as e:
            # Model rejected this thinking setting: try the next one (remembered for later calls)
            if think is not None and "think" in str(e).lower() and _thinking_mode["i"] < len(opts) - 1:
                _thinking_mode["i"] += 1
                continue
            raise


@lru_cache(maxsize=1)
def _groq_client():
    from groq import Groq
    return Groq(api_key=config.GROQ_API_KEY, timeout=TIMEOUT_S, max_retries=0)


def _groq(system: str, user: str, temperature: float, json_mode: bool) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    resp = _groq_client().chat.completions.create(
        model=config.GROQ_MODEL, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
    return resp.choices[0].message.content or ""


_PROVIDERS = {"gemini": (_gemini, "GEMINI_API_KEY"), "groq": (_groq, "GROQ_API_KEY")}


def _order() -> list[str]:
    first = config.LLM_PROVIDER if config.LLM_PROVIDER in _PROVIDERS else "gemini"
    order = [first] + [p for p in _PROVIDERS if p != first]
    return [p for p in order if getattr(config, _PROVIDERS[p][1])]


# ---------- public API ----------

def generate(system: str, user: str, temperature: float = 0.2, json_mode: bool = False) -> str:
    providers = _order()
    if not providers:
        raise LLMError("No API key set. Add GEMINI_API_KEY or GROQ_API_KEY to .env")
    errors = []
    for attempt in range(2):                       # second pass only if everything was rate-limited
        for name in providers:
            t0 = time.time()
            try:
                out = _PROVIDERS[name][0](system, user, temperature, json_mode)
                LAST.update(provider=name, seconds=round(time.time() - t0, 1))
                return out
            except Exception as e:             # fall through to the next provider at once
                errors.append(f"{name}: {str(e)[:200]}")
        if not all(_is_rate_limit(Exception(err)) for err in errors) or attempt == 1:
            break
        time.sleep(3)
    raise LLMError("All providers failed -> " + " | ".join(errors))


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def generate_json(system: str, user: str, schema: Type[T], retries: int = 2, mock=None) -> T:
    """Call the LLM and validate output against a Pydantic schema; retry on failure.

    `mock` is an optional zero-arg function returning a dict. When LLM_MOCK=1 is set,
    it is used instead of a real API call, so the app can be tested with no quota."""
    if config.LLM_MOCK and mock is not None:
        return schema.model_validate(mock())
    last_err = None
    for _ in range(retries + 1):
        raw = generate(system, user, temperature=0.0, json_mode=True)
        try:
            return schema.model_validate(_extract_json(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            user = f"{user}\n\nYour previous output was invalid ({e}). Return ONLY valid JSON."
    raise LLMError(f"Schema validation failed after retries: {last_err}")


def active_provider() -> str:
    if config.LLM_MOCK:
        return "mock (no API calls)"
    p = _order()
    return p[0] if p else "none"
