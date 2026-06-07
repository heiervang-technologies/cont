"""Multi-judge teacher backed by the OpenRouter free tier + local fallback.

Drop-in replacement for ``teacher_oss120b.judge`` that pools across the
free-tier models configured in ``lile/teach/free_models.json``. The pool
rotates round-robin; on per-model 4xx/5xx (rate limit, transient error,
model decommissioned) the dispatcher falls through to the next entry.

When the entire remote pool exhausts (or the env var ``LILE_TEACHER_LOCAL_ONLY``
is set), the dispatcher falls back to the local cluster's gemma-4-31b-iq4
endpoint (read from ``$LILE_LOCAL_TEACHER_URL``). The local backend is
treated as a *shared* resource — concurrency is capped at 1 and a small
inter-call gap is honored so we don't starve other users on the same GPU.

API
---

::

    from lile.teach.teacher_free_pool import judge

    result = await judge(
        prompt="What is 17 * 23?",
        rollouts=["391", "390", "I'm not sure"],
        domain="math",
    )
    # result["grades"]          -> ["correct", "wrong", "ambiguous"]
    # result["critiques"]       -> [None, "Off by 1 ...", None]
    # result["counterfactuals"] -> [None, "390", None]
    # result["demonstration"]   -> "17 * 23 = 391"
    # result["judge_model"]     -> "qwen/qwen-2.5-coder-32b-instruct:free"

Schema matches ``teacher_oss120b.judge`` so the existing
``rlvr_loop._step_one`` consumer works unchanged — drop in via
``--teacher free_pool`` (PR follow-up wires the CLI; for now construct
``RLVRScheduler`` with ``teacher_callable=teacher_free_pool.judge``).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "free_models.json"

# Module-level state — lazy-loaded on first judge() call.
_CONFIG: dict | None = None
_REMOTE_SEM: asyncio.Semaphore | None = None
_LOCAL_SEM: asyncio.Semaphore | None = None
_RR_CYCLE: itertools.cycle | None = None  # round-robin iterator over remote slugs


# ---------------------------------------------------------------------- prompt
_JUDGE_INSTRUCTION = """You are a grading oracle. Given a USER PROMPT and K
CANDIDATE responses, judge each candidate as exactly one of:

  correct   — the candidate is right; no notes.
  wrong     — the candidate is incorrect or unverifiable.
  ambiguous — the candidate is partially right, partially wrong, or off-topic.

Return STRICT JSON ONLY, no markdown, no commentary, in this exact shape:

  {
    "grades":          ["correct"|"wrong"|"ambiguous", ...],   // length K
    "critiques":       [null|"<short critique>", ...],         // length K
    "counterfactuals": [null|"<corrected response>", ...],     // length K
    "demonstration":   "<canonical correct response>"
  }

USER PROMPT:
{prompt}

CANDIDATES (K={k}):
{candidates}
"""


# ---------------------------------------------------------------------- config
def _load_config() -> dict:
    global _CONFIG, _REMOTE_SEM, _LOCAL_SEM, _RR_CYCLE
    if _CONFIG is not None:
        return _CONFIG
    _CONFIG = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    t = _CONFIG["throttle"]
    _REMOTE_SEM = asyncio.Semaphore(t["max_concurrent_remote"])
    _LOCAL_SEM = asyncio.Semaphore(t["max_concurrent_local"])
    slugs = [m["slug"] for m in _CONFIG["models"]]
    _RR_CYCLE = itertools.cycle(slugs)
    log.info("teacher_free_pool: %d remote models + 1 local fallback", len(slugs))
    return _CONFIG


# ---------------------------------------------------------------------- format
def _build_prompt(user_prompt: str, rollouts: list[str]) -> str:
    blocks = "\n".join(f"[{i}] {r.strip()}\n" for i, r in enumerate(rollouts))
    return _JUDGE_INSTRUCTION.format(prompt=user_prompt, k=len(rollouts), candidates=blocks)


def _parse_envelope(content: str, k: int) -> dict[str, Any]:
    """Robust JSON parse; tolerates markdown fences and trailing prose."""
    txt = content.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt)
        txt = re.sub(r"\s*```$", "", txt)
    # Find the first { ... } that parses.
    start = txt.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in teacher reply: {txt[:200]}")
    for end in range(len(txt), start, -1):
        try:
            obj = json.loads(txt[start:end])
            break
        except json.JSONDecodeError:
            continue
    else:
        raise ValueError(f"could not parse JSON envelope: {txt[:200]}")
    # Sanity-pad lengths
    grades = list(obj.get("grades", []))[:k] + ["ambiguous"] * (k - len(obj.get("grades", [])))
    critiques = list(obj.get("critiques", []))[:k] + [None] * (k - len(obj.get("critiques", [])))
    cfs = list(obj.get("counterfactuals", []))[:k] + [None] * (k - len(obj.get("counterfactuals", [])))
    return {
        "grades": grades,
        "critiques": [c if c else None for c in critiques],
        "counterfactuals": [c if c else None for c in cfs],
        "demonstration": obj.get("demonstration") or "",
    }


# ---------------------------------------------------------------------- HTTP
async def _call_openrouter(slug: str, body_prompt: str, *, max_tokens: int = 800) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = {
        "model": slug,
        "messages": [{"role": "user", "content": body_prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Title": "lile-rlvr-teacher-pool",
            },
            json=payload,
        )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def _call_local(model: str, body_prompt: str, *, max_tokens: int = 800) -> str:
    url = os.environ.get("LILE_LOCAL_TEACHER_URL")
    if not url:
        raise RuntimeError("LILE_LOCAL_TEACHER_URL not set — no local fallback configured")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": body_prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------- judge
async def judge(
    prompt: str,
    rollouts: list[str],
    domain: str = "general",
    *,
    max_tokens: int = 800,
) -> dict[str, Any]:
    """Dispatch a grading call across the pool; first model that returns a
    parseable JSON envelope wins.

    Returns the standard ``{grades, critiques, counterfactuals, demonstration}``
    shape plus a ``judge_model`` field naming the slug that actually answered.
    """
    cfg = _load_config()
    t = cfg["throttle"]
    body = _build_prompt(prompt, rollouts)
    last_err: Exception | None = None

    # ---- remote pool: round-robin ----
    pool_size = len(cfg["models"])
    if not os.environ.get("LILE_TEACHER_LOCAL_ONLY"):
        for _ in range(pool_size):
            slug = next(_RR_CYCLE)  # type: ignore[arg-type]
            for attempt in range(t["max_retries"] + 1):
                async with _REMOTE_SEM:  # type: ignore[union-attr]
                    try:
                        content = await _call_openrouter(slug, body, max_tokens=max_tokens)
                        envelope = _parse_envelope(content, k=len(rollouts))
                        envelope["judge_model"] = slug
                        return envelope
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code in t["retry_on_status"] and attempt < t["max_retries"]:
                            await asyncio.sleep(t["backoff_initial_s"] * (t["backoff_factor"] ** attempt))
                            continue
                        last_err = exc
                        break  # next slug
                    except Exception as exc:
                        last_err = exc
                        break  # next slug
                await asyncio.sleep(t["per_call_gap_remote_s"])

    # ---- local fallback ----
    local = cfg.get("local_fallback") or {}
    if local.get("endpoint_env") and os.environ.get(local["endpoint_env"]):
        async with _LOCAL_SEM:  # type: ignore[union-attr]
            try:
                content = await _call_local(local["model"], body, max_tokens=max_tokens)
                envelope = _parse_envelope(content, k=len(rollouts))
                envelope["judge_model"] = f"local:{local['model']}"
                return envelope
            except Exception as exc:
                last_err = exc
        await asyncio.sleep(t["per_call_gap_local_s"])

    raise RuntimeError(
        f"teacher_free_pool: every model failed. last_err={type(last_err).__name__}: {last_err}"
    )
