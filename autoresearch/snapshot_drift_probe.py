"""Snapshot-bracket determinism probe.

R-004 claimed snapshot/load is byte-exact under both weak and strong
memorize regimes. But across the 25+ experiments on autoresearch/may16 we
observe probe_cold drifting (0.47 → 0.53 → 0.60 → 0.67 → 0.73 under stable
code with `idle_replay=False`). That drift compromises score comparability.

This probe isolates the drift by asking: with NO training and NO other work
between calls, does a snapshot save → load cycle change the eval output?

Procedure (under temperature=0.0 greedy decoding):
1. Eval probe set → baseline_a.
2. Snapshot save under name "drift_probe" → load same.
3. Eval probe set → after_one_cycle.
4. Snapshot save (overwrite) → load.
5. Eval probe set → after_two_cycles.
6. Compare per-task: are responses byte-identical across (a, one, two)?
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DAEMON = "http://127.0.0.1:8768"
PROBE = ROOT / "autoresearch" / "probe_v0.json"
EVAL = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "enable_thinking": False}


async def chat(client: httpx.AsyncClient, prompt: str) -> str:
    payload = {"model": "Qwen3-8B", "messages": [{"role": "user", "content": prompt}], **EVAL}
    r = await client.post(f"{DAEMON}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def snapshot(client: httpx.AsyncClient, action: str, name: str) -> None:
    r = await client.post(f"{DAEMON}/v1/state/snapshot/{action}",
                          json={"name": name}, timeout=120.0)
    r.raise_for_status()


async def eval_probe(client: httpx.AsyncClient, tasks: list[dict], label: str) -> list[str]:
    out = []
    for t in tasks[:5]:  # 5 is enough; we want byte-equality not pass-rate
        resp = await chat(client, t["prompt"])
        out.append(resp)
        print(f"  [{label}] {t['task_id']}: {len(resp)} chars, first 60={resp[:60]!r}",
              flush=True)
    return out


async def memorize(client: httpx.AsyncClient, prompt: str, response: str) -> None:
    payload = {
        "prompt": prompt,
        "response": response,
        "max_steps": 100,
        "threshold": 0.95,
        "plateau_patience": 10,
        "lr": 0.002,
        "weight": 1.0,
    }
    r = await client.post(f"{DAEMON}/v1/train/memorize", json=payload, timeout=600.0)
    r.raise_for_status()


async def main() -> int:
    probe_tasks = json.loads(PROBE.read_text())["tasks"]
    async with httpx.AsyncClient(timeout=300.0) as client:
        print("[drift] PHASE A — initial eval", flush=True)
        a = await eval_probe(client, probe_tasks, "a")
        print()

        print("[drift] save → train → load (autoresearch's actual bracket)", flush=True)
        await snapshot(client, "save", "drift_probe")
        await memorize(client, "Drift test: what is 2+2?", "Answer: 4")
        await snapshot(client, "load", "drift_probe")
        print("[drift] PHASE B — after save→train→load (R-004 byte-exact claim)", flush=True)
        b = await eval_probe(client, probe_tasks, "b")
        print()

        print("[drift] second save → train → load cycle", flush=True)
        await snapshot(client, "save", "drift_probe")
        await memorize(client, "Drift test: what is 3+3?", "Answer: 6")
        await snapshot(client, "load", "drift_probe")
        print("[drift] PHASE C — after second save→train→load", flush=True)
        c = await eval_probe(client, probe_tasks, "c")
        print()

    drift_ab = sum(1 for x, y in zip(a, b) if x != y)
    drift_ac = sum(1 for x, y in zip(a, c) if x != y)
    drift_bc = sum(1 for x, y in zip(b, c) if x != y)
    print("=" * 60, flush=True)
    print(f"differences a↔b (1 save→train→load): {drift_ab}/5", flush=True)
    print(f"differences a↔c (2 save→train→load): {drift_ac}/5", flush=True)
    print(f"differences b↔c (incremental):       {drift_bc}/5", flush=True)
    if drift_ab == 0 and drift_ac == 0 and drift_bc == 0:
        print("VERDICT: save → train → load IS byte-exact", flush=True)
        print("         → R-004 confirmed; probe drift across autoresearch experiments must come from another source", flush=True)
        print("         → look at: eval-config changes (CoT vs no-CoT), background work between experiments", flush=True)
    else:
        print("VERDICT: save → train → load is NOT byte-exact", flush=True)
        print("         → R-004 partial; restoration leaves residual model state from training", flush=True)
        print("         → score comparability across experiments is brittle on this axis", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
