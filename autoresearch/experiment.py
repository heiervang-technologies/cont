"""Autoresearch experiment runner — measure how a training recipe affects
held-out logical-task accuracy on lile.

Reads:
    autoresearch/config.json — the agent's knob set
Side effects:
    1. Snapshots the daemon as ``autoresearch_baseline`` so every experiment
       starts from the same byte-exact state (per R-004's confirmed restore).
    2. Runs ``cfg.training.n_steps`` of RLVR with the configured weights / k /
       source against the logical-tasks train split.
    3. Evals on the held-out 10-task split, scoring with the local logical
       verifier (no teacher needed for eval — it's deterministic regex).
    4. Loads the baseline snapshot back so the next experiment starts from
       the same state.
    5. Prints ``score: <heldout_pass_rate>`` so the autoresearch loop can
       grep it out of ``run.log``.

Exit codes:
    0 = clean run, score printed
    1 = crash (e.g. daemon unreachable, snapshot failure)

Notes for the agent:
    * The training step itself is stubbed in v1 — the runner currently
      does snapshot + eval-only, not actual RLVR training. Wiring the
      training pulse is the FIRST experiment to commit (modify the
      ``_run_training_pulse`` body to call into ``lile.teach.rlvr_loop``).
    * Once training is wired, the metric is "post-training held-out
      accuracy after N steps starting from the baseline snapshot" — a
      direct measure of training recipe quality.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Make the repo root importable so we can pull from lile/ without an
# install step — autoresearch is a side-tool, not a daemon-side import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import httpx  # noqa: E402

from lile.objectives.verifiers import verify as registry_verify  # noqa: E402
from lile.teach.logical import get_split  # noqa: E402

_CONFIG_PATH = Path(__file__).parent / "config.json"


def _load_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


async def _snapshot_save(client: httpx.AsyncClient, base: str, name: str) -> None:
    r = await client.post(f"{base}/v1/state/snapshot/save", json={"name": name})
    r.raise_for_status()


async def _snapshot_load(client: httpx.AsyncClient, base: str, name: str) -> None:
    r = await client.post(f"{base}/v1/state/snapshot/load", json={"name": name})
    r.raise_for_status()


async def _chat(client: httpx.AsyncClient, base: str, prompt: str, sampling: dict) -> str:
    r = await client.post(
        f"{base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}], **sampling},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def _eval_heldout(
    client: httpx.AsyncClient, base: str, tasks: list[dict], sampling: dict,
) -> dict:
    """Run all held-out tasks through the daemon, score each with the
    local verifier, return aggregate stats.
    """
    passed = 0
    per_task: list[dict] = []
    for t in tasks:
        t0 = time.time()
        try:
            candidate = await _chat(client, base, t["prompt"], sampling)
        except Exception as exc:
            per_task.append({"task_id": t["task_id"], "passed": None,
                             "error": f"{type(exc).__name__}: {exc}",
                             "wall_s": time.time() - t0})
            continue
        verdict = registry_verify("logical", t["prompt"], candidate)
        if verdict is True:
            passed += 1
        per_task.append({
            "task_id": t["task_id"],
            "domain": t["domain"],
            "passed": bool(verdict) if verdict is not None else None,
            "candidate_chars": len(candidate or ""),
            "wall_s": time.time() - t0,
        })
    score = passed / len(tasks) if tasks else 0.0
    return {"score": score, "passed": passed, "total": len(tasks), "per_task": per_task}


async def _run_training_pulse(client: httpx.AsyncClient, base: str, cfg: dict) -> dict:
    """STUB. v1 does no training — see module docstring.

    The first real experiment should fill this in by importing
    ``lile.teach.rlvr_loop.RLVRScheduler`` + ``RLVRConfig``, building a
    config from ``cfg['training']``, and running ``n_steps``. Use either
    ``lile.teach.teacher_oss120b.judge`` or
    ``lile.teach.teacher_free_pool.judge`` based on ``cfg['teacher']``.
    Return the scheduler's stats dict for inclusion in run.log.
    """
    return {"step": "stub", "n_steps": 0,
            "note": "training pulse not wired yet — see _run_training_pulse docstring"}


async def main() -> int:
    cfg = _load_config()
    base = cfg["daemon_url"].rstrip("/")
    baseline = cfg["baseline_snapshot"]
    sampling = cfg["eval"]
    train_tasks, heldout = get_split()

    print(f"[autoresearch] config: {json.dumps(cfg, sort_keys=True)}", flush=True)
    print(f"[autoresearch] split: {len(train_tasks)} train / {len(heldout)} heldout", flush=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Pre-flight: daemon alive?
        try:
            h = await client.get(f"{base}/health", timeout=3.0)
            h.raise_for_status()
        except Exception as exc:
            print(f"[autoresearch] FATAL: daemon unreachable at {base} — {exc}", flush=True)
            return 1

        # Snapshot the starting state so the experiment is reversible.
        print(f"[autoresearch] snapshot/save → {baseline}", flush=True)
        await _snapshot_save(client, base, baseline)

        try:
            print("[autoresearch] training pulse …", flush=True)
            train_stats = await _run_training_pulse(client, base, cfg)
            print(f"[autoresearch] training done: {json.dumps(train_stats, sort_keys=True)}",
                  flush=True)

            print("[autoresearch] eval on held-out …", flush=True)
            t0 = time.time()
            eval_stats = await _eval_heldout(client, base, heldout, sampling)
            eval_stats["wall_s"] = time.time() - t0
            print(f"[autoresearch] eval done: passed={eval_stats['passed']}/{eval_stats['total']} "
                  f"({eval_stats['score']*100:.1f}%) in {eval_stats['wall_s']:.1f}s",
                  flush=True)
        finally:
            # Always rewind to baseline so the next experiment starts clean.
            print(f"[autoresearch] snapshot/load ← {baseline}", flush=True)
            await _snapshot_load(client, base, baseline)

    # The autoresearch loop grep-extracts this line. Keep the format stable.
    print(f"score: {eval_stats['score']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
