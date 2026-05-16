"""Autoresearch experiment runner — measure how a training recipe drives
*continual learning* on lile: simultaneously improve held-out task accuracy
AND preserve prior competence.

The metric is a composite that penalizes either failure mode:

    forward = heldout_pass_rate (logical/get_split() held-out partition)
    prior   = probe_pass_rate   (autoresearch/probe_v0.json, frozen)
    degrade = max(0, prior_cold - prior_post)   # only count regressions

    score = forward - LAMBDA_DEGRADE * degrade

LAMBDA_DEGRADE (default 1.0) is the trade-off: a recipe that gains +0.1
on held-out but loses 0.1 on the probe set scores zero — i.e. a wash.
We do NOT use a harmonic mean because we want to allow probe degradation
*if* the held-out gain is large enough; the linear penalty lets the agent
search the Pareto frontier.

Reads:
    autoresearch/config.json — the agent's knob set
    autoresearch/probe_v0.json — frozen non-degradation probe (do NOT edit)
Side effects:
    1. Cold-evals the probe set (baseline measurement, pre-training).
    2. Snapshots the daemon as ``autoresearch_baseline`` (byte-exact
       restore per R-004) so every experiment starts from the same state.
    3. Runs ``cfg.training.n_steps`` of RLVR with the configured weights /
       k / source against the logical-tasks train split.
    4. Evals on the held-out 10-task logical split → ``forward``.
    5. Re-evals the probe set → ``prior_post``; computes ``degrade``.
    6. Loads the baseline snapshot back so the next experiment starts
       from the same state.
    7. Prints ``score: <composite>``, plus the three component metrics
       and full per-task breakdown.

Exit codes:
    0 = clean run, score printed
    1 = crash (e.g. daemon unreachable, snapshot failure)

Notes for the agent:
    * The training step itself is stubbed in v1 — the runner currently
      does snapshot + eval-only, not actual RLVR training. Wiring the
      training pulse is the FIRST experiment to commit (modify the
      ``_run_training_pulse`` body to call into ``lile.teach.rlvr_loop``).
      Until then, ``score = forward - degrade`` where both reflect the
      cold model (degrade ≈ 0 by definition, modulo eval-time stochasticity).
    * The probe set is INTENTIONALLY small (~15 prompts) — the goal is
      catastrophic-forgetting detection, not benchmark coverage. Cheap
      to run after every iteration.
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

from lile.objectives.verifiers import _logical  # noqa: E402, F401 — ensures registration
from lile.objectives.verifiers import verify as registry_verify  # noqa: E402
from lile.teach.logical import get_split  # noqa: E402

_CONFIG_PATH = Path(__file__).parent / "config.json"
_PROBE_PATH = Path(__file__).parent / "probe_v0.json"
_LAMBDA_DEGRADE = 1.0  # trade-off: prior-task degradation penalty weight


def _load_probe() -> list[dict]:
    raw = json.loads(_PROBE_PATH.read_text(encoding="utf-8"))
    return raw["tasks"] if "tasks" in raw else raw


def _verify_probe(prompt: str, candidate: str, task: dict) -> bool | None:
    """Probe tasks aren't in the logical-corpus prompt-hash table, so
    ``registry_verify('logical', ...)`` would abstain (None). We instead
    call the verifier's per-mode compare directly with the probe's metadata.
    Avoids duplicating compare logic by reusing the private helpers.
    """
    got = _logical._extract(candidate, task.get("extract"))
    if got is None:
        return False
    return _logical._compare(got, task["expected"], task.get("compare", "exact"))


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


async def _eval_set(
    client: httpx.AsyncClient,
    base: str,
    tasks: list[dict],
    sampling: dict,
    *,
    use_registry: bool,
    label: str,
) -> dict:
    """Run all tasks through the daemon, grade each, return aggregate stats.

    ``use_registry=True`` routes through ``registry_verify("logical", ...)``,
    which expects prompts cataloged in lile/teach/logical/tasks_v0.json
    (held-out split).
    ``use_registry=False`` is for the probe set: bypass the hash-table claims
    check and use the task's compare metadata directly.
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
        if use_registry:
            verdict = registry_verify("logical", t["prompt"], candidate)
        else:
            verdict = _verify_probe(t["prompt"], candidate, t)
        if verdict is True:
            passed += 1
        per_task.append({
            "task_id": t["task_id"],
            "domain": t.get("domain", "unknown"),
            "passed": bool(verdict) if verdict is not None else None,
            "candidate_chars": len(candidate or ""),
            "wall_s": time.time() - t0,
        })
    rate = passed / len(tasks) if tasks else 0.0
    print(f"[autoresearch] {label}: passed={passed}/{len(tasks)} ({rate*100:.1f}%)",
          flush=True)
    fails = [pt["task_id"] for pt in per_task if pt["passed"] is not True]
    if fails:
        print(f"[autoresearch] {label} fails: {fails}", flush=True)
    return {"rate": rate, "passed": passed, "total": len(tasks), "per_task": per_task}


async def _run_training_pulse(client: httpx.AsyncClient, base: str, cfg: dict) -> dict:
    """Dispatch on ``cfg['training']['_mechanism']``.

    ``memorize`` — POST /v1/train/memorize on the first n_train_samples
    tasks (sorted by task_id) from the train split. The response of each
    task is wrapped in the same `Answer: <value>` grammar the verifier
    uses, so memorize learns to emit in the answer format eval scores.

    ``rlvr`` — stubbed; promote when memorize ceilings.
    """
    tcfg = cfg["training"]
    mechanism = tcfg.get("_mechanism", "memorize")

    if mechanism == "memorize":
        train_tasks, _ = get_split()
        mcfg = tcfg["memorize"]
        n = int(mcfg.get("n_train_samples", 5))
        cot_chains = mcfg.get("cot_chains") or {}
        task_ids_override = mcfg.get("task_ids")
        if task_ids_override:
            by_id = {t["task_id"]: t for t in train_tasks}
            samples = [by_id[tid] for tid in task_ids_override if tid in by_id]
            n = len(samples)
        elif mcfg.get("stratify_by_domain"):
            # Round-robin across domains so K=N covers up to N distinct domains.
            # When the train set has fewer unique domains than N, falls back to
            # extra picks per domain in domain-sorted order.
            buckets: dict[str, list[dict]] = {}
            for t in train_tasks:
                buckets.setdefault(t.get("domain", "unknown"), []).append(t)
            samples = []
            i = 0
            while len(samples) < n:
                for d in sorted(buckets):
                    if i < len(buckets[d]):
                        samples.append(buckets[d][i])
                        if len(samples) >= n:
                            break
                i += 1
                if i > 100:
                    break
        else:
            samples = train_tasks[:n]
        per_sample: list[dict] = []
        t_start = time.time()
        for i, t in enumerate(samples):
            t0 = time.time()
            chain = cot_chains.get(t["task_id"])
            response = (f"{chain}\n\nAnswer: {t['expected']}" if chain
                        else f"Answer: {t['expected']}")
            payload = {
                "prompt": t["prompt"],
                "response": response,
                "max_steps": int(mcfg.get("max_steps", 30)),
                "threshold": float(mcfg.get("threshold", 0.95)),
                "plateau_patience": int(mcfg.get("plateau_patience", 3)),
                "weight": float(mcfg.get("weight", 1.0)),
            }
            if mcfg.get("lr") is not None:
                payload["lr"] = float(mcfg["lr"])
            r = await client.post(f"{base}/v1/train/memorize", json=payload, timeout=600.0)
            r.raise_for_status()
            res = r.json()
            per_sample.append({
                "i": i,
                "task_id": t["task_id"],
                "steps": res.get("steps"),
                "reason": res.get("reason"),
                "commit_token": res.get("commit_token"),
                "wall_s": time.time() - t0,
            })
        return {
            "mechanism": "memorize",
            "n_train_samples": n,
            "per_sample": per_sample,
            "wall_s": time.time() - t_start,
        }

    if mechanism == "hybrid_presample_unlike":
        # Feedback-guided loss (SOTA track 1), PRE-sample variant of the
        # hybrid. Earlier hybrid_memorize_unlike found that after memorize
        # the sampling distribution is too tight to produce wrong rollouts
        # (the unlike pass never fires). This variant samples FIRST (at
        # the still-cold distribution), pushes unlike on whatever's wrong,
        # then memorizes on the demonstration. Tests whether killing off
        # the cold wrong-modes BEFORE memorize improves final convergence.
        train_tasks, _ = get_split()
        hcfg = tcfg["hybrid_presample_unlike"]
        n = int(hcfg.get("n_train_samples", 5))
        k = int(hcfg.get("k_rollouts", 4))
        samples = train_tasks[:n]
        per_sample: list[dict] = []
        t_start = time.time()
        for i, t in enumerate(samples):
            t0 = time.time()
            # PHASE A — sample at cold distribution + unlike on wrong.
            rollouts = []
            for _ in range(k):
                r = await client.post(
                    f"{base}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": t["prompt"]}],
                        "temperature": float(hcfg.get("sampling_temperature", 0.8)),
                        "top_p": float(hcfg.get("sampling_top_p", 0.95)),
                        "max_tokens": int(hcfg.get("max_new_tokens", 384)),
                        "enable_thinking": False,
                    },
                    timeout=120.0,
                )
                r.raise_for_status()
                rollouts.append(r.json()["choices"][0]["message"]["content"])
            grades = [registry_verify("logical", t["prompt"], c) for c in rollouts]
            wrong = [c for c, g in zip(rollouts, grades) if g is False]
            unlike_commit = None
            if wrong:
                spec = {
                    "objective": "unlike",
                    "samples": [{"prompt": t["prompt"], "response": c} for c in wrong],
                    "chunk_size": len(wrong),
                }
                r = await client.post(f"{base}/v1/train", json=spec, timeout=300.0)
                r.raise_for_status()
                unlike_commit = r.json().get("commit_token")
            # PHASE B — memorize on demonstration (positive signal).
            mem_payload = {
                "prompt": t["prompt"],
                "response": f"Answer: {t['expected']}",
                "max_steps": int(hcfg.get("memorize_max_steps", 100)),
                "threshold": float(hcfg.get("memorize_threshold", 0.95)),
                "plateau_patience": int(hcfg.get("memorize_plateau_patience", 10)),
                "weight": float(hcfg.get("memorize_weight", 1.0)),
            }
            if hcfg.get("memorize_lr") is not None:
                mem_payload["lr"] = float(hcfg["memorize_lr"])
            r = await client.post(f"{base}/v1/train/memorize", json=mem_payload, timeout=600.0)
            r.raise_for_status()
            mem_res = r.json()
            per_sample.append({
                "i": i,
                "task_id": t["task_id"],
                "k_rollouts": k,
                "cold_wrong_count": len(wrong),
                "unlike_commit": unlike_commit,
                "mem_steps": mem_res.get("steps"),
                "mem_reason": mem_res.get("reason"),
                "mem_commit": mem_res.get("commit_token"),
                "wall_s": time.time() - t0,
            })
        return {
            "mechanism": "hybrid_presample_unlike",
            "n_train_samples": n,
            "k_rollouts": k,
            "per_sample": per_sample,
            "wall_s": time.time() - t_start,
        }

    if mechanism == "self_synth":
        # Self-synthesis / mental modeling (SOTA track 3). For each train
        # task, ask the model to generate M PARAPHRASES of the question.
        # Test each paraphrase at T=0; keep ones that produce the correct
        # answer (model still solves the paraphrased version). Memorize on
        # each kept (paraphrase, expected) pair. Net: same fact, multiple
        # surface forms — the model trains itself by reformulating the
        # original question. Tests whether the model can act as its own
        # data-augmenter for tighter generalization.
        train_tasks, _ = get_split()
        scfg = tcfg["self_synth"]
        n = int(scfg.get("n_train_samples", 5))
        m = int(scfg.get("m_paraphrases", 3))
        samples = train_tasks[:n]
        per_sample: list[dict] = []
        t_start = time.time()
        for i, t in enumerate(samples):
            t0 = time.time()
            # Ask the model to rephrase the question, preserving meaning.
            paraphrase_prompt = (
                f"Rephrase the following question {m} different ways. Preserve "
                f"the exact meaning so the answer stays the same. Each rephrasing "
                f"must end with: End your reply with: Answer: <value>.\n"
                f"Number each rephrasing 1..{m}, one per line, no extra prose.\n\n"
                f"Original: {t['prompt']}"
            )
            r = await client.post(
                f"{base}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": paraphrase_prompt}],
                    "temperature": 0.9,
                    "top_p": 0.95,
                    "max_tokens": 768,
                    "enable_thinking": False,
                },
                timeout=120.0,
            )
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            # Naive parse: split by lines starting with N. or N).
            import re as _re
            paraphrases = []
            for line in raw.split("\n"):
                line = line.strip()
                if _re.match(r"^\d+[\.\)]\s+", line):
                    paraphrases.append(_re.sub(r"^\d+[\.\)]\s+", "", line))
            paraphrases = paraphrases[:m]
            # Verify each paraphrase: ask the model at T=0, check verifier.
            kept: list[str] = []
            for p in paraphrases:
                r = await client.post(
                    f"{base}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": p}],
                        "temperature": 0.0,
                        "max_tokens": 384,
                        "enable_thinking": False,
                    },
                    timeout=60.0,
                )
                if r.status_code != 200:
                    continue
                cand = r.json()["choices"][0]["message"]["content"]
                # Treat the original task's expected answer as ground truth.
                # Re-use the compare logic via probe verify helper.
                if _verify_probe(p, cand, t):
                    kept.append(p)
            # Memorize on the ORIGINAL plus each kept paraphrase.
            response = f"Answer: {t['expected']}"
            mem_results = []
            for variant_idx, prompt_variant in enumerate([t["prompt"]] + kept):
                mem_payload = {
                    "prompt": prompt_variant,
                    "response": response,
                    "max_steps": int(scfg.get("memorize_max_steps", 100)),
                    "threshold": float(scfg.get("memorize_threshold", 0.95)),
                    "plateau_patience": int(scfg.get("memorize_plateau_patience", 10)),
                    "weight": float(scfg.get("memorize_weight", 1.0)),
                }
                if scfg.get("memorize_lr") is not None:
                    mem_payload["lr"] = float(scfg["memorize_lr"])
                r = await client.post(f"{base}/v1/train/memorize", json=mem_payload, timeout=600.0)
                r.raise_for_status()
                res = r.json()
                mem_results.append({
                    "variant": "original" if variant_idx == 0 else f"paraphrase_{variant_idx-1}",
                    "steps": res.get("steps"),
                    "reason": res.get("reason"),
                })
            per_sample.append({
                "i": i,
                "task_id": t["task_id"],
                "m_requested": m,
                "m_parsed": len(paraphrases),
                "m_kept": len(kept),
                "mem_results": mem_results,
                "wall_s": time.time() - t0,
            })
        return {
            "mechanism": "self_synth",
            "n_train_samples": n,
            "m_paraphrases": m,
            "per_sample": per_sample,
            "wall_s": time.time() - t_start,
        }

    if mechanism == "entity_masked_sft":
        # SOTA track 2 supplement (sparse-supervision SFT). Memorize loss
        # currently covers the entire response including the "Answer: "
        # boilerplate. With T3.1 span_prefix, the loss masks past the
        # prefix and only supervises the final answer span. Hypothesis:
        # gradient sparsity → tighter learning + better generalization
        # because the model isn't wasting gradient on prefix tokens it
        # already knows how to emit.
        #
        # Implementation: bypass memorize, call /v1/train weighted_sft
        # with span_prefix="Answer: " per sample, n_passes_per_task SGD
        # steps to match memorize's depth budget.
        train_tasks, _ = get_split()
        ecfg = tcfg["entity_masked_sft"]
        n = int(ecfg.get("n_train_samples", 5))
        passes = int(ecfg.get("n_passes_per_task", 30))
        samples = train_tasks[:n]
        per_sample: list[dict] = []
        t_start = time.time()
        for i, t in enumerate(samples):
            t0 = time.time()
            response = f"Answer: {t['expected']}"
            for _ in range(passes):
                spec = {
                    "objective": "weighted_sft",
                    "samples": [{
                        "prompt": t["prompt"],
                        "response": response,
                        "span_prefix": "Answer: ",
                        "weight": float(ecfg.get("weight", 1.0)),
                    }],
                    "chunk_size": 1,
                }
                r = await client.post(f"{base}/v1/train", json=spec, timeout=300.0)
                r.raise_for_status()
            per_sample.append({
                "i": i,
                "task_id": t["task_id"],
                "passes": passes,
                "span_prefix": "Answer: ",
                "wall_s": time.time() - t0,
            })
        return {
            "mechanism": "entity_masked_sft",
            "n_train_samples": n,
            "passes_per_task": passes,
            "per_sample": per_sample,
            "wall_s": time.time() - t_start,
        }

    if mechanism == "rlvr":
        return {"mechanism": "rlvr", "step": "stub",
                "note": "rlvr arm not wired yet — use hybrid_presample_unlike or self_synth"}

    if mechanism == "stacked":
        # Run multiple sub-mechanisms back-to-back inside the same snapshot
        # bracket. Hypothesis: memorize first lifts forward (held-out) to 0.60,
        # then entity_masked_sft tightens probe to 0.93 without losing the
        # forward gain. Cheaper than designing a new objective and tests
        # whether the two recipes' strengths compose.
        stacked = tcfg.get("stacked", {})
        phases = stacked.get("phases", ["memorize", "entity_masked_sft"])
        sub_results = []
        t_start = time.time()
        for phase in phases:
            sub_cfg = {**cfg, "training": {**tcfg, "_mechanism": phase}}
            sub = await _run_training_pulse(client, base, sub_cfg)
            sub_results.append({"phase": phase, "result": sub})
        return {
            "mechanism": "stacked",
            "phases": phases,
            "sub_results": sub_results,
            "wall_s": time.time() - t_start,
        }

    raise ValueError(f"unknown training mechanism: {mechanism!r}")


async def main() -> int:
    cfg = _load_config()
    base = cfg["daemon_url"].rstrip("/")
    baseline = cfg["baseline_snapshot"]
    sampling = cfg["eval"]
    train_tasks, heldout = get_split()
    probe_tasks = _load_probe()

    print(f"[autoresearch] config: {json.dumps(cfg, sort_keys=True)}", flush=True)
    print(f"[autoresearch] tasks: {len(train_tasks)} train / {len(heldout)} heldout / "
          f"{len(probe_tasks)} probe", flush=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # Pre-flight: daemon alive?
        try:
            h = await client.get(f"{base}/health", timeout=3.0)
            h.raise_for_status()
        except Exception as exc:
            print(f"[autoresearch] FATAL: daemon unreachable at {base} — {exc}", flush=True)
            return 1

        # PHASE 0 — cold probe (baseline for non-degradation)
        print("[autoresearch] cold probe (pre-training prior competence) …", flush=True)
        t0 = time.time()
        probe_cold = await _eval_set(client, base, probe_tasks, sampling,
                                     use_registry=False, label="probe_cold")
        probe_cold["wall_s"] = time.time() - t0

        # PHASE 1 — snapshot
        print(f"[autoresearch] snapshot/save → {baseline}", flush=True)
        await _snapshot_save(client, base, baseline)

        try:
            # PHASE 2 — training pulse
            print("[autoresearch] training pulse …", flush=True)
            t0 = time.time()
            train_stats = await _run_training_pulse(client, base, cfg)
            train_stats["wall_s"] = time.time() - t0
            print(f"[autoresearch] training done: {json.dumps(train_stats, sort_keys=True)}",
                  flush=True)

            # PHASE 3 — held-out eval (forward learning)
            print("[autoresearch] held-out eval (forward learning) …", flush=True)
            t0 = time.time()
            heldout_stats = await _eval_set(client, base, heldout, sampling,
                                            use_registry=True, label="heldout_post")
            heldout_stats["wall_s"] = time.time() - t0

            # PHASE 4 — probe re-eval (non-degradation check)
            print("[autoresearch] probe re-eval (non-degradation check) …", flush=True)
            t0 = time.time()
            probe_post = await _eval_set(client, base, probe_tasks, sampling,
                                         use_registry=False, label="probe_post")
            probe_post["wall_s"] = time.time() - t0
        finally:
            # Always rewind to baseline so the next experiment starts clean.
            print(f"[autoresearch] snapshot/load ← {baseline}", flush=True)
            await _snapshot_load(client, base, baseline)

    # Composite score: forward learning minus weighted degradation.
    forward = heldout_stats["rate"]
    prior_cold = probe_cold["rate"]
    prior_post = probe_post["rate"]
    degrade = max(0.0, prior_cold - prior_post)
    score = forward - _LAMBDA_DEGRADE * degrade

    print(
        f"[autoresearch] forward={forward:.4f}  prior_cold={prior_cold:.4f}  "
        f"prior_post={prior_post:.4f}  degrade={degrade:.4f}  "
        f"lambda={_LAMBDA_DEGRADE}",
        flush=True,
    )
    # The autoresearch loop grep-extracts this line. Keep the format stable.
    print(f"score: {score:.4f}", flush=True)
    print(f"components: {json.dumps({'forward': forward, 'prior_cold': prior_cold, 'prior_post': prior_post, 'degrade': degrade, 'lambda': _LAMBDA_DEGRADE}, sort_keys=True)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
