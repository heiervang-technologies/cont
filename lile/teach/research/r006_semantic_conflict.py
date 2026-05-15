"""R-006 experiment runner: memorize retention under semantic conflict.

Tests whether two competing answers to the same prompt (response_A then
response_B) cause the model to forget A, interpolate, or converge on B.

Usage::

    python -m lile.teach.research.r006_semantic_conflict \
        --endpoint http://127.0.0.1:8768 \
        --n-pairs 20 \
        --seed 42 \
        --baseline-name R006_baseline \
        --out lile_data/research/R006/results.jsonl

    # Smoke (3 pairs, no intervening facts):
    python -m lile.teach.research.r006_semantic_conflict \
        --endpoint http://127.0.0.1:8768 \
        --n-pairs 3 \
        --K 0 \
        --baseline-name R006_smoke \
        --out lile_data/research/R006_smoke/results.jsonl

Design
------
For each collision triple (prompt, A, B):
1. Snapshot baseline.
2. Memorize A, eval recall(A) as baseline_ref.
3. **Warm-LoRA arm:** memorize B directly (continuing on the same LoRA that
   was just trained on A). Eval recall(A) and recall(B) at K in {0, 1, 5, 10}
   intervening unrelated facts (if K > 0).
4. Load baseline snapshot.
5. **Cold-LoRA arm:** memorize B on a *fresh* LoRA (snapshot restored to
   pre-A state, so the model has no prior exposure to this prompt). Eval
   recall(A) and recall(B) at the same K values.
6. Load baseline snapshot (restore for next triple).

This isolates the warm-LoRA effect: if recall(A) collapses in arm 3 but
not in arm 5, the overwrite is a LoRA-capacity artifact, not a semantic-
conflict property. If both arms show the same pattern, the conflict is
genuinely about the competing answer.

Two learning regimes:
- Default:  max_steps=30, threshold=0.95, plateau_patience=3
- Strong:   max_steps=100, plateau_patience=10, lr=5e-4, threshold=0.95
  (--regime strong)

Prerequisites
-------------
- A running lile daemon with ``/v1/eval/greedy_rank`` (PR #6) and
  ``/v1/train/memorize``.
- Daemon must have a snapshot baseline that includes the base model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx


def _post_json(client: httpx.Client, path: str, payload: dict[str, Any],
               timeout: float = 300.0) -> dict[str, Any]:
    r = client.post(path, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _memorize_and_eval(
    client: httpx.Client,
    prompt: str,
    response: str,
    memorize_params: dict[str, Any],
) -> dict[str, Any]:
    """Memorize a fact and return the memorize result + eval on the same fact."""
    mem = _post_json(client, "/v1/train/memorize", {
        "prompt": prompt,
        "response": response,
        **memorize_params,
    })
    ev = _post_json(client, "/v1/eval/greedy_rank", {
        "prompt": prompt,
        "response": response,
    })
    return {
        "mem_steps": mem.get("steps"),
        "mem_reason": mem.get("reason"),
        "mem_commit_token": mem.get("commit_token"),
        "eval_fraction": ev.get("fraction"),
        "eval_matched": ev.get("matched"),
        "eval_total": ev.get("total"),
        "eval_commit_token": ev.get("commit_token"),
    }


def run_experiment(
    endpoint: str,
    n_pairs: int = 20,
    seed: int = 42,
    baseline_name: str = "R006_baseline",
    out_path: str = "lile_data/research/R006/results.jsonl",
    regimes: list[str] | None = None,
    K_values: list[int] | None = None,
    n_intervening: int = 5,
    warm_lora: bool = True,
    cold_lora: bool = True,
) -> Path:
    """Run the R-006 semantic conflict experiment.

    Parameters
    ----------
    endpoint : str
        Daemon base URL.
    n_pairs : int
        Number of collision triples.
    seed : int
        RNG seed for fixture generation.
    baseline_name : str
        Snapshot name for baseline save/load.
    out_path : str
        Path for results JSONL.
    regimes : list[str] | None
        Learning regimes: "default", "strong", or both (default).
    K_values : list[int] | None
        Intervening-fact counts to test. Default [0, 1, 5, 10].
    n_intervening : int
        Number of unrelated facts available for K > 0 tests.
    warm_lora : bool
        Run warm-LoRA arm (memorize B on same LoRA as A).
    cold_lora : bool
        Run cold-LoRA arm (memorize B on fresh LoRA from snapshot).
    """
    from lile.teach.research_fixtures.collision_facts import generate_collision_pairs
    from lile.teach.research_fixtures.mythical_facts import generate_facts

    if regimes is None:
        regimes = ["default", "strong"]
    if K_values is None:
        K_values = [0, 1, 5, 10]

    _REGIME_PARAMS = {
        "default": {"max_steps": 30, "threshold": 0.95, "plateau_patience": 3},
        "strong":  {"max_steps": 100, "threshold": 0.95, "plateau_patience": 10, "lr": 5e-4},
    }

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Generate collision triples + unrelated facts for K > 0 tests.
    pairs = generate_collision_pairs(n_pairs=n_pairs, seed=seed)
    # Generate extra unrelated facts for K tests (different seed to avoid overlap).
    unrelated = generate_facts(n=n_intervening, seed=seed + 1)

    header = {
        "kind": "R006_header",
        "n_pairs": n_pairs,
        "seed": seed,
        "n_intervening": n_intervening,
        "regimes": regimes,
        "K_values": K_values,
        "warm_lora": warm_lora,
        "cold_lora": cold_lora,
        "baseline_name": baseline_name,
    }
    with out_p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")

    with httpx.Client(base_url=endpoint.rstrip("/")) as client:
        # --- snapshot baseline ---
        print(f"[R006] snapshot/save → {baseline_name}")
        _post_json(client, "/v1/state/snapshot/save", {"name": baseline_name})

        for idx, triple in enumerate(pairs):
            prompt = triple["prompt"]
            a_text = triple["response_A"]
            b_text = triple["response_B"]
            family = triple.get("family", "unknown")

            for regime in regimes:
                params = _REGIME_PARAMS[regime]

                for arm in ("warm", "cold"):
                    if arm == "cold" and not cold_lora:
                        continue
                    if arm == "warm" and not warm_lora:
                        continue

                    # For the cold-LoRA arm, restore baseline so LoRA is clean.
                    if arm == "cold":
                        _post_json(client, "/v1/state/snapshot/load",
                                   {"name": baseline_name})

                    # --- step 1: memorize A, eval A ---
                    t0 = time.time()
                    mem_a = _memorize_and_eval(client, prompt, a_text, params)
                    baseline_a = mem_a["eval_fraction"]

                    # --- step 2: memorize B ---
                    mem_b = _memorize_and_eval(client, prompt, b_text, params)

                    # --- step 3: eval A and B at each K ---
                    eval_a_at_k: list[float] = []
                    eval_b_at_k: list[float] = []
                    for ki, K in enumerate(K_values):
                        if K > 0 and ki > 0:
                            # Insert intervening unrelated facts
                            for j in range(K - (K_values[ki - 1] if ki > 0 else 0)):
                                u = unrelated[(idx + j) % len(unrelated)]
                                _post_json(client, "/v1/train/memorize", {
                                    "prompt": u["prompt"],
                                    "response": u["response"],
                                    **params,
                                })

                        # Eval A and B at this K point
                        ev_a = _post_json(client, "/v1/eval/greedy_rank", {
                            "prompt": prompt, "response": a_text,
                        })
                        ev_b = _post_json(client, "/v1/eval/greedy_rank", {
                            "prompt": prompt, "response": b_text,
                        })
                        eval_a_at_k.append(ev_a.get("fraction"))
                        eval_b_at_k.append(ev_b.get("fraction"))

                    elapsed = time.time() - t0

                    record = {
                        "i": idx,
                        "family": family,
                        "regime": regime,
                        "arm": arm,
                        "response_A": a_text,
                        "response_B": b_text,
                        "mem_A_steps": mem_a["mem_steps"],
                        "mem_A_reason": mem_a["mem_reason"],
                        "mem_A_commit_token": mem_a["mem_commit_token"],
                        "baseline_A_fraction": baseline_a,
                        "mem_B_steps": mem_b["mem_steps"],
                        "mem_B_reason": mem_b["mem_reason"],
                        "mem_B_commit_token": mem_b["mem_commit_token"],
                        "mem_B_fraction": mem_b["eval_fraction"],
                        "eval_A_at_K": eval_a_at_k,
                        "eval_B_at_K": eval_b_at_k,
                        "wall_s": elapsed,
                    }
                    with out_p.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                    progress = (idx + 1) / n_pairs * 100
                    print(
                        f"[R006] {idx+1}/{n_pairs} ({progress:.0f}%) "
                        f"{regime}/{arm} "
                        f"A_steps={mem_a['mem_steps']} "
                        f"B_steps={mem_b['mem_steps']} "
                        f"A_postB={eval_a_at_k[0]:.3f} "
                        f"B_postB={eval_b_at_k[0]:.3f} "
                        f"wall={elapsed:.1f}s"
                    )

                    # If we're in the warm-LoRA arm, restore baseline before
                    # the cold-LoRA arm (which will load baseline on its own).
                    # Actually, cold-LoRA arm loads baseline itself above.
                    # But for warm-LoRA, restore now to start next triple clean.
                    if arm == "warm":
                        _post_json(client, "/v1/state/snapshot/load",
                                   {"name": baseline_name})

        # --- restore baseline ---
        print(f"[R006] snapshot/load ← {baseline_name}")
        _post_json(client, "/v1/state/snapshot/load", {"name": baseline_name})

    return out_p


def main() -> None:
    parser = argparse.ArgumentParser(description="R-006: semantic conflict")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8768")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-name", default="R006_baseline")
    parser.add_argument("--out", default="lile_data/research/R006/results.jsonl")
    parser.add_argument("--regime", choices=["default", "strong", "both"],
                        default="both")
    parser.add_argument("--K", type=int, nargs="*",
                        help="K values for intervening facts (default: 0 1 5 10)")
    parser.add_argument("--n-intervening", type=int, default=5,
                        help="Number of unrelated facts available for K tests")
    parser.add_argument("--no-warm-lora", action="store_true",
                        help="Skip warm-LoRA arm")
    parser.add_argument("--no-cold-lora", action="store_true",
                        help="Skip cold-LoRA arm")
    args = parser.parse_args()

    regimes = ["default", "strong"] if args.regime == "both" else [args.regime]
    K_values = args.K if args.K else [0, 1, 5, 10]

    path = run_experiment(
        endpoint=args.endpoint,
        n_pairs=args.n_pairs,
        seed=args.seed,
        baseline_name=args.baseline_name,
        out_path=args.out,
        regimes=regimes,
        K_values=K_values,
        n_intervening=args.n_intervening,
        warm_lora=not args.no_warm_lora,
        cold_lora=not args.no_cold_lora,
    )
    print(f"[R006] done → {path}")


if __name__ == "__main__":
    main()
