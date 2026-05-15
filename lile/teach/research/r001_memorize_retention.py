"""R-001 experiment runner: memorize retention under sequential insertion.

Usage::

    python -m lile.teach.research.r001_memorize_retention \
        --endpoint http://127.0.0.1:8768 \
        --n-facts 100 \
        --seed 42 \
        --baseline-name R001_baseline \
        --out lile_data/research/R001/results.jsonl

Prerequisites:
- A running lile daemon on the given endpoint.
- Daemon must have the ``/v1/eval/greedy_rank`` route (added by the R-001
  claim PR).

Steps (mirrors BACKLOG.md R-001 Experiment):
1. Generate ``n`` synthetic (prompt, response) pairs from
   ``mythical_facts.py``.
2. Snapshot the daemon to ``baseline_name``.
3. For i in 0..n-1:
   a. ``POST /v1/train/memorize`` on pair i (greedy SFT until threshold).
   b. ``POST /v1/eval/greedy_rank`` on pair 0 (retention probe).
   c. ``POST /v1/eval/greedy_rank`` on pair i (instantaneous retention).
   d. Append the triple to JSONL.
4. ``POST /v1/state/snapshot/load`` ``baseline_name`` to restore.
5. Print summary + path to the JSONL.

The script is restart-safe: the JSONL header is written with mode='w' (overwrites
any prior file), and loop records are appended incrementally so a mid-run crash
preserves data. A clean re-run always starts from a fresh file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def _post_json(client: httpx.Client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = client.base_url.join(path)
    r = client.post(path, json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


def run_experiment(
    endpoint: str,
    n_facts: int = 100,
    seed: int = 42,
    baseline_name: str = "R001_baseline",
    out_path: str = "lile_data/research/R001/results.jsonl",
) -> Path:
    from lile.teach.research_fixtures.mythical_facts import generate_facts

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    facts = generate_facts(n=n_facts, seed=seed)
    if not facts:
        raise RuntimeError("generated zero facts — check mythical_facts.py")

    with httpx.Client(base_url=endpoint.rstrip("/")) as client:
        # --- step 0: baseline eval (pre-training) ---
        print(f"[R001] baseline eval on pair 0 and pair {n_facts-1}")
        pair0_baseline = _post_json(client, "/v1/eval/greedy_rank", {
            "prompt": facts[0]["prompt"],
            "response": facts[0]["response"],
        })
        pair_last_baseline = _post_json(client, "/v1/eval/greedy_rank", {
            "prompt": facts[-1]["prompt"],
            "response": facts[-1]["response"],
        })
        header = {
            "kind": "R001_header",
            "n_facts": n_facts,
            "seed": seed,
            "baseline_name": baseline_name,
            "pair0_baseline_fraction": pair0_baseline.get("fraction"),
            "pair_last_baseline_fraction": pair_last_baseline.get("fraction"),
        }
        # Write header fresh (mode='w') so restarts don't duplicate it.
        with out_p.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(header, ensure_ascii=False) + "\n")

        # --- step 1: baseline snapshot (overwrite if exists)
        print(f"[R001] snapshot/save → {baseline_name}")
        _post_json(client, "/v1/state/snapshot/save", {"name": baseline_name})

        # --- step 2: sequential memorize + eval loop
        results: list[dict[str, Any]] = []
        for i, fact in enumerate(facts):
            t0 = time.time()

            # a) memorize
            memorize_res = _post_json(client, "/v1/train/memorize", {
                "prompt": fact["prompt"],
                "response": fact["response"],
                "max_steps": 30,
                "threshold": 0.95,
            })

            # b) eval pair 0 (retention probe)
            pair0_eval = _post_json(client, "/v1/eval/greedy_rank", {
                "prompt": facts[0]["prompt"],
                "response": facts[0]["response"],
            })

            # c) eval pair i (instantaneous)
            pairi_eval = _post_json(client, "/v1/eval/greedy_rank", {
                "prompt": fact["prompt"],
                "response": fact["response"],
            })

            elapsed = time.time() - t0
            record = {
                "i": i,
                "memorize_steps": memorize_res.get("steps"),
                "memorize_reason": memorize_res.get("reason"),
                "memorize_commit_token": memorize_res.get("commit_token"),
                "pair0_fraction": pair0_eval.get("fraction"),
                "pair0_matched": pair0_eval.get("matched"),
                "pair0_total": pair0_eval.get("total"),
                "pair0_commit_token": pair0_eval.get("commit_token"),
                "pairi_fraction": pairi_eval.get("fraction"),
                "pairi_matched": pairi_eval.get("matched"),
                "pairi_total": pairi_eval.get("total"),
                "pairi_commit_token": pairi_eval.get("commit_token"),
                "prompt_family": _infer_family(fact["prompt"]),
                "wall_s": elapsed,
            }
            results.append(record)
            # append to JSONL incrementally so a crash mid-run preserves data.
            with out_p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress = (i + 1) / n_facts * 100
            print(
                f"[R001] {i+1}/{n_facts} ({progress:.0f}%) "
                f"mem_steps={record['memorize_steps']} "
                f"pair0={record['pair0_fraction']:.3f} "
                f"pairi={record['pairi_fraction']:.3f} "
                f"wall={record['wall_s']:.1f}s"
            )

        # --- step 3: restore baseline
        print(f"[R001] snapshot/load ← {baseline_name}")
        _post_json(client, "/v1/state/snapshot/load", {"name": baseline_name})

    return out_p


def _infer_family(prompt: str) -> str:
    if "capital" in prompt:
        return "capital"
    if "flag" in prompt:
        return "flag"
    if "ruler" in prompt:
        return "ruler"
    if "motto" in prompt:
        return "motto"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="R-001: memorize retention")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8768")
    parser.add_argument("--n-facts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-name", default="R001_baseline")
    parser.add_argument("--out", default="lile_data/research/R001/results.jsonl")
    args = parser.parse_args()

    path = run_experiment(
        endpoint=args.endpoint,
        n_facts=args.n_facts,
        seed=args.seed,
        baseline_name=args.baseline_name,
        out_path=args.out,
    )
    print(f"[R001] done → {path}")


if __name__ == "__main__":
    main()
