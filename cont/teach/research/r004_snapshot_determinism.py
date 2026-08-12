"""R-004 experiment runner: snapshot-load determinism after memorize.

Usage::

    python -m cont.teach.research.r004_snapshot_determinism \
        --endpoint http://127.0.0.1:8768 \
        --seed 42 \
        --n-chats 10

Prerequisites:
- A running trainfer daemon on the given endpoint.
- Daemon must have the ``/v1/eval/greedy_rank`` route.

Steps (mirrors BACKLOG.md R-004 Experiment):
1. Load _autosave (clean baseline).
2. Generate fact F and fact G (two distinct mythical facts).
3. Baseline eval: greedy-rank on F and G (pre-memorize).
4. Memorize fact F. Eval greedy-rank on F → F_recall_at_save.
5. Save snapshot R004_mid.
6. Run n unrelated chats.
7. Memorize fact G. Eval greedy-rank on F and G → F_recall_after_G, G_recall_after_mem.
8. Save snapshot R004_after.
9. Load snapshot R004_mid.
10. Eval greedy-rank on F and G → F_recall_after_load, G_recall_after_load.
11. Compare: F_recall_after_load == F_recall_at_save (byte-exact),
    G_recall_after_load ≈ G_baseline (not memorized yet at R004_mid time).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

# Unrelated chat prompts for the "interference" step
_UNRELATED_PROMPTS = [
    "Write a haiku about rain.",
    "What is 7 times 8?",
    "Name three primary colors.",
    "Explain why the sky is blue in one sentence.",
    "What is the speed of light approximately?",
    "Give me a recipe for scrambled eggs.",
    "Who wrote Hamlet?",
    "What is the capital of Japan?",
    "Describe a sunset in two sentences.",
    "What is 2 + 2?",
    "Name a mammal that lives in the ocean.",
    "What year did the Titanic sink?",
    "Explain osmosis briefly.",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water?",
]


def _post_json(
    client: httpx.Client, path: str, payload: dict[str, Any]
) -> dict[str, Any]:
    r = client.post(path, json=payload, timeout=300.0)
    r.raise_for_status()
    return r.json()


def _eval_greedy(client: httpx.Client, prompt: str, response: str) -> dict[str, Any]:
    return _post_json(
        client, "/v1/eval/greedy_rank", {"prompt": prompt, "response": response}
    )


def run_experiment(
    endpoint: str = "http://127.0.0.1:8768",
    seed: int = 42,
    n_chats: int = 10,
    out_dir: str = "data/research/R004",
) -> Path:
    from cont.teach.research_fixtures.mythical_facts import generate_facts

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    results_path = out_p / "results.jsonl"

    # Generate two facts from different families to ensure surface-form disjointness
    facts_F = generate_facts(n=1, seed=seed, families=["capital"])
    facts_G = generate_facts(n=1, seed=seed + 1000, families=["motto"])
    fact_F = facts_F[0]
    fact_G = facts_G[0]

    log: list[dict[str, Any]] = []

    with httpx.Client(base_url=endpoint.rstrip("/")) as client:
        # Step 1: Load clean baseline
        print("[R004] loading _autosave baseline")
        _post_json(client, "/v1/state/snapshot/load", {"name": "_autosave"})

        # Step 2: Baseline eval
        print("[R004] baseline eval on F and G")
        F_baseline = _eval_greedy(client, fact_F["prompt"], fact_F["response"])
        G_baseline = _eval_greedy(client, fact_G["prompt"], fact_G["response"])
        log.append({"step": "baseline", "fact": "F", **F_baseline})
        log.append({"step": "baseline", "fact": "G", **G_baseline})
        print(f"  F baseline fraction={F_baseline['fraction']:.4f}")
        print(f"  G baseline fraction={G_baseline['fraction']:.4f}")

        # Step 3: Memorize fact F
        print("[R004] memorizing fact F")
        t0 = time.time()
        mem_payload_F = {
            "prompt": fact_F["prompt"],
            "response": fact_F["response"],
            "max_steps": 100,
            "threshold": 0.70,
            "plateau_patience": 10,
        }
        F_mem = _post_json(client, "/v1/train/memorize", mem_payload_F)
        F_mem_wall = time.time() - t0
        print(
            f"  F memorize: steps={F_mem.get('steps')}, reason={F_mem.get('reason')}, wall={F_mem_wall:.1f}s",
            flush=True,
        )

        # Step 4: Eval F after memorize (this is the gold standard we must recover)
        F_recall_at_save = _eval_greedy(client, fact_F["prompt"], fact_F["response"])
        log.append({"step": "post_memorize_F", "fact": "F", **F_recall_at_save})
        print(
            f"  F recall at save: fraction={F_recall_at_save['fraction']:.4f} matched={F_recall_at_save['matched']}/{F_recall_at_save['total']}"
        )

        # Step 5: Save snapshot R004_mid
        print("[R004] snapshot/save → R004_mid")
        _post_json(client, "/v1/state/snapshot/save", {"name": "R004_mid"})

        # Step 6: Run n unrelated chats
        print(f"[R004] running {n_chats} unrelated chats")
        for ci in range(n_chats):
            prompt = _UNRELATED_PROMPTS[ci % len(_UNRELATED_PROMPTS)]
            chat_res = _post_json(
                client,
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                },
            )
            content = (
                chat_res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")[:80]
            )
            print(f"  chat {ci + 1}/{n_chats}: {content[:50]}...")

        # Step 7: Memorize fact G
        print("[R004] memorizing fact G")
        t0 = time.time()
        mem_payload_G = {
            "prompt": fact_G["prompt"],
            "response": fact_G["response"],
            "max_steps": 100,
            "threshold": 0.70,
            "plateau_patience": 10,
        }
        G_mem = _post_json(client, "/v1/train/memorize", mem_payload_G)
        G_mem_wall = time.time() - t0
        print(
            f"  G memorize: steps={G_mem.get('steps')}, reason={G_mem.get('reason')}, wall={G_mem_wall:.1f}s",
            flush=True,
        )

        # Eval F and G after G memorize
        F_recall_after_G = _eval_greedy(client, fact_F["prompt"], fact_F["response"])
        G_recall_after_mem = _eval_greedy(client, fact_G["prompt"], fact_G["response"])
        log.append({"step": "after_G_memorize", "fact": "F", **F_recall_after_G})
        log.append({"step": "after_G_memorize", "fact": "G", **G_recall_after_mem})
        print(f"  F recall after G: fraction={F_recall_after_G['fraction']:.4f}")
        print(
            f"  G recall after memorize: fraction={G_recall_after_mem['fraction']:.4f}"
        )

        # Step 8: Save snapshot R004_after
        print("[R004] snapshot/save → R004_after")
        _post_json(client, "/v1/state/snapshot/save", {"name": "R004_after"})

        # Step 9: Load R004_mid (the rollback)
        print("[R004] snapshot/load ← R004_mid")
        _post_json(client, "/v1/state/snapshot/load", {"name": "R004_mid"})

        # Step 10: Eval F and G after load
        F_recall_after_load = _eval_greedy(client, fact_F["prompt"], fact_F["response"])
        G_recall_after_load = _eval_greedy(client, fact_G["prompt"], fact_G["response"])
        log.append({"step": "after_load_R004_mid", "fact": "F", **F_recall_after_load})
        log.append({"step": "after_load_R004_mid", "fact": "G", **G_recall_after_load})
        print(
            f"  F recall after load: fraction={F_recall_after_load['fraction']:.4f} matched={F_recall_after_load['matched']}/{F_recall_after_load['total']}"
        )
        print(
            f"  G recall after load: fraction={G_recall_after_load['fraction']:.4f} matched={G_recall_after_load['matched']}/{G_recall_after_load['total']}"
        )

        # Step 11: Verdict
        F_exact = (
            F_recall_after_load["fraction"] == F_recall_at_save["fraction"]
            and F_recall_after_load["matched"] == F_recall_at_save["matched"]
            and F_recall_after_load["total"] == F_recall_at_save["total"]
        )
        G_at_baseline = G_recall_after_load["fraction"] <= G_baseline["fraction"] + 0.01

        verdict = {
            "F_fact": fact_F,
            "G_fact": fact_G,
            "F_baseline": F_baseline,
            "G_baseline": G_baseline,
            "F_mem_result": F_mem,
            "G_mem_result": G_mem,
            "F_recall_at_save": F_recall_at_save,
            "F_recall_after_G": F_recall_after_G,
            "G_recall_after_mem": G_recall_after_mem,
            "F_recall_after_load": F_recall_after_load,
            "G_recall_after_load": G_recall_after_load,
            "F_recall_delta_load_vs_save": F_recall_after_load["fraction"]
            - F_recall_at_save["fraction"],
            "G_recall_delta_load_vs_baseline": G_recall_after_load["fraction"]
            - G_baseline["fraction"],
            "invariant4_holds": F_exact and G_at_baseline,
            "F_exact_match": F_exact,
            "G_at_baseline": G_at_baseline,
        }

        print("\n[R004] === VERDICT ===")
        print(
            f"  F recall at save:    {F_recall_at_save['fraction']:.4f} ({F_recall_at_save['matched']}/{F_recall_at_save['total']})"
        )
        print(
            f"  F recall after load: {F_recall_after_load['fraction']:.4f} ({F_recall_after_load['matched']}/{F_recall_after_load['total']})"
        )
        print(f"  F exact match: {F_exact}")
        print(f"  G baseline:    {G_baseline['fraction']:.4f}")
        print(f"  G after load:  {G_recall_after_load['fraction']:.4f}")
        print(f"  G at baseline: {G_at_baseline}")
        print(
            f"  Invariant 4 (snapshot round-trip): {'PASS' if verdict['invariant4_holds'] else 'FAIL'}"
        )

    # Write results
    with results_path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"kind": "R004_header", "seed": seed, "n_chats": n_chats},
                ensure_ascii=False,
            )
            + "\n"
        )
        for entry in log:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        fh.write(
            json.dumps({"kind": "R004_verdict", **verdict}, ensure_ascii=False) + "\n"
        )

    print(f"\n[R004] done → {results_path}")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R-004: snapshot-load determinism after memorize"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8768")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-chats", type=int, default=10)
    parser.add_argument("--out-dir", default="data/research/R004")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    run_experiment(
        endpoint=args.endpoint,
        seed=args.seed,
        n_chats=args.n_chats,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
