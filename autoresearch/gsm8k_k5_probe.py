"""GSM8K-K5 probe: published-benchmark version of our winning recipe.

Question Stop-hook is asking: does the autoresearch may16 winning recipe
(memorize K=5 STRONG + enable_thinking=true at eval) transfer to a
published benchmark, or is it a property of our custom verifiable-logic
corpus?

Procedure:
- Load GSM8K (8.5K grade-school math problems with verifiable integer
  answers).
- Snapshot save → memorize on first 5 train problems → eval 100 held-out
  test problems → snapshot load.
- Compare cold pass rate (n_train=0 control) to post-train pass rate.
- Report against published baselines (LIMO 6.5→63.3 on AIME24 with 817
  examples; we use 5 on GSM8K).

GSM8K answer format is "#### <int>". For training response we use
"Answer: <int>" (our recipe's grammar). Eval extracts the same regex.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets")

from datasets import load_dataset  # noqa: E402

DAEMON = "http://127.0.0.1:8768"
BASELINE = "autoresearch_baseline"
EVAL = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 3000, "enable_thinking": True}
MEMORIZE = {
    "max_steps": 100,
    "threshold": 0.95,
    "plateau_patience": 10,
    "lr": 0.002,
    "weight": 1.0,
}
N_HELDOUT = 50  # 100 would take >1h with CoT eval; 50 still gives ±7pp on a binomial

GOLD_RE = re.compile(r"####\s*(-?\d+)")
ANSWER_RE = re.compile(r"(?is)Answer\s*[:=]\s*(-?\d+)")


def extract_gold(answer_field: str) -> str:
    m = GOLD_RE.search(answer_field)
    return m.group(1) if m else ""


def extract_pred(response: str) -> str:
    if not response:
        return ""
    matches = ANSWER_RE.findall(response)
    return matches[-1].strip() if matches else ""


async def chat(client: httpx.AsyncClient, prompt: str) -> str:
    payload = {"model": "Qwen3-8B",
               "messages": [{"role": "user", "content": prompt}],
               **EVAL}
    r = await client.post(f"{DAEMON}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def snapshot(client: httpx.AsyncClient, action: str, name: str) -> None:
    r = await client.post(f"{DAEMON}/v1/state/snapshot/{action}",
                          json={"name": name}, timeout=120.0)
    r.raise_for_status()


def fmt_prompt(question: str) -> str:
    return question + "\n\nEnd your reply with: Answer: <integer>."


async def train_memorize(client: httpx.AsyncClient, train_examples: list[dict]) -> list[dict]:
    out = []
    for ex in train_examples:
        gold = extract_gold(ex["answer"])
        if not gold:
            continue
        payload = {
            "prompt": fmt_prompt(ex["question"]),
            "response": f"Answer: {gold}",
            **MEMORIZE,
        }
        t0 = time.time()
        r = await client.post(f"{DAEMON}/v1/train/memorize", json=payload, timeout=600.0)
        r.raise_for_status()
        res = r.json()
        out.append({
            "question": ex["question"][:60],
            "gold": gold,
            "steps": res.get("steps"),
            "reason": res.get("reason"),
            "wall_s": time.time() - t0,
        })
    return out


async def eval_set(client: httpx.AsyncClient, examples: list[dict], label: str) -> dict:
    n_correct = 0
    per_task = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        gold = extract_gold(ex["answer"])
        if not gold:
            continue
        resp = await chat(client, fmt_prompt(ex["question"]))
        pred = extract_pred(resp)
        ok = pred == gold
        if ok:
            n_correct += 1
        per_task.append({"i": i, "gold": gold, "pred": pred, "ok": ok})
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(examples)}: running={n_correct}/{i+1} "
                  f"({100*n_correct/(i+1):.1f}%)", flush=True)
    rate = n_correct / len(examples)
    print(f"  [{label}] FINAL: {n_correct}/{len(examples)} ({100*rate:.1f}%) in "
          f"{time.time()-t0:.0f}s", flush=True)
    return {"rate": rate, "n_correct": n_correct, "total": len(examples), "per_task": per_task}


async def main() -> int:
    print(f"[gsm8k-k5] daemon={DAEMON} eval={EVAL}", flush=True)
    print("[gsm8k-k5] loading GSM8K main…", flush=True)
    ds = load_dataset("openai/gsm8k", "main", cache_dir="/tmp/hf-datasets")
    train = list(ds["train"])[:5]
    heldout = list(ds["test"])[:N_HELDOUT]
    print(f"[gsm8k-k5] K=5 train tasks, n={N_HELDOUT} heldout", flush=True)
    for ex in train:
        gold = extract_gold(ex["answer"])
        print(f"  train: gold={gold!r} | {ex['question'][:80]}…", flush=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        print()
        print("[gsm8k-k5] snapshot/save → autoresearch_baseline", flush=True)
        await snapshot(client, "save", BASELINE)
        try:
            print()
            print("[gsm8k-k5] PHASE A — COLD eval on 50 heldout (no training)", flush=True)
            cold = await eval_set(client, heldout, "cold")

            print()
            print("[gsm8k-k5] PHASE B — memorize K=5 on GSM8K-train", flush=True)
            train_results = await train_memorize(client, train)
            for tr in train_results:
                print(f"  trained: gold={tr['gold']!r} steps={tr['steps']} reason={tr['reason']} "
                      f"in {tr['wall_s']:.1f}s", flush=True)

            print()
            print("[gsm8k-k5] PHASE C — POST eval on same 50 heldout", flush=True)
            post = await eval_set(client, heldout, "post")
        finally:
            print()
            print("[gsm8k-k5] snapshot/load ← autoresearch_baseline", flush=True)
            await snapshot(client, "load", BASELINE)

    delta = post["rate"] - cold["rate"]
    print()
    print("=" * 60, flush=True)
    print(f"GSM8K cold pass rate : {cold['n_correct']}/{cold['total']} ({100*cold['rate']:.1f}%)", flush=True)
    print(f"GSM8K post pass rate : {post['n_correct']}/{post['total']} ({100*post['rate']:.1f}%)", flush=True)
    print(f"K=5 memorize delta   : {100*delta:+.1f}pp", flush=True)
    print("=" * 60, flush=True)
    print("comparison to literature:", flush=True)
    print(f"  LIMO Qwen2.5-32B  (n_train=817):  AIME24 6.5→63.3 (+56.8pp) — different model & benchmark", flush=True)
    print(f"  s1   Qwen2.5-32B  (n_train=1000): AIME24 +27pp over o1-preview — different model & benchmark", flush=True)
    print(f"  OURS Qwen3-8B-4bit (n_train=5):    GSM8K {100*cold['rate']:.0f}→{100*post['rate']:.0f} ({100*delta:+.0f}pp) — directly comparable axis: data efficiency", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
