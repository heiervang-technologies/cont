"""Fact-use-vs-reproduce diagnostic for the autoresearch may16 winning recipe.

Question the metric does not answer: when memorize K=5 + CoT-eval scores 0.70
forward, is the model *reasoning* through the held-out problems, or just
*reproducing* the trained `Answer: <value>` format on prompts that happen to
share surface structure with train?

Discriminator: take each of the 5 train tasks the model trained on, swap the
*numbers* (keep template identical), and re-evaluate. If the model still
returns the *trained* answer for the *new* numbers, it memorized. If it
returns the *correct* answer for the new numbers, it learned the algorithm.

Output is intentionally per-task verbose so we can audit which tasks
generalize and which do not.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainfer.objectives.verifiers.corpora.logical import get_split  # noqa: E402

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

# Number-substituted variants of the 5 K=5 train tasks. Same template, new
# numbers. The "memorized" answer is the original train answer (what the
# model should NOT say if it has merely memorized). The "correct" answer is
# what the model should say if it has learned the underlying algorithm.
VARIANTS = [
    {
        "task_id": "logical/arith/0",
        "original_prompt": ("A train travels 60 km in the first hour, then doubles its speed "
                            "for the second hour. How many kilometers does it travel in 2 hours? "
                            "End your reply with: Answer: <integer>."),
        "original_answer": "180",
        "variant_prompt": ("A train travels 80 km in the first hour, then doubles its speed "
                           "for the second hour. How many kilometers does it travel in 2 hours? "
                           "End your reply with: Answer: <integer>."),
        "variant_correct": "240",
    },
    {
        "task_id": "logical/arith/1",
        "original_prompt": ("Alice has 3 times as many apples as Bob. Together they have 28 apples. "
                            "How many apples does Bob have? End your reply with: Answer: <integer>."),
        "original_answer": "7",
        "variant_prompt": ("Alice has 4 times as many apples as Bob. Together they have 30 apples. "
                           "How many apples does Bob have? End your reply with: Answer: <integer>."),
        "variant_correct": "6",
    },
    {
        "task_id": "logical/bool_eval/0",
        "original_prompt": ("Given a=1, b=0, c=1, evaluate (a AND b) OR (c AND NOT b). "
                            "End your reply with: Answer: <0|1>."),
        "original_answer": "1",
        "variant_prompt": ("Given a=0, b=1, c=0, evaluate (a AND b) OR (c AND NOT b). "
                           "End your reply with: Answer: <0|1>."),
        "variant_correct": "0",  # (0 AND 1)=0, (0 AND NOT 1)=(0 AND 0)=0, 0 OR 0 = 0
    },
    {
        "task_id": "logical/bool_eval/1",
        "original_prompt": ("Given p=true, q=true, r=false, evaluate (p AND (q OR r)) AND NOT (p AND r). "
                            "End your reply with: Answer: <true|false>."),
        "original_answer": "true",
        "variant_prompt": ("Given p=true, q=false, r=true, evaluate (p AND (q OR r)) AND NOT (p AND r). "
                           "End your reply with: Answer: <true|false>."),
        # p=true, q=false, r=true. (q OR r)=(false OR true)=true. (p AND true)=true.
        # (p AND r)=(true AND true)=true. NOT true=false. true AND false = false.
        "variant_correct": "false",
    },
    {
        "task_id": "logical/counting/0",
        "original_prompt": ("How many distinct 3-letter strings can be formed from the letters A, B, C "
                            "where each letter is used exactly once? End your reply with: Answer: <integer>."),
        "original_answer": "6",
        "variant_prompt": ("How many distinct 4-letter strings can be formed from the letters A, B, C, D "
                           "where each letter is used exactly once? End your reply with: Answer: <integer>."),
        "variant_correct": "24",  # 4! = 24
    },
]

ANSWER_RE = re.compile(r"(?is)Answer\s*[:=]\s*([^\n<]+)")


def extract_answer(text: str) -> str:
    if not text:
        return ""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].strip().lower().rstrip(".")


async def chat(client: httpx.AsyncClient, prompt: str) -> str:
    payload = {
        "model": "Qwen3-8B",
        "messages": [{"role": "user", "content": prompt}],
        **EVAL,
    }
    r = await client.post(f"{DAEMON}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def snapshot(client: httpx.AsyncClient, action: str, name: str) -> None:
    r = await client.post(f"{DAEMON}/v1/state/snapshot/{action}",
                          json={"name": name}, timeout=120.0)
    r.raise_for_status()


async def train_memorize(client: httpx.AsyncClient, train_tasks: list[dict]) -> None:
    for t in train_tasks[:5]:
        payload = {
            "prompt": t["prompt"],
            "response": f"Answer: {t['expected']}",
            **MEMORIZE,
        }
        r = await client.post(f"{DAEMON}/v1/train/memorize", json=payload, timeout=600.0)
        r.raise_for_status()


def classify(predicted: str, original: str, variant: str) -> str:
    p = predicted.lower().strip()
    o = original.lower().strip()
    v = variant.lower().strip()
    if p == v:
        return "CORRECT (algorithm)"
    if p == o:
        return "PARROT (memorized)"
    if not p:
        return "EMPTY"
    return f"OTHER ({predicted!r})"


async def main() -> int:
    print(f"[fact-use] daemon={DAEMON} baseline={BASELINE}", flush=True)
    train_tasks, _ = get_split()
    train_tasks = sorted(train_tasks, key=lambda t: t["task_id"])

    async def eval_variants(client: httpx.AsyncClient, label: str) -> dict:
        n_orig = 0
        n_var = 0
        n_parrot = 0
        per_task = []
        for v in VARIANTS:
            orig_resp = await chat(client, v["original_prompt"])
            orig_pred = extract_answer(orig_resp)
            var_resp = await chat(client, v["variant_prompt"])
            var_pred = extract_answer(var_resp)
            orig_ok = orig_pred == v["original_answer"]
            var_ok = var_pred == v["variant_correct"]
            parrot = var_pred == v["original_answer"]
            if orig_ok:
                n_orig += 1
            if var_ok:
                n_var += 1
            if parrot:
                n_parrot += 1
            per_task.append({
                "task_id": v["task_id"],
                "orig_pred": orig_pred,
                "orig_ok": orig_ok,
                "var_pred": var_pred,
                "var_ok": var_ok,
                "parrot": parrot,
            })
            print(f"  [{label}] {v['task_id']}", flush=True)
            print(f"    original → expected={v['original_answer']!r:8s} got={orig_pred!r:8s} "
                  f"{'OK' if orig_ok else f'MISS ({orig_pred!r})'}", flush=True)
            print(f"    variant  → expected={v['variant_correct']!r:8s} got={var_pred!r:8s} "
                  f"{classify(var_pred, v['original_answer'], v['variant_correct'])}", flush=True)
            print()
        return {"n_orig": n_orig, "n_var": n_var, "n_parrot": n_parrot, "per_task": per_task}

    async with httpx.AsyncClient(timeout=300.0) as client:
        print("[fact-use] snapshot/save → autoresearch_baseline (overwrite)", flush=True)
        await snapshot(client, "save", BASELINE)
        try:
            print("[fact-use] PHASE A — cold eval on originals + variants (pre-train baseline)", flush=True)
            print()
            cold = await eval_variants(client, "cold")

            print("[fact-use] memorize K=5 STRONG …", flush=True)
            t0 = time.time()
            await train_memorize(client, train_tasks)
            print(f"[fact-use] train done in {time.time()-t0:.1f}s", flush=True)

            print("[fact-use] PHASE B — post-train eval on originals + variants", flush=True)
            print()
            post = await eval_variants(client, "post")
        finally:
            print("[fact-use] snapshot/load ← autoresearch_baseline", flush=True)
            await snapshot(client, "load", BASELINE)


    print("=" * 60, flush=True)
    print("            cold → post   delta", flush=True)
    print(f"originals: {cold['n_orig']}/5  → {post['n_orig']}/5    ({post['n_orig']-cold['n_orig']:+d})", flush=True)
    print(f"variants : {cold['n_var']}/5  → {post['n_var']}/5    ({post['n_var']-cold['n_var']:+d})   ← algorithm", flush=True)
    print(f"parrots  : {cold['n_parrot']}/5  → {post['n_parrot']}/5    ({post['n_parrot']-cold['n_parrot']:+d})   ← memorization signature", flush=True)
    print("=" * 60, flush=True)
    print("interpretation:", flush=True)
    print("  cold variants >= 4 and post == cold: model already knew algorithm; training only adds format", flush=True)
    print("  post variants > cold variants: training expanded algorithmic competence — REAL SOTA claim", flush=True)
    print("  post parrots > cold parrots: training induces format-mimicry — degenerate generalization", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
