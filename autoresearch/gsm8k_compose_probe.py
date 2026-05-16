"""Fine-tune + ICL composition probe.

Final test in the SOTA-claim falsification chain. We have:
- 0-shot cold GSM8K              : 32%   (from gsm8k_k5_probe.py phase A)
- K=5 fine-tune 0-shot eval      : 44%   (from gsm8k_k5_probe.py phase C)
- 5-shot ICL no training         : 96%   (from gsm8k_icl_probe.py)

This probe answers: does K=5 fine-tune add anything to 5-shot ICL? I.e.
does fine-tune + ICL > ICL alone?

If composition > 96%: the fine-tune contributes something the demos
don't (e.g. faster format convergence, stricter Answer: emission).
If composition == 96%: fine-tune is neutral when demos are present.
If composition <  96%: fine-tune is harmful when combined with demos.

Procedure:
1. Snapshot save → memorize K=5 → eval 50 heldout with 5-shot ICL demos
   in prompt → snapshot load.
2. Compare to the prior runs.
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
N_HELDOUT = 50

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


def fmt_zero(question: str) -> str:
    return question + "\n\nEnd your reply with: Answer: <integer>."


def fmt_demo(question: str, answer: str) -> str:
    gold = extract_gold(answer)
    return f"Q: {question}\n\nA: Answer: {gold}"


def fmt_few(demos: list[dict], question: str) -> str:
    parts = ["Here are some examples of the format expected:"]
    for ex in demos:
        parts.append(fmt_demo(ex["question"], ex["answer"]))
    parts.append("Now answer the following:")
    parts.append(f"Q: {question}\n\nEnd your reply with: Answer: <integer>.")
    return "\n\n".join(parts)


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


async def train_memorize(client: httpx.AsyncClient, examples: list[dict]) -> None:
    for ex in examples:
        gold = extract_gold(ex["answer"])
        if not gold:
            continue
        payload = {
            "prompt": fmt_zero(ex["question"]),
            "response": f"Answer: {gold}",
            **MEMORIZE,
        }
        r = await client.post(f"{DAEMON}/v1/train/memorize", json=payload, timeout=600.0)
        r.raise_for_status()


async def eval_few(client: httpx.AsyncClient, demos: list[dict],
                    heldout: list[dict], label: str) -> dict:
    n_correct = 0
    t0 = time.time()
    for i, ex in enumerate(heldout):
        gold = extract_gold(ex["answer"])
        if not gold:
            continue
        resp = await chat(client, fmt_few(demos, ex["question"]))
        pred = extract_pred(resp)
        if pred == gold:
            n_correct += 1
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(heldout)}: running={n_correct}/{i+1} "
                  f"({100*n_correct/(i+1):.1f}%)", flush=True)
    rate = n_correct / len(heldout)
    print(f"  [{label}] FINAL: {n_correct}/{len(heldout)} ({100*rate:.1f}%) in "
          f"{time.time()-t0:.0f}s", flush=True)
    return {"rate": rate, "n_correct": n_correct, "total": len(heldout)}


async def main() -> int:
    print(f"[compose] daemon={DAEMON}", flush=True)
    ds = load_dataset("openai/gsm8k", "main", cache_dir="/tmp/hf-datasets")
    demos = list(ds["train"])[:5]
    heldout = list(ds["test"])[:N_HELDOUT]

    async with httpx.AsyncClient(timeout=300.0) as client:
        print(f"[compose] snapshot/save → {BASELINE}", flush=True)
        await snapshot(client, "save", BASELINE)
        try:
            print("[compose] memorize K=5 on GSM8K-train", flush=True)
            t0 = time.time()
            await train_memorize(client, demos)
            print(f"[compose] train done in {time.time()-t0:.1f}s", flush=True)

            print("[compose] eval — 5-shot ICL with fine-tuned model", flush=True)
            result = await eval_few(client, demos, heldout, "ft+icl")
        finally:
            print(f"[compose] snapshot/load ← {BASELINE}", flush=True)
            await snapshot(client, "load", BASELINE)

    pct = 100 * result["rate"]
    print()
    print("=" * 60, flush=True)
    print(f"0-shot cold (prior probe)        : 16/50 (32.0%)", flush=True)
    print(f"K=5 fine-tune 0-shot (prior)     : 22/50 (44.0%)", flush=True)
    print(f"5-shot ICL no training (prior)   : 48/50 (96.0%)", flush=True)
    print(f"K=5 fine-tune + 5-shot ICL (this): "
          f"{result['n_correct']}/{result['total']} ({pct:.1f}%)", flush=True)
    print("=" * 60, flush=True)
    if pct > 96.0:
        print(f"VERDICT: fine-tune is ADDITIVE to ICL ({pct:.0f}% > 96%)", flush=True)
        print(f"         → the parameter update encodes something demos don't", flush=True)
        print(f"         → defensible claim: K=5 fine-tune contributes "
              f"+{pct-96.0:.0f}pp on top of 5-shot ICL", flush=True)
    elif pct == 96.0:
        print(f"VERDICT: fine-tune is NEUTRAL with ICL (tied at 96%)", flush=True)
        print(f"         → demos already saturate; fine-tune isn't helping or hurting", flush=True)
    else:
        print(f"VERDICT: fine-tune is SUBTRACTIVE with ICL ({pct:.0f}% < 96%)", flush=True)
        print(f"         → fine-tune is actively harmful when combined with ICL demos", flush=True)
        print(f"         → recipe interferes with the model's ICL ability", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
