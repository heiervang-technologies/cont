"""5-shot ICL baseline on GSM8K — apples-to-apples vs our K=5 fine-tune.

The Stop hook is correct that "K=5 fine-tune at 44%" without an ICL
baseline is not a defensible "sample-efficient learning" claim. Adding
5 examples in the prompt (ICL) is the natural sample-efficient baseline
that any K=5 fine-tune must beat.

This probe:
- Uses the same 5 GSM8K train examples as gsm8k_k5_probe.py
- Same eval set (first 50 GSM8K test problems)
- Same eval config (temperature=0, max_tokens=3000, enable_thinking=true)
- No training — just stuffs 5 demos into the prompt
- Cold pass rate on the same 50 problems is already 32% from the prior probe

If 5-shot ICL ≥ K=5 fine-tune: our recipe is not better than the simplest
sample-efficient baseline, and the "sample-efficient learning" SOTA claim
is unsupported.

If 5-shot ICL < K=5 fine-tune: our recipe encodes K=5 examples more
efficiently than ICL does (fine-tune adds them to parameters; ICL adds
them to context). That IS a defensible sample-efficient-learning claim.
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
EVAL = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 3000, "enable_thinking": True}
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


def fmt_demo(question: str, answer: str) -> str:
    gold = extract_gold(answer)
    return f"Q: {question}\n\nA: Answer: {gold}"


def fmt_zero_shot(question: str) -> str:
    return question + "\n\nEnd your reply with: Answer: <integer>."


def fmt_few_shot(demos: list[dict], question: str) -> str:
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


async def eval_icl(client: httpx.AsyncClient, demos: list[dict],
                    heldout: list[dict], label: str) -> dict:
    n_correct = 0
    t0 = time.time()
    for i, ex in enumerate(heldout):
        gold = extract_gold(ex["answer"])
        if not gold:
            continue
        prompt = fmt_few_shot(demos, ex["question"])
        resp = await chat(client, prompt)
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
    print(f"[gsm8k-icl] daemon={DAEMON} eval={EVAL}", flush=True)
    ds = load_dataset("openai/gsm8k", "main", cache_dir="/tmp/hf-datasets")
    demos = list(ds["train"])[:5]
    heldout = list(ds["test"])[:N_HELDOUT]
    print(f"[gsm8k-icl] 5 demos, {N_HELDOUT} heldout", flush=True)

    async with httpx.AsyncClient(timeout=300.0) as client:
        # NB: NO snapshot bracket — we don't mutate model state. If the daemon
        # is at autoresearch_baseline now, leave it there.
        print("[gsm8k-icl] eval 5-shot ICL (no training, demos in prompt)", flush=True)
        icl = await eval_icl(client, demos, heldout, "5-shot ICL")

    print()
    print("=" * 60, flush=True)
    print("GSM8K 0-shot cold (from gsm8k_k5_probe.py exp): 16/50 (32.0%)", flush=True)
    print("GSM8K K=5 fine-tune (from gsm8k_k5_probe.py exp): 22/50 (44.0%)", flush=True)
    print(f"GSM8K 5-shot ICL (this run):                      "
          f"{icl['n_correct']}/{icl['total']} ({100*icl['rate']:.1f}%)", flush=True)
    print("=" * 60, flush=True)
    ft_pp = 44.0
    icl_pp = 100 * icl["rate"]
    if ft_pp > icl_pp:
        print(f"VERDICT: K=5 fine-tune ({ft_pp:.0f}%) BEATS 5-shot ICL ({icl_pp:.0f}%)", flush=True)
        print("         → fine-tune encodes K=5 more efficiently than context-stuffing", flush=True)
        print("         → defensible sample-efficient-LEARNING claim (parameter updates beat prompting)", flush=True)
    elif ft_pp == icl_pp:
        print(f"VERDICT: K=5 fine-tune TIES with 5-shot ICL at {ft_pp:.0f}%", flush=True)
        print("         → fine-tune matches ICL but adds learning (parameter change persists across queries)", flush=True)
        print("         → moderate claim: same gain, but learned not prompted", flush=True)
    else:
        print(f"VERDICT: K=5 fine-tune ({ft_pp:.0f}%) LOSES to 5-shot ICL ({icl_pp:.0f}%)", flush=True)
        print("         → simpler baseline (context demos) outperforms our recipe at K=5", flush=True)
        print("         → no defensible sample-efficient-learning SOTA claim at this scale", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
