# autoresearch may16 — report

Branch: [`autoresearch/may16`](https://github.com/heiervang-technologies/agi/tree/autoresearch/may16) ([PR #17](https://github.com/heiervang-technologies/agi/pull/17)) · 28 fine-tune experiments + 3 reproducible probes · ~5h wall.

## Executive summary

The session's stated goal was *"novel SOTA in sample-efficient learning for LLMs on verifiable tasks."* **That goal was not met.** When the fairest baseline (5-shot in-context learning) was actually tested, it dominated our fine-tune recipe by 52 percentage points on GSM8K (96% vs 44%) at the same K=5. What the session *did* produce is a clean continual-learning recipe with reproducible diagnostics — useful for `lile`'s product (a live-learning local LLM daemon) but not a new SOTA.

## The headline chart

GSM8K, first 50 test problems, Qwen3-8B-bnb-4bit, identical eval config across all four conditions (temperature=0, max_tokens=3000, enable_thinking=true):

```
condition                                   pass rate   inference overhead
─────────────────────────────────────────  ──────────  ──────────────────────────────
0-shot cold (no training, plain prompt)     16/50 32%   base prompt only
K=5 fine-tune, 0-shot eval                  22/50 44%   base prompt only (params changed)
5-shot ICL, no training                     48/50 96%   base + 5 demos per query
K=5 fine-tune + 5-shot ICL                  48/50 96%   base + 5 demos per query

deltas
─────
fine-tune 0-shot vs cold:    +12 pp
ICL vs cold:                 +64 pp
fine-tune + ICL vs ICL:        0 pp   ← fine-tune is NEUTRAL once demos are in context
```

Same `progress.png` chart from the autoresearch loop ([`autoresearch/progress.png`](progress.png)) shows the 28-experiment trajectory on our *custom* 30-task verifiable-logic corpus, where the same recipe peaks at score 0.70 (forward 0.70, zero degradation). That number is real but is not directly comparable to published benchmarks.

## What was actually validated

1. **A continual-learning recipe works.** K=5 `memorize` + CoT eval gives +40pp on our custom verifiable-logic corpus with zero degradation on a 15-prompt frozen probe set. The recipe is sample-efficient *in the fine-tune-only family*; it's beaten by ICL when ICL is allowed.
2. **The fine-tune teaches algorithmic format, not parrot memorization.** Number-substituted variant probe: cold 0/5 → post 5/5 with zero parrots (`autoresearch/fact_use_probe.py`).
3. **R-004 byte-exactness has a caveat.** `save→load` is byte-exact; `save→memorize→load` propagates ≤1-char drift, stabilizes after one cycle (`autoresearch/snapshot_drift_probe.py`).

## What was falsified

| Track                       | Verdict                              | Result on autoresearch corpus | Result on GSM8K (where measured) |
|-----------------------------|--------------------------------------|------------------------------|----------------------------------|
| Feedback-guided loss (hybrid) | Tied with simpler memorize baseline | 0.70 (tied)                  | not measured                     |
| Sample efficiency (K=5 ft)  | Dominated by 5-shot ICL by 52pp       | 0.70 best                    | 44% (vs ICL 96%)                 |
| Self-synthesization (m=3)   | Regressed below baseline              | 0.40                         | not measured                     |
| Fine-tune + ICL composition | Neutral (no additive contribution)    | not measured                 | 96% = ICL alone                  |

## Honest framing for a third party

We set out to claim novel SOTA in sample-efficient learning by demonstrating that few fine-tune examples produce strong gains. The honest read of the data: **few-shot prompting (ICL) is a strictly better sample-efficient baseline at this K and model class**, and our fine-tune does not add to it when both are combined. The fine-tune's value is *not* in absolute accuracy — it's in the fact that the gain persists in the model's parameters without re-supplying demos at every query (an inference-cost trade-off relevant to `lile`'s deployment scenario, not a research SOTA claim).

The recipe is shipping-quality for `lile`. It is not a research result that beats the literature.

## Path ahead

Three concrete next campaigns that *could* yield a defensible sample-efficient-learning claim. None are in scope for this tag.

1. **ICL × K cross-over study.** ICL wins at K=5. At some K (probably 50-200, where the context window pressure starts to bite) fine-tune should overtake. Finding that cross-over is the legitimate research question. Run K ∈ {5, 20, 50, 100, 200} fine-tune vs K-shot ICL on the same GSM8K-test slice. The K where fine-tune beats ICL is the publishable result.
2. **Cost-amortized comparison.** At N queries, ICL pays the demo-token cost N times; fine-tune pays a one-time training cost then 0 per query. At what N does fine-tune amortize cheaper than ICL even when both produce the same accuracy? This is the deployment-relevant version of the sample-efficiency question and a defensible engineering claim.
3. **Multi-task accumulation.** ICL is per-query; fine-tune persists. If we sequentially fine-tune on multiple K=5 task families (arithmetic, then code, then logic) and measure whether the model can use all three at once, fine-tune wins by construction (ICL would need 15 demos in the prompt to match). This is the continual-learning angle that matches `lile`'s actual product.

## Artifacts

- [`autoresearch/results.tsv`](results.tsv) — 28-experiment ledger (gitignored; reproduce locally)
- [`autoresearch/progress.png`](progress.png) — running-best plot on custom corpus
- [`autoresearch/fact_use_probe.py`](fact_use_probe.py) — algorithmic-vs-mimicry diagnostic
- [`autoresearch/gsm8k_k5_probe.py`](gsm8k_k5_probe.py) — K=5 fine-tune on GSM8K (32% → 44%)
- [`autoresearch/gsm8k_icl_probe.py`](gsm8k_icl_probe.py) — 5-shot ICL baseline (96%)
- [`autoresearch/gsm8k_compose_probe.py`](gsm8k_compose_probe.py) — fine-tune + ICL composition (96%, neutral)
- [`autoresearch/snapshot_drift_probe.py`](snapshot_drift_probe.py) — R-004 byte-exact follow-up
- [`lile/docs/research/JOURNAL.md`](../lile/docs/research/JOURNAL.md) — full narrative including the falsification entry

## Bottom line

What we built: a documented, reproducible, continual-learning recipe with bounded variance and validated algorithmic learning. Useful for `lile`.

What we did not build: a novel sample-efficient-learning SOTA. The bar for that goal — beating the best published K-shot or few-shot result on a recognized benchmark — was not cleared. The session's value is the negative results and the diagnostics, not a record-setting recipe.
