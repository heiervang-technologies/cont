# Research backlog

Append-only. Each item is a falsifiable hypothesis with a concrete experiment that runs against the local lile daemon and a measurable outcome.

Statuses: `unclaimed` → `in-progress` → `done` (or `parked` with reason). See [`../../AGENTS.md`](../../../AGENTS.md) for the lifecycle.

---

## R-001 — `memorize.py` retention under sequential insertion

- **Owner:** unclaimed
- **Status:** unclaimed
- **Hypothesis:** When `memorize.greedy_memorize` is invoked N times in a row on distinct (prompt, response) pairs, the greedy-recall fraction on the *first* memorized pair decays as N grows. Decay shape (linear / log / cliff) tells us whether the SFT-until-greedy loop has implicit catastrophic-forgetting risk for the "auto-SFT on implicit OK" chat-UI flow.
- **Experiment:**
  1. Snapshot the daemon to `R001_baseline`.
  2. Generate 100 synthetic (prompt, response) pairs with disjoint surface forms (use Brixolia-style mythical-fact templates so the model has zero prior).
  3. For i in 0..99: call `/v1/train memorize` on pair i, then probe greedy-recall on pair 0 and pair i. Append both to JSONL.
  4. Plot recall(pair 0, after N) and recall(pair i-1, after pair i) as functions of N.
  5. Snapshot/load back to R001_baseline at the end.
- **Outcome:** retention curve for pair 0; instantaneous retention curve for "last fact." Two numbers per N. Decay shape inferred from the curves.
- **Anchor:** anything ≥ 90% retention at N=100 is a strong positive; ≤ 50% confirms forgetting risk and motivates R-002.

## R-002 — KL anchor strength vs memorize retention

- **Owner:** unclaimed
- **Status:** unclaimed  *(depends on R-001 baseline numbers)*
- **Hypothesis:** Cranking the KL anchor coefficient during the memorize loop reduces forgetting (R-001 decay) at a measurable cost to sample efficiency (more steps to reach greedy-match on the current pair). There is a sweet spot where retention improves materially while sample count grows sub-2×.
- **Experiment:** Same protocol as R-001 but sweep `kl_anchor_weight ∈ {0.0, 0.1, 0.5, 1.0, 2.0}`. Snapshot/load between sweeps so each starts from the same base.
- **Outcome:** Pareto curve of `retention(pair 0, N=100)` against `mean train-tokens-per-memorize-call`. Identify the knee.

## R-003 — context-in-prompt vs memorize: sample-efficiency crossover

- **Owner:** unclaimed
- **Status:** unclaimed
- **Hypothesis:** For a single fact, there is a query-count K above which `/v1/train memorize` (weights) is more sample-efficient — in total tokens billed + total latency — than re-supplying the fact as system-prompt context on every chat. K depends on response length, model size, and per-call KV-cache cost.
- **Experiment:** Pick one fact F. For each query count K ∈ {1, 5, 10, 25, 50, 100}:
  - Branch A (context): every chat sends F as system prompt; measure total prompt tokens + total wall.
  - Branch B (memorize): one memorize call on F, then K plain chats; measure memorize tokens + chat tokens + total wall.
  - Verify both branches answer all K queries correctly (greedy decode, T=0.0).
- **Outcome:** crossover K under our defaults (Qwen3-8B-bnb-4bit, seq_len=4096, lora_r=16). Report total tokens, total wall, and inference latency p50/p95 per branch.

## R-004 — snapshot-load determinism after memorize

- **Owner:** unclaimed
- **Status:** unclaimed
- **Hypothesis:** `snapshot/save` taken mid-memorize, then `snapshot/load`'d, yields *byte-exact* recall on a held-out probe — confirming the "checkpoint as rollback for failed memorize" pattern is safe. Failure here would invalidate R-002's per-sweep reset assumption.
- **Experiment:** Memorize fact F. Save snapshot `R004_mid`. Run 10 unrelated chats. Memorize fact G (which we expect to *partially* overwrite F under R-001). Save snapshot `R004_after`. Load `R004_mid`. Probe greedy-recall on F and G. Expected: F at the same recall as at save time, G at base-model recall.
- **Outcome:** snapshot/load byte-exactness verdict for the memorize path. Either invariant 4 (snapshot round-trip) holds for the memorize path or it doesn't — single boolean, but with a recall delta as the proof.

---

*New items: append below this line with the next `R-NNN` slug.*
