# Research backlog

Append-only. Each item is a falsifiable hypothesis with a concrete experiment that runs against the local trainfer daemon and a measurable outcome.

Statuses: `unclaimed` → `in-progress` → `done` (or `parked` with reason). See [`../../AGENTS.md`](../../../AGENTS.md) for the lifecycle.

---

## R-001 — `memorize.py` retention under sequential insertion

- **Owner:** glm
- **Status:** done — [JOURNAL entry](../../JOURNAL.md#2026-05-15--r-001--memorize-retention--glm)
- **Hypothesis:** When `memorize.greedy_memorize` is invoked N times in a row on distinct (prompt, response) pairs, the greedy-recall fraction on the *first* memorized pair decays as N grows. Decay shape (linear / log / cliff) tells us whether the SFT-until-greedy loop has implicit catastrophic-forgetting risk for the "auto-SFT on implicit OK" chat-UI flow.
- **Result:** Falsified (forgetting direction). pair0 recall *increased* with N (0.78→1.00). No decay signal at default settings. Subsequent R-001b tested
  stronger per-call learning (2.4× mem_steps, explicit lr) and low-threshold
  (0.7) — retroactive consolidation held across all three regimes. The
  catastrophic-forgetting hypothesis is falsified at n=100 on
  Qwen3-8B-bnb-4bit regardless of learning strength.
- **Data quality:** Clean (JSONL mode='w' fix applied in R-001b).

## R-001b — `memorize.py` retention with stronger per-call learning

- **Owner:** kimi
- **Status:** done — [JOURNAL entry](../../JOURNAL.md#2026-05-15--r-001b--memorize-retention-stronger--kimi)
- **Hypothesis:** At stronger per-call learning (`plateau_patience=10`, `max_steps=100`, explicit `lr=5e-4`), `memorize.greedy_memorize` will reach threshold on most facts, producing a stronger per-fact weight change. Under these conditions, the pair-0 decay hypothesized in R-001 may emerge — i.e., catastrophic forgetting is a function of insertion strength, not of sequential insertion per se.
- **Experiment:** Two arms: (i) strong-learning — `plateau_patience=10`, `max_steps=100`, `lr=5e-4`, `threshold=0.95`; (ii) low-threshold — `threshold=0.7`, others default. Both arms independently bracketed (snapshot/save → run → snapshot/load). 100 facts per arm.
- **Outcome:** Catastrophic forgetting does NOT emerge at stronger per-call learning. pair0 never dropped below baseline in either arm. arm 1: pair0 mean=0.873, pairI mean=0.957 (80% at 1.000), mem_steps mean=8.0 (2.4× R-001). arm 2: pair0 mean=0.849, pairI mean=0.727, mem_steps mean=1.2 (63% at 0 steps — facts already above 0.7 threshold). Total wall: 12.8 min for both arms.
- **Implication:** Retroactive consolidation is robust across weak, strong, and low-threshold regimes at n=100 on Qwen3-8B-bnb-4bit. The memorize loop is structurally safe for the auto-SFT-on-implicit-OK chat-UI flow at this model scale.

## R-002 — KL anchor strength vs memorize retention

- **Owner:** unclaimed
- **Status:** unclaimed  *(parked pending re-evaluation — R-001b found no forgetting across three regimes, so KL-anchor's forgetting-reduction motivation is weakened; see R-001b JOURNAL for discussion)*
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

- **Owner:** glm
- **Status:** done — [JOURNAL entry](../../JOURNAL.md#2026-05-15--r-004--snapshot-determinism--glm)
- **Hypothesis:** `snapshot/save` taken mid-memorize, then `snapshot/load`'d, yields *byte-exact* recall on a held-out probe — confirming the "checkpoint as rollback for failed memorize" pattern is safe. Failure here would invalidate R-002's per-sweep reset assumption.
- **Result:** Confirmed. Invariant 4 holds under both weak (3-step) and stronger (10-step) memorize. F recall after load matches save-time value byte-exact. G recall after load matches baseline. Cross-fact interference from G's training is fully erased by the rollback.
- **Experiment:** Memorize fact F. Save snapshot `R004_mid`. Run 10 unrelated chats. Memorize fact G (which we expect to *partially* overwrite F under R-001). Save snapshot `R004_after`. Load `R004_mid`. Probe greedy-recall on F and G. Expected: F at the same recall as at save time, G at base-model recall.
- **Outcome:** snapshot/load byte-exactness verdict for the memorize path. Either invariant 4 (snapshot round-trip) holds for the memorize path or it doesn't — single boolean, but with a recall delta as the proof.

---

*New items: append below this line with the next `R-NNN` slug.*

---

## R-006 — memorize semantic conflict (competing responses, same prompt)

- **Owner:** kimi
- **Status:** in-progress  *(daemon handoff after R-003)*
- **Hypothesis:** When two memorize calls target the same prompt with conflicting
  responses (`response_A` at i=0, `response_B` at i=K with K small), the model
  retains exactly one of them, both partially via interpolation, or neither
  reliably. Outcome distribution tells us whether memorize is robust under
  genuine semantic conflict (the steel-man of forgetting).
- **Experiment:** `collision_facts(seed=42, n_pairs=20)` yields 20 `(prompt,
  response_A, response_B)` triples where response_A and response_B are disjoint
  surface forms with similar token length. For each triple:
  1. Memorize A, eval recall(A) → save baseline.
  2. Memorize B.
  3. Eval recall(A) and recall(B) immediately after B (K=0).
  4. Run K ∈ {1, 5, 10} intervening unrelated (disjoint) facts, then re-eval
     both recall(A) and recall(B) — tests whether conflict resolves quickly or
     lingers.
  
  Use the daemon-default memorize params from R-001 (max_steps=30,
  threshold=0.95, plateau_patience=3) and the stronger params from R-001b
  (max_steps=100, plateau_patience=10, lr=5e-4) in a second arm.
- **Outcome:** Distribution of post-B recall(A) across 20 pairs × 2 learning
  regimes. If recall(A) post-B ≤ 0.2 (close to base-model on novel prompt),
  memorize collapses to last-write-wins — finding. If recall(A) and recall(B)
  both ≥ 0.7, partial interpolation. If A wins or B wins consistently, that's
  a recency-bias finding.
- **Anchor:** Same as R-001: ≥90% retention of A post-B = strong positive
  (memorize survives direct conflict). ≤50% = genuine interference risk.
