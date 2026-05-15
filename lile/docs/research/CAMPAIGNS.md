# Training campaigns (C-NNN)

Each campaign is a concrete training run against the live lile daemon that
produces a measurable, verifiable, citable capability delta. Campaigns use
the C-NNN slug space (separate from R-NNN research probes which investigate
a hypothesis without a training target).

Lifecycle: `unclaimed` → `in-progress` → `done` (or `parked`).

## C-001 — HumanEval-subset RLVR

- **Status:** scoping  *(architect reconciling HumanEval adapter vs GSM8K pivot)*
- **Goal:** pp improvement on 64 held-out HumanEval problems via RLVR on the
  other 100, with McNemar paired p<0.05 as the success criterion.
- **Prerequisite:** HumanEval verifier (evalplus or custom sandbox adapter).
- **Risk:** Cold pass rate >70% = insufficient headroom; <20% = increase k.
- **(Pending adapter decision)**

## Campaign anatomy

Every campaign should specify:

1. **Domain & verifier** — what task, how correctness is determined
   (sandboxed exec, exact-match, teacher grade).
2. **North star** — the single number that decides pass/fail (pp delta on a
   held-out set, McNemar p-value, pass@k).
3. **Data split** — train / held-out, pinned for reproducibility.
4. **Training protocol** — source, n, k, weights, halt rule.
5. **Evaluation protocol** — cold eval snippet, post-training eval snippet,
   paired comparison method.
6. **Budget** — expected teacher calls, wall time, OpenRouter cost.
7. **Risk hedges** — what triggers abort or pivot.

## What makes a good campaign (lessons from R-001 series)

| Factor | Good | Bad |
|--------|------|-----|
| Headroom | Cold accuracy 30-70% | Cold >80% or <10% |
| Verifiability | Sandboxed exec or exact-match integer | Teacher-only judgment |
| Reproducibility | Pinned dataset, seeded split | Live API data |
| Cost | <$10 OpenRouter | >$50 |
| Wall time | Overnight (8h) | Multi-day |
| Signal | Single binary metric (pass/fail per problem) | Soft rubric / multi-axis |
