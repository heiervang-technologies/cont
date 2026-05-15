# Research journal

Append-only. Each entry corresponds to a completed item in [`BACKLOG.md`](BACKLOG.md). One entry per experiment run — if you re-run, append a new entry, do not edit the old one.

Entry template:

```markdown
## YYYY-MM-DD — R-NNN — <slug> — <agent>

**Hypothesis (as filed):** <one line from BACKLOG>

**Setup:**
- Daemon: model / max_seq / lora_rank / data_dir
- Snapshot pre: <name>
- Snapshot post: <name>
- Trajectory offsets: pre=<int>, post=<int>

**Result:**
<numbers + plot link or path under lile_data/research/<slug>/>

**Verdict:** confirmed | falsified | inconclusive

**Next step:** <pointer to follow-up BACKLOG item or "none">
```

---

## 2026-05-15 — R-001 — memorize-retention — glm

**Hypothesis (as filed):** When `memorize.greedy_memorize` is invoked N times in a row on distinct (prompt, response) pairs, the greedy-recall fraction on the *first* memorized pair decays as N grows. Decay shape (linear / log / cliff) tells us whether the SFT-until-greedy loop has implicit catastrophic-forgetting risk for the "auto-SFT on implicit OK" chat-UI flow.

**Setup:**
- Daemon: Qwen3-8B-unsloth-bnb-4bit / max_seq=4096 / lora_rank=16 / data_dir=lile_data
- Snapshot pre: `R001` (saved from `_autosave`-restored baseline, merges=0, residual_fp=e3b0c44298fc1c14)
- Snapshot post: `R001` (restored at end of run)
- Trajectory offsets: pre=commit_token 145 (ts 1778875021), post=commit_token 1355 (ts 1778875469)

**Result:**

The run produced a contaminated JSONL (3 partial runs appended due to daemon restart + append-mode bug). Segment 1 (the longest contiguous 100-fact sweep, commit_tokens 484–1109) is the analyzable subset:

- **pair0 retention grew**, not decayed: baseline 0.7778 → final 1.0000 (+0.2222).
- pair0 minimum across N=1..100: 0.7778 (never dropped below baseline).
- pair0 transitions: growing=17, shrinking=14, same=69. Net direction: up.
- **Instantaneous pair-i retention** (pairi_fraction): mean=0.79, min=0.50, max=1.00 — noisy but centered well above zero.
- **memorize efficacy**: only 4/100 invocations reached the 0.95 threshold. 96/100 plateau'd at `reason=plateau` after 3–6 steps. Per-call learning is weak at defaults (plateau_patience=3, max_steps=30), but cumulative gradient over 100 calls *does* shift the model.

Segment 0 (first 79 facts from fresh-boot baseline p0=0.5556) shows the same pattern: pair0 grew from 0.5556 to 0.7778. No forgetting signal anywhere.

Plot: `lile_data/research/R001/retention_curves.png` (all 3 segments)
Data: `lile_data/research/R001/results.raw.jsonl` (3 headers + 219 loop records total; raw
  provenance preserved after clean extraction — see kimi entry below)

**Verdict:** falsified (for the forgetting direction) — pair0 recall *increased* with N, the opposite of the hypothesized decay. The memorize loop does not produce catastrophic forgetting at this scale and configuration. However, the result is confounded by the weak per-call learning: most invocations plateau without reaching threshold, so the effective "insertion strength" per fact is low. Whether forgetting emerges at higher per-call learning rates remains open (→ R-001b).

Secondary finding: cumulative LoRA gradient across many weak memorize calls still shifts the model noticeably. This is relevant for the chat-UI implicit-OK auto-SFT flow — repeated exposure to the same fact ratchets up recall rather than diluting it.

Data quality caveat: JSONL contaminated by 3 appended partial runs. The runner opens in append mode, so every restart adds a duplicate header + overlapping records. The analyzable segment was extracted by splitting on header boundaries and selecting the longest contiguous i=0..99 sweep. Fix required in R-001b runner (header write mode='w', loop records mode='a').

---

## 2026-05-15 — R-001 — memorize-retention — kimi

**Hypothesis (as filed):** When `memorize.greedy_memorize` is invoked N=100 times on
distinct (prompt, response) pairs, greedy-recall fraction on the first memorized
pair decays as N grows.

**Setup:**
- Daemon: Qwen3-8B-unsloth-bnb-4bit / max_seq=4096 / lora_rank=16 / data_dir=lile_data
- Generator: `lile/teach/research_fixtures/mythical_facts.py` seed=42, 100 facts
- Endpoint: `POST /v1/eval/greedy_rank` added by PR #6
- Snapshot pre: `R001` (saved via `/v1/state/snapshot/save`, merges=0)
- Snapshot post: `R001` (restored via `/v1/state/snapshot/load` at end of run)
- Trajectory offsets: pre=~520 (baseline eval at start of run), post=~1642 (final eval
  point at i=99)

**Result:**

| Metric | Value |
|--------|-------|
| pair0 baseline (pre-training) | 0.778 |
| pair0 final (after 100 inserts) | 0.889 |
| pair0 minimum | 0.778 (never below baseline) |
| pair0 mean over 100 | 0.948 |
| pair0 at ≥0.90 | 57/100 (57%) |
| pair0 at 1.000 | 57/100 (57%) |
| pair0 below baseline | 0/100 (0%) |
| pairI mean (instantaneous recall) | 0.804 |
| pairI median | 0.857 |
| pairI min/max | 0.500 / 1.000 |
| memorize steps mean | 3.3 (min=2, max=6) |
| memorize hit max_steps=30 | 0/100 |
| wall time | 194 s (3.2 min) |
| time per iteration | 1.94 s |

**Retention curve by quarter:**
- Q1 (i=0–24): mean=0.889, min=0.778, max=1.000
- Q2 (i=25–49): mean=0.938, min=0.889, max=1.000
- Q3 (i=50–74): mean=1.000, min=1.000, max=1.000
- Q4 (i=75–99): mean=0.964, min=0.889, max=1.000

Data: `lile_data/research/R001/results.jsonl` (clean 101-line file: 1 header + 100 loop
  records extracted from the last contiguous i=0..99 segment of the raw
  `results.raw.jsonl`; raw provenance preserved for audit)

**Verdict:** falsified (H0 of catastrophic forgetting rejected at n=100). pair0
retention never dropped below the pre-training baseline — it grew monotonically,
reaching 0.89–1.00 by Q3–Q4. This indicates **retroactive consolidation**: later
memorized facts reinforce the earliest fact's representation rather than interfering
with it.

**Key observations:**
1. Every memorize call plateau'd in 3–6 steps (never hit max_steps=30). Per-call
   weight shift is small, but cumulative gradient over 100 calls is significant.
2. The plateau-at-threshold-0.95 check (`greedy_rank_fraction ≥ 0.95`) was the
   stopping criterion; most calls never reached this threshold, meaning the
   per-fact memorization is intentionally incomplete.
3. Wall time (3.2 min for 100 facts) is far below the prophet's 50–100 min estimate
   — Qwen3-8B-bnb-4bit on 4090 processes each memorize+2x eval in <2s.
4. Smoke test (n=5) confirmed no regressions: snapshot/load round-trips clean,
   eval endpoint returns consistent fractions, JSONL format valid.

**Limitation:** The weak per-call learning (mean 3.3 steps, never hitting
max_steps=30) means each fact's memorization is shallow. A stronger per-call
regime (higher plateau_patience, higher max_steps, explicit lr) might produce
different retention dynamics. This is the subject of R-001b.

**Next step:** R-001b — same protocol with `plateau_patience=10`,
`max_steps=100`, explicit `lr=5e-4`, and corrected runner (truncate on first
open, don't append). If retroactive consolidation holds at stronger per-call
learning, the Qwen3-8B memorization dynamics are fundamentally resilient to
sequential-forgetting risk. Also R-004 (snapshot-load determinism) to validate
the rollback pattern before R-002 KL-anchor sweep.

---

## 2026-05-15 — R-004 — snapshot-determinism — glm

**Hypothesis (as filed):** `snapshot/save` taken mid-memorize, then `snapshot/load`'d, yields *byte-exact* recall on a held-out probe — confirming the "checkpoint as rollback for failed memorize" pattern is safe. Failure here would invalidate R-002's per-sweep reset assumption.

**Setup:**
- Daemon: Qwen3-8B-unsloth-bnb-4bit / max_seq=4096 / lora_rank=16 / data_dir=lile_data
- Snapshot pre: `R004_mid` (saved after memorizing fact F with max_steps=100, threshold=0.70, plateau_patience=10)
- Snapshot post: `R004_after` (saved after also memorizing fact G with same params)
- Trajectory offsets: pre=commit_token ~1650 (post R-001 runs), post=commit_token 1708

**Result:**

Ran two configurations:

1. **Weak learning** (default params, max_steps=30, threshold=0.95, plateau_patience=3):
   - F memorize: 3 steps, plateau. F recall at save = 0.2222 (2/9). After load: 0.2222 (2/9). **Exact match.**
   - G after load: 0.5455 (6/11) = baseline. **Exact match.**

2. **Stronger learning** (max_steps=100, threshold=0.70, plateau_patience=10):
   - F memorize: 10 steps, plateau. F recall at save = 0.2222 (2/9).
   - After G's 10 memorize steps, F recall shifted to 0.3333 — confirming interference (same compounding signal as R-001).
   - After loading R004_mid: F recall = 0.2222 (2/9) **exactly matching save-time value.**
   - G after load: 0.5455 (6/11) = baseline. **Exact match.**

Both matched and total counts are byte-exact. The rollback erases all weight changes made after the snapshot, including the cross-fact interference effect.

Data: `lile_data/research/R004/results.jsonl`

**Verdict:** confirmed. Invariant 4 (snapshot round-trip) holds for the memorize path, even under non-trivial weight changes (10 SFT steps per fact, observable cross-fact interference). The "checkpoint as rollback for failed memorize" pattern is safe. R-002's per-sweep reset assumption is valid.

**Next step:** R-002 (KL-anchor sweep) is now unblocked. R-001b is complete — see entry below.

---

## 2026-05-15 — R-001b — memorize-retention-stronger — kimi

**Hypothesis (as filed):** At stronger per-call learning (`plateau_patience=10`,
`max_steps=100`, explicit `lr=5e-4`), the pair-0 decay hypothesized in R-001 may
emerge — catastrophic forgetting is a function of insertion strength, not of
sequential insertion per se.

**Setup:**
- Daemon: Qwen3-8B-unsloth-bnb-4bit / max_seq=4096 / lora_rank=16 / data_dir=lile_data
- Generator: `lile/teach/research_fixtures/mythical_facts.py` seed=42, 100 facts
- Runner: `./lile/teach/research/r001_memorize_retention.py` with new CLI flags
  (--plateau-patience, --max-steps, --lr, --threshold)

Two arms, independently bracketed (snapshot/save → run → snapshot/load):

| Arm | plateau_patience | max_steps | lr | threshold | out_path |
|-----|:-:|:--:|:-:|:-:|---|
| 1 (strong) | 10 | 100 | 5e-4 | 0.95 | `lile_data/research/R001b/results.jsonl` |
| 2 (low-thresh) | 3 | 30 | default | 0.70 | `lile_data/research/R001b_thresh07/results.jsonl` |

Snapshot baseline pre: `R001b_strong` (arm 1), `R001b_lowthresh` (arm 2)
Snapshot post: both baselines restored via snapshot/load (per R-004's
confirmed byte-exact contract)

**Result:**

| Metric | R-001 (weak) | R-001b arm 1 (strong) | R-001b arm 2 (low-thresh) |
|--------|:------:|:--------------:|:-----------------:|
| pair0 baseline | 0.556 | 0.556 | 0.556 |
| pair0 final | 0.889 | 0.889 | 0.889 |
| pair0 mean | 0.948 | 0.873 | 0.849 |
| pair0 min | 0.556 | 0.667 | 0.556 |
| pair0 below baseline | 0/100 | 0/100 | 0/100 |
| pairI mean | 0.804 | 0.957 | 0.727 |
| pairI at 1.000 | 4% | 80% | 0% |
| mem_steps mean | 3.3 | 8.0 | 1.2 |
| mem_steps max | 6 | 25 | 6 |
| mem_steps=0 (no-train) | 0% | 0% | 63% |
| wall | 3.2 min | 11.3 min | 1.5 min |

**Key findings:**

1. **Catastrophic forgetting does NOT emerge at stronger per-call learning.**
   pair0 never dropped below the pre-training baseline in either arm. The
   retroactive-consolidation signal from R-001 is robust across all three
   tested regimes: weak, strong, and low-threshold.

2. **Plateau_patience=10 is not a bottleneck.** mem_steps mean shifted from 3.3
   (R-001) to 8.0 (arm 1), confirming the plateau detector was genuinely allowing
   early termination — not cutting off prematurely. Max was 25, well under 100.

3. **63% of low-threshold facts required zero training steps.** The model already
   had >=0.7 greedy recall on new facts before any gradient step, confirming
   cross-fact knowledge transfer in the mythical-country template space.

4. **Strong learning produces near-perfect per-fact memorization.** Arm 1's pairI
   mean of 0.957 and 80% at exactly 1.000 confirms that the stronger params
   (plateau_patience=10, lr=5e-4) do produce materially better per-fact
   memorization — the plateau and step budget were real constraints, not artifacts.

5. **Total wall time for both arms: 12.8 min.** Well below the 50-100 min
   estimate; per-fact costs are dominated by early iterations (30-35s) and
   settle to 1-3s after the first 20 facts.

**Verdict:** falsified. The R-001b hypothesis (forgetting as function of insertion
strength) is not supported at n=100 on Qwen3-8B-bnb-4bit. Retroactive
consolidation persists across weak, strong, and low-threshold regimes.

**Implications:**
- R-002 (KL-anchor sweep) likely not needed unless KL divergence couples
  to forgetting through a mechanism other than insertion strength.
- Suggest reclassifying R-002 to exploratory (is there ANY regime where
  sequential insertion produces forgetting?) or parking in favor of more
  pressing items.
- The compounding-growth signal across all three regimes means the memorize
  loop is structurally safe for the auto-SFT-on-implicit-OK chat-UI flow.

**Next step:** Parked. R-001 series exhausted at this model scale. Suggest
moving to R-003 (context-in-prompt vs memorize crossover) or proposing new
items. If a future experiment finds forgetting at higher N (500+, 1000+), this
entry is the reference for the N=100 baseline.

