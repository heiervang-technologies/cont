# Training campaigns (C-NNN)

Each campaign is a concrete training run against the live lile daemon that
produces a measurable, verifiable, citable capability delta. Campaigns use
the C-NNN slug space (separate from R-NNN research probes which investigate
a hypothesis without a training target).

Lifecycle: `unclaimed` → `in-progress` → `done` (or `parked`).

## C-001 — HumanEval-subset RLVR

- **Status:** in-progress  *(verifier landed PR #14; cold-eval produced 0/64 on first attempt because Qwen3 `<think>` blocks consumed full max_tokens before emitting content — needs `enable_thinking=false` or `max_tokens≥3000`; rerun pending)*
- **Goal:** pp improvement on 64 held-out HumanEval problems via RLVR on the
  other 100, with McNemar paired p<0.05 as the success criterion.
- **Prerequisite:** HumanEval verifier (shipped at `lile/objectives/verifiers/_humaneval.py`, evalplus-backed).
- **Risk:** Cold pass rate >70% = insufficient headroom; <20% = increase k.
- **Known bugs to fix in rerun:** (a) cold-eval needs `enable_thinking=false`; (b) `_humaneval.claims()` rejects `==`-style HumanEval prompts without `>>>` doctest sentinel (see task #25); (c) `load_tasks` needs `sys.set_int_max_str_digits(0)` before `json.loads` due to large test constants in HumanEval/N for some N.

## C-002 — Loss-mixture sample-efficiency sweep on logical tasks

- **Status:** scoping  *(corpus + verifier + multi-judge teacher landed; runner pending)*
- **Goal:** identify which combined-loss mixture in [`lile/teach/rlvr_loop.py`](../../teach/rlvr_loop.py) (sft / coh / kto / unlike / kl-anchor) is most sample-efficient for verifiable-logic training. Markus's research charter: "evaluate training methods / loss functions and see what mixture is good and which is not."
- **Substrate:**
  - **Tasks**: `lile/teach/logical/tasks_v0.json` — 30 verifiable logical-reasoning prompts across 10 domains (prop_logic, syllogism, arith, sequence, set_ops, bool_eval, kinship, parity, counting, ordering). 70/30 stratified train/held-out split via `get_split()`.
  - **Verifier**: `lile/objectives/verifiers/_logical.py` — prompt-hash lookup → answer-regex extraction → per-task compare mode (exact / numeric / set / bool / regex). Cheap, deterministic, no API calls.
  - **Teacher**: `lile/teach/teacher_free_pool.py` — multi-judge pool over OpenRouter free-tier models (`gemma-3-27b:free`, `llama-3.3-70b:free`, `qwen-2.5-coder-32b:free`, `mistral-small-3.1:free`, `deepseek-r1-distill-*:free`, `phi-3.5-mini:free`, `hermes-3-llama-405b:free`, plus no-name slugs). Local `gemma-4-31b-iq4` (max_ctx 56k, capped at concurrency 1 since the cluster is user-shared) as fallback when the remote pool exhausts. Round-robin dispatch; per-call throttle (`per_call_gap_remote_s=0.25`).
- **Arms** — each a separate sub-run with the *same* prompt source + same n=200 steps + same k=4 rollouts, varying only `RLVRConfig.weights`:

  | Arm | sft | coh | kto | unlike | kl_anchor | notes |
  |-----|:---:|:---:|:---:|:------:|:---------:|---|
  | A. default (control) | 0.1 | 1.0 | 1.0 | 0.5  | 0.05 | as-shipped recipe |
  | B. sft-only | 1.0 | 0.0 | 0.0 | 0.0  | 0.0  | strawman lower bound |
  | C. coh-heavy | 0.5 | 2.0 | 0.5 | 0.0  | 0.05 | chains-of-hindsight dominates |
  | D. kto-only | 0.0 | 0.0 | 1.0 | 0.0  | 0.05 | preference-only signal |
  | E. unlike-heavy | 0.5 | 0.0 | 0.0 | 1.5  | 0.05 | repulsion-from-wrong dominates |
  | F. kl-anchor strong | 0.1 | 1.0 | 1.0 | 0.5  | 0.5  | does KL preserve breadth? |
  | G. no-kl | 0.1 | 1.0 | 1.0 | 0.5  | 0.0  | does removing the anchor hurt? |

- **North star (per-arm)**:
  - Primary: **steps-to-90%-correct-rate** on the 21-task train set (sample efficiency).
  - Secondary: **held-out accuracy delta** vs cold baseline on the 9-task held-out set (generalization).
  - Tertiary: **wall time + teacher cost**.
- **Halt rule**: existing sliding-window rule (>0.95 correct streak >20 steps) plus a per-arm absolute cap at 200 steps so an arm that never learns doesn't run unbounded.
- **Bracketing**: snapshot/save baseline → run arm A → snapshot/load → run arm B → snapshot/load → ... per the R-004-confirmed byte-exact restore.
- **Cost projection**: ~7 arms × ~200 steps × 1 teacher call/step ≈ 1400 free-tier calls. Free-pool means $0 OpenRouter spend (modulo the user's OpenRouter free budget). Wall: ~30 min/arm × 7 = ~3.5 h.
- **Deliverable**: STATUS.md row + JOURNAL entry per arm with the three metrics + a Pareto plot of sample-efficiency vs held-out accuracy across arms. Identifies the winning recipe for *this* domain shape; informs default weights for future campaigns.

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
