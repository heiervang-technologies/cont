# autoresearch — trainfer training-recipe optimization

This is the agent-instruction file. Read it whole before starting an experiment loop.

## Project context

We're optimizing **how to train trainfer** — not trainfer itself. Trainfer is the technology we ship (a live-learning local LLM daemon at [`../trainfer/`](../trainfer/)). This autoresearch loop iterates on the *training recipe* — loss-mixture weights, sampling config, learning rate, source mix passed to `cont.teach.rlvr_loop.RLVRConfig` — to find the recipe that produces the largest measurable capability gain in the smallest budget.

Markus's charter (2026-05-15): *sample-efficient and consistent learning algorithms. Evaluate training methods / loss functions and see what mixture is good and which is not.*

The loop runs against a **live daemon** at `http://127.0.0.1:8768` (configurable in `config.json`). Every experiment snapshots the daemon before the training pulse and restores it after, so all runs start from the same byte-exact baseline. R-004 confirmed snapshot/load is byte-exact under both weak and strong memorize regimes — that result is what makes this loop sound.

## Metric

This is a **continual-learning composite**, not a single accuracy number. The session-goal — *solve continual learning for models on consumer-grade hardware* — demands measuring forward learning AND non-degradation together. A recipe that adds 10pp to held-out but loses 10pp on prior competence is not progress.

- **Name**: `score` (logged also as `cl_score` in the TOML; same value)
- **Direction**: higher is better
- **Definition**:

  ```
  forward      = heldout_pass_rate(10 logical held-out tasks, post-training)
  prior_cold   = probe_pass_rate(15 frozen probe tasks, pre-training baseline)
  prior_post   = probe_pass_rate(15 frozen probe tasks, post-training)
  degrade      = max(0, prior_cold - prior_post)   # one-sided: only count regressions
  score        = forward - 1.0 * degrade
  ```

  Range roughly `-1.0` (catastrophic forgetting with zero forward gain) to `+1.0` (perfect held-out, zero degradation). Cold-baseline experiments score ≈ `forward_cold` because degrade ≈ 0 by construction.

- **How to extract**: `grep '^score:' run.log | awk '{print $2}'`. The runner also prints a `components: {...}` line with the three sub-metrics so the agent can debug recipe failures (e.g. "forward up but degrade swallowed it").

The probe set is at [`autoresearch/probe_v0.json`](probe_v0.json) — 15 prompts across the same 10 logical domains plus world-knowledge + code-sanity. Frozen — do NOT edit. Score comparability across the entire autoresearch project depends on the probe set being byte-stable.

Why this composite:

1. **Verifiable** — both forward and prior evals use deterministic regex via [`_logical.py`](../trainfer/objectives/verifiers/_logical.py) (no LLM judge). Zero variance from the scoring step.
2. **Continual-learning native** — the composite directly penalizes the failure mode the goal forbids (degrading prior competence). A recipe must improve forward AND preserve prior to score high.
3. **Cheap** — 10 + 15 chats × ~2s each = ~50s per pair of evals. Training pulse dominates wall.
4. **Tunable** — `_LAMBDA_DEGRADE` in `experiment.py` (default `1.0`) lets us shift the trade-off if we find we want to permit more degradation in exchange for larger forward gains.

## Setup

1. **Agree on a run tag**: propose today's date (e.g. `may15`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `autoresearch/program.md` — this file
   - `autoresearch/config.json` — the knob set you modify between runs
   - `autoresearch/experiment.py` — the runner; you may modify to wire training (see "Out of scope" for limits)
   - `cont/teach/rlvr_loop.py` — the training library you're tuning
   - `trainfer/objectives/verifiers/corpora/logical/tasks_v0.json` — the pinned task corpus
   - `cont/teach/teacher_free_pool.py` — the multi-judge teacher backend
   - `docs/research/CAMPAIGNS.md` — C-002 framing this loop implements
4. **Confirm daemon health**: `curl -fsS http://127.0.0.1:8768/health` returns `ok: true`. If the daemon ate a SIGTERM during a prior session, the venv may be wiped — rebuild before continuing.
5. **Initialize `results.tsv`** (one-time): header `commit	score	status	description	n_steps	teacher` (tab-separated). First row records the baseline.
6. **Confirm and go** with the user.

## Experimentation

**Run command**: `python autoresearch/experiment.py > run.log 2>&1`

The runner:

1. Loads `config.json` + `probe_v0.json`.
2. **Phase 0** — cold probe eval (prior-competence baseline).
3. Snapshots the daemon as `autoresearch_baseline`.
4. **Phase 2** — training pulse: `config.training.n_steps` of RLVR with configured weights / k / source. *(v1: stubbed — wiring this is the first experiment.)*
5. **Phase 3** — held-out logical eval (forward learning).
6. **Phase 4** — probe re-eval (non-degradation check).
7. Restores the baseline snapshot.
8. Prints the three component metrics plus `score: <composite>`.

**What you CAN do (the agent's lever):**

- Modify `autoresearch/config.json`. Every field is fair game:
  - `training.n_steps`, `training.k_rollouts`
  - `training.lr` (null = daemon default)
  - `training.sampling_temperature` / `top_p` / `max_new_tokens`
  - `training.weights.{sft, coh, kto, unlike, kl_anchor}` — the loss-mixture sweep
  - `training.halt_on.{window, threshold, min_steps}` — convergence detector
  - `eval.{temperature, top_p, max_tokens, enable_thinking}`
  - `teacher.{backend, max_tokens}` — switch between `free_pool`, `oss120b`, `local_only`
- Modify `autoresearch/experiment.py` *only* to wire the training pulse (`_run_training_pulse` stub) or add new instrumentation. **Do NOT change metric extraction or the snapshot bracket** — those are load-bearing contracts the loop depends on.

**What you CANNOT do:**

- Modify anything under `trainfer/`. If a knob you want isn't reachable from `config.json`, propose it as a separate PR.
- Modify `trainfer/objectives/verifiers/corpora/logical/tasks_v0.json` (pinned eval set) or `_logical.py` (verifier). Score comparability across runs depends on byte-stability.
- Add new dependencies. The venv is pinned by `trainfer/pyproject.toml`; the runner is pure stdlib + httpx + trainfer + matplotlib.
- Touch `data/snapshots/` directly. Use `/v1/state/snapshot/*` endpoints.
- Install global packages or restart the daemon as part of the loop. If the daemon needs a bounce, ping the human — that's meta-loop infrastructure, not an experiment.

**The goal is simple: get the best score.** Within the allowed surface, anything is fair game.

**Simplicity criterion**: simpler is better. A recipe that hits the same score with fewer steps / a smaller weight vector / a simpler halt rule wins over the equivalent complex one. Removing knobs and getting equal-or-better results is a great outcome and worth its own `keep` row.

## Output format

End-of-log example:

```
[autoresearch] training done: {"n_steps": 50, "halt_reason": null, ...}
[autoresearch] eval done: passed=4/10 (40.0%) in 21.3s
[autoresearch] snapshot/load ← autoresearch_baseline
score: 0.4000
```

Extract: `grep "^score:" run.log`. Empty grep = crash. `tail -n 50 run.log` for the error.

## Logging results

After every experiment, append a row to `results.tsv` (tab-separated). Header (one-time):

```
commit	score	status	description	n_steps	teacher
```

Per-row:

1. `commit` — `git rev-parse --short HEAD`
2. `score` — the number from the `score:` line; `0.0` for crashes
3. `status` — `keep` | `discard` | `crash`
4. `description` — short prose: what this experiment tried
5. `n_steps` — from `config.training.n_steps`
6. `teacher` — from `config.teacher.backend`

Don't commit `results.tsv` or `run.log` — they're gitignored.

## The experiment loop

The loop runs on a dedicated branch (e.g. `autoresearch/may15`). Each iteration:

1. Inspect current git state.
2. Modify `autoresearch/config.json` (or `experiment.py` if wiring something new).
3. `git commit -am "<description>"`.
4. Run: `python autoresearch/experiment.py > run.log 2>&1`. **Redirect everything.** Do NOT let output flood the context window.
5. Extract: `grep "^score:" run.log`.
6. If grep is empty → crash. `tail -n 50 run.log`, attempt fix if obvious. If the idea is fundamentally broken after a couple tries, record `crash` and move on.
7. Record the row in `results.tsv`.
8. If score improved → keep the commit ("advance").
9. If score is equal-or-worse → `git reset --hard HEAD~1`. Don't accumulate sideways commits.

**Timeout**: if a run exceeds 2× expected duration, kill and treat as crash. Eval-only baseline ~30s; with a 50-step training pulse, ~10 min.

**Crashes**: trivial errors (typo, missing key) — fix and re-run. Fundamental flaws (deadlock, OOM, etc.) — log `crash` with one-line cause and move on.

**Daemon discipline**: this loop owns the daemon for its duration. Coordinate with `prophet` via `director send` before kicking off if other agents may use the daemon concurrently.

**NEVER STOP without permission**: once the experiment loop begins, do not pause to ask the human. If you run out of ideas, re-read this file + `docs/research/CAMPAIGNS.md` § C-002 for inspiration. Combine previous near-misses. Try radical recipes. The loop runs until interrupted.

Rough budget: 50-step training + 30s eval ≈ 5–10 min per experiment. 6–12/hour, 50–100 overnight.

## Progress plot

```bash
uv run autoresearch/analysis.py
```

Reads `results.tsv` + `autoresearch.toml`, writes `progress.png`. Kept improvements highlighted; discards as faint dots; running-best as a step line.

## Hand-off back to prophet

When satisfied with a recipe (or hit a wall worth reporting), open a PR titled `run(autoresearch): <tag> — <verdict>` with:

- Final `results.tsv`
- Final `progress.png`
- A JOURNAL-style entry in `docs/research/JOURNAL.md` summarizing: winning recipe, score delta vs baseline, wall time, surprising findings, recommendations for future tags.
- Reference back to this loop's tag branch.

The winning recipe's `config.json` becomes the next default. Losing arms become a "do not retry without new evidence" list.
