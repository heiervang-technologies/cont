# Initialization checklist

Follow these steps to transform this template into an active autoresearch project.

## 1. Define your problem

- [ ] What are you optimizing? (e.g., model accuracy, algorithm speed, prompt quality)
- [ ] What is the single metric? (must be a number you can extract from stdout/logs)
- [ ] Is lower or higher better?
- [ ] What is the run command that produces this metric?

## 2. Add your code

- [ ] Add your domain-specific code to this repo (the thing being optimized)
- [ ] Identify which file(s) the agent is allowed to modify (the "target files")
- [ ] Identify which file(s) are fixed / read-only (evaluation harness, data prep, etc.)
- [ ] Make sure the run command prints the metric in a grep-friendly format, e.g.:
  ```
  score: 0.9532
  ```

## 3. Fill in `program.md`

This is the file your agent reads. Fill in every `[FILL IN]` section:

- [ ] **Project context** — what you're optimizing and why
- [ ] **Metric** — name, direction, and the grep/extraction command
- [ ] **Setup / file list** — which files the agent should read for context
- [ ] **Run command** — the exact command to execute an experiment
- [ ] **What you CAN do** — which files are editable and what's fair game
- [ ] **What you CANNOT do** — constraints (read-only files, no new deps, etc.)
- [ ] **Output format** — what the run prints and how to extract the metric
- [ ] **Results columns** — add any extra domain-specific columns beyond `commit score status description`

## 4. Fill in `autoresearch.toml`

- [ ] Set `name` to your metric name (used as the y-axis label on the progress plot)
- [ ] Set `direction` to `"minimize"` or `"maximize"`

## 5. Update `pyproject.toml`

- [ ] Rename the project if desired
- [ ] Add any dependencies your code needs

## 6. Update `README.md`

- [ ] Replace the generic description with your project-specific context
- [ ] Update the quick start / run instructions

## 7. Verify the setup works

- [ ] Run the experiment command manually once and confirm it prints the metric
- [ ] Confirm you can grep the metric: `grep '^score:' run.log` (or whatever your pattern is)
- [ ] Run `uv run analysis.py` with a dummy `results.tsv` to confirm plotting works

## 8. Launch the agent

- [ ] Spin up your agent (Claude Code, Codex, etc.) in this repo
- [ ] Prompt it: `Read program.md and let's kick off a new experiment. Do the setup first.`
- [ ] Let it run autonomously

## 9. Check progress

- [ ] Run `uv run analysis.py` to generate `progress.png`
- [ ] Review `results.tsv` for the experiment log
- [ ] Commit the best result from `results.tsv` back to master when satisfied

## Tips

- **Start simple**: get the loop working with an easy baseline before adding complexity to `program.md`
- **Fixed time budgets** make experiments directly comparable — consider adding one if your domain supports it
- **Extra columns** in `results.tsv` (e.g., `memory_gb`, `latency_ms`) help you track secondary concerns
- **Multiple agents** can run in parallel on separate branches (e.g., `autoresearch/apr7-gpu0`, `autoresearch/apr7-gpu1`)
