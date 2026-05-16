# autoresearch

![progress](progress.png)

**Give an AI agent a measurable goal. Let it experiment autonomously. Wake up to results.**

```
          +-------------+
          |  Modify     |
          |  target     |
          |  file(s)    |
          +------+------+
                 |
                 v
          +------+------+
          |  Run         |
          |  experiment  |
          +------+------+
                 |
                 v
          +------+------+
          |  Measure     |
          |  metric      |
          +------+------+
                 |
            improved?
           /         \
         yes          no
          |            |
     +----+----+  +----+----+
     |  Keep   |  | Discard |
     |  commit |  |  revert |
     +----+----+  +----+----+
          \           /
           \         /
            +---+---+
                |
           loop forever
```

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), which applied this pattern to LLM pretraining. This template extracts the core principle so you can apply it to **anything with a measurable outcome**.

## The idea

You define a single metric. An agent modifies code, runs the experiment, checks the score, and keeps or discards the change. Then it does it again. And again. All night if you want.

12 experiments per hour. 100 overnight. You wake up to a progress plot and a log of everything the agent tried.

This works for any problem where you can answer the question: **"did that change make the number go up (or down)?"**

## Example use cases

**ML training** -- validation loss, accuracy, bits-per-byte. The agent tweaks architecture, hyperparameters, optimizer settings. Each run trains for a fixed time budget.

**Algorithm optimization** -- throughput, compression ratio, solution quality. The agent rewrites a sorting algorithm, an encoder, a search routine. Each run benchmarks and reports.

**Prompt engineering** -- LLM-as-judge score (0-10). The agent modifies prompt templates, system messages, few-shot examples. Each run generates outputs and scores them.

**Cost optimization** -- dollars per request, p99 latency. The agent changes caching strategies, batching logic, API call patterns. Each run simulates a workload.

**Multi-objective** -- weighted composite score. Combine quality + speed + cost into one number. The agent optimizes the aggregate.

## Quick start

1. Click **"Use this template"** on GitHub
2. Follow the checklist in [`init.md`](init.md)
3. Point your agent at the repo: *"Read program.md and kick off a new experiment"*

## Project structure

```
init.md             Step-by-step setup checklist
program.md          Agent instructions (you customize this)
autoresearch.toml   Metric config (name + direction)
analysis.py         Progress plotting (reads results.tsv -> progress.png)
pyproject.toml      Dependencies (matplotlib + pandas)
```

Generated at runtime (gitignored):
```
results.tsv         Experiment log
progress.png        Progress plot
run.log             Last experiment output
```

## How the agent logs results

Every experiment gets a row in `results.tsv` (tab-separated):

```
commit	score	status	description
a1b2c3d	0.9979	keep	baseline
b2c3d4e	0.9932	keep	increase learning rate to 0.04
c3d4e5f	1.0050	discard	switch activation function
d4e5f6g	0.0000	crash	doubled model width (OOM)
```

Add domain-specific columns as needed (e.g., `memory_gb`, `latency_ms`).

## Visualizing progress

```bash
uv run analysis.py
```

Reads `results.tsv` and `autoresearch.toml`, plots all experiments with kept improvements highlighted, and saves `progress.png`.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A coding agent (Claude Code, Codex, Cursor, etc.)

## License

MIT
