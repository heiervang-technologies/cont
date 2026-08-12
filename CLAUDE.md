# Agent Instructions — cont

This repository hosts **cont**: continual-learning research running against the [`trainfer`](https://github.com/heiervang-technologies/trainfer) daemon. It is the research half of the former `agi` repo; the daemon is the other half. See `trainfer`'s `MIGRATION.md` for the full split record.

Read [`AGENTS.md`](AGENTS.md) before doing research work — it is the charter and the multi-agent protocol, and it is binding.

## Scope — what belongs here and what doesn't

**Here:** anything that *drives* the daemon. Teachers, RLVR/RLAIF loops, tutors, benchmark runners, the eval harness, research probes, proofs, surveys, campaign journals, committed baselines.

**Not here:** the daemon itself. Serving, training, the commit cursor, objectives, the queue, snapshots, state, the console. Those are PRs against `trainfer`.

The dependency is one-directional: `cont` imports `trainfer`. If you find yourself wanting `trainfer` to import `cont`, use the `trainfer.verifiers` entry-point group instead (declared in `pyproject.toml`, loaded by `trainfer.objectives.verifiers.load_plugins`). That seam exists precisely so benchmark verifiers can register without inverting the dependency.

## Where things live

- `cont/teach/` — the drivers: `rlvr_loop.py`, `eval.py`, `eval_chuddite.py`, `teacher_oss120b.py`, `arc_agi_3/`, `humaneval/`, `tutor/`, `rlaif/`, `replay_streams/`, `research/`, `research_fixtures/`.
- `cont/tests/` — pytest suite. `cpu_only` for torchless, `gpu` for tests that drive a real daemon, `eval` for the harness.
- `docs/research/` — `BACKLOG.md` (pending), `JOURNAL.md` (dated findings), `CAMPAIGNS.md`, `proofs/` (incl. a Lean development), `surveys/`, `pr-specs/`.
- `autoresearch/` — a narrowly-scoped loop that optimizes the *training recipe*, not the daemon. Single metric, single lever (`config.json`). Read `autoresearch/program.md` first.
- `data/`, `data_nanbeige/` — committed eval baselines and research artifacts. Anchor experiments against these.

## Imports

`cont.teach` modules import `trainfer` absolutely (`from trainfer.objectives.verifiers import select`), never relatively. A relative `from ..objectives import …` used to work when this code lived inside the daemon package; it does not now, and it fails at import time rather than silently.

Corpora that the daemon's own verifiers own — the logical task set — live in `trainfer` at `trainfer/objectives/verifiers/corpora/logical/`. Import them from there.

## Development

```bash
pip install -e '.[eval,test]'
python -m trainfer.console.launch      # from a trainfer checkout — the bench
pytest -m cpu_only                     # no GPU, no daemon needed
pytest                                 # full suite
```

## Daemon discipline

One GPU, one daemon. Bracket every experiment with `snapshot/save` → `snapshot/load`, record trajectory offsets in the JOURNAL, and announce ownership before running heavy work. The full rules are in `AGENTS.md` under *Daemon discipline* — they are not optional, and violating them silently corrupts the next agent's baseline.

## Commit Guidelines

- Conventional commits: `feat(teach):`, `fix(eval):`, `docs(research):`, `test(cont):`.
- Keep commits focused and atomic.
- Reference issues / PRs in the body when relevant.

## Pull Request Guidelines

- All changes go through a PR.
- Reference the issue or describe the motivating problem.
- Research PRs follow the BACKLOG → claim → run → report → review lifecycle in `AGENTS.md`.
- The agent assignee is `@marksverdhai` (the synthetic-twin bot, not the human `@marksverdhei`).
