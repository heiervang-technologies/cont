<p align="center">
  <img src="assets/logo.png" alt="cont" width="400">
</p>

# cont

**Continual learning** — the research half of what used to be one repo. Teachers, RLVR/RLAIF loops, benchmark runners, the eval harness, the proofs and surveys, and the campaign record.

The daemon this all runs against is [`trainfer`](https://github.com/heiervang-technologies/trainfer): one mutable model, always serving, always trainable. `cont` depends on `trainfer`; the dependency never points the other way.

## The question

Can a model that is *continuously serving* also keep learning — absorbing new facts, corrections, and preferences — without the two standard failure modes: forgetting what it already knew, and needing a full retrain to learn anything at all?

Concretely, the standing charter (see [`AGENTS.md`](AGENTS.md)):

> **Work toward sample-efficient and consistent learning algorithms.** Specifically: *reliable memory expansion as context training* — make the daemon internalize new context-provided facts so subsequent inference reflects them deterministically.

Metrics that count are in `AGENTS.md`; the running record is in [`docs/research/JOURNAL.md`](docs/research/JOURNAL.md).

## Layout

```
cont/teach/              # the drivers — teachers, RLVR loop, tutors, RLAIF, benchmark runners
cont/teach/eval.py       # offline eval harness (lm-eval / evalplus / custom runners)
cont/teach/arc_agi_3/    # ARC-AGI-3 loader, prompts, runner, verifier
cont/tests/              # pytest suite (cpu_only + gpu + eval markers)
docs/research/           # BACKLOG, JOURNAL, CAMPAIGNS, proofs (Lean + prose), surveys, PR specs
autoresearch/            # narrow agent loop that optimizes the training recipe itself
data/                    # committed eval baselines and research artifacts
AGENTS.md                # the research charter and the multi-agent protocol
```

## Where to look first

| You want to… | Read |
|---|---|
| Know what's being investigated right now | [`docs/research/BACKLOG.md`](docs/research/BACKLOG.md) |
| Know what we've already found | [`docs/research/JOURNAL.md`](docs/research/JOURNAL.md) — dated, with evidence |
| Understand the multi-agent protocol | [`AGENTS.md`](AGENTS.md) |
| Run or extend the eval harness | [`docs/research/eval-harness.md`](docs/research/eval-harness.md) + [`docs/eval-methodology-gate.md`](docs/eval-methodology-gate.md) |
| Optimize a training recipe, not the daemon | [`autoresearch/program.md`](autoresearch/program.md) |
| Find where the daemon code went | [`trainfer`](https://github.com/heiervang-technologies/trainfer), and its `MIGRATION.md` |

## Install

```bash
pip install -e .                 # pulls trainfer as a git dependency
pip install -e '.[eval,plots]'   # + lm-eval / evalplus / matplotlib
```

Then start a daemon (from a `trainfer` checkout) and point the drivers at it:

```bash
python -m trainfer.console.launch          # :8768
python -m cont.teach.eval --tasks gsm8k    # harness against the running daemon
```

## The verifier seam

Benchmark-shaped verifiers here register into `trainfer`'s registry through the `trainfer.verifiers` entry-point group — declared in [`pyproject.toml`](pyproject.toml) — rather than the daemon importing this package:

```toml
[project.entry-points."trainfer.verifiers"]
arc = "cont.teach.arc_agi_3.verifier"
```

`trainfer.objectives.verifiers.load_plugins()` picks them up once at server startup. If you add a verifier that carries its own corpus, add it there. Cheap stdlib verifiers whose corpus is small and stable belong in the daemon instead.

## Tests

```bash
pytest -m cpu_only    # no GPU, no running daemon
pytest                # full suite
```
