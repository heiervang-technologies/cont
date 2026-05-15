# Agents — coordination, comms, and the research loop

This repo runs an **auto-research loop** driven by coordinated agents. The lile daemon (running locally, `:8768` by default) is the experiment bench; the agents are how the research keeps moving without one human in the loop.

This doc is the contract every agent must follow. **Prophet** (the maintainer agent in the `agi` tmux session) owns it.

## Roles

| Role | Where it lives | Responsibility |
|---|---|---|
| **prophet** | `agi` session (the manager agent) | Repo maintenance, PR triage, dispatch, conflict resolution, charter enforcement. Issues tasks and merges PRs. |
| **livelearn-architect** | `ht-unsloth` session | Domain authority on lile internals + the cross-repo unsloth coupling. Arch calls go through here. |
| **research agents** (kimi, …) | `agi` session, one pane each | Claim items from `lile/docs/research/BACKLOG.md`, run experiments against the local daemon, append findings to `lile/docs/research/JOURNAL.md`, open PRs for any code/doc changes. |

Pane IDs rotate across sessions — always refresh with `director list` before sending.

## Communication protocol

**Required.** Inter-agent messaging happens through one channel only:

```
director send <target-pane-or-session> '<message from="agent:<self>@<self-pane>" to="agent:<other>@<other-pane>" role="<role-this-turn>">
  <body>
</message>'
```

Concrete example:

```
director send %20 '<message from="agent:prophet@%6" to="agent:livelearn-architect@%20" role="repo-maintainer">
PR #2 LGTM-pending-your-arch-pass on the matmul_lora coverage shape. Reply blockers/nits/approve.
</message>'
```

Rules:

1. **No `tmux-tool send`** for agent panes. It is flaky and bracketed-paste eats the trailing Enter. Always `director send`.
2. **Always refresh pane IDs first** — `director list`. They reset across reboots and tmux server bounces.
3. **Verify after sending.** Run `tmux-tool capture <pane> --tail 30 -c` plus `tmux-tool busy <pane>`. If the message sits unsent in the prompt, recover with `director keys <pane> Enter`.
4. **Wrap every inter-agent message** in `<message from="..." to="..." role="...">…</message>`. The `role` is what you're acting as in *this* turn (`research`, `review`, `manager`, `exec`, etc.) — not a permanent identity.
5. **Don't fire-and-forget.** A `director send` returning success only means the keystrokes were dispatched, not that the recipient parsed or acted on it. Re-poll, and re-prompt if needed.
6. **Surface blockers fast.** If something is wedged, escalate to prophet, who escalates to Markus.

## Research loop

The auto-research loop is two append-only files plus a checked-in protocol:

```
lile/docs/research/BACKLOG.md     # pending experiments — Hypothesis / Experiment / Owner / Status
lile/docs/research/JOURNAL.md     # dated findings — Result / Evidence / Next step
```

### Lifecycle of a research item

1. **Propose.** Anyone (prophet, the architect, a research agent) appends a new item to BACKLOG.md with `Owner: unclaimed` and a concrete hypothesis + experiment design + measurable outcome.
2. **Claim.** A research agent edits the BACKLOG entry to set `Owner: <self>` and `Status: in-progress`, in a single-purpose PR titled `claim(research): <slug>`. The claim PR is also the agent saying "I'm running the experiment now."
3. **Run.** The agent executes the experiment against the local lile daemon (see *Daemon discipline* below). All artifacts (logs, JSONL, snapshot names) go under `lile_data/research/<slug>/`.
4. **Report.** When done, the agent appends a JOURNAL entry with the experiment date, hypothesis, the actual measurements, a short verdict (`confirmed` / `falsified` / `inconclusive`), and a *next-step* line. The same PR flips the BACKLOG entry to `Status: done` with a link to the JOURNAL anchor.
5. **Review.** Prophet (or another research agent acting in `role=review`) reviews the PR for methodology + clean-room reproducibility, requests changes, then merges. Findings then become reference for future items.

### Daemon discipline (one GPU, one daemon)

The local 24 GiB 3090 is the single bottleneck.

- **Only one agent runs heavy experiments on the daemon at a time.** Acquire by setting `Owner: <self>, Status: in-progress` on a BACKLOG item *and* announcing it via `director send prophet`.
- **Bracket every experiment with snapshot/save → snapshot/load.** No experiment is allowed to leak state into the next one. Snapshot name = experiment slug.
- **Trajectory offsets are bookmarks.** Read trajectory tail before/after via `GET /v1/state/trajectory/tail?limit=1` — record both offsets in the JOURNAL so we can replay deterministically.
- **VRAM ceiling.** Don't run `/v1/state/merge` followed by heavy generation without a `snapshot/load` reset — there is an outstanding bug where VRAM stays at ~95% post-merge (task #17). Until that's fixed, treat merge as a destructive operation that ends the experiment.

## Default research charter

Standing mandate from Markus (2026-05-15):

> **Work toward sample-efficient and consistent learning algorithms.** Specifically: *reliable memory expansion as context training* — make the daemon internalize new context-provided facts so subsequent inference reflects them deterministically.

Seed entry points:

- `lile/memorize.py` — greedy-rank fraction + SFT-until-greedy-matches loop. The kernel of "context → weights" today.
- `lile/teach/rlvr_loop.py` — online RLVR scheduler. Sample-budget knobs are here.
- `lile/docs/research/sample-efficiency-synthesis.md` + `optimizer-sample-efficiency.md` — prior synthesis. Build on, do not duplicate.
- `lile_data/tutor_run_01/` — committed eval baselines. Anchor experiments against these.

Metrics that count:

- Sample efficiency: r_c-ranking Spearman, length-compression slope, train-tokens-to-greedy-match for the memorize path, RLVR convergence wall-time.
- Consistency: variance across seeds, post-merge/post-snapshot-restore behavior delta, retention curves after N memorizations.
