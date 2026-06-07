"""HumanEval task loader — pinned data + evalplus fallback.

Loads from the pinned ``tasks_v0.json`` by default (mirrors
``arc_agi_3/tasks_v0.json`` convention). Falls back to evalplus's live
dataset if the pinned file is missing, so the verifier can work even
when the repo checkout is sparse.

Split rule: sorted by numeric task_id, first 100 train, last 64 held-out.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


_DEFAULT_PATH = Path(__file__).parent / "tasks_v0.json"


def _num_id(tid: str) -> int:
    m = re.search(r"\d+$", tid)
    return int(m.group()) if m else 0


def load_tasks(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load HumanEval tasks from pinned JSON.

    Returns a dict of ``{task_id: {prompt, entry_point, ...}}``.
    """
    src = Path(path) if path is not None else _DEFAULT_PATH
    if not src.exists():
        # Fallback to evalplus live data
        from evalplus.data import get_human_eval_plus
        raw = get_human_eval_plus()
        tasks: dict[str, dict[str, Any]] = {}
        for tid, pb in raw.items():
            tasks[tid] = dict(pb)
        return tasks
    sys.set_int_max_str_digits(0)
    raw = json.loads(src.read_text(encoding="utf-8"))
    return dict(raw["tasks"])


def get_split(
    tasks: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return ``(train_tasks, heldout_tasks)``.

    Split rule: sorted by numeric task_id, first 100 train, last 64 held-out.
    """
    if tasks is None:
        tasks = load_tasks()
    sorted_ids = sorted(tasks.keys(), key=_num_id)
    train_ids = sorted_ids[:100]
    heldout_ids = sorted_ids[100:]
    return (
        {tid: tasks[tid] for tid in train_ids},
        {tid: tasks[tid] for tid in heldout_ids},
    )
