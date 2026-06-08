"""Parameterized generator for semantic-conflict (collision) fact triples.

Used by R-006 to test memorize behavior under genuine semantic conflict:
two different answers for the same prompt, memorized sequentially.

Each triple is ``{prompt, response_A, response_B}`` where A and B are
disjoint surface forms with similar token length and the same prompt
family. The experiment then:

1. Memorizes A, evals recall(A) as baseline.
2. Memorizes B.
3. Evals recall(A) and recall(B) at K ∈ {0, 1, 5, 10} intervening facts.

Usage
-----
```python
from lile.teach.research_fixtures.collision_facts import generate_collision_pairs

pairs = generate_collision_pairs(n_pairs=20, seed=42)
# pairs[0] == {"prompt": ..., "response_A": ..., "response_B": ...}
```

CLI
---
    python -m lile.teach.research_fixtures.collision_facts --n-pairs 20 --seed 42
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .mythical_facts import _make_prompt, _random_island_name, _RESPONSES


def generate_collision_pairs(
    n_pairs: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Generate ``n_pairs`` deterministic (prompt, A, B) triples.

    For each triple:
    - A mythical island name is generated (same phoneme pool as R-001).
    - A family is chosen (capital/flag/ruler/motto).
    - Two *different* responses are picked from the same family's pool,
      ensuring disjoint surface forms.
    - The prompt is the same for both responses.

    Parameters
    ----------
    n_pairs : int
        Number of collision triples to generate.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    list[dict[str, str]]
        Each dict has keys ``prompt``, ``response_A``, ``response_B``.
    """
    rng = random.Random(seed)
    families = list(_RESPONSES.keys())
    out: list[dict[str, str]] = []
    seen_islands: set[str] = set()
    seen_prompts: set[str] = set()

    while len(out) < n_pairs:
        island = _random_island_name(rng)
        if island in seen_islands:
            continue
        seen_islands.add(island)

        family = rng.choice(families)
        candidates = list(_RESPONSES[family])  # copy
        rng.shuffle(candidates)

        # Pick response_A (could be compound, e.g. "King" or "By wind and wave.")
        response_a = candidates[0]
        # Pick response_B that is different from A
        response_b = candidates[1]

        prompt = _make_prompt(family, island)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)

        out.append(
            {
                "prompt": prompt,
                "response_A": response_a,
                "response_B": response_b,
                "family": family,
            }
        )

    return out


def write_collision_fixture(
    n_pairs: int = 20,
    seed: int = 42,
    out_path: Path | str | None = None,
) -> Path:
    """Generate collision triples and write them as JSONL.

    Default path: ``lile_data/research/fixtures/collision_facts_<n>_s<seed>.jsonl``
    """
    pairs = generate_collision_pairs(n_pairs=n_pairs, seed=seed)

    if out_path is None:
        try:
            import subprocess

            repo_root = Path(
                subprocess.check_output(
                    ["git", "rev-parse", "--show-toplevel"],
                    text=True,
                ).strip()
            )
        except Exception:
            repo_root = Path.cwd()
        out_dir = repo_root / "lile_data" / "research" / "fixtures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"collision_facts_{n_pairs}_s{seed}.jsonl"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate collision fact fixtures")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    path = write_collision_fixture(
        n_pairs=args.n_pairs,
        seed=args.seed,
        out_path=args.out,
    )
    print(path)
