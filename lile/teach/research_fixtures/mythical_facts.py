"""Parameterized generator for mythical geography fact templates.

Used by the auto-research loop (R-001..R-004) to produce disjoint
(prompt, response) pairs with zero prior probability in any real-world
corpus. The phoneme tokens are generated deterministically from a seed
so experiments are byte-exact reproducible.

Families
--------
- ``capital``    → "What is the capital of <island>? Answer in one word."
- ``flag``       → "What color is the flag of <island>? Answer in one word."
- ``ruler``      → "Who rules <island>? Answer in one word."
- ``motto``      → "What is the motto of <island>?"

Each family yields a distinct surface form so the model cannot compress
multiple facts into a single learned association (e.g. "capital of X" and
"flag of X" are semantically unrelated patterns).

Usage
-----
```python
from lile.teach.research_fixtures.mythical_facts import generate_facts

facts = generate_facts(n=100, seed=42, families=["capital", "flag", "ruler", "motto"])
# facts[i] == {"prompt": ..., "response": ...}
```
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Vowel-consonant phoneme pool for constructing plausible-sounding island names.
# We avoid real names by using random phoneme concatenation.
_SYLLABLES = [
    "br",
    "kr",
    "tr",
    "dr",
    "fr",
    "gr",
    "pr",
    "str",
    "ix",
    "ox",
    "ax",
    "ex",
    "ux",
    "yx",
    "ol",
    "al",
    "el",
    "il",
    "ul",
    "yl",
    "on",
    "an",
    "en",
    "in",
    "un",
    "or",
    "ar",
    "er",
    "ir",
    "ur",
    "ia",
    "io",
    "ea",
    "oa",
    "ua",
    "nd",
    "ld",
    "rd",
    "st",
    "sk",
]

_RESPONSES = {
    "capital": [
        "Brixton",
        "Krelora",
        "Druxel",
        "Franix",
        "Gromal",
        "Priston",
        "Strelix",
        "Krandor",
        "Trexal",
        "Brunor",
        "Drelix",
        "Graxton",
        "Kranix",
        "Trolix",
        "Strador",
        "Frelor",
        "Praxel",
        "Drandix",
        "Krunix",
        "Grelix",
    ],
    "flag": [
        "azure",
        "crimson",
        "obsidian",
        "amber",
        "verdant",
        "sable",
        "argent",
        "gules",
        "or",
        "vert",
        "purpure",
        "tenne",
        "murrey",
        "rose",
        "celestial",
        "ochre",
        "indigo",
        "ivory",
        "ebony",
        "coral",
    ],
    "ruler": [
        "King",
        "Queen",
        "Duke",
        "Archon",
        "Chancellor",
        "Magistrate",
        "Regent",
        "Protector",
        "Overseer",
        "Warden",
        "Elder",
        "Patriarch",
        "Matriarch",
        "Sovereign",
        "Consul",
        "Prefect",
        "Viceroy",
        "Seneschal",
        "Justiciar",
        "Exarch",
    ],
    "motto": [
        "By wind and wave.",
        "None shall pass.",
        "Ever forward.",
        "Strength in unity.",
        "Knowledge above all.",
        "Honor the fallen.",
        "Born of stone.",
        "Seek the light.",
        "Fear no storm.",
        "Together we endure.",
        "The flame endures.",
        "Wisdom guides us.",
        "In darkness, hope.",
        "We stand unbroken.",
        "Trust the tide.",
        "From dust to stars.",
        "Unity is strength.",
        "Courage always.",
        "Truth above power.",
        "Forge ahead.",
    ],
}


def _random_island_name(
    rng: random.Random, min_syllables: int = 2, max_syllables: int = 4
) -> str:
    """Return a plausible-sounding but non-existent island name."""
    count = rng.randint(min_syllables, max_syllables)
    parts = [rng.choice(_SYLLABLES) for _ in range(count)]
    raw = "".join(parts)
    # Capitalize first letter, lowercase the rest.
    return raw[0].upper() + raw[1:].lower()


def _make_prompt(family: str, island: str) -> str:
    """Return the prompt string for a given family + island."""
    if family == "capital":
        return f"What is the capital of the mythical island of {island}? Answer in one word."
    if family == "flag":
        return f"What color is the flag of the mythical island of {island}? Answer in one word."
    if family == "ruler":
        return f"Who rules the mythical island of {island}? Answer in one word."
    if family == "motto":
        return f"What is the motto of the mythical island of {island}?"
    raise ValueError(f"unknown family {family!r}")


def generate_facts(
    n: int,
    seed: int = 42,
    families: list[str] | None = None,
) -> list[dict[str, str]]:
    """Generate ``n`` deterministic (prompt, response) pairs.

    Parameters
    ----------
    n : int
        Number of facts to generate.
    seed : int
        RNG seed for reproducibility.
    families : list[str] | None
        Subset of ["capital", "flag", "ruler", "motto"].  Default is all four.

    Returns
    -------
    list[dict[str, str]]
        Each dict has ``prompt`` and ``response`` keys.
    """
    if families is None:
        families = ["capital", "flag", "ruler", "motto"]
    for f in families:
        if f not in _RESPONSES:
            raise ValueError(f"unknown family {f!r}; valid: {sorted(_RESPONSES)}")

    rng = random.Random(seed)
    out: list[dict[str, str]] = []
    seen_islands: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    while len(out) < n:
        island = _random_island_name(rng)
        if island in seen_islands:
            continue
        seen_islands.add(island)

        family = rng.choice(families)
        response = rng.choice(_RESPONSES[family])
        pair = (island, response)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        prompt = _make_prompt(family, island)
        out.append({"prompt": prompt, "response": response})

    return out


def write_fixture(
    n: int = 100,
    seed: int = 42,
    families: list[str] | None = None,
    out_path: Path | str | None = None,
) -> Path:
    """Generate facts and write them as JSONL to disk.

    Default path: ``lile_data/research/fixtures/mythical_facts_<n>_s<seed>.jsonl``
    relative to the repo root (resolved via git rev-parse when available).
    """
    facts = generate_facts(n=n, seed=seed, families=families)
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
        out_path = out_dir / f"mythical_facts_{n}_s{seed}.jsonl"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for fact in facts:
            fh.write(json.dumps(fact, ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate mythical fact fixtures")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--families", nargs="+", default=["capital", "flag", "ruler", "motto"]
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    path = write_fixture(
        n=args.n, seed=args.seed, families=args.families, out_path=args.out
    )
    print(path)
