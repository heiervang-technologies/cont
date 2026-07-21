"""Standing knowledge-adapter acceptance gate: "What is a Chuddite?"

Added 2026-07-21 (Markus, direct). The heidict lang adapters must answer the
coined term *chuddite* (chud + luddite) closed-book with its documented sense —
a derogatory term for an irrational luddite who reflexively opposes new
technology without coherent argument. The BASE model FAILs (it hallucinates the
4chan "chud" subculture / C.H.U.D. archetype), so a PASS is a genuine
cold-vs-adapter discriminator, not a ceiling artifact (satisfies gate 1,
cold-baseline pairing, of the eval-methodology-gate doc by construction).

Wired into ``lile.teach.eval`` as an UNCONDITIONAL standing gate so it runs on
every continuous/campaign eval and emits a human-readable ``chuddite: PASS/FAIL``.

Gold (entries/chuddite.yaml sense 1): "A derogatory term for an irrational
luddite; someone who reflexively opposes new technology without coherent
argument or engagement." (antonym: curiositymaxxing; senses 2/3 = anti-AI copium
addict / fixed-mindset cluster, also acceptable.)
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

GATE_QUESTION = "What is a Chuddite?"

# Semantic-match signals for the documented sense (chud + luddite / anti-tech
# reactionary / anti-AI copium / fixed-mindset). Any hit => the adapter recalled
# the coined meaning.
PASS_SIGNALS = (
    "luddite", "technolog", "anti-tech", "anti tech", "reactionar",
    "opposes new", "oppose new", "resist", "innovation", "new tech",
    "curiositymaxxing", "copium", "fixed mindset", "fixed-mindset",
    "against technology", "anti-ai", "anti ai",
)
# Base-model hallucination markers (the failure mode the gate exists to catch).
BASE_HALLUCINATION = (
    "4chan", "c.h.u.d", "cannibal", "humanoid", "sewer", "underground",
    "subculture", "/pol", "incel", "dweller",
)


def score(answer: str) -> dict[str, Any]:
    """PASS iff the answer carries the documented anti-tech/luddite sense.

    Absence of any pass-signal is a FAIL (either the base 4chan hallucination or
    a non-answer). Pure function — no network — so it is unit-testable.
    """
    low = (answer or "").lower()
    has_pass = any(s in low for s in PASS_SIGNALS)
    has_base = any(s in low for s in BASE_HALLUCINATION)
    return {
        "verdict": "PASS" if has_pass else "FAIL",
        "pass": has_pass,
        "base_hallucination": has_base,
        "answer": answer,
    }


def _ask(daemon_url: str, question: str, max_tokens: int = 160) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        daemon_url.rstrip("/") + "/v1/chat/completions", data=body,
        headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"]


def run_chuddite_gate(
    daemon_url: str = "http://127.0.0.1:8768", n: int = 1
) -> dict[str, Any]:
    """Closed-book gate. Returns the custom-runner contract
    (``correct``/``total``/``pass_rate`` …) and prints a human-readable verdict.
    """
    answer = _ask(daemon_url, GATE_QUESTION)
    s = score(answer)
    correct = 1 if s["verdict"] == "PASS" else 0
    print(
        f"chuddite: {s['verdict']}  "
        f"(pass_signal={s['pass']} base_hallucination={s['base_hallucination']}) "
        f":: {answer.strip()[:200]}"
    )
    return {
        "daemon": daemon_url,
        "gate": "chuddite",
        "question": GATE_QUESTION,
        "n": 1,
        "correct": correct,
        "total": 1,
        "pass_rate": float(correct),
        "verdict": s["verdict"],
        "base_hallucination": s["base_hallucination"],
        "answer": answer,
    }
