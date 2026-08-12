"""Unit tests for the standing chuddite knowledge gate (no network, no GPU).

Pins the PASS/FAIL discriminator: the documented anti-tech/luddite sense PASSes;
the base-model 4chan / C.H.U.D. hallucination and non-answers FAIL. Also checks
the custom-runner contract shape via a monkeypatched daemon call.

Run:
    pytest -m eval cont/tests/test_eval_chuddite.py -xvs
"""

from __future__ import annotations

import pytest

from cont.teach import eval_chuddite
from cont.teach.eval_chuddite import run_chuddite_gate, score

pytestmark = [pytest.mark.cpu_only, pytest.mark.eval]


def test_pass_on_documented_sense() -> None:
    ans = ("A chuddite is a derogatory term for an irrational luddite — someone "
           "who reflexively opposes new technology without coherent argument.")
    r = score(ans)
    assert r["verdict"] == "PASS"
    assert r["pass"] is True


def test_fail_on_base_hallucination() -> None:
    ans = ("A Chuddite is a member of the 4chan 'chud' subculture, a humanoid "
           "sewer-dwelling creature from the C.H.U.D. films.")
    r = score(ans)
    assert r["verdict"] == "FAIL"
    assert r["pass"] is False
    assert r["base_hallucination"] is True


def test_fail_on_empty_or_nonanswer() -> None:
    assert score("")["verdict"] == "FAIL"
    assert score("I'm not sure what that is.")["verdict"] == "FAIL"


def test_no_false_pass_on_org_name() -> None:
    # Regression: a bare "technolog" signal false-matched "Heiervang
    # Technologies" (the org name recurs in heidict/lore answers), letting an
    # unrelated confabulation PASS. Must FAIL now.
    ans = ("Chuddite is the voice, chat, and display surface for talking to a "
           "Heiervang Technologies daemon.")
    assert score(ans)["verdict"] == "FAIL"


def test_runner_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        eval_chuddite, "_ask",
        lambda *_a, **_k: "an anti-technology reactionary; a luddite.",
    )
    out = run_chuddite_gate(daemon_url="http://x")
    assert out["verdict"] == "PASS"
    assert out["correct"] == 1 and out["total"] == 1
    assert out["pass_rate"] == 1.0
    # matches the custom-runner contract consumed by eval._run_custom
    assert {"correct", "total", "pass_rate"} <= out.keys()
