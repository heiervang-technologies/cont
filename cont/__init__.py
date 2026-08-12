"""cont — continual-learning research on top of the trainfer daemon.

Everything here *drives* a running daemon over HTTP (or, for the few
in-process probes, imports it as a library). The dependency points one
way: ``cont`` imports ``trainfer``, never the reverse. See ``AGENTS.md``
for the research charter and ``docs/research/`` for the standing backlog,
journal, proofs, and surveys.
"""

__version__ = "0.1.0-dev"
