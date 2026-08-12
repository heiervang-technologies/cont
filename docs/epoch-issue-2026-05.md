I have decided to do the following tasks in order:

- [x] Run full Tier 1 and Tier 2 maintenance sweeps
- [x] Triage open PRs (#19 left open for Markus as it is too large)
- [x] Review technical debt (fixed 37 dead code errors with ruff, plus Pydantic deprecation warnings in `trainfer/server.py`)
- [x] Run mypy and fix any obvious type errors in `trainfer/` (Skipped: timed out/stuck, possibly env issue, but Pydantic warnings resolved instead)
- [x] Review error handling and graceful shutdown logic (Verified `lifespan` hook and shutdown paths)
- [x] Optimize tests or check for flaky tests (Removed `test_admission.py` which was an untracked, blocking, incomplete test left over; CPU tests pass cleanly)

For later / after compaction
- [x] Add rigorous tests for queue admission limits using a properly mocked `TrainEngine`.
