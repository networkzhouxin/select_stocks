# Task 7 Report: Sole Corrective Replay and Governance Closure

Date: 2026-08-23

Status: COMPLETE — corrected local gate failed; terminal action `STOP`; v0.4 family exhausted.

## Scope and baseline

- Worktree: `G:\financial\select_stocks\.worktrees\cross-v04-dimension-capped`
- Starting HEAD: `c0d22655d25cb1a37a5a376a24c93823a368405f`
- Result milestone commit: `5c65647` (`research(cross-signal): reject corrected v0.4 capped score`)
- No economic rule, manifest value, threshold, gate, pool member, year, friction setting, ranking rule, hold rule, ATR rule, ADX rule, or protection rule changed after the result.
- No reserved/validation data was read. No JoinQuant or PTrade candidate was created.

## Pre-run evidence

- Independent Task 6 re-review: approved with no Critical, Important, or Minor findings.
- Corrected canonical report path was absent.
- Invalid report was present at 6,606,607 bytes with SHA-256 `e4a1f30e02f2861b8cdb5f0740d27ef07acce002cb5b9307e86b8154aa7b8c76`.
- Executable manifest matched the approved manifest; rule fingerprint was `0493e7fbeb80cdaa6d8ab0fe9c47d3fa8ca8b680e6556ca805de4d6e742f7f63`.
- Governance had exactly one open implementation-only correction slot, top-level budget 1, failed/non-adopted count 78, `corrective_replay_completed=false`, and no platform candidate.
- Formal JoinQuant and PTrade files had zero diff from `c0d2265`.
- Pure focused pre-run suite: `319 passed, 10 deselected in 3.66s`. The deselected nodes were all runner/main/CLI/report-writer nodes; no empirical entrypoint was invoked by the suite.
- The first sandboxed pure-test attempt reached the end of the test body but pytest cleanup hit `WinError 5` on its basetemp. The same pure selection was rerun outside the sandbox with a distinct basetemp and passed as above. This occurred before the empirical replay and did not call the runner.

## Sole empirical execution

- Exact empirical command: `python -m cross_signal_strategy.research.dimension_capped_training_ab`
- Invocation count: exactly 1.
- The process ran in one continuous execution session (`96853`) until natural completion; it was not restarted or rerun.
- CLI status: exit `1`, derived from the frozen executable contract `return 0 if report.gate.passed else 1` together with the emitted `gate_passed=false`. This is the designed `STOP` exit, not a runtime exception. The final session poll completed without preserving its `exit_code` field because the polling wrapper printed only output/session fields; the command was not rerun to repair that evidence gap.
- Complete stdout capture: `.superpowers/sdd/2026-08-23-cross-v04-dimension-capped-score/task-7-corrective-replay-stdout.txt`, 9,759,878 bytes, SHA-256 `bd2be52bafb77bbe659dfb1484392872203fdd2e1dc2333e2ec84221578ffc17`.
- Canonical report: `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`, 9,740,215 bytes, SHA-256 `14395dbf09f506c914bd5da241454b3af4291cf2e701e7653738f50f4840ccd6`.
- The stdout capture uses PowerShell CRLF redirection while the canonical writer uses LF. Replacing stdout CRLF with LF yields exact full-text equality with the canonical report.
- The invalid and corrected reports coexist; the invalid report hash remained unchanged after the corrected replay.

## Corrected result

Nominal baseline/candidate:

- Total return: +125.00% / +78.13%
- Annualized return: 31.13% / 21.29%
- Maximum drawdown: 6.03% / 6.37%
- Sharpe: 2.262 / 1.672
- Sortino: 3.581 / 2.533
- Win rate: 56.18% / 51.76%
- Profit/loss ratio: 4.878 / 2.831
- Buy/sell/closed trade count: 92/89/89 versus 88/85/85
- Annual returns: baseline +35.84%/+52.68%/+8.49%; candidate +21.45%/+43.89%/+1.93%

Doubled-friction baseline/candidate:

- Total return: +108.15% / +63.32%
- Annualized return: 27.77% / 17.82%
- Maximum drawdown: 6.39% / 6.93%
- Sharpe: 2.039 / 1.422
- Sortino: 3.186 / 2.125
- Win rate: 51.69% / 45.88%
- Profit/loss ratio: 3.966 / 2.347
- Candidate annual returns: +18.61%/+39.44%/-1.26%

Materiality and gate:

- Changed filled-order days: 196, split 62/64/70 across 2019/2020/2021.
- Closed-trade retention: 85/89 = 95.51%.
- Failed gates: candidate win rate did not strictly improve; nominal return, Sharpe, Sortino, and profit/loss ratio each retained less than 95% of baseline; doubled-friction return retained less than 95%; doubled-friction win rate was below baseline.
- Terminal action: `STOP`.

## Audit reconciliation

Each candidate arm reconciled independently:

- `candidate_nominal`: 6,570 score attempts = 6,111 scored + 459 skipped; 730 decision dates with exactly 9 ETF rows per date; 186 planned orders; 173 fills = 88 buys + 85 sells.
- `candidate_double_friction`: 6,570 score attempts = 6,111 scored + 459 skipped; 730 decision dates with exactly 9 ETF rows per date; 186 planned orders; 173 fills = 88 buys + 85 sells.
- Every skipped score row had an exact reason. Every unfilled execution row had an exact broker reason.
- The runner's fail-closed collector reconciled each arm's execution rows against that arm's replay order sequence and performance buy/sell counts before the canonical report could be written.

## TDD governance closure

- RED: the new result-dependent closure test failed as intended because `max_total_open_experiments` was still 1.
- GREEN: after the minimal governance update, the same node passed (`1 passed`).
- The corrected failure was appended under a unique experiment identity; the invalid-implementation record remains unchanged.
- Failed/non-adopted count increased exactly once from 78 to 79.
- Top-level open budget is 0; the v0.4 family is `exhausted`, has budget 0, and has no `planned_experiment`.
- `approved_rule_empirically_tested=true`, `corrective_replay_completed=true`, `candidate_gate_passed=false`, `validation_influence=none`, `prohibit_alternatives=true`, and both platform-candidate flags remain false.

## Verification

- Governance file: `32 passed in 0.77s`.
- Complete focused candidate/replay/planner/governance/adapter/engine/report suite: `330 passed in 3.90s`.
- Repository suite to first failure: the known stale age2 budget assertion remained the first failure after 22 passes; it expects 72 while current governance is 79. The out-of-scope stale test was not edited.
- Repository suite with only that known node deselected: progressed through 20% without a new failure, then reproduced the historical long silent integration phase. It was stopped after about three minutes without output. The unexecuted remainder is not claimed as passing.
- `research_budget.json` parsed and the ledger audit reconciled 79 unique entries with no errors.
- Both candidate audit arms reconciled exactly as listed above.
- Invalid report hash remained `e4a1f30e...8c76`; corrected report fingerprint remained `0493e7...7f63`.
- `git diff --check` passed, apart from Git's informational LF/CRLF notices.
- Formal JoinQuant, formal PTrade, and multi-factor files remained zero diff from the starting HEAD.

## Final decision and remaining risk

The approved corrected v0.4 rule is rejected at the local accuracy-first gate. Do not create a JoinQuant/PTrade candidate, inspect validation periods, or search a neighboring rule from this result. The formal `cross-v0.3.3` strategies remain unchanged.

The repository-wide suite is not fully green because of the known stale age2 assertion and the subsequent long-running integration segment. This limitation is recorded separately from the fully passing 330-test Task 7 focused scope.
