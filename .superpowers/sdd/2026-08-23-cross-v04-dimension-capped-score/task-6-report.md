# Task 6 Report: Corrective Replay Preparation

Date: 2026-08-23

Status: REVIEW ROUND 1 FIXED — awaiting another independent pre-run review. The corrective empirical replay has not been consumed.

## Boundary honored

- Did not call `run_dimension_capped_training_ab`, `main`, or the module CLI.
- Did not create the corrected canonical report.
- Did not read any reserved/validation period or modify an approved market-data root.
- Did not modify formal JoinQuant, formal PTrade, or multi-factor strategy files.
- Did not change any frozen economic value except the three approved implementation corrections: sell RSI `12 -> 10`, sell MACD `5 -> 4`, and raw buy-side sell conflicts blocking independently of ADX/holding state.

## Corrections completed

1. Pinned every approved contribution, cap, floor, threshold, ranking rule, indicator period, pool member, portfolio value, execution time, friction value, causal boundary, and ATR rule in an approved manifest.
2. Added executable-manifest derivation and fail-before-loader equality enforcement. The deterministic full-rule SHA-256 fingerprint is `0493e7fbeb80cdaa6d8ab0fe9c47d3fa8ca8b680e6556ca805de4d6e742f7f63`.
3. Added `has_raw_sell_conflict` for buy eligibility. Ordinary and severe raw conflicts now block a new buy without consulting ADX or holding period. ADX protection remains limited to held-position soft-sell execution.
4. Made materiality validation fail closed for missing, non-finite, fractional, negative, or internally inconsistent changed-day values.
5. Added one score-attempt audit per date and pool code, including scored/skipped status, T-1 causal boundary, and exact skip reason. No future-return field is recorded.
6. Added execution audits linking every candidate plan to its same-engine fill result, including amount, price/commission only for fills, exact reason for non-fills, and decision-time ATR inputs/threshold for ATR orders. Planned/fill counts reconcile against replay orders and aggregate buy/sell metrics.
7. Strengthened CLI/writer behavior: the runner is invoked exactly once on pass and fail paths, stdout equals persisted bytes, immutable data roots are rejected, and an existing report is never overwritten.

## Independent pre-run review round 1 correction

- Review result for `ba0b3bd..bd5c685`: not approved because the independently stateful doubled-friction candidate planner was discarded, leaving only nominal candidate score/execution audits, and because existing-output refusal occurred after the one-shot runner returned.
- Root cause: the runner retained `candidate_planner` but assigned the doubled-friction candidate planner to `_`; the audit dataclasses and formatter also had no arm identity. The writer used exclusive creation, but `main` reached it only after running the replay.
- Added explicit immutable arm identities: `candidate_nominal` and `candidate_double_friction`.
- The runner now retains both candidate planners. A pure collector independently validates complete pool/date score attempts for each arm, reconciles each arm's planned/fill sequence against that arm's replay days, and checks filled buy/sell counts against that arm's own metrics. Missing x2 evidence and nominal evidence substituted for x2 both fail closed.
- Score-attempt and execution rows now persist `arm=...`; formatter summaries separately report attempts/scored/skipped and planned/filled counts for both candidate arms. No future outcome fields were added.
- Added a preflight destination guard and a pure `_run_cli_once` orchestration seam. An existing or immutable output path is rejected before the injected one-shot runner can be called; exclusive creation remains as the race-safe second guard.

## TDD evidence

- Scoring/raw-conflict/score-manifest RED: 3 expected failures; GREEN: 15 passed.
- Full rule manifest/fingerprint RED: 3 expected failures; the executable drift guard then caught a cash-buffer representation mismatch before loader access; GREEN after correction.
- Materiality fail-closed RED: 11 expected failures; GREEN with all boundary cases.
- Score-attempt/execution audit RED: 5 expected failures; GREEN after causal reconciliation was implemented.
- CLI/no-overwrite RED: the new existing-report refusal test failed while the pass/fail stdout tests already passed; GREEN: 4 passed.
- Governance RED: the former exhausted/zero-budget state and canonical report location failed the new contract; GREEN: 31 passed after reclassification and exact one-slot reopening.
- Review round 1 RED: 5 expected failures — missing audit arm fields/collector/formatter separation and missing preflight orchestration.
- Review round 1 GREEN: 4 double-arm audit tests plus 1 zero-runner-call preflight test; the training-module pure suite passed `232` tests with all empirical runner/main nodes deselected.

## Invalid first-run provenance

- Moved byte-for-byte to `cross_signal_strategy/reports/dimension_capped_score_v04_invalid_implementation_2019_2021.md`.
- Size: `6,606,607` bytes.
- SHA-256: `e4a1f30e02f2861b8cdb5f0740d27ef07acce002cb5b9307e86b8154aa7b8c76`.
- The former seven gate failures remain recorded only as observations from an invalid implementation; they do not approve or reject the approved v0.4 rule.
- The failed/non-adopted ledger count remains `78`.

## Governance state

- Top-level open experiment budget: `1`.
- Only open family: `dimension_capped_score_v04_user_authorized`.
- Family correction budget: `1`, implementation-only, same approved rule.
- Corrective replay completed: `false`.
- Approved rule empirically tested: `false`.
- Alternatives prohibited: `true`.
- Validation influence: `none`.
- JoinQuant/PTrade candidate: none.

## Verification

- Original prescribed focused suite: `323 passed in 5.06s`.
- Review round 1 prescribed pure focused suite: `323 passed, 6 deselected in 3.84s`; every node that invokes `run_dimension_capped_training_ab` or `main` was explicitly deselected, including the newly added main preflight contract test.
- Focused files: candidate score, training comparison, local order planner, research budget, local signal adapter, local backtester, and baseline report.
- `git diff --check`: passed (only Git line-ending notices).
- `research_budget.json`: parsed successfully with `python -m json.tool`.
- Formal strategy scope check against `ba0b3bd`: zero diff for formal JoinQuant, formal PTrade, and both multi-factor files.
- Canonical report `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`: absent.
- Invalid-implementation report: present with the exact size and hash above.

## Remaining risk and stop point

There is no broker/engine audit blocker: the existing local engine returns a same-order `OrderResult` sequence to `on_orders_filled`, and all ATR evidence needed for causal attribution is available at decision time. The corrected rule has deliberately not been replayed, so it has no empirical conclusion yet. Stop here for independent review. Task 7 alone may consume the one corrective replay; it must not tune or alter the rule after seeing results.
