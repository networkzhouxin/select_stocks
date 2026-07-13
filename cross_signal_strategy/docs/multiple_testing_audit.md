# Cross-Signal Multiple-Testing Risk Audit

Date: 2026-07-13

## Scope

This audit evaluates whether the frozen `cross-v0.3.2` training result remains
statistically distinguishable from zero after a conservative correction for the
strategy variants that can be proved from the retained research ledger. It uses
only the approved 2018 warm-up buffer and the 2019-2021 training period. It is
not out-of-sample validation, and it does not authorize any strategy change.

## Evidence Boundary

- `docs/failed_experiments.md` retains 47 failed or non-adopted experiments.
- The selected frozen mainline contributes one additional selection.
- minimum trial count: 48
- The count is a lower bound. Early or adopted variants may not all have a
  retained candidate return series or a failed-experiment record.
- Validation-period data was not read, inspected, or used.

## Methods

The audit reports two complementary training-only approximations:

1. A probabilistic Sharpe ratio (PSR) against a zero Sharpe benchmark, corrected
   with a Bonferroni bound using the minimum retained trial count.
2. A one-sided Newey-West/HAC test of positive mean daily return, using the
   standard automatic lag rule and the same Bonferroni trial correction.

Both corrections are deliberately simple and reproducible. They do not replace
reserved-window validation and should not be interpreted as a probability that
the strategy will remain profitable.

## Results

- Training total return: 120.61%
- Training annualized return: 30.27%
- Training annualized Sharpe: 2.172
- Annual returns: 2019 35.84%, 2020 49.74%, 2021 8.46%
- Single-trial PSR: 0.999876
- Single-trial PSR p-value: 0.000123988
- Minimum-48 Bonferroni PSR p-value: 0.00595144
- maximum trials passing the 5% PSR/Bonferroni approximation: 403
- Newey-West/HAC automatic lag: 6
- Newey-West/HAC t-statistic: 3.837
- Single-trial HAC p-value: 0.0000622008
- Minimum-48 Bonferroni HAC p-value: 0.00298564

At the provable lower bound of 48 trials, both approximations remain below the
5% family-wise threshold. This is evidence that the frozen training result is
not merely a marginal winner among the retained experiments. Because the true
trial count may be higher, the corrected confidence values are optimistic upper
bounds. Under the PSR/Bonferroni approximation, the result ceases to pass at 404
trials.

## Unavailable Canonical Measures

- Canonical DSR: unavailable. The complete cross-trial Sharpe distribution for
  every tried variant was not retained, so the expected maximum Sharpe cannot
  be estimated in the canonical Deflated Sharpe Ratio formulation.
- PBO: unavailable. Probability of Backtest Overfitting requires aligned daily
  candidate return curves across all variants; that matrix was not retained.

Reconstructing either measure from only the winning curve and failed-experiment
prose would create false precision. Future experiments should retain aligned
daily returns and a candidate manifest so these measures can be computed without
retroactive inference.

## Decision

Keep official `cross-v0.3.2` frozen. Record the audit as research-governance
infrastructure, not as a new strategy experiment and not as permission to reopen
an exhausted indicator or parameter family. The remaining uncertainty must be
resolved by genuinely reserved validation or future prospective performance,
not by further mining of 2019-2021.
