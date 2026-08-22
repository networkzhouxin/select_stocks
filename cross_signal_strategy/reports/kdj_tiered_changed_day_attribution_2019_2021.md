# KDJ moderate-points changed-day attribution: 2019-2021

## Scope and causality

This is an observation-only attribution of the already rejected candidate:

- Buy: K<=20 +20; 20<K<=30 +10.
- Sell: K>=80 +10; 70<=K<80 +5.
- Current causal T-1 state only; no retention.
- Official price confirmation, ADX protection, and five-session minimum hold
  remain active.

The baseline and candidate were replayed once under normal friction using the
same cached official T-1 signals. Only the approved 2018 warm-up and 2019-2021
training roots were read. No validation or recent-market data was read.

Forward 1/3/5/10/20-session returns below are ex-post diagnostic labels. They
were never supplied to signal calculation, ranking, order planning, execution,
or parameter selection.

## Root cause

The 15 changed filled-order days contained 16 changed order events:

- 4 direct buy-bonus events.
- 0 direct sell-bonus events.
- 12 subsequent portfolio-chain events.

All four direct events came from the strong buy tier K<=20. In every case the
official buy score was 41 and the +20 bonus raised it to 61, just above the
formal 60-point threshold. The near buy tier +10 did not directly create a
filled buy. The sell bonuses did not directly create any filled sell after the
official price-confirmation and ADX rules were applied.

Therefore the observed degradation is attributable to the strong oversold buy
bonus, followed by position/cash/sizing chain effects. It is not attributable
to an early KDJ-overbought sell in this candidate.

## Four direct added buys

| Buy date | ETF | K | Score | Exit date/reason | P&L | Return |
|---|---:|---:|---:|---|---:|---:|
| 2019-05-16 | 513100 | 16.32 | 41 -> 61 | 2019-06-03 ATR stop | -416.00 | -5.41% |
| 2019-09-30 | 512100 | 17.04 | 41 -> 61 | 2019-10-21 signal sell | -130.00 | -1.49% |
| 2020-04-09 | 159985 | 18.92 | 41 -> 61 | 2020-04-29 signal sell | -312.60 | -3.46% |
| 2021-03-04 | 513880 | 18.52 | 41 -> 61 | 2021-03-23 signal sell | -71.80 | -0.48% |

All four closed at a loss. Their realized P&L totals -930.40 yuan. This amount
is comparable to 84.3% of the final portfolio shortfall, but it is not a fully
additive causal allocation because the added positions also changed later
position sizes, commissions, replacement orders, and mark-to-market exposure.

## Ex-post return labels from the added-buy time

Returns use the decision-day 09:35 market price as the anchor and later daily
closes as labels.

| Buy date / ETF | +1 | +3 | +5 | +10 | +20 sessions |
|---|---:|---:|---:|---:|---:|
| 2019-05-16 / 513100 | +1.12% | -0.34% | -1.49% | -3.10% | 0.00% |
| 2019-09-30 / 512100 | -1.04% | +1.34% | +2.98% | -0.89% | +1.04% |
| 2020-04-09 / 159985 | +0.20% | -0.51% | -1.94% | -1.83% | -3.16% |
| 2021-03-04 / 513880 | -2.07% | -1.44% | -0.80% | +2.15% | -0.64% |
| Mean | -0.45% | -0.24% | -0.31% | -0.92% | -0.69% |

Only one of four observations was positive at each of the +3, +5, +10, and
+20 horizons. The sample is small, but it is directionally consistent with
the closed-trade losses: K<=20 plus an otherwise insufficient 41-point setup
selected falling or weakly stabilizing prices rather than proven reversals.

## Portfolio-chain order events

These are consequences of the four direct buys, not additional direct KDJ
triggers:

| Date | Path | Side | ETF | Reason |
|---|---|---|---:|---|
| 2019-06-03 | candidate only | sell | 513100 | ATR stop |
| 2019-10-15 | baseline only | buy | 512100 | buy signal |
| 2019-10-21 | candidate only | sell | 512100 | signal sell |
| 2019-10-28 | candidate only | buy | 518880 | buy signal |
| 2019-11-01 | baseline only | sell | 512100 | signal sell |
| 2019-11-05 | baseline only | buy | 159915 | buy signal |
| 2019-11-07 | candidate only | sell | 518880 | signal sell |
| 2019-11-12 | baseline only | sell | 159915 | signal sell |
| 2020-04-17 | baseline only | buy | 513050 | buy signal |
| 2020-04-29 | candidate only | sell | 159985 | signal sell |
| 2020-04-29 | candidate only | buy | 513050 | buy signal |
| 2021-03-23 | candidate only | sell | 513880 | signal sell |

The 2020 path shows why a later order must not be misclassified as a direct
KDJ effect: the baseline bought 513050 on 2020-04-17, while the candidate was
still holding the added losing 159985 position and bought 513050 only after
selling it on 2020-04-29.

## Portfolio outcome and decision

| Path | Terminal value | Total return |
|---|---:|---:|
| Baseline | 45,000.50 | 125.00% |
| Candidate | 43,896.80 | 119.48% |
| Difference | -1,103.70 | -5.52pp |

Decision: close the KDJ current-state bonus branch.

- Do not add the K<=20 or 20<K<=30 buy-state points.
- Do not add sell-state points merely for symmetry; they created no direct
  benefit in this screen and would add inactive complexity.
- Keep the official KDJ cross contributions. A cross describes a state change;
  an extreme K level alone does not prove reversal.
- Do not run additional nearby point searches from this result.

No formal JoinQuant or PTrade file was modified.
