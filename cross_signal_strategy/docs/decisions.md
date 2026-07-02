# Strategy Decisions

Use this file to record design decisions before and after each experiment.

## Decision Template

```text
Date:
Decision:
Reason:
Evidence:
Affected files:
Allowed validation influence: training only / multiple validation periods / none
Status: proposed / adopted / rejected / reverted
```

## Initial Decisions

### Isolate The New Strategy

Date: 2026-07-02
Decision: Build the cross-signal strategy in `cross_signal_strategy/` instead of modifying the current production multi-factor strategy.
Reason: The new strategy has a different trading philosophy. Mixing it into the production strategy would make results hard to interpret.
Evidence: User specifically requested a new file and independent folder.
Affected files: `cross_signal_strategy/*`
Allowed validation influence: none
Status: adopted

### Use 2019-2021 As Development Window

Date: 2026-07-02
Decision: Use 2019-01-01 to 2021-12-31 as the development/training period.
Reason: This period includes rising, crash/rebound, and structural-divergence behavior, but keeps other periods reserved for out-of-sample validation.
Evidence: Research protocol agreed in conversation.
Affected files: `cross_signal_strategy/README.md`, `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Freeze v0.1 As Cross-Signal Timing, Not Momentum Rotation

Date: 2026-07-02
Decision: First implementation will use RSI/MACD/KDJ cross resonance as the primary buy/sell driver, with MA/BOLL location, light trend context, volume confirmation, and ATR risk control.
Reason: The experiment is meant to test earlier swing turning points, not to retune the existing momentum-led multi-factor strategy.
Evidence: User described repeated live concern that momentum confirmation may enter after a short swing has already risen for several days.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Disable Profit Floor In v0.1

Date: 2026-07-02
Decision: v0.1 will disable profit-floor protection while keeping ATR stop logic.
Reason: The first test should isolate whether cross-signal sell timing has value before adding the current production strategy's profit-protection layer.
Evidence: Profit floor is already validated in production strategy; adding it immediately would obscure whether new sell signals work.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted

### Use Equal-Weight Sizing In v0.1

Date: 2026-07-02
Decision: v0.1 will use equal-weight sizing across selected holdings rather than volatility-inverse sizing.
Reason: The first experiment should evaluate signal quality before adding sizing complexity.
Evidence: User's recent concern includes whether volatility-based sizing can misallocate capital; v0.1 should not inherit that complexity before signal validation.
Affected files: `cross_signal_strategy/docs/strategy_spec.md`
Allowed validation influence: none
Status: adopted
