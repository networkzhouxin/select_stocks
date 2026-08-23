# Cross-v0.4 Dimension-Capped Score Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate the single pre-registered `cross-v0.4.0-dimension-capped-candidate` on the approved 2019-2021 local training replay without changing either formal platform strategy.

**Architecture:** Decorate the official causal T-1 score snapshot with three capped buy dimensions and two capped sell dimensions, then execute it through an isolated research-only planner that preserves the existing pool, ATR state, five-day signal hold, portfolio stress scaling, costs, and 09:35 execution. Compare the unchanged v0.3.3 path and the single v0.4 candidate under nominal and doubled friction, apply the frozen accuracy-first gate, and stop before creating a JoinQuant or PTrade file.

**Tech Stack:** Python 3, pandas, dataclasses, pytest, the existing local loader/backtester/broker, and the frozen JoinQuant v0.3.3 snapshot/ATR helpers.

**Spec:** `cross_signal_strategy/docs/2026-08-23-v04-dimension-capped-score-design.md`

## Global Constraints

- Scope is `cross_signal_strategy`, its tests, reports, and research governance only; no multi-factor file may change.
- Formal `cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py` and `cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py` remain byte-for-byte unchanged in this plan.
- Candidate identity is exactly `cross-v0.4.0-dimension-capped-candidate`; only one structure and one parameter set may run.
- Training performance dates are exactly `2019-01-01` through `2021-12-31`; 2018 data is indicator warm-up only.
- Read only `G:\financial\history_data\cross_signal_train_2019_2021` and `G:\financial\history_data\cross_signal_warmup_2018`; never write, delete, clean, or cache under either root.
- Every 09:35 signal uses T-1 or earlier completed daily bars. T-day prices are execution/mark inputs only.
- Freeze RSI(6,12,24), MACD(12,26,9), KDJ(9,3,3), BOLL(20,2), ATR14, ADX14, MA(5,10,20,60), and `cross_window=3`.
- Freeze the nine-ETF pool, `max_hold=3`, `base_ratio=0.95`, five-day signal hold, 5% cash buffer, ATR stop construction, highest-close anchor, same-day ATR re-buy guard, and portfolio ATR-stress scale.
- Volume can affect only the final tie-break for A-share ETFs. Candidate target value is equal weight times the existing portfolio ATR-stress scale; it must not call the v0.3.3 zero-volume position-size reducer.
- Do not add 14:45, historical IOPV, profit taking, replacement rotation, ETF/year exceptions, nearby scores, nearby thresholds, indicator deletion, or a second ranking order.
- Every correction starts with one focused failing test, observes the red result, adds the minimum implementation, and observes the green result.
- A failed empirical gate is a valid result. Do not alter rules or rerun a nearby variant after seeing it.
- JoinQuant authority work is a separate conditional plan only if the local gate passes. PTrade mapping, the existing live 8% IOPV override, reserved validation, and formal promotion are outside this plan.

## File Map

- Create `cross_signal_strategy/research/dimension_capped_score_candidate.py`: pure direction resolution, capped dimension scoring, buy eligibility, deterministic ranking, and soft/severe signal-sell decisions.
- Create `cross_signal_strategy/research/dimension_capped_training_ab.py`: research-only planner, fixed replay configuration, nominal/doubled-friction A/B, frozen gate, report formatter, and CLI.
- Create `tests/test_cross_signal_dimension_capped_score_candidate.py`: unit tests for every score cap, conflict, boundary, filter, ordering, and ADX rule.
- Create `tests/test_cross_signal_dimension_capped_training_ab.py`: planner, T-1 metadata, ATR, five-day hold, replay configuration, gate, and report tests.
- Modify `tests/test_cross_signal_research_budget.py`: prove exactly one family opens before implementation and closes after the one run.
- Modify `cross_signal_strategy/docs/research_budget.json`: pre-register and then consume exactly one structural candidate.
- Modify `cross_signal_strategy/docs/research_budget.md`: mirror the machine-readable research state and final outcome.
- Modify `cross_signal_strategy/docs/backtest_notes.md`: append the frozen hypothesis, exact metrics, gate reasons, and next permitted action.
- Modify `cross_signal_strategy/docs/failed_experiments.md` only if the local gate fails; otherwise modify `cross_signal_strategy/docs/decisions.md` with the local qualification and explicit JoinQuant dependency.
- Create `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`: deterministic output from the one empirical run.

---

### Task 1: Pre-register exactly one v0.4 research family

**Files:**
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `cross_signal_strategy/docs/research_budget.json:8-9`
- Modify: `cross_signal_strategy/docs/research_budget.md`

**Interfaces:**
- Consumes: `load_research_budget(path)` and `evaluate_experiment_request(budget, family_key, planned_variants)`.
- Produces: open family key `dimension_capped_score_v04_user_authorized` with one permitted variant.

- [ ] **Step 1: Add the failing governance test**

Append this test and reuse the file's existing `json`, `BUDGET`, `load_research_budget`, and `evaluate_experiment_request` imports:

```python
def test_dimension_capped_v04_is_the_only_open_research_family():
    budget = load_research_budget(BUDGET)
    families = {item.key: item for item in budget.families}
    family = families["dimension_capped_score_v04_user_authorized"]

    assert budget.max_total_open_experiments == 1
    assert family.status == "open"
    assert family.max_new_experiments == 1
    assert family.planned_experiment == (
        "one fixed v0.4 dimension-capped buy/sell score structure with "
        "40-point buy and 24-point ordinary sell gates"
    )
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=1
    ).allowed is True
    assert evaluate_experiment_request(
        budget, family.key, planned_variants=2
    ).allowed is False
    assert [item.key for item in budget.families if item.status == "open"] == [
        "dimension_capped_score_v04_user_authorized"
    ]

    payload = json.loads(BUDGET.read_text(encoding="utf-8"))
    raw = next(item for item in payload["families"] if item["key"] == family.key)
    assert raw["candidate_name"] == "cross-v0.4.0-dimension-capped-candidate"
    assert raw["candidate_variants"] == 1
    assert raw["buy_threshold"] == 40
    assert raw["ordinary_sell_threshold"] == 24
    assert raw["severe_damage_threshold"] == 18
    assert raw["validation_influence"] == "none"
    assert raw["data_scope"] == "2018_warmup_plus_2019_2021_training_only"
    assert raw["prohibit_alternatives"] is True
```

- [ ] **Step 2: Run the test and observe the missing-family failure**

Run:

```powershell
pytest -q tests/test_cross_signal_research_budget.py::test_dimension_capped_v04_is_the_only_open_research_family
```

Expected: FAIL because `dimension_capped_score_v04_user_authorized` is absent and the top-level open budget is still zero.

- [ ] **Step 3: Add the exact open-family record**

Set top-level `max_total_open_experiments` to `1` and append this family object:

```json
{
  "key": "dimension_capped_score_v04_user_authorized",
  "label": "User-authorized v0.4 dimension-capped score structure",
  "status": "open",
  "max_new_experiments": 1,
  "rationale": "The user approved one independent score architecture that caps correlated indicator dimensions and separates sell weakness from price damage. This does not reopen exhausted indicator, threshold, extreme-zone, ADX, or ranking searches.",
  "planned_experiment": "one fixed v0.4 dimension-capped buy/sell score structure with 40-point buy and 24-point ordinary sell gates",
  "candidate_name": "cross-v0.4.0-dimension-capped-candidate",
  "candidate_variants": 1,
  "buy_threshold": 40,
  "ordinary_sell_threshold": 24,
  "severe_damage_threshold": 18,
  "validation_influence": "none",
  "data_scope": "2018_warmup_plus_2019_2021_training_only",
  "prohibit_alternatives": true
}
```

Update the readable map to identify this as the only open family and explicitly state that KDJ tier variants, direct extreme exits, MACD changes, score rebalance, indicator deletion, and nearby thresholds remain exhausted.

- [ ] **Step 4: Run the complete governance test file**

```powershell
pytest -q tests/test_cross_signal_research_budget.py
```

Expected: PASS with `expected_failed_experiment_count` still exactly `77` because opening a candidate is not a failure record.

- [ ] **Step 5: Commit the research authorization milestone**

```powershell
git add -- tests/test_cross_signal_research_budget.py cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md
git commit -m "docs(cross-signal): preregister v0.4 capped score"
```

---

### Task 2: Implement pure capped-dimension scoring and decisions

**Files:**
- Create: `tests/test_cross_signal_dimension_capped_score_candidate.py`
- Create: `cross_signal_strategy/research/dimension_capped_score_candidate.py`

**Interfaces:**
- Consumes: one official T-1 score snapshot dictionary produced by `LocalSignalAdapter`.
- Produces: `DimensionCappedScoreAdapter.score`, `is_dimension_capped_buy_candidate`, `sort_dimension_capped_candidates`, and `should_dimension_capped_signal_sell`.

Start the test file with these complete helpers; later tests in this task use only these names:

```python
from copy import deepcopy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticAdapter:
    def __init__(self, score):
        self.score_value = score

    def score(self, code, current_date, return_reason=False):
        value = deepcopy(self.score_value) if self.score_value is not None else None
        reason = None if value is not None else "no_data"
        return (value, reason) if return_reason else value


def _candidate_module():
    from cross_signal_strategy.research import dimension_capped_score_candidate
    return dimension_capped_score_candidate


def _snapshot(**overrides):
    values = {
        "code": "513100",
        "current_date": "2019-01-08",
        "signal_date": "2019-01-07",
        "max_data_date": "2019-01-07",
        "rsi6_cross_rsi12_up": False,
        "rsi6_cross_rsi24_up": False,
        "rsi6_cross_rsi12_down": False,
        "rsi6_cross_rsi24_down": False,
        "macd_cross_up": False,
        "macd_cross_down": False,
        "kdj_k_cross_up": False,
        "kdj_j_cross_up": False,
        "kdj_k_cross_down": False,
        "kdj_j_cross_down": False,
        "k": 50.0,
        "close_between_boll_lower_mid": False,
        "close_cross_boll_mid_up": False,
        "close_near_ma20": False,
        "close_far_above_ma20": False,
        "ma5_gt_ma10": False,
        "ma10_gt_ma20": False,
        "ma20_slope_non_negative": False,
        "close_gt_ma60": False,
        "downside_continuation": False,
        "close_below_falling_ma10": False,
        "close_below_ma20": False,
        "close_below_boll_mid": False,
        "fell_back_inside_boll": False,
        "buy_allowed": True,
        "volume_score": 0.0,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
        "atr": 0.1,
        "buy_score": 0.0,
        "sell_score": 0.0,
        "reversal_score": 0.0,
        "location_score": 0.0,
        "trend_score": 0.0,
        "sell_reversal_score": 0.0,
        "sell_risk_score": 0.0,
    }
    values.update(overrides)
    return values


def _eligible_snapshot(**overrides):
    values = _snapshot(
        rsi6_cross_rsi12_up=True,
        kdj_k_cross_up=True,
        close_between_boll_lower_mid=True,
        ma5_gt_ma10=True,
        ma10_gt_ma20=True,
    )
    values.update(overrides)
    return values
```

- [ ] **Step 1: Create official-like snapshot fixtures and failing direction tests**

Create a static adapter that returns defensive copies and a `_snapshot(**overrides)` fixture containing all buy/sell flags, `code`, `signal_date`, `max_data_date`, `k`, DMI/ADX values, and official score fields. Add these tests:

```python
def test_rsi_and_kdj_groups_neutralize_mixed_directions():
    module = _candidate_module()
    mixed = _snapshot(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi24_down=True,
        kdj_k_cross_up=True,
        kdj_j_cross_down=True,
    )
    score = module.DimensionCappedScoreAdapter(StaticAdapter(mixed)).score(
        "513100", "2019-01-08"
    )
    assert score["rsi_direction"] is None
    assert score["kdj_direction"] is None
    assert score["buy_rsi_group_score"] == 0
    assert score["sell_rsi_group_score"] == 0
    assert score["buy_kdj_group_score"] == 0
    assert score["sell_kdj_group_score"] == 0


def test_same_direction_multiple_crosses_count_once():
    module = _candidate_module()
    bullish = _snapshot(
        rsi6_cross_rsi12_up=True,
        rsi6_cross_rsi24_up=True,
        kdj_k_cross_up=True,
        kdj_j_cross_up=True,
        k=19.0,
        macd_cross_up=True,
    )
    score = module.DimensionCappedScoreAdapter(StaticAdapter(bullish)).score(
        "513100", "2019-01-08"
    )
    assert score["buy_rsi_group_score"] == 12
    assert score["buy_kdj_group_score"] == 6
    assert score["buy_kdj_state_score"] == 10
    assert score["buy_macd_confirmation_score"] == 5
    assert score["reversal_score"] == 25
```

- [ ] **Step 2: Run the new test file and observe the import failure**

```powershell
pytest -q tests/test_cross_signal_dimension_capped_score_candidate.py
```

Expected: FAIL with `ModuleNotFoundError` for `dimension_capped_score_candidate`.

- [ ] **Step 3: Implement direction resolution and reversal/weakness caps**

Create these constants and public adapter skeleton:

```python
CANDIDATE_NAME = "cross-v0.4.0-dimension-capped-candidate"
BUY_THRESHOLD = 40.0
BUY_REVERSAL_MIN = 12.0
BUY_LOCATION_MIN = 7.0
BUY_TREND_MIN = 6.0
ORDINARY_SELL_THRESHOLD = 24.0
SELL_WEAKNESS_MIN = 10.0
SELL_DAMAGE_MIN = 8.0
SEVERE_DAMAGE_MIN = 18.0
SEVERE_WEAKNESS_MIN = 6.0


@dataclass(frozen=True)
class DimensionCappedScoreAdapter:
    source: object

    def score(self, code: str, current_date: str, return_reason: bool = False):
        base = self.source.score(code, current_date, return_reason=return_reason)
        if return_reason:
            snapshot, reason = base
            return (None, reason) if snapshot is None else (score_snapshot(snapshot), reason)
        return None if base is None else score_snapshot(base)
```

Implement `resolve_rsi_direction` and `resolve_kdj_direction` with exclusive-direction semantics. Implement current-T-1 K tiers exactly as `K<=20: buy 10`, `20<K<=30: buy 5`, `70<=K<80: sell 4`, `K>=80: sell 8`, otherwise zero. Missing, NaN, or infinite K contributes zero. Cap buy reversal at 25 and sell weakness at 20. Preserve the original snapshot under a deep copy and store every raw contribution separately.

- [ ] **Step 4: Add failing location/trend/damage tests**

```python
def test_location_and_damage_take_the_strongest_item_instead_of_accumulating():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(
        close_between_boll_lower_mid=True,
        close_cross_boll_mid_up=True,
        close_near_ma20=True,
        downside_continuation=True,
        close_below_falling_ma10=True,
        close_below_ma20=True,
        close_below_boll_mid=True,
        fell_back_inside_boll=True,
    ))
    assert score["location_score"] == 10
    assert score["trend_score"] == 0
    assert score["sell_damage_score"] == 20


def test_trend_is_additive_only_inside_its_twenty_point_cap():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(
        ma5_gt_ma10=True,
        ma10_gt_ma20=True,
        ma20_slope_non_negative=True,
        close_gt_ma60=True,
    ))
    assert score["trend_score"] == 20
```

Run those two node IDs and verify they fail before implementing price location as `max(10,8,7)`, trend as `6+6+5+3`, and damage as `max(20,18,15,12,8)`.

- [ ] **Step 5: Add failing buy eligibility, conflict, and ranking tests**

```python
def test_buy_requires_all_three_dimension_floors_and_total_forty():
    module = _candidate_module()
    eligible = module.score_snapshot(_eligible_snapshot())
    assert eligible["buy_score"] == 40
    assert eligible["buy_macd_confirmation_score"] == 0
    assert module.is_dimension_capped_buy_candidate(eligible, held_codes=set())

    for field, value in (
        ("reversal_score", 11),
        ("location_score", 6),
        ("trend_score", 5),
        ("buy_score", 39),
    ):
        rejected = dict(eligible, **{field: value})
        assert not module.is_dimension_capped_buy_candidate(rejected, held_codes=set())


def test_buy_hard_blocks_chasing_downside_weak_repair_and_sell_conflict():
    module = _candidate_module()
    base = module.score_snapshot(_eligible_snapshot())
    cases = [
        dict(base, close_far_above_ma20=True),
        dict(base, downside_continuation=True),
        dict(base, weak_repair_blocked=True),
        dict(base, buy_allowed=False),
        dict(base, sell_weakness_score=10, sell_damage_score=14, sell_score=24),
        dict(base, sell_weakness_score=6, sell_damage_score=18, sell_score=24),
        dict(base, code="513100"),
    ]
    for index, item in enumerate(cases):
        held = {"513100"} if index == len(cases) - 1 else set()
        assert not module.is_dimension_capped_buy_candidate(item, held)


def test_ranking_uses_only_the_frozen_keys():
    module = _candidate_module()
    items = [
        dict(code="513100", buy_score=40, location_score=8, reversal_score=13, volume_rank_score=0),
        dict(code="159915", buy_score=40, location_score=8, reversal_score=13, volume_rank_score=6),
        dict(code="510300", buy_score=40, location_score=10, reversal_score=12, volume_rank_score=0),
        dict(code="513050", buy_score=41, location_score=7, reversal_score=12, volume_rank_score=0),
    ]
    assert [item["code"] for item in module.sort_dimension_capped_candidates(items)] == [
        "513050", "510300", "159915", "513100"
    ]
```

Implement `weak_repair_blocked` by evaluating the official `strategy.is_blocked_entry_combo` against the untouched base snapshot before overwriting candidate scores. For A-share codes use the existing base `volume_score` as `volume_rank_score`; for every other code force it to zero. Do not use volume in `buy_score` or eligibility.

- [ ] **Step 6: Add failing soft/severe sell tests**

```python
def test_soft_sell_can_be_protected_but_severe_damage_cannot():
    module = _candidate_module()
    strong_adx = dict(adx=30.0, plus_di=35.0, minus_di=10.0, ma20_slope_non_negative=True)
    soft = dict(strong_adx, sell_weakness_score=12, sell_damage_score=12, sell_score=24)
    severe = dict(strong_adx, sell_weakness_score=6, sell_damage_score=18, sell_score=24)
    assert not module.should_dimension_capped_signal_sell(soft)
    assert module.should_dimension_capped_signal_sell(severe)


def test_high_k_without_price_damage_does_not_sell():
    module = _candidate_module()
    score = module.score_snapshot(_snapshot(k=85.0))
    assert score["sell_weakness_score"] == 8
    assert score["sell_damage_score"] == 0
    assert not module.should_dimension_capped_signal_sell(score)
```

Implement the severe branch first: `damage>=18 and weakness>=6` returns true without ADX. The ordinary branch requires `weakness>=10`, `damage>=8`, and total `>=24`; when damage is below 18 it returns false if `strategy.is_strong_adx_uptrend(score)` is true. ATR is not part of this function.

- [ ] **Step 7: Prove metadata isolation and run the focused suite**

Add these tests to prove defensive copying and current-state-only KDJ behavior:

```python
def test_adapter_preserves_t_minus_one_metadata_and_source_snapshot():
    source = _snapshot(nested={"values": [1]})
    original = deepcopy(source)
    adapter = _candidate_module().DimensionCappedScoreAdapter(StaticAdapter(source))
    first, reason = adapter.score("513100", "2019-01-08", return_reason=True)
    first["nested"]["values"].append(2)
    second = adapter.score("513100", "2019-01-08")
    assert reason is None
    assert source == original
    assert second["nested"] == {"values": [1]}
    assert second["signal_date"] == "2019-01-07"
    assert second["max_data_date"] == "2019-01-07"


def test_missing_or_nonfinite_k_contributes_zero_and_is_audited():
    module = _candidate_module()
    for value in (None, float("nan"), float("inf")):
        score = module.score_snapshot(_snapshot(k=value))
        assert score["buy_kdj_state_score"] == 0
        assert score["sell_kdj_state_score"] == 0
        assert "invalid_k" in score["candidate_input_warnings"]


def test_kdj_state_uses_current_t_minus_one_only_without_retention():
    previous = _candidate_module().score_snapshot(_snapshot(k=19.0))
    current = _candidate_module().score_snapshot(_snapshot(k=50.0))
    assert previous["buy_kdj_state_score"] == 10
    assert current["buy_kdj_state_score"] == 0
    assert current["sell_kdj_state_score"] == 0
```

The invalid-K warning is diagnostic only and cannot add favorable points.

Then run:

```powershell
pytest -q tests/test_cross_signal_dimension_capped_score_candidate.py tests/test_cross_signal_strategy.py
```

Expected: PASS; the formal strategy regression remains unchanged.

- [ ] **Step 8: Commit the pure scoring milestone**

```powershell
git add -- cross_signal_strategy/research/dimension_capped_score_candidate.py tests/test_cross_signal_dimension_capped_score_candidate.py
git commit -m "feat(cross-signal): add capped score candidate"
```

---

### Task 3: Implement the isolated candidate order planner

**Files:**
- Create: `tests/test_cross_signal_dimension_capped_training_ab.py`
- Create: `cross_signal_strategy/research/dimension_capped_training_ab.py`

**Interfaces:**
- Consumes: `DimensionCappedScoreAdapter`, the pure candidate filters/sorter/sell decider, and inherited ATR/state helpers from `LocalCrossSignalOrderPlanner`.
- Produces: `DimensionCappedOrderPlanner.plan_orders` with the same callback signature required by `LocalBacktestEngine.run`.

Start the test file with these concrete helpers:

```python
from copy import deepcopy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cross_signal_strategy.local.local_backtester import LocalBroker, Position


class FakeSignalAdapter:
    def __init__(self, scores):
        self.scores = scores

    def score(self, code, current_date, return_reason=False):
        value = deepcopy(self.scores.get(code))
        reason = None if value is not None else "no_data"
        return (value, reason) if return_reason else value


def _training_module():
    from cross_signal_strategy.research import dimension_capped_training_ab
    return dimension_capped_training_ab


def _candidate_score(code, **overrides):
    values = {
        "code": code,
        "buy_allowed": True,
        "buy_score": 40.0,
        "reversal_score": 18.0,
        "location_score": 10.0,
        "trend_score": 12.0,
        "volume_rank_score": 0.0,
        "sell_score": 0.0,
        "sell_weakness_score": 0.0,
        "sell_damage_score": 0.0,
        "close_far_above_ma20": False,
        "downside_continuation": False,
        "weak_repair_blocked": False,
        "adx": 10.0,
        "plus_di": 20.0,
        "minus_di": 10.0,
        "ma20_slope_non_negative": True,
        "atr": 0.1,
    }
    values.update(overrides)
    return values


def _six_trade_dates():
    return [
        "2019-07-01", "2019-07-02", "2019-07-03",
        "2019-07-04", "2019-07-05", "2019-07-08",
    ]


def _held_severe_sell_fixture(buy_date):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score(
                "510300", buy_score=10.0,
                sell_score=24.0, sell_weakness_score=6.0, sell_damage_score=18.0,
            )
        }),
        etf_pool=["510300"],
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)
    return planner, broker


def _held_atr_stop_fixture(buy_date):
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score("510300", buy_score=44.0),
            "159915": _candidate_score("159915", buy_score=42.0),
        }),
        etf_pool=["510300", "159915"],
        buy_dates={"510300": buy_date},
        trade_dates=_six_trade_dates(),
    )
    planner.highest_since_buy["510300"] = 10.0
    planner.entry_atr["510300"] = 1.0
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 9.0)
    return planner, broker
```

- [ ] **Step 1: Write failing buy/sell orchestration tests**

Use `FakeSignalAdapter`, `LocalBroker`, and `Position` fixtures patterned after `tests/test_cross_signal_local_order_planner.py`. Add:

```python
def test_candidate_planner_sells_first_then_buys_ranked_empty_slots():
    module = _training_module()
    planner = module.DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "510300": _candidate_score("510300", buy_score=10, sell_weakness_score=10, sell_damage_score=14),
            "513100": _candidate_score("513100", buy_score=44, location_score=10),
            "159915": _candidate_score("159915", buy_score=42, location_score=8, volume_rank_score=6),
        }),
        etf_pool=["510300", "513100", "159915"],
        buy_dates={"510300": "2019-06-20"},
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=12000.0)
    broker.positions["510300"] = Position("510300", 1000, 3.0)

    orders = planner.plan_orders("2019-07-08", "2019-07-05", broker)
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "dimension_capped_signal_sell"}
    assert [item["code"] for item in orders[1:]] == ["513100", "159915"]
```

Add a second test proving a QDII and an A-share candidate with equal core scores both receive the same `total_value * 0.95 / 3` target when the ATR-stress scale is 1.0, regardless of `volume_rank_score`.

```python
def test_candidate_target_is_equal_weight_and_volume_only_breaks_rank_ties():
    planner = _training_module().DimensionCappedOrderPlanner(
        FakeSignalAdapter({
            "159915": _candidate_score("159915", volume_rank_score=10.0),
            "513100": _candidate_score("513100", volume_rank_score=0.0),
        }),
        etf_pool=["159915", "513100"],
        trade_dates=_six_trade_dates(),
    )
    broker = LocalBroker(initial_cash=20000.0)
    orders = planner.plan_orders("2019-07-01", None, broker)
    assert [item["code"] for item in orders] == ["159915", "513100"]
    assert orders[0]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
    assert orders[1]["target_value"] == pytest.approx(20000.0 * 0.95 / 3)
```

- [ ] **Step 2: Run the planner test and observe the missing-module failure**

```powershell
pytest -q tests/test_cross_signal_dimension_capped_training_ab.py
```

Expected: FAIL with `ModuleNotFoundError` for `dimension_capped_training_ab`.

- [ ] **Step 3: Implement the planner without modifying shared local code**

Subclass `LocalCrossSignalOrderPlanner`, but override `_score_pool`, `_should_force_signal_sell`, and `plan_orders`. Reuse inherited `_atr_stop_codes`, `_total_value`, `_portfolio_atr_stress_buy_scale`, `on_orders_filled`, and `on_after_close`. The candidate buy target is:

```python
def _candidate_target_value(self, broker, current_prices, current_date):
    equal_weight = (
        self._total_value(broker, current_prices)
        * float(self.params["base_ratio"])
        / int(self.params["max_hold"])
    )
    return equal_weight * self._portfolio_atr_stress_buy_scale(current_date)
```

Use reason strings `atr_stop`, `dimension_capped_signal_sell`, and `dimension_capped_buy`. Preserve the same-day ATR stop exclusion and available-slot calculation. Never call `strategy.filter_buy_candidates` or `strategy.calc_buy_target_value`, because both encode v0.3.3 scoring/sizing behavior.

- [ ] **Step 4: Add failing five-day and ATR-priority tests**

```python
def test_candidate_signal_sell_waits_five_trading_days():
    planner, broker = _held_severe_sell_fixture(buy_date="2019-07-01")
    assert planner.plan_orders("2019-07-05", "2019-07-04", broker) == []
    assert planner.plan_orders("2019-07-08", "2019-07-05", broker)[0]["reason"] == "dimension_capped_signal_sell"


def test_candidate_atr_stop_ignores_five_day_signal_hold_and_blocks_same_day_rebuy():
    planner, broker = _held_atr_stop_fixture(buy_date="2019-07-01")
    orders = planner.plan_orders(
        "2019-07-02", "2019-07-01", broker,
        current_prices={"510300": 8.0, "159915": 4.0},
    )
    assert orders[0] == {"code": "510300", "target_value": 0.0, "reason": "atr_stop"}
    assert "510300" not in [item["code"] for item in orders[1:]]
```

Run the two nodes, observe failure, then enforce `min_signal_hold_days=5` through the frozen candidate configuration while leaving ATR checks before signal holds.

- [ ] **Step 5: Prove formal planner parity remains intact**

```powershell
pytest -q tests/test_cross_signal_dimension_capped_training_ab.py tests/test_cross_signal_local_order_planner.py
```

Expected: PASS; no shared planner file is changed.

- [ ] **Step 6: Commit the candidate planner milestone**

```powershell
git add -- cross_signal_strategy/research/dimension_capped_training_ab.py tests/test_cross_signal_dimension_capped_training_ab.py
git commit -m "feat(cross-signal): plan capped score orders"
```

---

### Task 4: Freeze the A/B configuration, metrics, and gate

**Files:**
- Modify: `cross_signal_strategy/research/dimension_capped_training_ab.py`
- Modify: `tests/test_cross_signal_dimension_capped_training_ab.py`

**Interfaces:**
- Consumes: approved loader/warm-up roots, `PrecomputedSignalAdapter`, `LocalBacktestEngine`, `build_baseline_report`, and the planner from Task 3.
- Produces: `DimensionCappedTrainingConfig`, `DimensionCappedPerformance`, `DimensionCappedGateDecision`, `DimensionCappedComparisonReport`, and `run_dimension_capped_training_ab`.

Use these exact immutable result shapes so replay, gate, formatter, and closure tests share one vocabulary:

```python
@dataclass(frozen=True)
class DimensionCappedPerformance:
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    win_rate: float
    profit_loss_ratio: float | None
    buy_count: int
    sell_count: int
    closed_trade_count: int
    annual_returns: dict[int, float]


@dataclass(frozen=True)
class DimensionCappedGateInputs:
    baseline: DimensionCappedPerformance
    candidate: DimensionCappedPerformance
    baseline_double_friction: DimensionCappedPerformance
    candidate_double_friction: DimensionCappedPerformance
    changed_order_days: int
    changed_days_by_year: dict[int, int]


@dataclass(frozen=True)
class DimensionCappedGateDecision:
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DimensionCappedDecisionAudit:
    decision_date: str
    signal_date: str
    max_data_date: str
    code: str
    held: bool
    buy_reversal: float
    buy_location: float
    buy_trend: float
    volume_rank: float
    buy_total: float
    sell_weakness: float
    sell_damage: float
    sell_total: float
    kdj_tier: str
    macd_confirmation: str
    raw_contributions: tuple[tuple[str, float], ...]
    adx_protected: bool
    atr_stop: bool
    min_hold_blocked: bool
    hard_block_reasons: tuple[str, ...]
    order_reason: str | None


@dataclass(frozen=True)
class DimensionCappedComparisonReport:
    config: DimensionCappedTrainingConfig
    inputs: DimensionCappedGateInputs
    gate: DimensionCappedGateDecision
    decision_audits: tuple[DimensionCappedDecisionAudit, ...]
```

`DimensionCappedOrderPlanner` owns a `decision_audits` list and appends one row per scored ETF/decision before orders are returned. Each row must expose the raw contribution fields already stored by `score_snapshot`; it must not query any future price path, MFE/MAE, post-sell return, or gate result.

- [ ] **Step 1: Write the failing frozen-configuration test**

```python
def test_dimension_capped_training_configuration_is_exact():
    config = _training_module().dimension_capped_training_config()
    assert config.candidate_name == "cross-v0.4.0-dimension-capped-candidate"
    assert config.training_start == "2019-01-01"
    assert config.training_end == "2021-12-31"
    assert config.initial_cash == pytest.approx(20000.0)
    assert config.execution_time == "09:35"
    assert config.buy_threshold == pytest.approx(40.0)
    assert config.ordinary_sell_threshold == pytest.approx(24.0)
    assert config.min_signal_hold_days == 5
    assert config.max_hold == 3
    assert config.base_ratio == pytest.approx(0.95)
    assert config.candidate_variants == 1
    assert config.training_root == pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021")
    assert config.warmup_root == pathlib.Path(r"G:\financial\history_data\cross_signal_warmup_2018")
```

Run the node and observe failure before adding the frozen dataclass and constructor.

- [ ] **Step 2: Write failing tests for every adoption gate**

Add the following complete helpers to the test file:

```python
from dataclasses import replace


def _performance(**overrides):
    module = _training_module()
    values = {
        "total_return": 1.0,
        "annualized_return": 0.25,
        "max_drawdown": 0.10,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 3.0,
        "win_rate": 0.55,
        "profit_loss_ratio": 4.0,
        "buy_count": 100,
        "sell_count": 100,
        "closed_trade_count": 100,
        "annual_returns": {2019: 0.20, 2020: 0.30, 2021: 0.15},
    }
    values.update(overrides)
    return module.DimensionCappedPerformance(**values)


def _passing_inputs():
    module = _training_module()
    return module.DimensionCappedGateInputs(
        baseline=_performance(),
        candidate=_performance(
            total_return=0.96,
            annualized_return=0.24,
            max_drawdown=0.104,
            sharpe_ratio=1.90,
            sortino_ratio=2.85,
            win_rate=0.56,
            profit_loss_ratio=3.80,
            closed_trade_count=80,
            annual_returns={2019: 0.19, 2020: 0.29, 2021: 0.14},
        ),
        baseline_double_friction=_performance(total_return=0.80, win_rate=0.50),
        candidate_double_friction=_performance(total_return=0.76, win_rate=0.50),
        changed_order_days=10,
        changed_days_by_year={2019: 4, 2020: 3, 2021: 3},
    )


def _failed_inputs(name):
    item = _passing_inputs()
    if name == "changed_total_9":
        return replace(item, changed_order_days=9, changed_days_by_year={2019: 3, 2020: 3, 2021: 3})
    if name == "changed_2019_1":
        return replace(item, changed_order_days=11, changed_days_by_year={2019: 1, 2020: 5, 2021: 5})
    if name == "closed_trade_79pct":
        return replace(item, candidate=replace(item.candidate, closed_trade_count=79))
    if name == "win_rate_equal":
        return replace(item, candidate=replace(item.candidate, win_rate=0.55))
    if name == "return_949pct":
        return replace(item, candidate=replace(item.candidate, total_return=0.949))
    if name == "drawdown_plus_051pp":
        return replace(item, candidate=replace(item.candidate, max_drawdown=0.1051))
    if name == "sharpe_949pct":
        return replace(item, candidate=replace(item.candidate, sharpe_ratio=1.898))
    if name == "sortino_949pct":
        return replace(item, candidate=replace(item.candidate, sortino_ratio=2.847))
    if name == "pl_949pct":
        return replace(item, candidate=replace(item.candidate, profit_loss_ratio=3.796))
    if name == "positive_year_to_zero":
        return replace(item, candidate=replace(
            item.candidate, annual_returns={2019: 0.19, 2020: 0.29, 2021: 0.0}
        ))
    if name == "x2_return_949pct":
        return replace(item, candidate_double_friction=replace(
            item.candidate_double_friction, total_return=0.759
        ))
    if name == "x2_win_lower":
        return replace(item, candidate_double_friction=replace(
            item.candidate_double_friction, win_rate=0.499
        ))
    raise AssertionError("unknown mutation: %s" % name)
```

Test the all-pass case, then parameterize these exact failures:

```python
@pytest.mark.parametrize("mutation, reason", [
    ("changed_total_9", "fewer than 10 changed filled-order days"),
    ("changed_2019_1", "2019 has fewer than 2 changed filled-order days"),
    ("closed_trade_79pct", "candidate retains fewer than 80% of closed trades"),
    ("win_rate_equal", "candidate win rate does not strictly improve"),
    ("return_949pct", "candidate retains less than 95% of baseline return"),
    ("drawdown_plus_051pp", "candidate maximum drawdown worsens by more than 0.5pp"),
    ("sharpe_949pct", "candidate Sharpe ratio retains less than 95%"),
    ("sortino_949pct", "candidate Sortino ratio retains less than 95%"),
    ("pl_949pct", "candidate profit/loss ratio retains less than 95%"),
    ("positive_year_to_zero", "a positive baseline year turns non-positive"),
    ("x2_return_949pct", "doubled-friction return retains less than 95%"),
    ("x2_win_lower", "doubled-friction win rate is below baseline"),
])
def test_training_gate_rejects_each_frozen_failure(mutation, reason):
    comparison = _failed_inputs(mutation)
    decision = _training_module().evaluate_dimension_capped_gate(comparison)
    assert not decision.passed
    assert reason in decision.reasons
```

Missing Sharpe, Sortino, or profit/loss ratio when the baseline has a value must fail with an explicit `metric is missing` reason. If both baseline and candidate lack a ratio because the sample cannot define it, report `not_applicable` and do not silently count it as an improvement.

- [ ] **Step 3: Implement deterministic replay and performance extraction**

Build one official cached score source for all nine ETF codes and all approved training dates. Run four independent engines/planners:

1. v0.3.3 nominal;
2. v0.4 nominal;
3. v0.3.3 with commission `0.0006`, minimum commission `10.0`, slippage `0.002`;
4. v0.4 with the same doubled friction.

Before loading data, assert the resolved roots equal the approved constants and every trade date is within the fixed training window. Candidate params may change only `buy_threshold=40`, `sell_threshold=24`, and `min_signal_hold_days=5`; all other pool/risk values come from a defensive copy of official defaults.

Use `BaselineReport.closed_trade_count`, `win_rate`, `profit_loss_ratio`, `total_return`, `max_drawdown`, `sharpe_ratio`, and `sortino_ratio`. Compute chained calendar-year returns from daily total values. Compute filled-order differences from `(date, code, side, abs(amount_delta))` signatures and require identical date lists before comparison. Exclude order-reason text from the materiality signature so renamed candidate reasons cannot create false changed days.

The public runner must follow this fixed orchestration; the private helpers each accept only the objects shown here and never reload market data:

```python
def run_dimension_capped_training_ab(
    loader=None,
    initial_cash=20000.0,
    warmup_root=APPROVED_WARMUP_ROOT,
):
    loader = loader or CrossSignalTrainingDataLoader()
    _assert_approved_loader(loader)
    _assert_approved_warmup_root(warmup_root)
    trade_dates = get_training_trade_dates(loader)
    _assert_training_dates(trade_dates)

    official_params = dict(strategy.get_default_params())
    candidate_params = dict(official_params)
    candidate_params.update({
        "buy_threshold": 40.0,
        "sell_threshold": 24.0,
        "min_signal_hold_days": 5,
    })
    pool = [code.split(".")[0] for code in strategy.get_default_etf_pool()]
    official_source = build_training_signal_adapter(loader, warmup_root=warmup_root)
    cached = PrecomputedSignalAdapter.from_source(
        official_source, trade_dates=trade_dates, codes=pool
    )

    baseline_days, baseline_planner = _run_arm(
        loader, cached, LocalCrossSignalOrderPlanner,
        official_params, pool, trade_dates, initial_cash, None,
    )
    candidate_source = DimensionCappedScoreAdapter(cached)
    candidate_days, candidate_planner = _run_arm(
        loader, candidate_source, DimensionCappedOrderPlanner,
        candidate_params, pool, trade_dates, initial_cash, None,
    )
    baseline_x2_days, _ = _run_arm(
        loader, cached, LocalCrossSignalOrderPlanner,
        official_params, pool, trade_dates, initial_cash, DOUBLE_FRICTION,
    )
    candidate_x2_days, _ = _run_arm(
        loader, candidate_source, DimensionCappedOrderPlanner,
        candidate_params, pool, trade_dates, initial_cash, DOUBLE_FRICTION,
    )

    inputs = _build_gate_inputs(
        baseline_days, candidate_days, baseline_x2_days, candidate_x2_days,
        initial_cash,
    )
    gate = evaluate_dimension_capped_gate(inputs)
    return DimensionCappedComparisonReport(
        config=dimension_capped_training_config(),
        inputs=inputs,
        gate=gate,
        decision_audits=tuple(candidate_planner.decision_audits),
    )
```

- [ ] **Step 4: Add a T-1 and immutable-root integration test**

```python
def test_runner_rejects_unapproved_root_and_preserves_t_minus_one_metadata():
    module = _training_module()
    with pytest.raises(ValueError, match="approved training data root"):
        module.run_dimension_capped_training_ab(
            loader=SimpleNamespace(root="G:/unapproved/training")
        )

    adapter = module.DimensionCappedScoreAdapter(_real_local_adapter())
    score, reason = adapter.score("510300", "2019-07-01", return_reason=True)
    assert reason is None
    assert score["signal_date"] == "2019-06-28"
    assert score["max_data_date"] == "2019-06-28"
```

- [ ] **Step 5: Add deterministic report formatting tests**

Construct one failed and one passing `DimensionCappedComparisonReport`. Assert the formatted report includes candidate identity, hypothesis, all four metric rows, 2019/2020/2021 returns, changed days by year, closed-trade retention, every gate reason, future-function audit, `authority=local_screen_only`, and exactly one terminal action: `STOP` or `ELIGIBLE_FOR_JOINQUANT_PLAN`.

Also add an audit test with one accepted buy, one soft sell protected by ADX, one severe sell, and one ATR stop. Assert each rendered audit line contains `decision_date`, `signal_date`, `max_data_date`, code, all five dimension totals, every named raw contribution, KDJ tier, MACD confirmation, ADX status, minimum-hold status, hard-block reasons, and order reason. This is causal decision evidence only; no post-trade outcome may appear in the audit object.

Use this concrete report fixture and terminal-action test:

```python
def _comparison_report(passed):
    module = _training_module()
    audit = module.DimensionCappedDecisionAudit(
        decision_date="2019-07-01",
        signal_date="2019-06-28",
        max_data_date="2019-06-28",
        code="510300",
        held=False,
        buy_reversal=18.0,
        buy_location=10.0,
        buy_trend=12.0,
        volume_rank=6.0,
        buy_total=40.0,
        sell_weakness=0.0,
        sell_damage=0.0,
        sell_total=0.0,
        kdj_tier="neutral",
        macd_confirmation="none",
        raw_contributions=(("buy_rsi_group", 12.0), ("buy_kdj_group", 6.0)),
        adx_protected=False,
        atr_stop=False,
        min_hold_blocked=False,
        hard_block_reasons=(),
        order_reason="dimension_capped_buy",
    )
    reasons = () if passed else ("candidate win rate does not strictly improve",)
    return module.DimensionCappedComparisonReport(
        config=module.dimension_capped_training_config(),
        inputs=_passing_inputs(),
        gate=module.DimensionCappedGateDecision(passed, reasons),
        decision_audits=(audit,),
    )


def test_formatter_emits_one_terminal_action_and_causal_audit():
    module = _training_module()
    passed = module.format_dimension_capped_comparison(_comparison_report(True))
    failed = module.format_dimension_capped_comparison(_comparison_report(False))
    assert "ELIGIBLE_FOR_JOINQUANT_PLAN" in passed
    assert "STOP" not in passed
    assert "STOP" in failed
    assert "ELIGIBLE_FOR_JOINQUANT_PLAN" not in failed
    for token in (
        "2019", "2020", "2021", "BASELINE", "CANDIDATE",
        "BASELINE_X2_FRICTION", "CANDIDATE_X2_FRICTION",
        "2019-06-28", "510300", "buy_rsi_group", "dimension_capped_buy",
        "authority=local_screen_only",
    ):
        assert token in passed
```

- [ ] **Step 6: Run focused integration and regression suites**

```powershell
pytest -q tests/test_cross_signal_dimension_capped_score_candidate.py tests/test_cross_signal_dimension_capped_training_ab.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_baseline_report.py
```

Expected: PASS before any empirical result is inspected.

- [ ] **Step 7: Commit the frozen replay/gate milestone**

```powershell
git add -- cross_signal_strategy/research/dimension_capped_training_ab.py tests/test_cross_signal_dimension_capped_training_ab.py
git commit -m "feat(cross-signal): gate capped score replay"
```

---

### Task 5: Run the single empirical candidate and close governance

**Files:**
- Modify: `cross_signal_strategy/research/dimension_capped_training_ab.py`
- Modify: `tests/test_cross_signal_dimension_capped_training_ab.py`
- Modify: `tests/test_cross_signal_research_budget.py`
- Modify: `cross_signal_strategy/docs/research_budget.json`
- Modify: `cross_signal_strategy/docs/research_budget.md`
- Modify: `cross_signal_strategy/docs/backtest_notes.md`
- Modify if failed: `cross_signal_strategy/docs/failed_experiments.md`
- Modify if passed: `cross_signal_strategy/docs/decisions.md`
- Create: `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`

**Interfaces:**
- Consumes: the fixed runner and evaluator from Task 4.
- Produces: one immutable training report, an exhausted family record, and a stop/JoinQuant-plan decision.

- [ ] **Step 1: Add the failing CLI contract test**

Use monkeypatch to supply a prebuilt report and a temporary repository report path:

```python
def test_cli_writes_exact_report_once_and_returns_gate_status(tmp_path, monkeypatch):
    module = _training_module()
    calls = []
    passing = _comparison_report(True)

    def fake_run():
        calls.append("run")
        return passing

    monkeypatch.setattr(module, "run_dimension_capped_training_ab", fake_run)
    output = tmp_path / "report.md"
    assert module.main(report_path=output) == 0
    assert calls == ["run"]
    assert output.read_text(encoding="utf-8") == (
        module.format_dimension_capped_comparison(passing) + "\n"
    )

    monkeypatch.setattr(
        module, "run_dimension_capped_training_ab", lambda: _comparison_report(False)
    )
    assert module.main(report_path=tmp_path / "failed.md") == 1


def test_report_writer_rejects_both_immutable_data_roots():
    module = _training_module()
    for root in (
        pathlib.Path(r"G:\financial\history_data\cross_signal_train_2019_2021"),
        pathlib.Path(r"G:\financial\history_data\cross_signal_warmup_2018"),
    ):
        with pytest.raises(ValueError, match="immutable data root"):
            module.write_report_text(root / "forbidden.md", "text")
```

- [ ] **Step 2: Implement the deterministic CLI**

`python -m cross_signal_strategy.research.dimension_capped_training_ab` must run the fixed configuration once, print the complete report, write identical UTF-8 text to `cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md`, and exit `0` only if every local gate passes. Exit `1` means `STOP`, not a runtime error.

- [ ] **Step 3: Run all focused tests before consuming the research slot**

```powershell
pytest -q tests/test_cross_signal_dimension_capped_score_candidate.py tests/test_cross_signal_dimension_capped_training_ab.py tests/test_cross_signal_research_budget.py tests/test_cross_signal_local_signal_adapter.py tests/test_cross_signal_local_backtester.py tests/test_cross_signal_local_order_planner.py tests/test_cross_signal_baseline_report.py
```

Expected: PASS. Fix only implementation defects that violate the already frozen tests; do not change economic rules.

- [ ] **Step 4: Execute the one fixed empirical A/B**

```powershell
python -m cross_signal_strategy.research.dimension_capped_training_ab
```

Preserve stdout and the generated report. Do not alter scores, thresholds, filters, ordering, pool, years, friction, or gate after reading the result.

- [ ] **Step 5: Add a failing closure test using the emitted result**

Append a test that requires `max_total_open_experiments == 0`, family status `exhausted`, budget zero, no `planned_experiment`, and exact JSON fields for `candidate_gate_passed`, nominal baseline/candidate metrics, doubled-friction metrics, annual returns, changed days, closed trades, gate reasons, `joinquant_candidate_created: false`, `validation_influence: none`, and `prohibit_alternatives: true`. Run this node and verify it fails while the family remains open.

- [ ] **Step 6: Close the research family with exact evidence**

Set top-level open budget back to zero and consume the family. In both result branches, remove `planned_experiment`, set `status: exhausted`, `max_new_experiments: 0`, and copy numeric values directly from the structured report without rounding beyond the formatter.

If the gate failed:

- set `candidate_gate_passed: false` and `joinquant_candidate_created: false`;
- append one uniquely named v0.4 entry to `failed_experiments.md` with every gate reason;
- increment `expected_failed_experiment_count` from `77` to `78` exactly once;
- record `STOP` in `backtest_notes.md` and the readable budget map.

If the gate passed:

- set `candidate_gate_passed: true` and `joinquant_candidate_created: false`;
- leave `expected_failed_experiment_count` at `77`;
- append the local qualification to `decisions.md`, explicitly saying local returns are not authoritative;
- record `ELIGIBLE_FOR_JOINQUANT_PLAN` in `backtest_notes.md` and the readable budget map.

- [ ] **Step 7: Run governance, focused, and full regression suites**

```powershell
pytest -q tests/test_cross_signal_research_budget.py tests/test_cross_signal_dimension_capped_score_candidate.py tests/test_cross_signal_dimension_capped_training_ab.py
pytest -q
```

Expected: all assertions PASS. Report permission warnings from old pytest/cache directories separately; do not call them strategy failures.

- [ ] **Step 8: Verify strict scope and formal-file immutability**

```powershell
git diff --check
git status --short
git diff --exit-code HEAD -- cross_signal_strategy/smart_trade_joinquant_cross_signal_etf.py cross_signal_strategy/smart_trade_ptrade_cross_signal_etf.py
git diff --name-only HEAD
```

Expected: the formal diff command exits zero; no multi-factor file or market-data root appears; all changed files are named in this plan.

- [ ] **Step 9: Commit the consumed result milestone**

For a failed gate:

```powershell
git add -- cross_signal_strategy/research/dimension_capped_training_ab.py tests/test_cross_signal_dimension_capped_training_ab.py tests/test_cross_signal_research_budget.py cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md cross_signal_strategy/docs/backtest_notes.md cross_signal_strategy/docs/failed_experiments.md
git commit -m "research(cross-signal): reject v0.4 capped score"
```

For a passing gate:

```powershell
git add -- cross_signal_strategy/research/dimension_capped_training_ab.py tests/test_cross_signal_dimension_capped_training_ab.py tests/test_cross_signal_research_budget.py cross_signal_strategy/reports/dimension_capped_score_v04_2019_2021.md cross_signal_strategy/docs/research_budget.json cross_signal_strategy/docs/research_budget.md cross_signal_strategy/docs/backtest_notes.md cross_signal_strategy/docs/decisions.md
git commit -m "research(cross-signal): qualify v0.4 capped score"
```

- [ ] **Step 10: Stop at the local research gate**

If failed, report the exact failed gates and do not generate JoinQuant/PTrade code or read a reserved window. If passed, report that only a separate JoinQuant 2019-2021 authority-candidate plan is authorized; do not create it inside this plan.

## Deferred Behind the Local Gate

- Independent JoinQuant candidate source, source-parity tests, version/log labels, and the authority 2019-2021 run require a new plan after a local pass.
- The 2022-2023, 2024-current, and 2015-2018 reserved windows remain unread until the JoinQuant training rule is frozen and passes.
- PTrade behavior, order lifecycle, state migration, release verification, and any mapping to the existing live 8% IOPV override require separate design approval after all authority gates.
- Formal promotion and live deployment require independent code review, Extra High reasoning review, release verification, and explicit user confirmation.
